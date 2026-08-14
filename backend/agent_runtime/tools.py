"""A3 — 4 个只读 Tool 的实现与注册。

- query_memory_facts：结构化事实（count/exists/first/last/date/group/media）。
- search_memories：检索 kernel 封装（视觉/文本/混合），返回 ResultSet 摘要。
- get_original_photos：当前 ResultSet 原图交付（A4 ResultSetStore 后完整可用）。
- inspect_photo：多模态复核（A0.6 已验证链路），结果 ephemeral 不写长期记忆。

Tool 观察只暴露模型可安全看到的内容；内部 asset_id 通过 handle 映射（A4 完整化）。
"""

from __future__ import annotations

import base64
import json
import os
import re
import threading
import time
from pathlib import Path

from .tool_registry import ToolSpec, register
from .capability import tool_capability_summary
from .intent import ocr_intent, visual_intent
from .ocr_tool import (_read_photo_text, ocr_telemetry_snapshot,
                       record_ocr_telemetry, small_ocr_available)
from .ocr_tool import bind_ocr_runtime

_RUNTIME: dict = {}


def bind_runtime(store, *, gamma=None, embedding_router=None, retrieval_config=None):
    from .result_set import ResultSetStore
    _RUNTIME["store"] = store
    _RUNTIME["gamma"] = gamma
    _RUNTIME["embedding_router"] = embedding_router
    _RUNTIME["retrieval_config"] = retrieval_config
    _RUNTIME["result_sets"] = ResultSetStore(store)
    bind_ocr_runtime(_RUNTIME)


def set_conversation_id(conversation_id):
    """D4：把当前 conversation_id 绑定到 tool 层（search_conversation_history 用）。"""
    _RUNTIME["conversation_id"] = conversation_id


def _kernel():
    from ..evidence_retrieval import EvidenceRetrievalKernel
    if _RUNTIME.get("embedding_router") is not None:
        from ..retrieval import RetrievalConfig, build_default_retrievers
        config = _RUNTIME.get("retrieval_config") or RetrievalConfig()
        retrievers = build_default_retrievers(_RUNTIME["store"], embedding_router=_RUNTIME["embedding_router"], config=config)
        return EvidenceRetrievalKernel(_RUNTIME["store"], retrievers=retrievers,
                                       embedding_router=_RUNTIME["embedding_router"], config=config)
    return EvidenceRetrievalKernel(_RUNTIME["store"])


def _cn_month(text: str) -> int | None:
    """把中文数字月份（十/十月/十一月/十二月）转成阿拉伯数字；阿拉伯数字原样。"""
    digits = {"一": 1, "二": 2, "三": 3, "四": 4, "五": 5,
              "六": 6, "七": 7, "八": 8, "九": 9, "十": 10}
    if text.isdigit():
        return int(text)
    if not text or any(ch not in digits for ch in text):
        return None
    if "十" not in text:
        return digits.get(text)
    tens, _, ones = text.partition("十")
    return (digits.get(tens, 1) if tens else 1) * 10 + digits.get(ones, 0)


def _resolve_time_expression(value: str) -> str | None:
    """把相对时间解析成可被 parse_time_expression 接受的绝对时间（含范围）。

    模型契约：相对时间必须原样传 filters.time，由这里确定性换算；不依赖模型算年份。
    支持：去年/今年/前年/这两年/近两年/最近一年/上个月/去年X月/去年春天等。
    """
    import re
    from .time_context import now
    now = now()
    v = (value or "").strip()
    if not v:
        return None
    y, m = now.year, now.month
    seasons = {
        "春天": (3, 5), "夏天": (6, 8), "秋天": (9, 11), "冬天": (12, 2),
        "春季": (3, 5), "夏季": (6, 8), "秋季": (9, 11), "冬季": (12, 2),
    }
    base_year = {"去年": y - 1, "今年": y, "前年": y - 2}
    month_rel = re.search(r"(去年|今年|前年)\s*(\d{1,2}|[一二三四五六七八九十]+)\s*月", v)
    if month_rel:
        month = _cn_month(month_rel.group(2))
        return f"{base_year[month_rel.group(1)]}年{month}月"
    season_rel = re.search(r"(去年|今年|前年)\s*(春天|夏天|秋天|冬天|春季|夏季|秋季|冬季)", v)
    if season_rel:
        base = base_year[season_rel.group(1)]
        sm, em = seasons[season_rel.group(2)]
        if sm <= em:
            return f"{base}年{sm}月-{base}年{em}月"
        return f"{base}年12月-{base + 1}年2月"
    if "这两年" in v or "近两年" in v or "最近两年" in v:
        return f"{y - 1}年-{y}年"
    if "最近一年" in v or "近一年" in v:
        prev_y, prev_m = (y - 1, m) if m > 1 else (y - 1, 12)
        return f"{prev_y}年{prev_m}月-{y}年{m}月"
    if "上上个月" in v:
        pm2 = (y, m - 2) if m > 2 else (y - 1, m + 10)
        return f"{pm2[0]}年{pm2[1]}月"
    if "上个月" in v:
        pm = (y, m - 1) if m > 1 else (y - 1, 12)
        return f"{pm[0]}年{pm[1]}月"
    if "去年" in v:
        return f"{y - 1}年"
    if "今年" in v:
        return f"{y}年"
    if "前年" in v:
        return f"{y - 2}年"
    if re.fullmatch(r"20\d{2}", v):
        return f"{v}年"
    if re.fullmatch(r"20\d{2}年", v):
        return v
    return v


def _draft_from_filters(filters: dict, *, answer_type="asset_set", group_by=None):
    """把工具 filters 转成 QueryParseDraft（只读 shadow 用，语义与 thin_agent 对齐）。"""
    from ..query_contracts import QueryParseDraft
    draft = QueryParseDraft(intent="answer", answer_target="general",
                            answer_type=answer_type)
    time_expr = _resolve_time_expression((filters or {}).get("time") or "")
    if time_expr:
        draft.time_expression = time_expr
    place = (filters or {}).get("place") or ""
    if place:
        draft.semantic_conditions.append({"dimension": "place", "value": place, "strictness": "semantic_required"})
    person = (filters or {}).get("person") or ""
    if person:
        draft.entity_names.append(person)
    media = (filters or {}).get("media") or ""
    if media:
        draft.media_expressions.append(media)
    query = (filters or {}).get("query") or ""
    if query and answer_type == "asset_set":
        draft.semantic_conditions.append({"dimension": "semantic", "value": query, "strictness": "semantic_required"})
    if group_by:
        draft.structured = {"aggregation": {"op": "group_by", "group_by": group_by}}
    return draft


def _spec_for(draft, scope_id, viewer_id):
    from ..query_contracts import build_query_spec
    return build_query_spec(
        draft, scope_id=scope_id, viewer_id=viewer_id,
        conversation_id="tool_loop", query_id=f"tool_{int(time.time()*1000)}",
        entity_resolver=lambda name: _resolve_entity(name, scope_id),
    )


def _resolve_entity(name, scope_id):
    store = _RUNTIME.get("store")
    if store is None:
        return None
    try:
        for entity in store.list_entities(status="confirmed", scope_id=scope_id or None):
            if entity.get("canonical_name") == name:
                return entity.get("id")
    except Exception:
        pass
    return None


# ---- Tool 1: query_memory_facts ----
_FACT_OPERATIONS = {"count", "exists", "first", "last", "date", "group", "meal"}


def _normalize_fact_arguments(arguments: dict) -> tuple[str, str]:
    """C11：operation/group_by 安全默认（model 只给 group_by 时补 operation=group）。"""
    operation = (arguments.get("operation") or "").strip()
    group_by = (arguments.get("group_by") or "").strip()
    if not operation and group_by:
        operation = "group"
    if operation not in _FACT_OPERATIONS:
        operation = "count"
    if operation == "group" and not group_by:
        group_by = "month"
    return operation, group_by


def _query_memory_facts(arguments: dict, *, context: dict | None = None) -> dict:
    operation, group_by = _normalize_fact_arguments(arguments)
    filters = arguments.get("filters") or {}
    scope_id = (context or {}).get("scope_id") or ""
    viewer_id = (context or {}).get("viewer_id") or "owner"
    if operation == "meal":
        # Phase C C5：饮食/活动聚合（事件级去重 + 食物证据分层）
        return _query_meal_evidence(filters, scope_id=scope_id, viewer_id=viewer_id)
    if operation == "group":
        draft = _draft_from_filters(filters, answer_type="grouped_list", group_by=group_by)
    else:
        answer_type = {
            "count": "count", "exists": "exists", "first": "first_occurrence",
            "last": "last_occurrence", "date": "date", "media": "count",
        }.get(operation, "count")
        draft = _draft_from_filters(filters, answer_type=answer_type)
    from ..structured_memory import StructuredMemoryExecutor
    executor = StructuredMemoryExecutor(_RUNTIME["store"])
    spec = _spec_for(draft, scope_id, viewer_id)
    result = executor.execute(draft, spec)
    out = {
        "operation": operation,
        "answer_type": result.answer_type,
        "value": result.value,
        "total": result.total,
        "rows": result.rows if operation == "group" else None,
        "filters_applied": result.filters_applied,
        "coverage": {"complete": True},
    }
    if operation == "group":
        # 分组结果不附任意样本：随机照片与分组内容不匹配会造成误导（如城市分组展示无关照片）。
        out["samples"] = []
    else:
        try:
            out["samples"] = executor._sample_observations(draft, spec, limit=3)
        except Exception:
            out["samples"] = []
    if operation == "group" and group_by == "place" and isinstance(out["rows"], list) and len(out["rows"]) > 12:
        # 地点分组只给模型前 12 个，避免超长 rows 干扰 12B 输出（month 分组本身 ≤12 不截断）
        out["rows_truncated"] = len(out["rows"])
        out["rows"] = out["rows"][:12]
    if operation == "group" and group_by == "place":
        # Phase C C4：地点聚合必须带 coverage（多少张有/没有可靠地点信息）
        rows = result.rows or []
        known = sum(r["count"] for r in rows if str(r.get("group") or "") != "未知")
        unknown = sum(r["count"] for r in rows if str(r.get("group") or "") == "未知")
        total_assets = sum(r["count"] for r in rows)
        out["coverage"] = {
            "complete": True,
            "known_location_assets": known,
            "unknown_location_assets": unknown,
            "total_assets": total_assets,
            "disclosure": ("还有部分照片没有可靠的地点信息。" if unknown else
                           "全部相关照片都有地点信息。"),
        }
    return out


# ---- Phase C C5：饮食 / 活动证据聚合 ----

_MEAL_ACTIVITY = (
    "吃|餐|饭|聚餐|火锅|烧烤|早餐|午餐|晚餐|夜宵|宴|宴请|下厨|做饭|煮|炒|煎|蒸|烤|"
    "dining|dinner|lunch|breakfast|eating|meal|bbq|hotpot|cook|cooking|party"
)
_MEAL_ACTIVITY_RE = None


def _meal_activity_re():
    global _MEAL_ACTIVITY_RE
    if _MEAL_ACTIVITY_RE is None:
        import re as _re
        _MEAL_ACTIVITY_RE = _re.compile(r"(" + _MEAL_ACTIVITY + r")", _re.I)
    return _MEAL_ACTIVITY_RE


def _query_meal_evidence(filters: dict, *, scope_id="home-default", viewer_id="owner") -> dict:
    """饮食/活动聚合（简化版）。

    食物来自数据层已产出的 objects_json（VLM 物体标签），不再用死代码食物词表
    去匹配 caption/ocr；“是不是用餐场景”由 activity/event_type/caption 判断。
    事件级去重：同一 event 的多张照片只算一次用餐。
    """
    from ..structured_memory import StructuredMemoryExecutor
    store = _RUNTIME.get("store")
    if store is None:
        return {"operation": "meal", "summary": "记忆库不可用。", "total": 0,
                "explicit_foods": [], "coverage": {"complete": False}}
    draft = _draft_from_filters(filters, answer_type="count")
    spec = _spec_for(draft, scope_id, viewer_id)
    executor = StructuredMemoryExecutor(store)
    start, end = executor._time_range(draft, spec)
    food_hint = str((filters or {}).get("food") or "").strip().lower()

    clauses = []
    params = []
    if scope_id:
        clauses.append("a.scope_id = ?")
        params.append(scope_id)
    if start:
        clauses.append("a.captured_at >= ?")
        params.append(start)
    if end:
        clauses.append("a.captured_at < ?")
        params.append(end)
    rows = store._rows(
        "SELECT o.id AS observation_id, o.asset_id, o.activity, o.event_type, o.caption, "
        "o.objects_json, a.captured_at FROM observations o "
        "JOIN assets a ON a.id = o.asset_id WHERE " + " AND ".join(clauses) +
        " ORDER BY a.captured_at", params)

    event_rows = store._rows(
        "SELECT observation_id, event_id FROM event_observations", ())
    obs_to_event = {}
    for row in event_rows:
        obs_to_event.setdefault(row["observation_id"], row["event_id"])

    def _event_key(observation_id):
        return obs_to_event.get(observation_id) or f"obs:{observation_id}"

    foods_by_event: dict[str, set[str]] = {}
    meal_observation_ids: list[str] = []
    meal_events_without_food = 0
    for row in rows:
        activity = str(row["activity"] or "")
        event_type = str(row["event_type"] or "")
        caption = str(row["caption"] or "")
        if not _meal_activity_re().search(" ".join([activity, event_type, caption])):
            continue
        meal_observation_ids.append(row["observation_id"])
        objects = []
        try:
            objects = json.loads(row["objects_json"] or "[]")
        except Exception:
            objects = []
        objs = [str(o).strip() for o in objects if isinstance(o, str) and str(o).strip()]
        if food_hint:
            objs = [o for o in objs if food_hint in o.lower()]
        if objs:
            foods_by_event.setdefault(_event_key(row["observation_id"]), set()).update(objs)
        else:
            meal_events_without_food += 1

    food_counts: dict[str, int] = {}
    for foods in foods_by_event.values():
        for food in foods:
            food_counts[food] = food_counts.get(food, 0) + 1
    top_foods = [{"food": food, "events": count}
                 for food, count in sorted(food_counts.items(), key=lambda kv: (-kv[1], kv[0]))]

    meal_event_keys = {_event_key(oid) for oid in meal_observation_ids}
    meal_samples = []
    for r in rows[:3]:
        meal_samples.append({
            "asset_id": r.get("asset_id"),
            "captured_at": r.get("captured_at"),
            "media_type": "image",
            "caption": (r.get("caption") or "")[:120],
        })
    return {
        "operation": "meal",
        "answer_type": "meal_summary",
        "value": top_foods,
        "total": len(meal_observation_ids),
        "samples": meal_samples,
        "time_range": {"start": start, "end": end} if (start or end) else None,
        "scanned_observations": len(rows),
        "total_meal_observations": len(meal_observation_ids),
        "event_count": len(meal_event_keys),
        "explicit_foods": top_foods[:20],
        "events_without_food_label": meal_events_without_food,
        "filters_applied": {"scope_id": scope_id or None,
                            "time_range": {"start": start, "end": end} if (start or end) else None,
                            "food_hint": food_hint or None},
        "coverage": {
            "complete": True,
            "disclosure": ("其中一部分用餐记录能确认'在吃饭'，但照片物体标签里没有具体菜品。"
                           if meal_events_without_food else
                           "已识别的用餐记录都有明确的食物线索。"),
        },
    }


# ---- Tool 2: search_memories ----
def _even_indices(total: int, n: int) -> list[int]:
    """在 [0, total) 内均匀取 n 个下标（representative 预览用，避免只展示最新几张），包含首尾。"""
    if total <= n:
        return list(range(total))
    if n <= 1:
        return [0]
    return [min(int(round(i * (total - 1) / (n - 1))), total - 1) for i in range(n)]


def _search_metadata_only(draft, spec, scope_id, query, mode, user_goal="") -> dict:
    """空 query 搜索：只按硬筛选（时间/媒体/地点/人物）返回资产，构建 ResultSet 预览。"""
    from ..structured_memory import StructuredMemoryExecutor
    executor = StructuredMemoryExecutor(_RUNTIME["store"])
    assets = executor._matching_assets(draft, spec, limit=500)
    asset_ids = [a["id"] for a in assets]
    rs = _RUNTIME["result_sets"].new(
        scope_id=scope_id, query=query or "(时间/地点筛选)", asset_ids=asset_ids,
        unresolved=[])
    handles = rs.handles()
    _RUNTIME["last_handles"] = handles
    indices = _even_indices(len(asset_ids), 6) if mode == "representative" \
        else list(range(min(6, len(asset_ids))))
    preview = []
    store = _RUNTIME.get("store")
    for i, idx in enumerate(indices):
        a = assets[idx]
        place = ""
        source_video_asset_id = None
        source_timestamp_sec = None
        if store is not None:
            try:
                asset = store.get_asset(a.get("id")) or {}
                place = _short_place_label(asset)
                if asset.get("derived_kind") == "video_keyframe":
                    source_video_asset_id = asset.get("parent_asset_id")
                    source_timestamp_sec = asset.get("source_timestamp_sec")
            except Exception:
                place = ""
        preview.append({
            "handle": f"photo_{idx + 1}",
            "captured_at": a.get("captured_at"),
            "level": "exact",
            "place": place,
            "source_video_asset_id": source_video_asset_id,
            "source_timestamp_sec": source_timestamp_sec,
            "condition_summary": {},
        })
    total = len(assets)
    return {
        "result_set_id": rs.result_set_id,
        "query": query,
        "mode": mode,
        "total": total,
        "asset_ids": asset_ids[:50],
        "evidence_count": len(asset_ids),
        "preview": preview,
        "has_more": total > len(preview),
        "remaining": max(0, total - len(preview)),
        "completeness": "complete",
        "gaps": [],
        "query_satisfaction": "full_support" if total else "no_match",
        "answerability": "full" if total else "none",
        "condition_summary": {},
        "can_inspect": len(preview) > 0,
        "inspect_hint": "preview 里的 handle（photo_1…）可直接用于 inspect_photo 复核视觉细节" if preview else "",
        "recommended_resolution": _recommended_resolution(query, preview,
                                                       "full_support" if total else "no_match",
                                                       user_goal=user_goal),
    }


_TIME_TOKEN_RE = re.compile(r"20\d{2}\s*年(?:\s*[01]?\d|\s*十[一二]?)?\s*月?")
_RELATIVE_TIMES = ("这两年", "近两年", "最近两年", "最近一年", "今年", "去年", "前年",
                   "上上个月", "上个月", "去年春天", "去年夏天", "去年秋天", "去年冬天")


def _extract_time_from_query(query: str) -> str | None:
    """C11：模型把时间写进 query 文本（而非 filters.time）时自动提取。"""
    m = _TIME_TOKEN_RE.search(query or "")
    if m:
        return m.group(0).replace(" ", "")
    # 先匹配更具体的"去年X月"，再退回相对时间词
    m = re.search(r"去年(?:[0-9一二三四五六七八九十]+)月", query or "")
    if m:
        return m.group(0)
    for expr in _RELATIVE_TIMES:
        if expr in (query or ""):
            return expr
    return None


def _search_memories(arguments: dict, *, context: dict | None = None) -> dict:
    query = arguments.get("query") or ""
    mode = arguments.get("mode") or "best"
    if mode not in {"best", "all", "representative"}:
        mode = "best"
    filters = dict(arguments.get("filters") or {})
    if not (filters.get("time") or ""):
        extracted = _extract_time_from_query(query)
        if extracted:
            filters["time"] = extracted
    scope_id = (context or {}).get("scope_id") or ""
    viewer_id = (context or {}).get("viewer_id") or "owner"
    draft = _draft_from_filters({**filters, "query": query}, answer_type="asset_set")
    draft.result_requirement = {"mode": mode}
    spec = _spec_for(draft, scope_id, viewer_id)
    if not (query or "").strip():
        # 纯时间/地点/人物/媒体筛选：走确定性元数据路径，不依赖 ANN 语义召回（生产多检索器下空 query 会 0 召回）
        user_goal = ((context or {}).get("task_state") or {}).get("user_goal") or ""
        return _search_metadata_only(draft, spec, scope_id, query, mode, user_goal=user_goal)
    packet = _kernel().retrieve(spec)
    assets = packet.assets or []
    asset_ids = [item.get("asset_id") for item in assets if item.get("asset_id")]
    rs = _RUNTIME["result_sets"].new(
        scope_id=scope_id, query=query, asset_ids=asset_ids,
        unresolved=[g.get("reason") for g in (packet.gaps or [])],
    )
    preview = []
    handles = rs.handles()
    store = _RUNTIME.get("store")
    for i, item in enumerate(assets[:6]):
        handle = f"photo_{i + 1}"
        place = ""
        source_video_asset_id = None
        source_timestamp_sec = None
        if store is not None:
            try:
                asset = store.get_asset(item.get("asset_id")) or {}
                place = _short_place_label(asset)
                if asset.get("derived_kind") == "video_keyframe":
                    source_video_asset_id = asset.get("parent_asset_id")
                    source_timestamp_sec = asset.get("source_timestamp_sec")
            except Exception:
                place = ""
        preview.append({
            "handle": handle,
            "captured_at": item.get("captured_at"),
            "level": item.get("level"),
            "place": place,
            "source_video_asset_id": source_video_asset_id,
            "source_timestamp_sec": source_timestamp_sec,
            "condition_summary": _condition_summary(item),
        })
    _RUNTIME["last_handles"] = handles
    cond, satisfaction, answerability = _truth_contract(packet, rs.total)
    return {
        "result_set_id": rs.result_set_id,
        "query": query,
        "mode": mode,
        "total": rs.total,
        "asset_ids": asset_ids[:50],
        "evidence_count": len(asset_ids),
        "preview": preview,
        "has_more": len(asset_ids) > len(preview),
        "remaining": max(0, len(asset_ids) - len(preview)),
        "completeness": "complete" if not (packet.gaps) else "partial",
        "gaps": rs.unresolved[:3],
        "query_satisfaction": satisfaction,
        "answerability": answerability,
        "condition_summary": cond,
        "can_inspect": len(preview) > 0,
        "inspect_hint": "preview 里的 handle（photo_1…）可直接用于 inspect_photo 复核视觉细节" if preview else "",
        "retrieval_timing": packet.retrieval_timing,
        "recommended_resolution": _recommended_resolution(
            query, preview, satisfaction,
            user_goal=((context or {}).get("task_state") or {}).get("user_goal") or ""),
    }


def _short_place_label(asset: dict) -> str:
    """从资产反地理编码取短地点标签（'秦皇岛市昌黎县' / 'Chiang Mai'），供 preview 证据展示。"""
    import json as _json
    metadata = asset.get("metadata_json") or {}
    if isinstance(metadata, str):
        try:
            metadata = _json.loads(metadata)
        except (TypeError, ValueError):
            metadata = {}
    geocode = metadata.get("reverse_geocode") or {}
    if isinstance(geocode, str):
        try:
            geocode = _json.loads(geocode)
        except (TypeError, ValueError):
            geocode = {}
    if not isinstance(geocode, dict):
        return ""
    parts = []
    for key in ("city", "district"):
        value = str(geocode.get(key) or "").strip()
        if value and value not in parts:
            parts.append(value)
    if parts:
        return "".join(parts)
    return str(geocode.get("label") or "")


def _condition_summary(item: dict) -> dict:
    out = {}
    for key, cond in (item.get("condition_results") or {}).items():
        label = key.split(":", 1)[-1]
        out[label] = cond.get("status")
    return out


def _recommended_resolution(query: str, preview: list, satisfaction: str,
                            user_goal: str = "") -> dict:
    """Evidence Finder（Phase E §8.2）：告诉 Agent 下一步证据解析方案。

    不替 Agent 做决定，只把"这条检索还需要什么"翻译成工具建议。
    """
    if not preview:
        return {"needed": False, "tool": None,
                "reason": "" if satisfaction == "full_support" else "没有候选可复核"}
    q = f"{query or ''} {user_goal or ''}"
    needs_ocr = ocr_intent(q)
    needs_visual = visual_intent(q)
    if needs_ocr:
        return {"needed": True, "tool": "read_photo_text",
                "reason": "问题需要读取照片中的文字/数字，请用 read_photo_text 复核 preview 里的照片"}
    if needs_visual:
        return {"needed": True, "tool": "inspect_photo",
                "reason": "问题需要查看照片中的视觉细节，请用 inspect_photo 复核 preview 里的照片"}
    return {"needed": False, "tool": None, "reason": ""}


def _truth_contract(packet, total: int) -> tuple[dict, str, str]:
    """确定性计算查询满足度（B2）：基于 condition_results 与 gaps，不交给模型判断。"""
    if total <= 0:
        return {}, "no_match", "none"
    condition_verdict: dict[str, dict] = {}
    for item in (packet.assets or []):
        for key, cond in (item.get("condition_results") or {}).items():
            label = key.split(":", 1)[-1]
            status = cond.get("status") or "unknown"
            bucket = condition_verdict.setdefault(label, {"confirmed": 0, "supported": 0,
                                                          "unknown": 0, "contradicted": 0})
            if status == "matched":
                bucket["confirmed"] += 1
            elif status == "possible":
                bucket["supported"] += 1
            elif status == "contradicted":
                bucket["contradicted"] += 1
            else:
                bucket["unknown"] += 1
    summary = {}
    for label, bucket in condition_verdict.items():
        if bucket["confirmed"] > 0:
            summary[label] = "confirmed"
        elif bucket["supported"] > 0:
            summary[label] = "supported"
        elif bucket["contradicted"] > 0 and bucket["confirmed"] == 0:
            summary[label] = "contradicted"
        else:
            summary[label] = "unknown"
    confirmed = sum(1 for v in summary.values() if v == "confirmed")
    unknown = sum(1 for v in summary.values() if v in {"unknown", "contradicted"})
    if not summary:
        satisfaction = "candidate_only"
    elif unknown == 0:
        satisfaction = "full_support"
    elif confirmed > 0:
        satisfaction = "partial_support"
    else:
        satisfaction = "candidate_only"
    answerability = "full" if satisfaction == "full_support" else ("partial" if confirmed else "limited")
    return summary, satisfaction, answerability


# ---- Tool 3: get_original_photos ----
def _get_original_photos(arguments: dict, *, context: dict | None = None) -> dict:
    task_state = (context or {}).get("task_state") or {}
    result_set_id = arguments.get("result_set_id") or (task_state or {}).get("current_result_set")
    handle = arguments.get("handle") or ""
    rs_store = _RUNTIME.get("result_sets")
    if not result_set_id or rs_store is None:
        return {"summary": "当前没有可交付的结果集。", "delivered": 0, "blocked": ["no_result_set"]}
    rs = rs_store.get(result_set_id)
    if rs is None:
        return {"summary": "结果集不存在。", "delivered": 0, "blocked": ["unknown_result_set"]}
    if rs.scope_id != ((context or {}).get("scope_id") or ""):
        return {"summary": "无权交付该结果集的原图。", "delivered": 0, "blocked": ["scope_mismatch"]}
    asset_id = rs_store.resolve_handle(result_set_id, handle) if handle else None
    if handle and not asset_id:
        return {"summary": "无法解析选中的照片。", "delivered": 0, "blocked": ["bad_handle"]}
    target = handle if asset_id else (rs.asset_ids[0] if rs.asset_ids else None)
    url = ""
    if target:
        url = (f"/api/assistant/result-set/{result_set_id}/photo?handle={target}"
               f"&scope_id={rs.scope_id}&original=1")
    return {
        "summary": f"已从结果集 {result_set_id} 授权原图交付。",
        "result_set_id": result_set_id,
        "handle": handle or "first",
        "delivered": 1 if asset_id else (1 if rs.asset_ids else 0),
        "total": rs.total,
        "scope_id": rs.scope_id,
        "url": url,
    }


def get_result_set_store():
    """B3.2：API 层访问 ResultSetStore 的公开入口（原图授权端点用）。"""
    return _RUNTIME.get("result_sets")


def resolve_handle_asset_id(handle: str, result_set_id: str | None = None,
                            scope_id: str | None = None) -> str | None:
    """D8：从 handle（+结果集）解析 asset_id；API 层 Photo Thread 用。"""
    rs_store = _RUNTIME.get("result_sets")
    if result_set_id and rs_store is not None:
        aid = rs_store.resolve_handle(result_set_id, handle)
        if aid:
            return aid
    return _handle_to_asset_id(handle)


def result_set_context(result_set_id: str, scope_id: str) -> str | None:
    """B3.1：给模型一段当前结果集的续接上下文（不暴露内部 ID 之外的敏感信息）。"""
    rs_store = _RUNTIME.get("result_sets")
    if not rs_store:
        return None
    rs = rs_store.get(result_set_id)
    if rs is None or (scope_id and rs.scope_id != scope_id):
        return None
    shown = min(rs.total, rs.shown or 0)
    return (f"当前结果集：{rs.result_set_id}，共 {rs.total} 张，已显示 {shown} 张，"
            f"还有 {max(0, rs.total - shown)} 张。查看更多用 get_result_page（page 从 1 开始）。")


# ---- Tool 3.5: get_result_page（B3.1 分页）----
def _get_result_page(arguments: dict, *, context: dict | None = None) -> dict:
    scope_id = (context or {}).get("scope_id") or ""
    task_state = (context or {}).get("task_state") or {}
    result_set_id = arguments.get("result_set_id") or task_state.get("current_result_set")
    try:
        page_no = max(1, int(arguments.get("page") or 1))
    except (TypeError, ValueError):
        page_no = 1
    try:
        page_size = min(20, max(1, int(arguments.get("page_size") or 6)))
    except (TypeError, ValueError):
        page_size = 6
    rs_store = _RUNTIME.get("result_sets")
    if not result_set_id or rs_store is None:
        return {"summary": "当前没有可用的结果集。", "total": 0, "blocked": ["no_result_set"]}
    rs = rs_store.get(result_set_id)
    if rs is None:
        return {"summary": "结果集不存在或已过期。", "total": 0, "blocked": ["unknown_result_set"]}
    if scope_id and rs.scope_id != scope_id:
        return {"summary": "无权访问该结果集。", "total": 0, "blocked": ["scope_mismatch"]}
    items = rs.page(page_no, page_size)
    shown = min(rs.total, (page_no - 1) * page_size + len(items))
    return {
        "result_set_id": rs.result_set_id,
        "page": page_no,
        "page_size": page_size,
        "total": rs.total,
        "shown": shown,
        "has_more": shown < rs.total,
        "remaining": max(0, rs.total - shown),
        "preview": [{"handle": h["handle"]} for h in items],
        "query": rs.query,
    }


# ---- Tool 4: inspect_photo ----
def _inspect_photo(arguments: dict, *, context: dict | None = None) -> dict:
    asset_handle = arguments.get("asset_handle") or ""
    question = arguments.get("question") or "请描述这张照片"
    scope_id = (context or {}).get("scope_id") or ""
    task_state = (context or {}).get("task_state") or {}
    if not asset_handle:
        # C11：未填 handle 时用当前结果集 preview 首个可复核 handle（安全默认）
        preview = (task_state.get("result_preview") or []) or []
        if preview:
            asset_handle = preview[0]
    # B3：handle 必须解析自当前结果集；失败再退回最近一次检索的 handle 映射
    asset_id = None
    result_set_id = task_state.get("current_result_set")
    rs_store = _RUNTIME.get("result_sets")
    if result_set_id and rs_store is not None:
        asset_id = rs_store.resolve_handle(result_set_id, asset_handle)
    if not asset_id:
        asset_id = _handle_to_asset_id(asset_handle)
    store = _RUNTIME.get("store")
    if not asset_id or store is None:
        return {"summary": "无法定位照片。", "certainty": "uncertain", "persisted": False,
                "blocked": ["unknown_handle"]}
    row = store.connection.execute(
        "SELECT path, scope_id FROM assets WHERE id = ?", (asset_id,)).fetchone()
    if row and scope_id and row["scope_id"] != scope_id:
        return {"summary": "无法复核该照片（不在当前相册范围）。", "certainty": "uncertain",
                "persisted": False, "blocked": ["scope_mismatch"]}
    if not row or not row["path"] or not Path(row["path"]).is_file():
        return {"summary": "照片文件不可用。", "certainty": "uncertain", "persisted": False,
                "blocked": ["file_unavailable"]}
    gamma = _RUNTIME.get("gamma")
    if gamma is None:
        return {"summary": "模型不可用。", "certainty": "uncertain", "persisted": False}
    model_call_metrics = []
    try:
        encoded, mime_type = gamma.encode_vision_image(row["path"])
        image = {"base64": encoded, "mime_type": mime_type}
        raw = gamma.chat(_INSPECT_PROMPT.format(question=question), images=[image],
                         json_mode=True, role="inspect")
    except Exception as exc:
        model_call_metrics = gamma.get_and_clear_call_metrics()
        return {"summary": f"图片复核失败：{exc}", "certainty": "uncertain", "persisted": False,
                "_model_call_metrics": model_call_metrics}
    model_call_metrics = gamma.get_and_clear_call_metrics()
    try:
        start, end = raw.find("{"), raw.rfind("}")
        parsed = json.loads(raw[start:end + 1]) if start >= 0 else {}
    except Exception:
        parsed = {}
    return {
        "asset_handle": asset_handle,
        "question": question,
        "observation": parsed.get("observation") or parsed.get("scene") or "",
        "certainty": parsed.get("certainty") or "supported",
        "confirms_visual_only": True,
        "source": "runtime_visual_inspection",
        "persisted": False,
        "_model_call_metrics": model_call_metrics,
    }
# ---- Tool 5: search_conversation_history（D4）----
def _search_conversation_history(arguments: dict, *, context: dict | None = None) -> dict:
    """检索历史对话：current（当前会话）/ recent（最近会话）/ all（全部历史会话）。"""
    from ..agent_conversation import ConversationStore
    query = (arguments.get("query") or "").strip()
    scope = str(arguments.get("scope") or "current").strip()
    if scope not in {"current", "recent", "all_user_conversations"}:
        scope = "current"
    store = _RUNTIME.get("store")
    if store is None or not query:
        return {"summary": "对话检索暂不可用。", "total": 0, "scope": scope}
    cs = ConversationStore(store)
    scope_id = (context or {}).get("scope_id") or ""
    current_cid = arguments.get("conversation_id") or _RUNTIME.get("conversation_id")
    matches = []
    if scope == "current":
        if current_cid:
            for m in cs.search_messages(query, conversation_id=current_cid, limit=8):
                matches.append({"role": m.get("role"), "text": _msg_text_short(m.get("content")),
                                "created_at": m.get("created_at")})
    elif scope == "recent":
        convs = cs.list_conversations(scope_id=scope_id, limit=5)
        seen = set()
        for conv in convs:
            if conv["conversation_id"] == current_cid:
                continue
            for m in cs.search_messages(query, conversation_id=conv["conversation_id"], limit=2):
                key = m.get("id")
                if key in seen:
                    continue
                seen.add(key)
                matches.append({"role": m.get("role"), "text": _msg_text_short(m.get("content")),
                                "created_at": m.get("created_at"),
                                "conversation_title": conv.get("title")})
    else:
        for m in cs.search_messages(query, scope_id=scope_id, limit=10):
            matches.append({"role": m.get("role"), "text": _msg_text_short(m.get("content")),
                            "created_at": m.get("created_at")})
    if not matches:
        return {"summary": "历史对话中没有找到相关内容。", "total": 0, "scope": scope}
    return {
        "summary": f"在{'当前' if scope == 'current' else '历史'}对话中找到 {len(matches)} 条相关内容。",
        "total": len(matches),
        "scope": scope,
        "matches": matches[:10],
        "note": "对话内容属于用户表述，不等于照片证据；回答时需说明这是'你之前说过'而非照片事实。",
    }


def _msg_text_short(content) -> str:
    if isinstance(content, dict):
        text = str(content.get("text") or content.get("content") or "")
    elif isinstance(content, str):
        text = content
    else:
        text = ""
    return text.strip().replace("\n", " ")[:160]


# ---- Tool 6: get_core_memory（D5）----
def _sync_core_memory_from_entities(scope_id: str) -> "CoreMemoryStore":
    """把已确认人物/关系同步为 Core Memory 卡片（只写人物名与家庭关系，用户授权数据）。"""
    from ..core_memory import CoreMemoryStore
    store = _RUNTIME.get("store")
    cms = CoreMemoryStore(store)
    if store is None:
        return cms
    try:
        entities = store.list_entities(status="confirmed", scope_id=scope_id or None)
        name_by_id = {e["id"]: e["canonical_name"] for e in entities}
        for ent in entities:
            card_id = cms.upsert_card(scope_id=scope_id or "home-default",
                                      subject_type="person", subject_id=ent["id"],
                                      display_name=ent["canonical_name"])
            role = (ent.get("family_role") or "").strip()
            if role:
                cms.upsert_item(card_id=card_id,
                                text=f"{ent['canonical_name']} 的家庭角色是 {role}。",
                                epistemic_type="confirmed_fact", source_type="entity",
                                source_ids=[ent["id"]], source_revisions={"entity": 1})
        rels = store.list_person_relationships(scope_id=scope_id or None)
        for rel in rels:
            if rel.get("status") != "active" and rel.get("status") != "confirmed":
                continue
            subj, obj = rel.get("subject_entity_id"), rel.get("object_entity_id")
            subj_name = name_by_id.get(subj) or rel.get("subject_name")
            obj_name = name_by_id.get(obj) or rel.get("object_name")
            predicate = (rel.get("predicate") or "").strip()
            if not (subj and obj and subj_name and obj_name and predicate):
                continue
            card_id = cms.upsert_card(scope_id=scope_id or "home-default",
                                      subject_type="person", subject_id=subj,
                                      display_name=subj_name)
            cms.upsert_item(card_id=card_id,
                            text=f"{subj_name} 和 {obj_name} 的关系是 {predicate}。",
                            epistemic_type="confirmed_fact", source_type="relationship",
                            source_ids=[rel["id"]],
                            source_revisions={"relationship": int(rel.get("revision") or 1)})
    except Exception:
        pass
    return cms


def _get_core_memory(arguments: dict, *, context: dict | None = None) -> dict:
    scope_id = (context or {}).get("scope_id") or ""
    viewer_id = (context or {}).get("viewer_id") or "owner"
    conversation_id = (context or {}).get("conversation_id")
    subject = (arguments.get("subject") or "").strip()
    topic = (arguments.get("topic") or "").strip()
    limit = max(1, min(10, int(arguments.get("limit") or 5)))
    cms = _sync_core_memory_from_entities(scope_id)
    store = _RUNTIME.get("store")
    if store is None:
        return {"summary": "长期记忆不可用。", "cards": [], "total": 0}
    subject_ids = None
    if subject:
        matched = []
        try:
            for ent in store.list_entities(status="confirmed", scope_id=scope_id or None):
                if subject in (ent.get("canonical_name") or "") or subject in (ent.get("family_role") or ""):
                    matched.append(ent["id"])
        except Exception:
            matched = []
        subject_ids = matched or None
    cards = cms.list_cards(scope_id=scope_id or None, subject_ids=subject_ids, limit=limit)
    if topic:
        topic = topic.lower()
        filtered = []
        for card in cards:
            keep = []
            for item in card.get("items") or []:
                if topic in (item.get("text") or "").lower():
                    keep.append(item)
            if keep:
                card = dict(card)
                card["items"] = keep
                filtered.append(card)
        cards = filtered
    public_cards = []
    for card in cards:
        try:
            cms.record_access(card_id=card["card_id"], conversation_id=conversation_id,
                              viewer_id=viewer_id)
        except Exception:
            pass
        public_cards.append({
            "subject_type": card.get("subject_type"),
            "subject_id": card.get("subject_id"),
            "display_name": card.get("display_name"),
            "items": [{
                "text": item.get("text"),
                "truth_status": item.get("epistemic_type"),
                "source_type": item.get("source_type"),
                "source_ids": item.get("source_ids"),
                "source_revisions": item.get("source_revisions"),
                "updated_at": item.get("created_at"),
            } for item in (card.get("items") or [])],
        })
    if not public_cards:
        return {"summary": "长期记忆中没有找到相关内容。", "cards": [], "total": 0,
                "note": "只读长期记忆；agent_inference 不会被当作 confirmed。"}
    return {
        "summary": f"找到 {len(public_cards)} 张长期记忆卡片。",
        "total": len(public_cards),
        "cards": public_cards,
        "note": "truth_status: confirmed_fact/user_assertion/agent_inference/observed_pattern；agent_inference 不能表述为 confirmed。",
    }


# ---- Tool 7: get_person_memory（D6）----
def _resolve_person_entity(person: str, scope_id: str):
    store = _RUNTIME.get("store")
    if store is None or not person:
        return None
    try:
        for ent in store.list_entities(status="confirmed", scope_id=scope_id or None):
            name = ent.get("canonical_name") or ""
            role = ent.get("family_role") or ""
            if person in name or person in role or name in person or role in person:
                return ent
    except Exception:
        return None
    return None


def _person_assets(entity_id: str, scope_id: str) -> list:
    store = _RUNTIME.get("store")
    if store is None:
        return []
    sql = ("SELECT DISTINCT a.id AS asset_id, a.captured_at, a.captured_location "
           "FROM assets a "
           "JOIN face_instances fi ON fi.asset_id = a.id "
           "JOIN face_clusters fc ON fc.id = fi.cluster_id AND fc.entity_id = ? "
           "WHERE a.scope_id = ?")
    try:
        rows = store.connection.execute(sql, (entity_id, scope_id or "")).fetchall()
    except Exception:
        rows = []
    sql2 = ("SELECT DISTINCT a.id AS asset_id, a.captured_at, a.captured_location "
            "FROM assets a "
            "JOIN observations o ON o.asset_id = a.id "
            "JOIN entity_mentions em ON em.observation_id = o.id AND em.entity_id = ? "
            "WHERE a.scope_id = ?")
    try:
        rows2 = store.connection.execute(sql2, (entity_id, scope_id or "")).fetchall()
    except Exception:
        rows2 = []
    seen = {}
    for row in list(rows) + list(rows2):
        seen.setdefault(row["asset_id"], row)
    return list(seen.values())


def _person_events(entity_id: str, scope_id: str) -> list:
    store = _RUNTIME.get("store")
    if store is None:
        return []
    try:
        rows = store.connection.execute(
            """SELECT DISTINCT ev.id AS event_id, ev.title, ev.time_start, ev.scope_id
               FROM events ev
               JOIN event_observations eo ON eo.event_id = ev.id
               JOIN observations o ON o.id = eo.observation_id
               JOIN entity_mentions em ON em.observation_id = o.id AND em.entity_id = ?
               WHERE ev.scope_id = ?
               ORDER BY ev.time_start""",
            (entity_id, scope_id or "")).fetchall()
        return [dict(r) for r in rows]
    except Exception:
        return []


def _person_co_occurrence(entity_id: str, scope_id: str) -> list:
    store = _RUNTIME.get("store")
    if store is None:
        return []
    try:
        rows = store.connection.execute(
            """SELECT e.canonical_name, COUNT(DISTINCT a.id) AS shared_assets
               FROM assets a
               JOIN observations o ON o.asset_id = a.id
               JOIN entity_mentions em2 ON em2.observation_id = o.id
               JOIN entities e ON e.id = em2.entity_id AND e.entity_type = 'person' AND e.status = 'confirmed'
               WHERE a.scope_id = ? AND em2.entity_id != ?
                 AND a.id IN (
                   SELECT DISTINCT a2.id FROM assets a2
                   JOIN observations o2 ON o2.asset_id = a2.id
                   JOIN entity_mentions em ON em.observation_id = o2.id AND em.entity_id = ?
                 )
               GROUP BY e.id ORDER BY shared_assets DESC LIMIT 10""",
            (scope_id or "", entity_id, entity_id)).fetchall()
        return [{"name": r["canonical_name"], "shared_assets": r["shared_assets"]} for r in rows]
    except Exception:
        return []


def _get_person_memory(arguments: dict, *, context: dict | None = None) -> dict:
    person = (arguments.get("person") or "").strip()
    operation = (arguments.get("operation") or "overview").strip()
    if operation not in {"overview", "first_occurrence", "last_occurrence",
                         "common_places", "co_occurrence", "events"}:
        operation = "overview"
    scope_id = (context or {}).get("scope_id") or ""
    ent = _resolve_person_entity(person, scope_id)
    if ent is None:
        return {"person": person, "readiness": "limited", "operation": operation,
                "insufficient_evidence": True,
                "summary": f"没有找到已确认人物「{person}」的记忆数据。",
                "note": "人物未确认或数据不足时返回 limited，不编造。"}
    assets = _person_assets(ent["id"], scope_id)
    if not assets:
        return {"person": person, "readiness": "limited", "operation": operation,
                "insufficient_evidence": True,
                "summary": f"「{person}」暂无足够的照片/观察数据。",
                "note": "数据不足时返回 limited，不编造。"}
    captured = [a.get("captured_at") for a in assets if a.get("captured_at")]
    captured.sort()
    places: dict[str, int] = {}
    for a in assets:
        loc = (a.get("captured_location") or "").strip()
        if loc:
            places[loc] = places.get(loc, 0) + 1
    common_places = [{"place": k, "count": v}
                     for k, v in sorted(places.items(), key=lambda kv: -kv[1])[:8]]
    events = _person_events(ent["id"], scope_id)
    co_occurrence = _person_co_occurrence(ent["id"], scope_id)
    base = {
        "person": ent["canonical_name"],
        "family_role": ent.get("family_role") or "",
        "readiness": "ready",
        "operation": operation,
        "asset_count": len(assets),
        "first_occurrence": captured[0] if captured else None,
        "last_occurrence": captured[-1] if captured else None,
        "common_places": common_places,
        "co_occurrence": co_occurrence,
        "events": [{"event_id": e["event_id"], "title": e["title"], "time_start": e.get("time_start")}
                   for e in events[:10]],
        "representative_events": [e["title"] for e in events[:3]],
        "observation_count": len(events) + len(assets),
        "event_count": len(events),
        "entity_binding_coverage": "confirmed",
        "note": "主观性格等无法从照片确认的问题应回答 insufficient evidence，不要臆测。",
    }
    if operation == "first_occurrence":
        return {k: base[k] for k in ("person", "readiness", "operation", "first_occurrence", "asset_count")}
    if operation == "last_occurrence":
        return {k: base[k] for k in ("person", "readiness", "operation", "last_occurrence", "asset_count")}
    if operation == "common_places":
        return {k: base[k] for k in ("person", "readiness", "operation", "common_places", "asset_count")}
    if operation == "co_occurrence":
        return {k: base[k] for k in ("person", "readiness", "operation", "co_occurrence", "asset_count")}
    if operation == "events":
        return {k: base[k] for k in ("person", "readiness", "operation", "events", "event_count", "asset_count")}
    base["summary"] = (f"「{ent['canonical_name']}」共出现在 {len(assets)} 张照片中，"
                       f"最早 {captured[0][:10]}，最近 {captured[-1][:10]}。")
    return base


def _base64_image(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def _mime_for(path: str) -> str:
    return "image/png" if str(path).lower().endswith(".png") else "image/jpeg"


def _handle_to_asset_id(handle: str) -> str | None:
    # shadow 期 handle -> asset_id 映射来自最近一次 search_memories 的 preview。
    return (_RUNTIME.get("last_handles") or {}).get(handle)


_INSPECT_PROMPT = """观察这张照片，输出 JSON：
{{"observation": "一句话描述", "certainty": "supported|uncertain"}}
问题：{question}"""



def _cap_hint(tool: str) -> str:
    """能力实测提示：无数据返回空串，不改变工具合同。"""
    try:
        hint = tool_capability_summary(tool)
        return ("\n" + hint) if hint else ""
    except Exception:
        return ""


def register_tools():

    register(ToolSpec(
        name="query_memory_facts",
        description=("确定性结构化事实查询（数量/存在性/首次/最近/日期/分组/饮食），不要用模型估算。"
                     "filters.time 原样写相对或具体时间，系统自动换算；不填表示全部。"
                     "group 必须填 group_by（month|place），place 分组需如实说明无地点照片数。"
                     "meal 用于'吃过什么/吃饭'类问题。菜单价格/招牌等视觉文字先用 search_memories 再 read_photo_text，不要用本工具猜。"
                     "filters.place 只填结构化地名（城市/区县/景区/地标），不要把目标/活动/主题当 place；不确定留空。"),
        input_schema={"operation": "count|exists|first|last|date|group|meal",
                      "filters": {"time": "去年/这两年/2023年 等相对或具体时间（原样写）",
                                  "person": "", "place": "", "media": "",
                                  "food": "可选：限定某种食物（如'火锅'）"},
                      "group_by": "month|place"},
        executor=_query_memory_facts, read_write="read", cost_class="cheap", readiness="ready",
    ))
    register(ToolSpec(
        name="search_memories",
        description=("检索照片（人/物/场景/衣着/颜色）。返回结果集摘要。"
                     "时间必须写 filters.time（'2024年'/'去年'等），系统会自动从 query 提取漏填的时间。"
                     "filters.place 只填结构化地名（城市/区县/景区/地标），不要把目标/活动/主题当 place；不确定留空。"),
        input_schema={"query": "", "mode": "best|all|representative",
                      "filters": {"time": "", "place": "", "person": ""}},
        executor=_search_memories, read_write="read", cost_class="medium", readiness="ready",
    ))
    register(ToolSpec(
        name="get_original_photos",
        description="交付当前结果集/选中照片的原图。",
        input_schema={"result_set_id": "", "handle": ""},
        executor=_get_original_photos, read_write="read", cost_class="cheap", readiness="limited",
        readiness_reason="ResultSetStore 就绪后完整可用（A4）",
    ))
    register(ToolSpec(
        name="get_result_page",
        description="查看 search_memories 结果集的下一页/指定页。result_set_id 用 search_memories 返回的，page 从 1 开始。",
        input_schema={"result_set_id": "", "page": 1, "page_size": 6},
        executor=_get_result_page, read_write="read", cost_class="cheap", readiness="ready",
    ))
    register(ToolSpec(
        name="inspect_photo",
        description=("复核已检索照片的视觉细节（物体/衣着/文字/场景）。asset_handle 使用 search_memories preview 里的 handle（photo_1…），可省略（默认用预览第一张）。昂贵，默认每轮最多 1 次。"
                     + _cap_hint("inspect_photo")),
        input_schema={"asset_handle": "", "question": ""},
        executor=_inspect_photo, read_write="read", cost_class="expensive", readiness="ready",
    ))
    register(ToolSpec(
        name="read_photo_text",
        description=("读取照片中的文字内容（菜单/价格/招牌/店名/电话/年份/小字）。"
                     "适用于'多少钱/价格/售价/店名/招牌/电话/写了什么/什么字/创始于哪一年'等需要看照片文字的题；"
                     "内部会把照片切块放大后 OCR。asset_handle 用 search_memories preview 里的 handle，可省略（默认预览第一张）。"
                     "昂贵，每轮最多 1 次。"
                     + _cap_hint("read_photo_text")),
        input_schema={"asset_handle": "", "question": ""},
        executor=_read_photo_text, read_write="read", cost_class="expensive", readiness="ready",
    ))
    register(ToolSpec(
        name="search_conversation_history",
        description=("检索历史对话内容：回答'我之前是不是说过…/上次聊到哪里/你之前说那张在哪'类问题。"
                     "scope=current 只查当前会话；scope=recent 查最近几个会话；scope=all_user_conversations 查全部历史会话。"
                     "对话内容属于用户表述，不等于照片证据；回答时应说'你之前说过'而不是当成照片事实。"),
        input_schema={"query": "", "scope": "current|recent|all_user_conversations",
                      "conversation_id": ""},
        executor=_search_conversation_history, read_write="read", cost_class="cheap", readiness="ready",
    ))
    register(ToolSpec(
        name="get_core_memory",
        description=("读取长期家庭记忆（已确认人物/家庭角色/关系/偏好等）。"
                     "subject 填人物名（如'明哥'）；topic 填话题关键词（如'西湖'）；都不填返回优先级最高的卡片。"
                     "每条记忆带 truth_status（confirmed_fact/user_stated/agent_inference/observed_pattern），"
                     "agent_inference 不能说成 confirmed。"),
        input_schema={"subject": "", "topic": "", "limit": 5},
        executor=_get_core_memory, read_write="read", cost_class="cheap", readiness="ready",
    ))
    register(ToolSpec(
        name="get_person_memory",
        description=("读取已确认人物的结构化记忆：首次/最近出现时间、出现次数、常去地点、同行人物、参与事件。"
                     "operation=overview(总览)/first_occurrence(首次)/last_occurrence(最近一次)/"
                     "common_places(常去地点)/co_occurrence(同行)/events(参与事件)。"
                     "人物未确认或数据不足时返回 limited，要如实说明，不要编造。"
                     "性格等主观问题：照片无法确认时回答 insufficient evidence。"),
        input_schema={"person": "", "operation": "overview|first_occurrence|last_occurrence|common_places|co_occurrence|events"},
        executor=_get_person_memory, read_write="read", cost_class="cheap", readiness="ready",
    ))
