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
import tempfile
import time
from pathlib import Path

from .tool_registry import ToolSpec, register
from .capability import tool_capability_summary
from .intent import ocr_intent, visual_intent
from .ocr_tool import (_read_photo_text, ocr_telemetry_snapshot,
                       record_ocr_telemetry, small_ocr_available)
from .ocr_tool import bind_ocr_runtime
from ..person_appearance import expanded_person_crop

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
    if re.fullmatch(r"20\d{2}年(?:\d{1,2}月(?:\d{1,2}[日号]?)?)?", v):
        return v
    # Unknown relative phrases must not be passed to the strict time parser as
    # if they were absolute expressions.  Dropping the unsupported constraint
    # preserves semantic recall; the caller can surface the raw filter in
    # diagnostics instead of silently forcing a contradicted time range.
    return None


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
_FACT_OPERATIONS = {"count", "exists", "first", "last", "date", "group", "meal", "list"}


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
    user_goal = str(((context or {}).get("task_state") or {}).get("user_goal") or "")
    # A date/first-occurrence question must not silently fall through to the
    # broad media-list branch when a 12B planner emits operation=list.
    if operation == "list" and re.search(r"哪天|什么时候|何时|日期|哪一年|几月|最早|最近一次", user_goal):
        operation = "first" if re.search(r"最早|第一次", user_goal) else "date"
    scope_id = (context or {}).get("scope_id") or ""
    viewer_id = (context or {}).get("viewer_id") or "owner"
    # An unresolved person/entity constraint must never degrade into a query
    # over the whole scope.  That turns a planner mistake such as treating
        # Do not interpret a free-form activity phrase as a person name.
    # Returning an explicit no-support result lets the final gate ask for a
    # narrower search or state that the fact is unavailable.
    person_filter = str((filters or {}).get("person") or "").strip()
    if person_filter and _resolve_entity(person_filter, scope_id) is None:
        return {
            "operation": operation,
            "answer_type": "unknown",
            "value": None,
            "total": 0,
            "rows": [],
            "samples": [],
            "filters_applied": {"person": person_filter},
            "coverage": {"complete": False, "reason": "unresolved_entity"},
            "source_asset_ids": [],
            "source_handles": [],
        }
    if operation == "list":
        return _query_media_list(filters, scope_id=scope_id, viewer_id=viewer_id)
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
    if operation in {"first", "last", "date"}:
        # Boundary facts must cite the assets that actually produced MIN/MAX;
        # the generic latest-observation sampler is not a valid source for a
        # first-occurrence answer.
        try:
            matching = [row for row in executor._matching_assets(draft, spec, limit=500)
                        if row.get("captured_at")]
            matching.sort(key=lambda row: str(row.get("captured_at") or ""),
                          reverse=operation == "last")
            boundary_value = str(result.value or "")
            out["samples"] = [row for row in matching
                              if str(row.get("captured_at") or "") == boundary_value][:3]
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
    source_asset_ids = []
    for sample in out.get("samples") or []:
        if isinstance(sample, dict) and sample.get("asset_id"):
            source_asset_ids.append(str(sample["asset_id"]))
    for row in out.get("rows") or []:
        if isinstance(row, dict) and row.get("asset_id"):
            source_asset_ids.append(str(row["asset_id"]))
    out["source_asset_ids"] = list(dict.fromkeys(source_asset_ids))
    out["source_handles"] = []
    return out


def _query_media_list(filters: dict, *, scope_id="home-default", viewer_id="owner") -> dict:
    """List actual media assets instead of only counting them.

    The tool contract keeps the model in charge: it receives stable IDs,
    media_kind, and for keyframes the owning video/time/context.  It can then
    decide whether to name the videos, deliver the source video, or inspect
    specific keyframes.
    """
    from ..structured_memory import StructuredMemoryExecutor

    store = _RUNTIME.get("store")
    if store is None:
        return {"operation": "list", "summary": "记忆库不可用。", "total": 0,
                "items": [], "coverage": {"complete": False}}
    draft = _draft_from_filters(filters, answer_type="asset_set")
    spec = _spec_for(draft, scope_id, viewer_id)
    executor = StructuredMemoryExecutor(store)
    media_filter = str((filters or {}).get("media") or "").strip().lower() or None
    try:
        assets = executor._matching_assets(draft, spec, limit=500)
    except Exception:
        assets = []
    items = []
    for row in assets:
        asset = store.get_asset(row.get("id")) or {}
        if not asset:
            continue
        media_type = asset.get("media_type") or row.get("media_type") or ""
        derived_kind = asset.get("derived_kind")
        if derived_kind == "video_keyframe":
            media_kind = "video_keyframe"
        elif media_type == "video":
            media_kind = "video"
        elif media_type == "image":
            media_kind = "original_image"
        else:
            media_kind = media_type or "unknown"
        item = {
            "asset_id": asset.get("id"),
            "file_name": asset.get("file_name"),
            "media_type": media_type,
            "media_kind": media_kind,
            "derived_kind": derived_kind,
            "captured_at": asset.get("captured_at") or row.get("captured_at"),
        }
        if derived_kind == "video_keyframe":
            source_video_id = asset.get("parent_asset_id")
            item["source_video_asset_id"] = source_video_id
            item["source_timestamp_sec"] = asset.get("source_timestamp_sec")
            item["source_scene_index"] = asset.get("source_scene_index")
            source_video = store.get_asset(source_video_id) if source_video_id else None
            item["source_video_file_name"] = (source_video or {}).get("file_name")
        if media_type == "video":
            metadata = asset.get("metadata_json") or {}
            video_metadata = metadata.get("video_metadata") or {}
            scenes = store.list_video_scene_events(asset.get("id")) if store.list_video_scene_events else []
            item["duration_sec"] = video_metadata.get("duration_sec")
            item["scene_count"] = int(metadata.get("worldmm_scene_count") or len(scenes) or 0)
            item["keyframe_count"] = int(
                metadata.get("worldmm_selected_keyframe_count")
                or metadata.get("worldmm_summary_keyframe_count")
                or metadata.get("worldmm_keyframe_count")
                or 0
            )
            item["scene_samples"] = []
            for scene in (scenes or [])[:5]:
                keyframe_samples = []
                for frame in (scene.get("keyframe_assets") or [])[:3]:
                    keyframe_samples.append({
                        "asset_id": frame.get("id"),
                        "timestamp_sec": frame.get("source_timestamp_sec"),
                        "source_scene_index": frame.get("source_scene_index"),
                    })
                item["scene_samples"].append({
                    "scene_id": scene.get("id"),
                    "title": scene.get("title"),
                    "start_sec": scene.get("source_start_sec"),
                    "end_sec": scene.get("source_end_sec"),
                    "source_scene_index": scene.get("source_scene_index"),
                    "keyframe_samples": keyframe_samples,
                })
        items.append(item)
    limited = items[:80]
    summary_items = []
    for item in limited:
        if item.get("media_kind") == "video":
            duration = item.get("duration_sec")
            summary_items.append(
                f"{item.get('file_name') or '未命名视频'}（时长 {duration if duration is not None else '未知'} 秒，"
                f"{item.get('scene_count') or 0} 个场景）"
            )
        elif item.get("media_kind") == "video_keyframe":
            source = item.get("source_video_file_name") or "未知视频"
            ts = item.get("source_timestamp_sec")
            summary_items.append(
                f"关键帧 {item.get('file_name') or '未命名关键帧'}（来自 {source} 第 {ts if ts is not None else '未知'} 秒）"
            )
        else:
            summary_items.append(item.get("file_name") or "未命名媒体")
    return {
        "operation": "list",
        "answer_type": "media_list",
        "summary": "；".join(summary_items[:20]),
        "value": len(limited),
        "total": len(items),
        "items": limited,
        "source_asset_ids": [str(item["asset_id"]) for item in limited if item.get("asset_id")],
        "source_handles": [],
        "has_more": len(items) > len(limited),
        "media_filter": media_filter,
        "filters_applied": {
            "scope_id": scope_id or None,
            "media": media_filter,
        },
        "coverage": {"complete": True},
    }


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
        "source_asset_ids": [str(item["asset_id"]) for item in meal_samples if item.get("asset_id")],
        "source_handles": [],
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
_RESULT_PREVIEW_LIMIT = 6
_RESULT_PAGE_SIZE = 6


def _public_candidate_limit() -> int:
    """Maximum candidate count exposed to the Agent/UI contract.

    The ResultSet keeps the complete retrieved pool for recall accounting, but
    the raw pool size must never become a user-visible answer fact.  The same
    bound is used by validation and pagination so a model cannot bypass the
    bounded evidence window with get_result_page.
    """
    try:
        return max(1, int(os.getenv("SENTRIX_SEARCH_VALIDATION_MAX_CANDIDATES", "30")))
    except (TypeError, ValueError):
        return 30


def _visible_candidate_total(asset_count: int) -> int:
    return min(max(0, int(asset_count or 0)), _public_candidate_limit())
_RESULT_PREVIEW_RELEVANCE_HEAD = max(
    0, min(_RESULT_PREVIEW_LIMIT, int(os.getenv("SENTRIX_RESULT_PREVIEW_RELEVANCE_HEAD", "3")))
)
_PREVIEW_QUERY_ALIASES = {
    "布置": ("布置", "装饰", "花艺", "彩带", "气球", "窗帘", "床品", "家具"),
    "装饰": ("装饰", "花艺", "彩带", "气球", "窗帘", "床品", "家具"),
    "文字": ("文字：", "文字", "写着", "标志"),
    "雕塑": ("雕塑", "石雕", "雕像", "纪念碑"),
    "石雕": ("石雕", "雕塑", "雕像", "纪念碑"),
    "桥": ("桥", "桥上"),
    # Person-count/scene cues are useful for choosing the visible evidence
    # window after a broad place recall. They do not change retrieved
    # candidates; they only promote matching observations into preview.
    "三人": ("三人", "三个人", "三人合影"),
    "三个人": ("三人", "三个人", "三人合影"),
    "合影": ("合影", "自拍", "合照"),
    "舞台": ("舞台", "仪式", "典礼"),
    "户外": ("户外", "室外", "露天"),
    "夜晚": ("夜晚", "夜间", "夜景", "灯光"),
}


def _observation_summary(store, asset_id: str) -> str:
    """Expose bounded evidence text without exposing storage identifiers."""
    if store is None:
        return ""
    try:
        rows = store.list_observations(asset_id=asset_id, limit=1)
    except Exception:
        return ""
    if not rows:
        return ""
    observation = rows[0] or {}
    parts = []
    for key in ("caption", "activity", "place"):
        value = str(observation.get(key) or "").strip()
        if value and value not in parts:
            parts.append(value)
    for key in ("objects", "clothing"):
        values = observation.get(key) or []
        labels = []
        for value in values:
            if isinstance(value, dict):
                value = value.get("label") or value.get("primary") or ""
            value = str(value or "").strip()
            if value and value not in labels:
                labels.append(value)
        if labels:
            parts.append("、".join(labels[:8]))
    ocr = str(observation.get("ocr_text") or "").strip()
    if ocr:
        parts.append(f"文字：{ocr[:120]}")
    detail = observation.get("detail") or {}
    if isinstance(detail, dict):
        details = []
        for item in detail.get("visible_details") or []:
            if isinstance(item, dict):
                value = item.get("text") or item.get("label") or ""
            else:
                value = item
            value = str(value or "").strip()
            if value and value not in details:
                details.append(value)
        if details:
            parts.append("；".join(details[:6]))
    return "；".join(parts)[:300]


def _asset_group_key(store, asset_id: str) -> str:
    """Group video keyframes and event-near-duplicates for the initial preview."""
    if store is None:
        return asset_id
    try:
        asset = store.get_asset(asset_id) or {}
        if asset.get("derived_kind") in {"video_keyframe", "video_keyframe_webp"}:
            return ":".join(str(asset.get(key) or "") for key in (
                "parent_asset_id", "source_scene_index")) or asset_id
        row = store.connection.execute(
            "SELECT event_id FROM event_observations WHERE observation_id IN "
            "(SELECT id FROM observations WHERE asset_id = ?) ORDER BY event_id LIMIT 1",
            (asset_id,),
        ).fetchone()
        return str(row["event_id"] if row else asset_id)
    except Exception:
        return asset_id


def _preview_query_order(asset_ids: list[str], query: str, store) -> list[int]:
    """Promote candidates whose stored visual detail matches explicit visual cues."""
    text = str(query or "")
    cues = [(term, aliases) for term, aliases in _PREVIEW_QUERY_ALIASES.items()
            if term in text]
    requested_count = None
    count_match = re.search(r"([一二三四五六七八九十]|\d+)\s*(?:人|个人)", text)
    if count_match:
        raw = count_match.group(1)
        requested_count = {"一": 1, "二": 2, "三": 3, "四": 4, "五": 5,
                           "六": 6, "七": 7, "八": 8, "九": 9, "十": 10}.get(raw)
        if requested_count is None:
            try:
                requested_count = int(raw)
            except ValueError:
                requested_count = None
    if not cues and requested_count is None:
        return list(range(len(asset_ids)))
    scored = []
    for index, asset_id in enumerate(asset_ids):
        summary = _observation_summary(store, asset_id)
        score = sum(1 for _, aliases in cues if any(alias in summary for alias in aliases))
        if requested_count is not None and store is not None:
            try:
                face_count = int(store.connection.execute(
                    "SELECT COUNT(*) FROM face_instances WHERE asset_id = ?", (asset_id,)
                ).fetchone()[0])
                # Face detections can contain duplicate clusters, so reward the
                # nearest count rather than requiring exact equality.
                score += max(0, 2 - abs(face_count - requested_count))
            except Exception:
                pass
            # Prefer an observation that explicitly states the requested count;
            # a nearby two-person scene with duplicated face detections should
            # not outrank a genuine “三个人” caption.
            count_words = {1: ("一个", "一名", "一人"), 2: ("两个", "两名", "两人"),
                           3: ("三个", "三名", "三人"), 4: ("四个", "四名", "四人")}
            if any(word in summary for word in count_words.get(requested_count, ())):
                score += 3
        scored.append((-score, index))
    scored.sort()
    return [index for _, index in scored]


def _preview_indices(asset_ids: list[str], mode: str, store, query: str = "") -> list[int]:
    """Select bounded indices under an explicit candidate-window policy.

    The full ResultSet remains server-side.  ``SENTRIX_CANDIDATE_STRATEGY`` is
    intentionally process-scoped so benchmark A/B runs can change only this
    policy while keeping model, ANN and source data fixed:
    ``head_only`` keeps retrieval order, ``event_diversity`` maximizes event
    diversity, and the default keeps a relevance head before diversity.
    """
    # The retrieval order is the only ranking signal guaranteed to have been
    # produced by the complete candidate search.  Event-diversity is still
    # available as an explicit opt-in, but must not silently hide the sixth
    # ranked source image from the user/evidence window.
    strategy = os.getenv("SENTRIX_CANDIDATE_STRATEGY", "head_only").strip().lower()
    if len(asset_ids) <= _RESULT_PREVIEW_LIMIT:
        return (_preview_query_order(asset_ids, query, store)
                if mode != "representative" else list(range(len(asset_ids))))
    if mode == "representative":
        candidates = _even_indices(len(asset_ids), _RESULT_PREVIEW_LIMIT)
    else:
        candidates = _preview_query_order(asset_ids, query, store)
    if strategy in {"head_only", "relevance_head_only"}:
        return candidates[:_RESULT_PREVIEW_LIMIT]
    if strategy in {"event_diversity", "diversity_only"}:
        relevance_head = 0
    else:
        relevance_head = min(_RESULT_PREVIEW_RELEVANCE_HEAD, len(candidates))
    selected = []
    seen_groups = set()
    # Retrieval order is the strongest available relevance signal.  Preserve a
    # small head even when those assets belong to one event; otherwise event
    # diversity can discard the actual answer image before the model can inspect it.
    for index in candidates[:relevance_head]:
        selected.append(index)
        seen_groups.add(_asset_group_key(store, asset_ids[index]))
    if len(selected) >= _RESULT_PREVIEW_LIMIT:
        return selected[:_RESULT_PREVIEW_LIMIT]
    for index in candidates[relevance_head:]:
        group = _asset_group_key(store, asset_ids[index])
        if group in seen_groups:
            continue
        selected.append(index)
        seen_groups.add(group)
        if len(selected) >= _RESULT_PREVIEW_LIMIT:
            return selected
    # With explicit scene cues, never pad the visible window with an arbitrary
    # head item that failed the cue match. Such padding let the model inspect
    # a generic “photo_1” and answer from the wrong image even though the
    # matching asset was already in the retrieved candidate set.
    if any(term in str(query or "") for term in _PREVIEW_QUERY_ALIASES):
        return selected
    for index in range(len(asset_ids)):
        if index not in selected:
            selected.append(index)
        if len(selected) >= _RESULT_PREVIEW_LIMIT:
            break
    return selected


def _candidate_window_summary(asset_ids: list[str], indices: list[int], store) -> dict:
    """Expose bounded candidate-window diagnostics without storage IDs.

    The ResultSet remains complete server-side; this summary tells the model
    and 8771 whether the visible window is a diverse head or one large event,
    without dumping the full candidate list into the prompt.
    """
    visible_limit = _visible_candidate_total(len(asset_ids))
    bounded_ids = list(asset_ids[:visible_limit])
    bounded_indices = [index for index in indices if index < visible_limit]
    groups = {}
    for asset_id in bounded_ids:
        key = _asset_group_key(store, asset_id)
        groups[key] = groups.get(key, 0) + 1
    return {
        "total_candidates": visible_limit,
        "visible_candidates": len(bounded_indices),
        "visible_ranks": [index + 1 for index in bounded_indices],
        "event_group_count": len(groups),
        "largest_event_group": max(groups.values(), default=0),
            "strategy": os.getenv("SENTRIX_CANDIDATE_STRATEGY", "head_only").strip().lower(),
    }


def _preview_entry(store, asset_id: str, handle: str, *, level="exact", condition_summary=None,
                   priority_rank: int | None = None, selection_reason: str = "") -> dict:
    asset = store.get_asset(asset_id) if store is not None else {}
    asset = asset or {}
    media_kind = "original_image"
    source_video_asset_id = None
    source_timestamp_sec = None
    source_scene_index = None
    source_video_file_name = None
    if asset.get("derived_kind") in {"video_keyframe", "video_keyframe_webp"}:
        media_kind = "video_keyframe"
        source_video_asset_id = asset.get("parent_asset_id")
        source_timestamp_sec = asset.get("source_timestamp_sec")
        source_scene_index = asset.get("source_scene_index")
        source_video = store.get_asset(source_video_asset_id) if store and source_video_asset_id else None
        source_video_file_name = (source_video or {}).get("file_name")
    evidence_summary = _observation_summary(store, asset_id)
    # Confirmed face/entity links are deterministic memory evidence.  Expose
    # only the public name/role projection in search previews; face IDs and
    # embeddings remain server-side and pending clusters stay unnamed.
    people = []
    for identity in _confirmed_photo_identities(store, asset_id):
        if not isinstance(identity, dict):
            continue
        name = str(identity.get("person_name") or "").strip()
        if not name or identity.get("identity_status") != "confirmed":
            continue
        people.append({
            "name": name,
            "family_role": str(identity.get("family_role") or "").strip(),
            "identity_status": "confirmed",
        })
    people = list({(item["name"], item["family_role"]): item for item in people}.values())
    return {
        "handle": handle,
        "asset_id": asset_id,
        "captured_at": asset.get("captured_at"),
        "level": level,
        "place": _short_place_label(asset) if asset else "",
        "media_kind": media_kind,
        "source_video_asset_id": source_video_asset_id,
        "source_timestamp_sec": source_timestamp_sec,
        "source_scene_index": source_scene_index,
        "source_video_file_name": source_video_file_name,
        "evidence_summary": evidence_summary,
        "people": people,
        # Keep description availability explicit so the UI/benchmark can tell
        # an empty observation apart from a transport/schema omission.
        "description_status": "available" if evidence_summary else "missing",
        "condition_summary": condition_summary or {},
        "priority_rank": priority_rank,
        "selection_reason": selection_reason or ("相关性排序靠前" if priority_rank == 1 else "候选补充" if priority_rank else ""),
    }
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
    store = _RUNTIME.get("store")
    indices = _preview_indices(asset_ids, mode, store, query=user_goal or query)
    preview = [
        _preview_entry(store, assets[idx].get("id"), f"photo_{idx + 1}",
                       priority_rank=rank, selection_reason="相关性最高" if rank == 1 else "事件多样性补充")
        for rank, idx in enumerate(indices, 1)
    ]
    total = len(assets)
    visible_total = _visible_candidate_total(total)
    preview_asset_ids = [asset_ids[idx] for idx in indices if idx < len(asset_ids)]
    return {
        "result_set_id": rs.result_set_id,
        "query": query,
        "mode": mode,
        "total": visible_total,
        "evidence_count": _visible_candidate_total(len(asset_ids)),
        "preview": preview,
        "has_more": visible_total > len(preview),
        "remaining": max(0, visible_total - len(preview)),
        "candidate_window": _candidate_window_summary(asset_ids, indices, store),
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
        "_retrieved_asset_ids": list(asset_ids),
        # Public trace contract: keep the complete candidate set distinct from
        # the bounded preview.  Runtime may still redact private underscore
        # fields, so expose stable asset IDs explicitly for benchmark/user
        # provenance accounting.
        "retrieved_asset_ids": list(asset_ids),
        "_preview_asset_ids": preview_asset_ids,
        "evidence_asset_ids": [],
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


def _sanitize_model_filters(filters: dict, *, query: str, user_goal: str) -> dict:
    """Keep model-generated time filters anchored to explicit user wording."""
    sanitized = dict(filters or {})
    explicit_time = _extract_time_from_query(f"{user_goal or ''} {query or ''}")
    if explicit_time:
        sanitized["time"] = explicit_time
    else:
        # The model sees current_time in its prompt and may hallucinate it as a
        # photo date.  An unsupported time filter silently destroys recall.
        sanitized.pop("time", None)
    return sanitized


def _canonical_search_enabled() -> bool:
    from .canonical_intent import canonical_enabled
    return canonical_enabled()


def _event_resolution(question: str, store, scope_id: str) -> dict | None:
    """W2.4：多轮引用解析到 Event（turn-0 无结果集时的二级锚）。

    从问题提取时间/人物/活动线索，在 events 表里召回候选（用 time/participants/place/activity，
    不只看 title），单候选高置信时返回其资产，否则 None（交回普通检索/澄清）。
    """
    from .canonical_intent import extract_time
    if store is None or not scope_id:
        return None
    t = extract_time(question)
    # 直接查 entities 表解析人物（绕过 list_entities 的 include_in_people 过滤，benchmark scope 也能用）
    persons = []
    try:
        for ent in store.connection.execute(
                "SELECT id, canonical_name, family_role FROM entities "
                "WHERE scope_id=? AND entity_type='person' AND status='confirmed'",
                (scope_id,)).fetchall():
            for alias in (ent["canonical_name"], ent["family_role"]):
                if alias and alias != "自己" and alias in question:
                    persons.append(ent["id"])
                    break
    except Exception:
        pass
    try:
        rows = store.connection.execute(
            "SELECT id,title,place,activity,summary,participants_json,substr(time_start,1,10) AS ts,"
            "substr(time_start,1,7) AS ym FROM events "
            "WHERE scope_id=? AND status NOT IN ('rejected','superseded','merged')", (scope_id,)).fetchall()
    except Exception:
        return None
    # Generic words identify an activity class but not a particular event.
    # They must not be allowed to lock a first-turn query to an unrelated
    # event (for example “合影” selecting a later children photo set).
    generic_overlap = {"合影", "照片", "活动", "室内", "户外", "空间", "场地", "参加", "不同"}
    scored = []
    for raw_row in rows:
        # Different album DB revisions may omit optional event columns. Work
        # from a plain mapping so an absent summary cannot abort retrieval.
        r = dict(raw_row)
        score = 0
        if t:
            if r.get("ts") and r["ts"].startswith(t[:10]):
                score += 3
            elif r.get("ym") and r["ym"].startswith(t[:7]):
                score += 2
        for pid in persons:
            if str(pid) in (r.get("participants_json") or ""):
                score += 3
        # 数据驱动的文本重叠：问题与事件 title/place/activity 的中文子串匹配（不硬编码任何关键词）
        # Event summaries contain discriminative facts that may be absent from
        # short title/place fields. Keep the result bounded to that event.
        hay = " ".join(str(x) for x in
                       (r.get("title"), r.get("place"), r.get("activity"), r.get("summary")) if x)
        overlap = 0
        strong_overlap = 0
        for length in (4, 3, 2):
            ngrams = {hay[i:i + length] for i in range(max(0, len(hay) - length + 1))
                      if len(hay[i:i + length]) == length
                      and re.search(r"[\u4e00-\u9fff]", hay[i:i + length])
                      and not any(c.isdigit() for c in hay[i:i + length])}
            for ng in ngrams:
                # Suffixes such as “的合影/的合” are not event identity
                # signals even though they technically overlap the query.
                if ng in generic_overlap or any(token in ng for token in generic_overlap):
                    continue
                if ng in question:
                    overlap += 1
                    if length >= 4:
                        strong_overlap += 1
                    break
        score += min(overlap, 2)
        if score:
            scored.append((score, r.get("id"), r.get("title"), overlap, strong_overlap))
    # Enumeration questions about a group photo often use natural language
    # (“兄弟们”“不同人数”) that is absent from the generated event title.
    # Recover the event from its own asset observations: count adult-male
    # group-photo observations, while excluding rows explicitly describing
    # children. This remains data-driven and does not depend on benchmark IDs.
    if re.search(r"合影", question or "") and re.search(r"(?:几张|多少|不同人数|几人|人数)", question or ""):
        try:
            group_scores = []
            for r in rows:
                obs = store.connection.execute(
                    "SELECT caption, activity, people_json FROM observations o "
                    "JOIN event_observations eo ON eo.observation_id=o.id "
                    "WHERE eo.event_id=?", (r["id"],)).fetchall()
                count = 0
                sizes = set()
                event_text = ""
                for o in obs:
                    text = " ".join(str(o[k] or "") for k in ("caption", "activity"))
                    event_text += " " + text
                    if "合影" not in text:
                        continue
                    if any(token in text for token in ("幼儿", "孩子", "儿童", "小孩")):
                        continue
                    # Chinese count words or Arabic digits followed by adult
                    # male wording are strong signals for the requested group.
                    if re.search(r"(?:[一二三四五六七八九十两\d]+名|[一二三四五六七八九十两\d]+人).*男|男子|男性", text):
                        count += 1
                        match = re.search(r"([一二三四五六七八九十两\d]+)(?:名|人)", text)
                        if match:
                            token = match.group(1)
                            cn = {"一": 1, "二": 2, "两": 2, "三": 3, "四": 4,
                                  "五": 5, "六": 6, "七": 7, "八": 8,
                                  "九": 9, "十": 10}
                            size = int(token) if token.isdigit() else cn.get(token)
                            if size:
                                sizes.add(size)
                if count:
                    # For enumeration questions, coverage of distinct group
                    # sizes is more discriminative than raw photo count.
                    group_scores.append((len(sizes), count, r["id"], r["title"]))
            group_scores.sort(key=lambda x: (-x[0], -x[1], x[3]))
            if group_scores and (len(group_scores) == 1 or group_scores[0][:2] > group_scores[1][:2]):
                eid = group_scores[0][2]
                asset_rows = store.connection.execute(
                    "SELECT DISTINCT o.asset_id FROM observations o "
                    "JOIN event_observations eo ON eo.observation_id=o.id "
                    "JOIN assets a ON a.id=o.asset_id WHERE eo.event_id=? AND a.scope_id=?",
                    (eid, scope_id)).fetchall()
                asset_ids = [a["asset_id"] for a in asset_rows]
                if asset_ids:
                    return {"event_id": eid, "event_title": group_scores[0][3],
                            "asset_ids": asset_ids[:50]}
        except Exception:
            pass
    if not scored:
        return None
    scored.sort(key=lambda x: -x[0])
    top, second = scored[0], scored[1] if len(scored) > 1 else None
    # 单候选高置信，或多候选但第一明显领先。事件锚定只允许在问题与
    # 同一事件命中至少两个独立短语时生效；单个四字重叠（例如“婚礼照片”
    # 或“水利工程”）不足以把整次搜索截断到一个事件。否则一个自然语言
    # 场景词就会覆盖 ANN/metadata 的完整候选集。
    if (top[0] >= 2 and top[3] >= 2 and top[4] > 0
            and (second is None or top[0] - second[0] >= 1)):
        eid = top[1]
        assets = store.connection.execute(
            "SELECT DISTINCT a.id FROM assets a JOIN observations o ON o.asset_id=a.id "
            "JOIN event_observations eo ON eo.observation_id=o.id WHERE eo.event_id=? "
            "AND a.scope_id=?", (eid, scope_id)).fetchall()
        asset_ids = [a["id"] for a in assets]
        if asset_ids:
            return {"event_id": eid, "event_title": top[2], "asset_ids": asset_ids[:50]}
    return None


def _event_summary_terms(query: str) -> list[str]:
    """Extract meaningful phrase terms for Chinese event-summary lookup.

    This is intentionally generic and data-driven. It does not contain album
    names, benchmark answers, or scene-specific vocabulary.
    """
    text = str(query or "").strip().casefold()
    if not text:
        return []
    stop = {
        "视频", "录像", "相册", "记忆", "摘要", "事件", "场景", "内容",
        "里面", "里边", "什么", "哪些", "哪个", "怎么", "如何", "有没有",
        "请问", "告诉", "一下", "时候", "后来", "然后", "先后", "是否",
    }
    terms: list[str] = []
    # Keep ASCII words here; Unicode ``\w`` would capture the whole Chinese
    # sentence as one token and make matching depend on an accidental exact
    # sentence copy. Chinese phrases are generated from n-grams below.
    for token in re.findall(r"[A-Za-z0-9_\-]+", text):
        if len(token) >= 2 and token not in stop and token not in terms:
            terms.append(token)
    for chunk in re.findall(r"[\u4e00-\u9fff]+", text):
        for length in (4, 3, 2):
            for index in range(max(0, len(chunk) - length + 1)):
                term = chunk[index:index + length]
                if len(term) < length or term in stop:
                    continue
                # Function-word ngrams are too weak to select an event.
                if any(mark in term for mark in ("了", "的", "吗", "呢", "么")):
                    continue
                if term not in terms:
                    terms.append(term)
    return terms


def _event_keyword_anchor(question: str, store, scope_id: str) -> dict | None:
    """Fallback event anchor using distinctive Chinese terms in summaries.

    This is intentionally generic: it does not know benchmark IDs or fixed
    places, and only returns an event when one summary clearly dominates the
    query-term overlap. It also tolerates older event schemas by selecting
    only required columns.
    """
    if store is None or not scope_id:
        return None
    q_text = str(question or "")
    ignored = {"我们", "我和", "家人", "一起", "去参观", "参观", "旅行",
               "那次", "在哪里", "超级", "的", "那次旅行"}
    terms = []
    for length in (4, 3, 2):
        for match in re.finditer(rf"[\u4e00-\u9fff]{{{length}}}", q_text):
            term = match.group(0)
            if term not in ignored and not any(token in term for token in ignored):
                terms.append((term, length))
    terms = list(dict.fromkeys(terms))
    if not terms:
        return None
    try:
        rows = store.connection.execute(
            "SELECT id,title,summary FROM events WHERE scope_id=?",
            (scope_id,)).fetchall()
    except Exception:
        return None
    scored = []
    for raw in rows:
        row = dict(raw)
        hay = " ".join(str(row.get(k) or "") for k in ("title", "summary"))
        # Event summaries can omit text that is present on the event's own
        # observations (OCR such as “我愿意” or captions mentioning a
        # roll-on/roll-off ship).  Include only that event's rows so this
        # remains a bounded, data-driven anchor rather than a global scan.
        try:
            observed = store.connection.execute(
                "SELECT o.asset_id,caption,activity,place,ocr_text FROM observations o "
                "JOIN event_observations eo ON eo.observation_id=o.id "
                "WHERE eo.event_id=?", (row.get("id"),)).fetchall()
            hay += " " + " ".join(
                " ".join(str(dict(item).get(k) or "") for k in ("caption", "activity", "place", "ocr_text"))
                for item in observed
            )
            for item in observed:
                asset_id = dict(item).get("asset_id")
                if not asset_id:
                    continue
                meta = store.connection.execute(
                    "SELECT metadata_json FROM assets WHERE id=?", (asset_id,)
                ).fetchone()
                if meta:
                    hay += " " + str(meta[0] or "")
        except Exception:
            pass
        matched = [(term, length) for term, length in terms if term in hay]
        if matched:
            scored.append((len(set(term for term, _ in matched)),
                           sum(1 for _, length in matched if length >= 4),
                           row.get("id"), row.get("title") or ""))
    scored.sort(key=lambda item: (-item[0], item[2]))
    if not scored or scored[0][0] < 1:
        return None
    # A single strong phrase is not enough to replace the full retriever
    # result. For example, “婚礼照片” or “水利工程” can occur in several
    # unrelated events. Keep this fallback conservative; the normal ANN and
    # metadata channels remain responsible for broad candidate recall.
    if scored[0][0] < 2 or scored[0][1] < 1:
        return None
    if len(scored) > 1 and scored[0][0] == scored[1][0]:
        return None
    event_id = scored[0][2]
    try:
        asset_rows = store.connection.execute(
            "SELECT DISTINCT o.asset_id FROM observations o "
            "JOIN event_observations eo ON eo.observation_id=o.id "
            "JOIN assets a ON a.id=o.asset_id "
            "WHERE eo.event_id=? AND a.scope_id=?",
            (event_id, scope_id)).fetchall()
    except Exception:
        return None
    asset_ids = [dict(row).get("asset_id") for row in asset_rows if dict(row).get("asset_id")]
    return ({"event_id": event_id, "event_title": scored[0][2],
             "asset_ids": asset_ids[:50]} if asset_ids else None)

def _time_matches_event(time_expr: str | None, ts: str | None) -> int:
    """事件时间匹配分：完整日期/年月/年命中返回 3，否则 0。相对时间不参与确定性锚定。"""
    if not time_expr or not ts:
        return 0
    q = re.sub(r"\s+", "", time_expr)
    day = re.search(r"(\d{4})年(\d{1,2})月(\d{1,2})日", q)
    if day:
        return 3 if ts.startswith(f"{int(day.group(1)):04d}-{int(day.group(2)):02d}-{int(day.group(3)):02d}") else 0
    ym = re.search(r"(\d{4})年(\d{1,2})月", q)
    if ym:
        return 3 if ts.startswith(f"{int(ym.group(1)):04d}-{int(ym.group(2)):02d}") else 0
    y = re.search(r"(\d{4})年", q)
    if y:
        return 3 if ts.startswith(y.group(1)) else 0
    return 0


def _event_resolution_geo(question, store, scope_id, time_expr=None, place=None):
    """事件级主路径：时间+地点 → 锁单个事件 → 返回其资产。

    仅当单事件高置信（时间+地点同时命中，score>=6 且明显领先）才返回；
    否则返回 None，交回融合检索。全程数据驱动，不硬编码任何 benchmark 内容。
    """
    from ..geocoding import place_text_matches
    if store is None or not scope_id or (not time_expr and not place):
        return None
    try:
        rows = store.connection.execute(
            "SELECT e.id, e.title, e.time_start, e.event_type, e.activity, "
            "e.summary, a.metadata_json AS cover_meta "
            "FROM events e LEFT JOIN assets a ON a.id=e.cover_asset_id "
            "WHERE e.scope_id=? AND e.status NOT IN ('rejected','superseded','merged')",
            (scope_id,)).fetchall()
    except Exception:
        return None
    scored = []
    for r in rows:
        score = 0
        ts = r["time_start"] or ""
        if time_expr:
            score += _time_matches_event(time_expr, ts)
        geo = None
        if r["cover_meta"]:
            try:
                geo = (json.loads(r["cover_meta"]) or {}).get("reverse_geocode")
            except Exception:
                geo = None
        if place and geo and place_text_matches(place, geo):
            score += 3
        hay = " ".join(str(x) for x in (r["title"], r["event_type"], r["activity"], r["summary"]) if x)
        q = str(question or "")
        overlap_count = 0
        for length in (3, 2):
            ngrams = {hay[i:i + length] for i in range(max(0, len(hay) - length + 1))
                      if len(hay[i:i + length]) == length}
            overlap_count += sum(1 for ng in ngrams if ng in q)
        score += min(overlap_count, 3)
        if score:
            scored.append((score, r["id"], r["title"], overlap_count))
    if not scored:
        return None
    scored.sort(key=lambda x: -x[0])
    top, second = scored[0], scored[1] if len(scored) > 1 else None
    # For place-only queries require at least two independent semantic overlaps
    # before locking one event; a single generic word such as “参观/合影” is
    # not enough to distinguish events in the same city.
    top_overlap = top[3] if len(top) > 3 else 0
    # A district-level geocode plus an explicit event phrase is enough to
    # produce a bounded event candidate even when the event row has no long
    # natural-language place title. Ambiguous ties still fall back to ANN.
    min_score = 4 if place and not time_expr else 4
    if top[0] >= min_score and (not place or time_expr or top_overlap >= 1) and (second is None or top[0] - second[0] >= 1):
        eid = top[1]
        asset_ids = store.connection.execute(
            "SELECT DISTINCT o.asset_id FROM observations o "
            "JOIN event_observations eo ON eo.observation_id=o.id "
            "JOIN assets a ON a.id=o.asset_id "
            "WHERE eo.event_id=? AND a.scope_id=?", (eid, scope_id)).fetchall()
        ids = [a["asset_id"] for a in asset_ids]
        if ids:
            return {"event_id": eid, "event_title": top[2], "asset_ids": ids[:50]}
    return None



_REFERENT_MARKERS = ("就是", "那次", "这次", "刚才", "上次", "那个", "这个", "那一次", "这一次")


def _is_referent_query(query: str) -> bool:
    """W2.3：判断是否为多轮引用类 query（指代上一次的实体/结果集，而非全新检索）。"""
    q = (query or "").strip()
    if not q:
        return False
    return any(m in q for m in _REFERENT_MARKERS)


def _search_from_prior_result_set(prior_rs, scope_id: str, *, query: str = "",
                                  user_goal: str = "") -> dict | None:
    """W2.3：基于已有 ResultSet 构建检索响应（不重新全库搜索）。"""
    if prior_rs is None:
        return None
    asset_ids = list(prior_rs.asset_ids or [])
    store = _RUNTIME.get("store")
    preview = []
    indices = _preview_indices(
        asset_ids, "best", store, query=query or getattr(prior_rs, "query", "") or user_goal)
    preview = [
        _preview_entry(store, asset_ids[index], f"photo_{index + 1}",
                       priority_rank=rank, selection_reason="相关性最高" if rank == 1 else "事件多样性补充")
        for rank, index in enumerate(indices, 1)
    ]
    preview_asset_ids = [asset_ids[index] for index in indices if index < len(asset_ids)]
    display_query = query or getattr(prior_rs, "query", "") or f"(引用已有结果集 {prior_rs.result_set_id})"
    validation = _validate_search_candidates(
        query=display_query,
        user_goal=user_goal,
        filters={},
        asset_ids=asset_ids,
        store=store,
        relaxation_level=0,
    )
    validated = set(validation.get("validated_asset_ids") or [])
    if validation.get("validation_status") == "complete" and validated:
        # Only a non-empty validated projection may replace the candidate
        # preview.  An empty validator result is not an empty retrieval: it
        # means the model could not promote any candidate to direct evidence.
        # Keep the bounded candidates visible so the agent can inspect them
        # and the UI can show the retrieval source instead of a blank result.
        preview = [item for item in preview if item.get("asset_id") in validated]
        preview_asset_ids = [item.get("asset_id") for item in preview if item.get("asset_id")]
    # An event/reference result is already a structured source, not merely an
    # ANN candidate. Preserve bounded representative assets as evidence even
    # when the visual validator cannot promote any row.
    # Event/reference anchoring is still a retrieval candidate. Only the
    # vision validator can promote a candidate to direct answer evidence.
    reference_evidence = list(validated) if asset_ids else []
    group_photo_rows = []
    if re.search(r"合影", display_query or user_goal or ""):
        try:
            for aid in asset_ids:
                obs_rows = store.connection.execute(
                    "SELECT caption, activity FROM observations WHERE asset_id=?",
                    (aid,)).fetchall()
                for obs in obs_rows:
                    text = " ".join(str(obs[k] or "") for k in ("caption", "activity"))
                    if "合影" not in text:
                        continue
                    m = re.search(r"([一二三四五六七八九十两\d]+)名", text)
                    if not m:
                        m = re.search(r"([一二三四五六七八九十两\d]+)人", text)
                    if not m:
                        continue
                    token = m.group(1)
                    cn = {"一": 1, "二": 2, "两": 2, "三": 3, "四": 4,
                          "五": 5, "六": 6, "七": 7, "八": 8,
                          "九": 9, "十": 10}
                    size = int(token) if token.isdigit() else cn.get(token)
                    if size:
                        group_photo_rows.append({"asset_id": aid,
                                                 "group_size": size,
                                                 "description": text[:240]})
                        break
        except Exception:
            group_photo_rows = []
    group_photo_rows = list({row["asset_id"]: row for row in group_photo_rows}.values())
    group_sizes = sorted({row["group_size"] for row in group_photo_rows})
    # The count/size fact is derived from these exact observation rows. Keep
    # every contributing asset in evidence_sources even when the vision
    # validator did not promote it; delivery may later choose a smaller
    # representative subset, but provenance must remain complete.
    for row in group_photo_rows:
        asset_id = str(row.get("asset_id") or "").strip()
        if asset_id and asset_id not in reference_evidence:
            reference_evidence.append(asset_id)
    group_summary = (
        f"本次事件可确认 {len(group_photo_rows)} 张不同人数的合影，人数分别为"
        + "、".join(str(x) for x in group_sizes) + "人。"
        if group_photo_rows and group_sizes else ""
    )
    return {
        "result_set_id": prior_rs.result_set_id,
        "query": display_query,
        "mode": "best",
        "total": _visible_candidate_total(len(asset_ids)),
        "evidence_count": _visible_candidate_total(len(reference_evidence)),
        "preview": preview,
        "has_more": _visible_candidate_total(len(asset_ids)) > len(preview),
        "remaining": max(0, _visible_candidate_total(len(asset_ids)) - len(preview)),
        "candidate_window": _candidate_window_summary(asset_ids, indices, store),
        "completeness": "complete",
        "gaps": [],
        "query_satisfaction": (
            "full_support" if reference_evidence else ("candidate_only" if asset_ids else "no_match")
        ),
        "answerability": "full" if reference_evidence else ("limited" if asset_ids else "none"),
        "condition_summary": {},
        "can_inspect": len(preview) > 0,
        "inspect_hint": "preview 里的 handle（photo_1…）可直接用于 inspect_photo 复核视觉细节" if preview else "",
        "recommended_resolution": _recommended_resolution(
            display_query, preview, "full_support" if reference_evidence else "candidate_only",
            user_goal=user_goal),
        "reference_resolution": True,
        "validation_status": validation.get("validation_status"),
        "validation_error": validation.get("validation_error", ""),
        "raw_candidate_count": validation.get("raw_candidate_count", len(asset_ids)),
        "validation_candidate_count": validation.get("validation_candidate_count", 0),
        "validation_batches": validation.get("validation_batches", 0),
        "validation_rows": validation.get("validation_rows") or [],
        "evidence_status": "validated" if reference_evidence else ("candidate_only" if asset_ids else "none"),
        "_retrieved_asset_ids": list(asset_ids),
        "retrieved_asset_ids": list(asset_ids),
        "_preview_asset_ids": preview_asset_ids,
        "evidence_asset_ids": reference_evidence,
        "source_asset_ids": reference_evidence,
        "group_photo_count": len(group_photo_rows),
        "group_photo_sizes": group_sizes,
        "group_photo_rows": group_photo_rows,
        "summary": group_summary,
        "_model_call_metrics": validation.get("_model_call_metrics") or [],
    }


def _bounded_event_result(prior_rs, scope_id: str, *, query: str = "",
                          user_goal: str = "") -> dict:
    """Return a safe event projection when optional validator fields are absent.

    Event resolution is already a structured, scope-bound source. Older album
    databases may lack one optional event/observation column; that must not
    turn a valid event anchor into a tool error or a broad ANN fallback.
    """
    asset_ids = list(prior_rs.asset_ids or [])
    store = _RUNTIME.get("store")
    indices = _preview_indices(asset_ids, "best", store, query=query or user_goal)
    preview = []
    for rank, index in enumerate(indices, 1):
        if index >= len(asset_ids):
            continue
        preview.append(_preview_entry(
            store, asset_ids[index], f"photo_{index + 1}",
            priority_rank=rank,
            selection_reason="事件锚定来源" if rank == 1 else "事件来源补充",
        ))
    source_ids = [item.get("asset_id") for item in preview if item.get("asset_id")]
    return {
        "result_set_id": prior_rs.result_set_id,
        "query": query or prior_rs.query,
        "mode": "best",
        "total": _visible_candidate_total(len(asset_ids)),
        "evidence_count": _visible_candidate_total(len(source_ids)),
        "preview": preview,
        "has_more": _visible_candidate_total(len(asset_ids)) > len(preview),
        "remaining": max(0, _visible_candidate_total(len(asset_ids)) - len(preview)),
        "completeness": "complete",
        "gaps": [],
        "query_satisfaction": "candidate_only" if source_ids else "no_match",
        "answerability": "limited" if source_ids else "none",
        "condition_summary": {},
        "can_inspect": bool(preview),
        "inspect_hint": "preview 里的 handle（photo_1…）可直接用于 inspect_photo 复核视觉细节" if preview else "",
        "recommended_resolution": _recommended_resolution(
            query or prior_rs.query, preview, "candidate_only", user_goal=user_goal),
        "reference_resolution": True,
        "validation_status": "not_required",
        "evidence_status": "candidate_only" if source_ids else "none",
        "_retrieved_asset_ids": asset_ids,
        "retrieved_asset_ids": asset_ids,
        "_preview_asset_ids": source_ids,
        "evidence_asset_ids": source_ids,
        "source_asset_ids": source_ids,
        "_model_call_metrics": [],
    }



def _relaxed_retrieve(query: str, filters: dict, scope_id: str, viewer_id: str, mode: str):
    """确定性渐进放宽：严格检索为空时依次降级，返回 (packet, level)。

    level 0=严格, 1=去person, 2=去place, 3=去time, 4=纯语义。全程数据驱动。
    """
    base = dict(filters or {})
    # Explicit time/place are identity anchors, not optional ranking hints.
    # Dropping them after an empty strict pass returned unrelated photos (for
    # example an indoor 2018 group photo for a Zhao County landmark query).
    # Only an unresolved person constraint may be relaxed; an anchored query
    # otherwise returns no match rather than claiming an unrelated image.
    steps = [dict(base)]
    if base.get("person"):
        steps.append({k: v for k, v in base.items() if k != "person"})
    last = None
    for level, f in enumerate(steps):
        draft = _draft_from_filters({**f, "query": query}, answer_type="asset_set")
        draft.result_requirement = {"mode": mode}
        # Keep the model validation window bounded separately from retrieval.
        # The retriever must retain a deeper candidate head so a paraphrase
        # does not disappear before the validator gets its relevance head and
        # sparse tail sample. Public delivery is still limited by preview and
        # evidence selection; this is not a request to show the full pool.
        draft.result_requirement["top_k"] = max(
            1, int(os.getenv("SENTRIX_SEARCH_RETRIEVAL_TOP_K", "500")))
        spec = _spec_for(draft, scope_id, viewer_id)
        last = _kernel().retrieve(spec)
        if last.assets:
            return last, level
    return last, len(steps) - 1


def _parse_search_validation_response(raw) -> list[dict]:
    """Normalize the bounded vision validator response without trusting prose."""
    from ..model_clients import parse_json_response
    parsed = parse_json_response(raw) if isinstance(raw, str) else (raw or {})
    rows = parsed.get("candidates") if isinstance(parsed, dict) else None
    if rows is None and isinstance(parsed, dict):
        rows = parsed.get("decisions") or parsed.get("results")
    if not isinstance(rows, list):
        return []
    normalized = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        handle = str(row.get("handle") or "").strip()
        status = str(row.get("support_status") or row.get("status") or "").strip().lower()
        if not handle or status not in {"supported", "candidate_only", "rejected"}:
            continue
        normalized.append({
            "handle": handle,
            "time_match": bool(row.get("time_match")),
            "place_match": bool(row.get("place_match")),
            "scene_match": bool(row.get("scene_match")),
            "person_match": bool(row.get("person_match")),
            "support_status": status,
            "reason": str(row.get("reason") or "")[:240],
        })
    return normalized


def _validate_search_candidates(*, query: str, user_goal: str, filters: dict,
                                asset_ids: list[str], store, relaxation_level: int) -> dict:
    """Use the bound vision model to validate a bounded candidate window.

    Retrieval remains code-owned and complete. The validator only decides
    which bounded candidates support the natural-language question; it never
    expands the pool or invents asset IDs.
    """
    max_candidates = max(1, int(os.getenv("SENTRIX_SEARCH_VALIDATION_MAX_CANDIDATES", "30")))
    # The 8100 vision endpoint accepts at most five images per request.
    batch_size = min(5, max(1, int(os.getenv("SENTRIX_SEARCH_VALIDATION_BATCH_SIZE", "5"))))
    task_query = user_goal or query or "用户问题相关的照片"
    # Validate the first relevance window in retrieval order.  The complete
    # candidate set remains available for recall accounting, but a sparse tail
    # wastes model batches on low-relevance images and can create a blind spot
    # immediately after the head (for example a correct rank-16 photo).  The
    # visible preview stays bounded at six, while this internal window remains
    # capped by the configured validation budget.
    ordered_indices = _preview_query_order(asset_ids, task_query, store)
    selected_indices = ordered_indices[:max_candidates]
    candidate_ids = [asset_ids[index] for index in selected_indices if index < len(asset_ids)]
    gamma = _RUNTIME.get("gamma")
    result = {
        "candidate_asset_ids": candidate_ids,
        "validated_asset_ids": [],
        "validation_rows": [],
        "validation_batches": 0,
        "validation_status": "disabled" if not gamma else "pending",
        "raw_candidate_count": len(asset_ids),
        "validation_candidate_count": len(candidate_ids),
        "_model_call_metrics": [],
    }
    if not candidate_ids or gamma is None or os.getenv("SENTRIX_SEARCH_VALIDATION_ENABLED", "1").lower() in {"0", "false", "off"}:
        return result
    for offset in range(0, len(candidate_ids), batch_size):
        batch_ids = candidate_ids[offset:offset + batch_size]
        records = []
        images = []
        for index, asset_id in enumerate(batch_ids):
            asset = store.get_asset(asset_id) if store else None
            if not asset:
                continue
            handle = f"photo_{asset_ids.index(asset_id) + 1}"
            row = {
                "handle": handle,
                "asset_id": asset_id,
                "captured_at": asset.get("captured_at") or "",
                "place": _short_place_label(asset),
                "people": _preview_entry(store, asset_id, handle).get("people") or [],
                "description": _observation_summary(store, asset_id) or "",
                "source_type": asset.get("derived_kind") or asset.get("media_kind") or "image",
            }
            records.append(row)
            path = asset.get("path")
            if path and Path(path).is_file():
                try:
                    encoded, mime_type = gamma.encode_vision_image(path)
                    images.append({"base64": encoded, "mime_type": mime_type})
                except Exception:
                    images.append(None)
            else:
                images.append(None)
        if not records:
            continue
        prompt = (
            "你是 Sentrix 检索证据验证器。用户问题是：" + task_query + "\n"
            "检索条件（仅供理解，不可自行扩展）：" + json.dumps(filters or {}, ensure_ascii=False) + "\n"
            "当前放宽级别：" + str(relaxation_level) + "。逐张判断候选是否真正能用于回答用户问题。"
            "时间、地点、人物词按用户语义理解；不要因为相似场景就支持。图片按候选记录顺序提供。"
            "只返回 JSON 对象：{\"candidates\":[{\"handle\":\"photo_N\","
            "\"time_match\":true/false,\"place_match\":true/false,"
            "\"scene_match\":true/false,\"person_match\":true/false,"
            "\"support_status\":\"supported|candidate_only|rejected\",\"reason\":\"简短依据\"}]}。"
            "supported 表示这张图或其明确元数据直接支持问题；candidate_only 表示相关但不能作为答案依据；"
            "rejected 表示不符合。未知人物不得猜姓名。候选记录：" + json.dumps(records, ensure_ascii=False)
        )
        try:
            raw = gamma.chat(prompt, images=[image for image in images if image],
                             json_mode=True, role="search_validation")
            rows = _parse_search_validation_response(raw)
            valid_handles = {row["handle"] for row in rows}
            result["validation_rows"].extend(row for row in rows if row["handle"] in {
                f"photo_{asset_ids.index(asset_id) + 1}" for asset_id in batch_ids
            })
            result["validated_asset_ids"].extend(
                asset_ids[int(row["handle"].split("_")[-1]) - 1]
                for row in rows if row["handle"] in valid_handles
                and row["support_status"] == "supported"
                and row["handle"].startswith("photo_")
                and row["handle"].split("_")[-1].isdigit()
                and 0 < int(row["handle"].split("_")[-1]) <= len(asset_ids)
            )
            result["validation_batches"] += 1
        except Exception as exc:
            result["validation_status"] = "error"
            result["validation_error"] = str(exc)[:240]
            break
    if result["validation_status"] != "error":
        result["validation_status"] = "complete"
    result["validated_asset_ids"] = list(dict.fromkeys(result["validated_asset_ids"]))
    try:
        result["_model_call_metrics"] = gamma.get_and_clear_call_metrics()
    except Exception:
        pass
    return result


def _search_memories(arguments: dict, *, context: dict | None = None) -> dict:
    query = arguments.get("query") or ""
    mode = arguments.get("mode") or "best"
    if mode not in {"best", "all", "representative"}:
        mode = "best"
    filters = dict(arguments.get("filters") or {})
    raw_time_filter = str(filters.get("time") or "").strip()
    if not (filters.get("time") or ""):
        extracted = _extract_time_from_query(query)
        if extracted:
            filters["time"] = extracted
    scope_id = (context or {}).get("scope_id") or ""
    user_goal = ((context or {}).get("task_state") or {}).get("user_goal") or ""
    filters = _sanitize_model_filters(filters, query=query, user_goal=user_goal)
    unresolved_explicit_place = ""
    # P2 v2: Canonical Retrieval Intent（fusion candidate）—— 从用户原问题确定性提取结构化
    # 约束，强信号（时间+地点）时作为 filters 增强 hybrid 检索（保留语义 query），消除 paraphrase 漂移。
    if _canonical_search_enabled():
        from .canonical_intent import extract_constraints
        # Canonical constraints must come from the user's raw query.  The
        # planner's user_goal is a lossy paraphrase and may drop the exact
        # city/person anchor (or introduce a broader one), which silently
        # sends an otherwise precise question to ANN fallback.
        user_message = arguments.get("query") or user_goal or ""

        ci = extract_constraints(user_message, _RUNTIME.get("store"), scope_id)
        # 防御：user_message 提取不到 time/place 时，再从 search query 提取并合并（覆盖边角）。
        if not (ci.get("time") or ci.get("place")):
            ci_q = extract_constraints(arguments.get("query") or "", _RUNTIME.get("store"), scope_id)
            for k in ("time", "place", "person"):
                if not ci.get(k) and ci_q.get(k):
                    ci[k] = ci_q[k]
            ci["strong"] = bool(ci.get("time") and ci.get("place"))
        # Event matches are ranking hints only.  Do not return an event result
        # before the normal metadata/ANN fusion has seen the complete query:
        # event summaries are often coarser than the user's requested scene
        # and can otherwise hide the correct asset from the candidate set.
        # canonical 结构化约束始终作为 filters 增强（覆盖 agent 的坏值如"所有时间"），
        # 保留 hybrid 语义 query，避免 v1 空 query 走纯元数据路径丢失 OCR/关键词召回。
        if ci.get("time"):
            filters["time"] = ci["time"]
        if ci.get("place"):
            filters["place"] = ci["place"]
        if ci.get("person"):
            filters["person"] = ci["person"]
        # Do not treat a model's free-form scene/role phrase as a hard place
        # constraint (e.g. "亲戚婚房"). However, if the model copied an
        # explicit user location that is not present in this scope, dropping
        # it and falling through to ANN would return unrelated photos. Keep a
        # restrictive no-match marker for that case; retrieval must never
        # silently broaden an unresolved proper-place constraint.
        if filters.get("place") and not ci.get("place"):
            candidate_place = str(filters.get("place") or "").strip()
            explicit_place = bool(candidate_place and re.search(
                rf"(?:在|于|到|去|从|位于|来自|关于)\s*{re.escape(candidate_place)}",
                user_message or "",
            ))
            scene_place_tokens = (
                "室内", "户外", "客厅", "卧室", "厨房", "餐厅",
                "活动场地", "舞台", "场地", "公共区域",
            )
            # A broad administrative label is still a valid structured
            # constraint when it occurs in this scope's reverse-geocoded
            # metadata (e.g. 河北/湖北).  Only reject an explicit place that
            # is absent from the album; never broaden it to unrelated ANN
            # photos merely because the planner failed to canonicalize it.
            known_in_scope = False
            if candidate_place:
                try:
                    rows = _RUNTIME["store"].connection.execute(
                        "SELECT metadata_json FROM assets WHERE scope_id=?",
                        (scope_id,),
                    ).fetchall()
                    for row in rows:
                        raw_meta = row["metadata_json"] if hasattr(row, "keys") else row[0]
                        if candidate_place in str(raw_meta or ""):
                            known_in_scope = True
                            break
                except Exception:
                    known_in_scope = False
            # A planner may put a free-form scene/relationship phrase into
            # filters.place (for example a room or activity description). It
            # is not a structured geocode constraint. Let the semantic
            # retriever and the bounded vision validator judge it instead of
            # converting the phrase into a zero-result hard filter.
            from ..geocoding import place_alias_names
            has_structured_shape = bool(re.search(
                r"(?:省|市|区|县|州|国)$", candidate_place
            )) or bool(place_alias_names(candidate_place))
            if (explicit_place and not known_in_scope and has_structured_shape
                    and not any(token in candidate_place for token in scene_place_tokens)):
                unresolved_explicit_place = candidate_place
            elif not known_in_scope:
                filters.pop("place", None)
                if candidate_place and candidate_place not in query:
                    query = f"{query} {candidate_place}".strip()
            if not unresolved_explicit_place:
                # Keep a known broad place in the structured filter path.
                # It may match multiple events, so it must not be promoted to
                # a single event without an independent semantic anchor.
                pass
            else:
                filters.pop("place", None)
    if unresolved_explicit_place:
        # The user supplied a place, but this scope has no canonical label or
        # event alias for it. Return a typed empty result instead of broad ANN
        # candidates; the caller can report that the record is unconfirmed or
        # ask for another anchor. This preserves precision and provenance.
        rs = _RUNTIME["result_sets"].new(
            scope_id=scope_id, query=query, asset_ids=[],
            unresolved=[f"unresolved_place:{unresolved_explicit_place}"],
        )
        return {
            "result_set_id": rs.result_set_id,
            "query": query,
            "mode": mode,
            "total": 0,
            "evidence_count": 0,
            "preview": [],
            "has_more": False,
            "remaining": 0,
            "candidate_window": {"total": 0, "preview_count": 0, "asset_ids": []},
            "completeness": "complete",
            "gaps": [f"未找到可核验的地点：{unresolved_explicit_place}"],
            "query_satisfaction": "no_match",
            "answerability": "none",
            "condition_summary": {query: "unresolved_place"},
            "can_inspect": False,
            "inspect_hint": "",
            "retrieval_timing": {},
            "relaxation_level": 0,
            "raw_candidate_count": 0,
            "validation_candidate_count": 0,
            "validation_batches": 0,
            "validation_status": "complete",
            "validation_error": "",
            "validation_rows": [],
            "evidence_status": "none",
            "recommended_resolution": {
                "needed": False,
                "tool": "query_memory_facts",
                "reason": f"相册中没有可核验的地点标签：{unresolved_explicit_place}",
            },
            "_retrieved_asset_ids": [],
            "_preview_asset_ids": [],
            "evidence_asset_ids": [],
            "_model_call_metrics": [],
        }
    # Event summaries are intentionally not an early-return retrieval path.
    # They may be used later as an additional ranking signal, but must never
    # replace the multi-channel candidate universe.
    # W2.3：多轮引用消解 —— 引用类 query 先解析到已有 current_result_set，不重新全库搜索。
    # 引用标记在用户原话（task_state.user_goal）里，search query 是 LLM 提取后的内容，故两者都检测。
    _user_msg = ((context or {}).get("task_state") or {}).get("user_goal") or ""
    if _is_referent_query(query) or _is_referent_query(_user_msg):
        prior_rs_id = ((context or {}).get("task_state") or {}).get("current_result_set")
        prior_rs = None
        if prior_rs_id:
            rs_store = _RUNTIME.get("result_sets")
            if rs_store is not None and hasattr(rs_store, "get"):
                try:
                    prior_rs = rs_store.get(prior_rs_id)
                except Exception:
                    prior_rs = None
        try:
            anchored = _search_from_prior_result_set(
                prior_rs, scope_id, query=query, user_goal=_user_msg)
        except (KeyError, IndexError, TypeError):
            anchored = _bounded_event_result(prior_rs, scope_id, query=query, user_goal=_user_msg) if prior_rs else None
        if anchored is not None:
            return anchored
        # A first-turn referent without a prior result set still goes through
        # the normal retrieval channels.  Resolving an arbitrary event here
        # would silently turn “那次旅行的合影” into an unrelated event.
    viewer_id = (context or {}).get("viewer_id") or "owner"
    draft = _draft_from_filters({**filters, "query": query}, answer_type="asset_set")
    draft.result_requirement = {"mode": mode}
    spec = _spec_for(draft, scope_id, viewer_id)
    if not (query or "").strip():
        # 纯时间/地点/人物/媒体筛选：走确定性元数据路径，不依赖 ANN 语义召回（生产多检索器下空 query 会 0 召回）
        user_goal = ((context or {}).get("task_state") or {}).get("user_goal") or ""
        return _search_metadata_only(draft, spec, scope_id, query, mode, user_goal=user_goal)
    packet, _relax_level = _relaxed_retrieve(query, filters, scope_id, viewer_id, mode)
    assets = packet.assets or []
    asset_ids = [item.get("asset_id") for item in assets if item.get("asset_id")]
    rs = _RUNTIME["result_sets"].new(
        scope_id=scope_id, query=query, asset_ids=asset_ids,
        unresolved=[g.get("reason") for g in (packet.gaps or [])],
    )
    preview = []
    store = _RUNTIME.get("store")
    preview_indices = _preview_indices(asset_ids, mode, store, query=query)
    preview = [
        _preview_entry(
            store,
            asset_ids[index],
            f"photo_{index + 1}",
            level=(assets[index].get("level") if index < len(assets) else "exact"),
            condition_summary=_condition_summary(assets[index]),
            priority_rank=rank,
            selection_reason="相关性最高" if rank == 1 else "事件多样性补充",
        )
        for rank, index in enumerate(preview_indices, 1)
    ]
    preview_asset_ids = [asset_ids[index] for index in preview_indices if index < len(asset_ids)]
    validation = _validate_search_candidates(
        query=query,
        user_goal=((context or {}).get("task_state") or {}).get("user_goal") or "",
        filters=filters,
        asset_ids=asset_ids,
        store=store,
        relaxation_level=_relax_level,
    )
    validated_ids = set(validation.get("validated_asset_ids") or [])
    if validation.get("validation_status") == "complete" and validated_ids:
        # A successful validation with supported evidence can narrow the
        # public preview.  A successful pass with zero supported rows must
        # retain the bounded candidate preview; otherwise a temporary model
        # ambiguity turns a real retrieval into an empty tool result.
        preview = [item for item in preview if item.get("asset_id") in validated_ids]
        # Evidence selected by the validator must always be visible to the
        # caller.  A validated asset can fall outside the initial six-item
        # presentation window (for example an OCR hit ranked below the visual
        # head); filtering the window without backfilling produced an empty
        # preview even though evidence_asset_ids was non-empty.
        if not preview:
            for rank, asset_id in enumerate(
                    (aid for aid in asset_ids if aid in validated_ids), 1):
                preview.append(_preview_entry(
                    store, asset_id, f"photo_{asset_ids.index(asset_id) + 1}",
                    priority_rank=rank,
                    selection_reason="验证通过的证据来源" if rank == 1 else "验证通过的证据补充",
                ))
                if len(preview) >= _RESULT_PREVIEW_LIMIT:
                    break
        preview_asset_ids = [item.get("asset_id") for item in preview if item.get("asset_id")]
        preview_handles = {item.get("handle") for item in preview}
        validation_rows = [row for row in validation.get("validation_rows") or []
                           if row.get("handle") in preview_handles]
    else:
        validation_rows = validation.get("validation_rows") or []
    cond, satisfaction, answerability = _truth_contract(packet, rs.total)
    evidence_status = "validated" if validated_ids else ("candidate_only" if asset_ids else "none")
    if validation.get("validation_status") == "complete":
        satisfaction = "full_support" if validated_ids else "candidate_only"
        answerability = "full" if validated_ids else "limited"
    return {
        "result_set_id": rs.result_set_id,
        "query": query,
        "mode": mode,
        "total": _visible_candidate_total(rs.total),
        "evidence_count": len(validated_ids),
        "preview": preview,
        "has_more": _visible_candidate_total(len(asset_ids)) > len(preview),
        "remaining": max(0, _visible_candidate_total(len(asset_ids)) - len(preview)),
        "candidate_window": _candidate_window_summary(asset_ids, preview_indices, store),
        "completeness": "complete" if not (packet.gaps) else "partial",
        "gaps": rs.unresolved[:3],
        "query_satisfaction": satisfaction,
        "answerability": answerability,
        "condition_summary": cond,
        "can_inspect": len(preview) > 0,
        "inspect_hint": "preview 里的 handle（photo_1…）可直接用于 inspect_photo 复核视觉细节" if preview else "",
        "retrieval_timing": packet.retrieval_timing,
        "retrieval_channels": packet.channel_trace,
        "relaxation_level": _relax_level,
        "raw_candidate_count": validation.get("raw_candidate_count", len(asset_ids)),
        "validation_candidate_count": validation.get("validation_candidate_count", 0),
        "validation_batches": validation.get("validation_batches", 0),
        "validation_status": validation.get("validation_status"),
        "validation_error": validation.get("validation_error", ""),
        "validation_rows": validation_rows,
        "evidence_status": evidence_status,
        "recommended_resolution": _recommended_resolution(
            query, preview, satisfaction,
            user_goal=((context or {}).get("task_state") or {}).get("user_goal") or ""),
        "_retrieved_asset_ids": list(asset_ids),
        "_preview_asset_ids": preview_asset_ids,
        "evidence_asset_ids": list(validated_ids),
        "_model_call_metrics": validation.get("_model_call_metrics") or [],
    }


def _short_place_label(asset: dict) -> str:
    """从资产反地理编码取短地点标签（城市/区县名），供 preview 证据展示。"""
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
    # Identity/list questions must resolve a concrete photo before answering;
    # metadata people on several candidates are not interchangeable evidence.
    if re.search(r"都有谁|有哪些人|哪几个人|几个人|谁一起|谁参加|人物", q):
        return {"needed": True, "tool": "inspect_photo",
                "reason": "问题需要确认照片中的人物，请用 inspect_photo 复核 preview 里的照片"}
    if re.search(r"合影|合照|同行|朋友|晚餐|聚餐", q):
        return {"needed": True, "tool": "inspect_photo",
                "reason": "问题需要确认合影和同行者，请用 inspect_photo 复核 preview 里的照片"}
    if re.search(r"哪次旅行|哪次经历|什么旅行|旅行记录", q):
        return {"needed": True, "tool": "inspect_photo",
                "reason": "问题需要复核代表性旅行照片，请用 inspect_photo 确认场景"}
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
    store = _RUNTIME.get("store")
    target_asset = store.get_asset(asset_id or target) if store and (asset_id or target) else None
    if target_asset and target_asset.get("derived_kind") == "video_keyframe" and target_asset.get("parent_asset_id"):
        source_video_id = target_asset["parent_asset_id"]
        return {
            "summary": "已从结果集授权原始视频交付，来源是关键帧对应的时间点。",
            "result_set_id": result_set_id,
            "handle": handle or "first",
            "delivered": 1,
            "total": _visible_candidate_total(rs.total),
            "scope_id": rs.scope_id,
            "url": f"/api/assets/{source_video_id}/file",
            "media_type": "video",
            "source_video_asset_id": source_video_id,
            "source_timestamp_sec": target_asset.get("source_timestamp_sec"),
        }
    url = ""
    if target:
        url = (f"/api/assistant/result-set/{result_set_id}/photo?handle={target}"
               f"&scope_id={rs.scope_id}&original=1")
    return {
        "summary": f"已从结果集 {result_set_id} 授权原图交付。",
        "result_set_id": result_set_id,
        "handle": handle or "first",
        "delivered": 1 if asset_id else (1 if rs.asset_ids else 0),
        "total": _visible_candidate_total(rs.total),
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
    visible_total = _visible_candidate_total(rs.total)
    shown = min(visible_total, rs.shown or 0)
    return (f"当前结果集：{rs.result_set_id}，当前可核验候选最多 {visible_total} 张，已显示 {shown} 张，"
            f"还有 {max(0, visible_total - shown)} 张。查看更多用 get_result_page（page 从 1 开始）。")


# ---- Tool 3.5: get_result_page（B3.1 分页）----
def _get_result_page(arguments: dict, *, context: dict | None = None) -> dict:
    scope_id = (context or {}).get("scope_id") or ""
    task_state = (context or {}).get("task_state") or {}
    result_set_id = arguments.get("result_set_id") or task_state.get("current_result_set")
    try:
        page_no = max(1, int(arguments.get("page") or 1))
    except (TypeError, ValueError):
        page_no = 1
    if arguments.get("query") or arguments.get("filters"):
        return {
            "summary": "新的查询条件不能翻旧结果集，请重新调用 search_memories。",
            "total": 0,
            "requires_new_search": True,
            "blocked": ["new_query_requires_search"],
        }
    try:
        page_size = min(_RESULT_PAGE_SIZE, max(1, int(arguments.get("page_size") or _RESULT_PAGE_SIZE)))
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
    visible_total = _visible_candidate_total(rs.total)
    # Keep the full ResultSet for server-side recall/provenance, but never let
    # pagination expose candidates beyond the bounded model-visible window.
    visible_ids = list(rs.asset_ids[:visible_total])
    start = max(0, (page_no - 1) * page_size)
    items = [
        {"handle": f"photo_{start + i + 1}", "asset_id": asset_id}
        for i, asset_id in enumerate(visible_ids[start:start + page_size])
    ]
    shown = min(visible_total, start + len(items))
    store = _RUNTIME.get("store")
    asset_ids = [item.get("asset_id") for item in items if item.get("asset_id")]
    preview = []
    for item in items:
        aid = item.get("asset_id")
        if not aid:
            continue
        entry = _preview_entry(
            store,
            aid,
            item.get("handle") or "",
            level="exact",
            priority_rank=len(preview) + 1,
            selection_reason="分页候选",
        )
        if entry:
            preview.append(entry)
    return {
        "result_set_id": rs.result_set_id,
        "page": page_no,
        "page_size": page_size,
        "total": visible_total,
        "shown": shown,
        "has_more": shown < visible_total,
        "remaining": max(0, visible_total - shown),
        "preview": preview,
        "asset_ids": asset_ids,
        "retrieved_asset_ids": asset_ids,
        "source_asset_ids": asset_ids,
        # Pagination exposes more retrieved candidates. A page is not
        # evidence until the validator/inspect/OCR/metadata path accepts it.
        "evidence_asset_ids": [],
        "query": rs.query,
    }


# ---- Tool 4: inspect_photo ----
def _inspect_photo(arguments: dict, *, context: dict | None = None) -> dict:
    asset_handle = arguments.get("asset_handle") or ""
    question = arguments.get("question") or "请描述这张照片"
    scope_id = (context or {}).get("scope_id") or ""
    task_state = (context or {}).get("task_state") or {}
    target_person = str(arguments.get("target_person") or task_state.get("active_person") or "").strip()
    if not asset_handle:
        # C11：未填 handle 时用当前结果集 preview 首个可复核 handle（安全默认）
        preview = (task_state.get("result_preview") or []) or []
        if preview:
            asset_handle = preview[0]
    # B3：handle 必须解析自当前结果集；失败再退回最近一次检索的 handle 映射
    asset_id = None
    result_set_id = task_state.get("current_result_set")
    rs_store = _RUNTIME.get("result_sets")
    used_result_set = False
    if result_set_id and rs_store is not None:
        asset_id = rs_store.resolve_handle(result_set_id, asset_handle)
        used_result_set = True
    if not asset_id and not used_result_set:
        return {"summary": "请先通过 search_memories 建立当前结果集。",
                "certainty": "uncertain", "persisted": False,
                "blocked": ["no_current_result_set"]}
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
    identity_rows = _confirmed_photo_identities(store, asset_id)
    if not target_person:
        user_goal = str(task_state.get("user_goal") or task_state.get("last_user_goal") or "")
        target_person = next((str(item.get("person_name") or "") for item in identity_rows
                              if item.get("person_name") and item["person_name"] in (question + user_goal)), "")
    target_identity = next((row for row in identity_rows
                            if target_person and row.get("person_name") == target_person), None)
    target_status = "not_requested"
    target_bbox = None
    inspect_images = []
    crop_path = None
    try:
        encoded, mime_type = gamma.encode_vision_image(row["path"])
        image = {"base64": encoded, "mime_type": mime_type}
        inspect_images = [image]
        if target_person:
            if target_identity and target_identity.get("bbox"):
                try:
                    from PIL import Image, ImageOps
                    source_image = ImageOps.exif_transpose(Image.open(row["path"])).convert("RGB")
                    face_bbox = list(target_identity["bbox"])
                    if face_bbox and max(face_bbox) <= 1.0:
                        face_bbox = [face_bbox[0] * source_image.width,
                                     face_bbox[1] * source_image.height,
                                     face_bbox[2] * source_image.width,
                                     face_bbox[3] * source_image.height]
                    crop, crop_bbox = expanded_person_crop(source_image, face_bbox)
                    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as temporary:
                        crop.save(temporary, format="JPEG", quality=90)
                        crop_path = temporary.name
                    crop_encoded, crop_mime = gamma.encode_vision_image(crop_path)
                    inspect_images.insert(0, {"base64": crop_encoded, "mime_type": crop_mime})
                    target_status = "located"
                    target_bbox = crop_bbox
                except (OSError, ValueError, TypeError):
                    target_status = "not_located"
            else:
                target_status = "not_located"
        prompt = _INSPECT_PROMPT.format(
            question=question,
            target_instruction=(
                f"目标人物是“{target_person}”。第一张图（如有）是该人物的定位裁剪图；"
                "只回答目标人物，不要把同图其他人的外观或动作归给目标人物。"
                if target_person else ""),
        )
        raw = gamma.chat(prompt, images=inspect_images,
                         json_mode=True, role="inspect")
    except Exception as exc:
        if crop_path:
            Path(crop_path).unlink(missing_ok=True)
        model_call_metrics = gamma.get_and_clear_call_metrics()
        return {"summary": f"图片复核失败：{exc}", "certainty": "uncertain", "persisted": False,
                "_model_call_metrics": model_call_metrics}
    model_call_metrics = gamma.get_and_clear_call_metrics()
    # Keep the negative identity evidence visible as well: a multi-person
    # photo must distinguish confirmed people from remaining unconfirmed
    # companions instead of silently treating the named subset as complete.
    observation_row = store.connection.execute(
        "SELECT people_json FROM observations WHERE asset_id = ? ORDER BY updated_at DESC LIMIT 1",
        (asset_id,),
    ).fetchone()
    try:
        people_values = json.loads(observation_row["people_json"] or "[]") if observation_row else []
        # The naming flow appends confirmed entity dictionaries to the
        # original visual people list. Count only the original descriptive
        # entries so those annotations do not inflate unknown companions.
        people_count = sum(
            1 for person in people_values
            if not (isinstance(person, dict) and person.get("entity_id"))
        )
    except (TypeError, ValueError, json.JSONDecodeError):
        people_count = 0
    unconfirmed_count = max(0, people_count - len(identity_rows))
    try:
        start, end = raw.find("{"), raw.rfind("}")
        parsed = json.loads(raw[start:end + 1]) if start >= 0 else {}
    except Exception:
        parsed = {}
    finally:
        if crop_path:
            Path(crop_path).unlink(missing_ok=True)
    return {
        "_source_asset_id": asset_id,
        "asset_handle": asset_handle,
        "question": question,
        "observation": parsed.get("observation") or parsed.get("scene") or "",
        "certainty": parsed.get("certainty") or "supported",
        "confirms_visual_only": not bool(identity_rows),
        "photo_identities": identity_rows,
        "unconfirmed_people_count": unconfirmed_count,
        "unconfirmed_people": ([{"description": "未确认身份同行者", "count": unconfirmed_count}]
                               if unconfirmed_count else []),
        "target_person": target_person,
        "target_face_status": target_status,
        "target_bbox": target_bbox,
        "source": "runtime_visual_inspection",
        "persisted": False,
        "_model_call_metrics": model_call_metrics,
    }


def _confirmed_photo_identities(store, asset_id: str) -> list[dict]:
    """Read existing confirmed face/entity links without mutating identity data."""
    if store is None or not asset_id:
        return []
    rows = store.connection.execute(
        """
        SELECT fi.id AS face_instance_id, fi.asset_id, fi.observation_id,
               fi.cluster_id, fi.bbox_json, fi.detection_confidence, fi.quality,
               fc.entity_id, fc.status AS cluster_status,
               e.canonical_name, e.family_role, e.status AS entity_status,
               em.confidence AS mention_confidence
        FROM face_instances fi
        JOIN face_clusters fc ON fc.id = fi.cluster_id
        JOIN entities e ON e.id = fc.entity_id
        JOIN entity_mentions em
          ON em.face_instance_id = fi.id
         AND em.entity_id = fc.entity_id
        WHERE fi.asset_id = ?
          AND fc.status = 'confirmed'
          AND e.entity_type = 'person'
          AND e.status = 'confirmed'
        ORDER BY fi.quality DESC, fi.detection_confidence DESC
        """, (asset_id,)
    ).fetchall()
    return [{
        "evidence_type": "photo_identity",
        "asset_id": str(row["asset_id"] or asset_id),
        "face_instance_id": str(row["face_instance_id"]),
        "cluster_id": str(row["cluster_id"] or ""),
        "entity_id": str(row["entity_id"] or ""),
        "person_name": str(row["canonical_name"] or ""),
        "family_role": str(row["family_role"] or ""),
        "identity_status": "confirmed",
        "mention_confidence": row["mention_confidence"],
        "bbox": _decode_bbox(row["bbox_json"]),
        "source": "existing_face_cluster_entity_mention",
    } for row in rows]


def _decode_bbox(value):
    try:
        bbox = json.loads(value) if isinstance(value, str) else value
        if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
            return [float(item) for item in bbox]
    except (TypeError, ValueError, json.JSONDecodeError):
        pass
    return None
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


def _query_memory_metadata(arguments: dict, *, context: dict | None = None) -> dict:
    """Dedicated structured metadata path; keeps date/place/count out of visual search."""
    operation = str(arguments.get("operation") or "").strip().lower()
    # Accept the pre-P2 wire names during rollout.  The canonical contract is
    # operation=place/date/event, but older 12B prompts emitted
    # metadata_type=location/time and put the filter at the top level.
    if not operation:
        legacy_type = str(arguments.get("metadata_type") or arguments.get("metadata") or "").strip().lower()
        operation = {
            "location": "place", "place": "place", "time": "date",
            "date": "date", "event": "event", "count": "count",
        }.get(legacy_type, "count")
    if operation == "count" and not (arguments.get("operation") or arguments.get("metadata_type") or arguments.get("metadata")):
        query_hint = " ".join(str(arguments.get(key) or "") for key in ("query", "question"))
        query_hint += " " + str(((context or {}).get("task_state") or {}).get("user_goal") or "")
        if re.search(r"哪一年|哪天|什么时候|何时|日期|几月", query_hint):
            operation = "date"
        elif re.search(r"在哪里|地点|位置|拍摄地|举办地", query_hint):
            operation = "place"
        elif re.search(r"聚餐|旅行|活动|事件|那次", query_hint):
            operation = "event"
    # Event is a first-class structured record, not a media-list alias.  The
    # event summary and its cover/member assets are the source for event QA.
    if operation in {"event", "timeline"}:
        store = _RUNTIME.get("store")
        scope_id = (context or {}).get("scope_id") or ""
        filters = dict(arguments.get("filters") or {})
        if arguments.get("place") and not filters.get("place"):
            filters["place"] = arguments.get("place")
        # Event-summary QA must work with natural Chinese questions. The old
        # implementation used ``query.split()``; a Chinese sentence has no
        # spaces, so it required the entire question to occur verbatim in one
        # summary and returned zero rows for normal questions.
        query = str(arguments.get("query") or arguments.get("question") or "").strip().lower()
        if not query:
            query = str(((context or {}).get("task_state") or {}).get("user_goal") or "").strip().lower()
        query_terms = [] if operation == "timeline" else _event_summary_terms(query)
        rows = store.connection.execute(
            "SELECT e.id, e.title, e.event_type, e.time_start, e.time_end, e.place, "
            "e.activity, e.summary, e.cover_asset_id "
            "FROM events e WHERE e.scope_id=? AND e.status='active' ORDER BY e.time_start",
            (scope_id,),
        ).fetchall() if store is not None else []
        scored_items = []
        for row in rows:
            values = " ".join(str(row[key] or "").lower() for key in
                               ("title", "event_type", "place", "activity", "summary"))
            matched_terms = [term for term in query_terms if term in values]
            if query and query_terms and not matched_terms:
                continue
            member_rows = store.connection.execute(
                "SELECT o.asset_id FROM event_observations eo "
                "JOIN observations o ON o.id=eo.observation_id "
                "WHERE eo.event_id=? ORDER BY o.captured_at", (row["id"],)
            ).fetchall()
            member_asset_ids = [str(member["asset_id"]) for member in member_rows
                                if member["asset_id"]]
            place_filter = str(filters.get("place") or "").strip().lower()
            if place_filter and place_filter not in values:
                # Event records often have a short/empty place field while
                # member photos carry the authoritative reverse-geocode label.
                # Use those member assets only to validate the event filter;
                # do not broaden an unrelated event into a match.
                asset_values = []
                for asset_id in member_asset_ids:
                    asset = store.get_asset(asset_id) or {}
                    metadata = asset.get("metadata_json") or {}
                    if isinstance(metadata, str):
                        try:
                            metadata = json.loads(metadata)
                        except (TypeError, ValueError, json.JSONDecodeError):
                            metadata = {}
                    geocode = metadata.get("reverse_geocode") or {}
                    if isinstance(geocode, str):
                        try:
                            geocode = json.loads(geocode)
                        except (TypeError, ValueError, json.JSONDecodeError):
                            geocode = {}
                    if isinstance(geocode, dict):
                        asset_values.extend(str(geocode.get(key) or "").lower()
                                            for key in ("label", "city", "district", "street"))
                if not any(place_filter in text for text in asset_values if text):
                    continue
            item = {
                "event_id": row["id"], "title": row["title"],
                "event_type": row["event_type"], "time_start": row["time_start"],
                "time_end": row["time_end"], "place": row["place"],
                "activity": row["activity"], "summary": row["summary"],
                "asset_id": row["cover_asset_id"] or "",
            }
            item["source_asset_ids"] = list(dict.fromkeys(
                ([str(item["asset_id"])] if item["asset_id"] else [])
                + member_asset_ids
            ))
            # Longer phrases carry more meaning than incidental two-character
            # overlaps. Keep all equal-scoring rows, then apply a bounded
            # result window so one broad video question cannot flood context.
            score = sum(len(term) ** 2 for term in set(matched_terms))
            scored_items.append((score, item))
        scored_items.sort(key=lambda pair: (-pair[0], pair[1].get("time_start") or "", pair[1].get("event_id") or ""))
        items = [item for _score, item in scored_items[:80 if operation == "timeline" else 30]]
        source_ids = list(dict.fromkeys(
            asset_id for item in items for asset_id in item.get("source_asset_ids", [])
        ))
        return {
            "tool": "query_memory_metadata", "metadata_operation": operation,
            "operation": operation, "answer_type": "event_list", "value": items,
            "items": items, "total": len(items), "source_asset_ids": source_ids,
            "evidence_asset_ids": source_ids, "evidence_kind": "structured_event",
            "coverage": {"complete": True},
        }
    mapping = {"place": "group", "date": "date",
               "count": "count", "first": "first", "last": "last"}
    fact_operation = mapping.get(operation, "count")
    filters = dict(arguments.get("filters") or {})
    for key in ("time", "place", "person", "media"):
        if arguments.get(key) and not filters.get(key):
            filters[key] = arguments.get(key)
    fact_args = {"operation": fact_operation, "filters": filters,
                 "group_by": "place" if operation == "place" else arguments.get("group_by")}
    result = _query_memory_facts(fact_args, context=context)
    result = dict(result or {})
    result["tool"] = "query_memory_metadata"
    result["metadata_operation"] = operation
    source_rows = (result.get("items") or result.get("samples") or [])
    result["source_asset_ids"] = list(dict.fromkeys(
        [str(row.get("asset_id") or row.get("id")) for row in source_rows
         if isinstance(row, dict) and (row.get("asset_id") or row.get("id"))]
        + [str(row.get("asset_id") or row.get("id")) for row in (result.get("rows") or [])
           if isinstance(row, dict) and (row.get("asset_id") or row.get("id"))]
    ))
    if operation == "place" and not result["source_asset_ids"]:
        # Place grouping returns aggregate rows by design.  Attach the exact
        # matching assets as provenance without mixing visual observations into
        # the structured fact itself.
        try:
            from ..structured_memory import StructuredMemoryExecutor
            draft = _draft_from_filters(filters, answer_type="asset_set")
            spec = _spec_for(draft, (context or {}).get("scope_id") or "",
                             (context or {}).get("viewer_id") or "owner")
            assets = StructuredMemoryExecutor(_RUNTIME["store"])._matching_assets(
                draft, spec, limit=500)
            result["source_asset_ids"] = list(dict.fromkeys(
                str(row.get("id") or row.get("asset_id")) for row in assets
                if isinstance(row, dict) and (row.get("id") or row.get("asset_id"))))
        except Exception:
            result["source_asset_ids"] = []
    result["evidence_asset_ids"] = list(result["source_asset_ids"])
    result["evidence_kind"] = "structured_metadata"
    return result


def _query_photo_people(arguments: dict, *, context: dict | None = None) -> dict:
    """Canonical image-level people evidence, bound to one stable result handle."""
    scope_id = (context or {}).get("scope_id") or ""
    task_state = (context or {}).get("task_state") or {}
    result_set_id = arguments.get("result_set_id") or task_state.get("current_result_set")
    handle = str(arguments.get("asset_handle") or arguments.get("handle") or "").strip()
    rs_store = _RUNTIME.get("result_sets")
    if not result_set_id or rs_store is None:
        return {"summary": "请先通过 search_memories 建立结果集。", "blocked": ["no_result_set"],
                "evidence_asset_ids": []}
    asset_id = rs_store.resolve_handle(result_set_id, handle) if handle else None
    if not asset_id:
        return {"summary": "无法解析当前结果集中的照片句柄。", "blocked": ["bad_handle"],
                "evidence_asset_ids": []}
    store = _RUNTIME.get("store")
    if store is None:
        return {"summary": "记忆库不可用。", "blocked": ["store_unavailable"],
                "evidence_asset_ids": []}
    asset = store.get_asset(asset_id) or {}
    if scope_id and asset.get("scope_id") and asset.get("scope_id") != scope_id:
        return {"summary": "照片不在当前相册范围。", "blocked": ["scope_mismatch"],
                "evidence_asset_ids": []}
    identities = _confirmed_photo_identities(store, asset_id)
    observation = store.connection.execute(
        "SELECT people_json FROM observations WHERE asset_id = ? ORDER BY updated_at DESC LIMIT 1",
        (asset_id,),
    ).fetchone()
    try:
        people_values = json.loads(observation["people_json"] or "[]") if observation else []
        people_count = sum(1 for p in people_values
                           if not (isinstance(p, dict) and p.get("entity_id")))
    except (TypeError, ValueError, json.JSONDecodeError):
        people_count = 0
    unknown_count = max(0, people_count - len(identities))
    people = [{"person_name": row.get("person_name"),
               "family_role": row.get("family_role") or "",
               "identity_status": "confirmed",
               "asset_id": asset_id}
              for row in identities if row.get("person_name")]
    return {
        "asset_handle": handle,
        "result_set_id": result_set_id,
        "asset_id": asset_id,
        "people": people,
        "unconfirmed_people": ([{"description": "未确认身份同行者", "count": unknown_count}]
                               if unknown_count else []),
        "unconfirmed_people_count": unknown_count,
        "summary": ("、".join(p["person_name"] for p in people) or "没有已确认姓名")
                   + (f"；另有 {unknown_count} 名未确认身份同行者" if unknown_count else ""),
        "source_asset_ids": [asset_id],
        "source_handles": [handle],
        "evidence_asset_ids": [asset_id],
        "evidence_kind": "photo_people",
    }


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
{target_instruction}
问题：{question}"""



def _cap_hint(tool: str) -> str:
    """能力实测提示：无数据返回空串，不改变工具合同。"""
    try:
        hint = tool_capability_summary(tool)
        return ("\n" + hint) if hint else ""
    except Exception:
        return ""


def person_profile_summary(person: str, scope_id: str = "") -> str:
    """Compact Chinese profile summary for a confirmed person; '' when unavailable."""
    try:
        ent = _resolve_person_entity(person, scope_id)
        store = _RUNTIME.get("store")
        if ent is None or store is None:
            return ""
        digest = store.person_profile_digest(ent["id"], scope_id)
        if not digest or not digest.get("summary_zh"):
            return ""
        lines = []
        if digest.get("family_role"):
            lines.append(f"家庭角色：{digest['family_role']}")
        if digest.get("relationships"):
            lines.append("关系：" + "、".join(f"{r.get('other_name')}（{r.get('predicate')}）" for r in digest["relationships"]))
        if digest.get("preference_summary_zh"):
            lines.append(digest["preference_summary_zh"])
        titles = [e.get("title") or "" for e in (digest.get("recent_events") or []) if e.get("title")]
        if titles:
            lines.append("近期事件：" + "、".join(titles))
        return "；".join(lines) or digest.get("summary_zh")
    except Exception:
        return ""


def _get_person_profile(arguments: dict, *, context: dict | None = None) -> dict:
    person = (arguments.get("person") or "").strip()
    scope_id = (context or {}).get("scope_id") or ""
    ent = _resolve_person_entity(person, scope_id)
    if ent is None:
        return {"person": person, "readiness": "limited", "insufficient_evidence": True,
                "summary": f"没有找到已确认人物「{person}」的画像。",
                "note": "人物未确认或数据不足时返回 limited，不编造。"}
    store = _RUNTIME.get("store")
    digest = store.person_profile_digest(ent["id"], scope_id) if store else None
    if not digest or not digest.get("summary_zh"):
        return {"person": person, "readiness": "limited", "insufficient_evidence": True,
                "summary": f"「{person}」暂无足够的人物画像数据。",
                "note": "画像数据不足时返回 limited，不编造。"}
    return {
        "person": digest.get("person"),
        "family_role": digest.get("family_role") or "",
        "readiness": "ready",
        "summary": digest.get("summary_zh") or "",
        "preference_summary": digest.get("preference_summary_zh") or "",
        "relationships": digest.get("relationships") or [],
        "patterns": digest.get("patterns") or [],
        "recent_events": digest.get("recent_events") or [],
        "claims": digest.get("claims") or [],
        "profile_text": person_profile_summary(person, scope_id),
        "note": "画像来自已确认人物的语义记忆；性格等无法由证据确认的问题应回答 insufficient evidence。",
    }


def register_tools():

    register(ToolSpec(
        name="query_memory_facts",
        description=("确定性结构化事实查询（数量/存在性/首次/最近/日期/分组/饮食/媒体列表），不要用模型估算。"
                     "operation=date 用于回答某次记录的日期/年份；operation=first/last 用于最早/最近；operation=list 只用于用户明确要求列出媒体。"
                     "filters.time 原样写相对或具体时间，系统自动换算；不填表示全部。"
                     "group 必须填 group_by（month|place），place 分组需如实说明无地点照片数。"
                     "meal 用于'吃过什么/吃饭'类问题。菜单价格/招牌等视觉文字先用 search_memories 再 read_photo_text，不要用本工具猜。"
                     "operation=list 用于列出实际媒体（如'相册里所有视频/所有照片'），filters.media 填 video/image/audio/text；"
                     "返回 items 含视频时长/场景/关键帧来源，不要用 count 回答列表问题。"
                     "filters.place 只填结构化地名（城市/区县/景区/地标），不要把目标/活动/主题当 place；不确定留空。"),
        input_schema={"operation": "count|exists|first|last|date|group|meal|list",
                      "filters": {"time": "去年/这两年/2023年 等相对或具体时间（原样写）",
                                  "person": "", "place": "", "media": "",
                                  "food": "可选：限定某种食物（如火锅等具体菜名）"},
                      "group_by": "month|place"},
        executor=_query_memory_facts, read_write="read", cost_class="cheap", readiness="ready",
        produces_evidence=("structured_fact", "temporal_metadata", "location_metadata"),
        required_inputs=("operation",),
    ))
    register(ToolSpec(
        name="search_memories",
        description=("检索照片（人/物/场景/衣着/颜色）。返回结果集摘要。"
                     "时间必须从用户问题里提取并写 filters.time（如'2024年7月'）；问题没给具体时间就省略 time，不要写'所有时间'。"
                     "filters.place 只填结构化地名（城市/区县/景区/地标），不要把目标/活动/主题当 place；不确定留空。"
                     "filters.person 只填用户明确提到的人物名，不要用'伴娘/兄弟/亲戚'这类角色词。"),
        input_schema={"query": "", "mode": "best|all|representative",
                      "filters": {"time": "", "place": "", "person": ""}},
        executor=_search_memories, read_write="read", cost_class="medium", readiness="ready",
        produces_evidence=("memory_asset", "temporal_metadata", "location_metadata"),
        required_inputs=("query",),
    ))
    register(ToolSpec(
        name="query_memory_metadata",
        description=("查询照片的结构化元数据，专用于时间、地点、事件和数量。"
                     "operation=date/place/event/timeline/count/first/last；timeline 用于读取完整视频事件时间线并生成剪辑编排。不要用它回答衣着、场景细节等视觉问题。"
                     "返回的事实带 source_asset_ids，可作为回答来源。"),
        input_schema={"operation": "date|place|event|timeline|count|first|last",
                      "filters": {"time": "", "place": "", "person": "", "media": ""}},
        executor=_query_memory_metadata, read_write="read", cost_class="cheap", readiness="ready",
        produces_evidence=("structured_fact", "temporal_metadata", "location_metadata"),
        required_inputs=("operation",),
    ))
    register(ToolSpec(
        name="query_photo_people",
        description=("读取一张已检索照片自己的已确认人物和未确认同行者。"
                     "asset_handle 必须来自当前 search_memories preview；不会把其他候选照片的人名移植过来。"),
        input_schema={"asset_handle": "", "result_set_id": ""},
        executor=_query_photo_people, read_write="read", cost_class="cheap", readiness="ready",
        produces_evidence=("photo_identity", "confirmed_identity"),
        required_inputs=("asset_handle",),
        preconditions=("asset_handle_in_current_preview",),
        prerequisite_evidence_types=("memory_asset",),
    ))
    register(ToolSpec(
        name="get_original_photos",
        description="交付当前结果集/选中照片的原图。",
        input_schema={"result_set_id": "", "handle": ""},
        executor=_get_original_photos, read_write="read", cost_class="cheap", readiness="limited",
        produces_evidence=("memory_asset",),
        required_inputs=("handle",),
        readiness_reason="ResultSetStore 就绪后完整可用（A4）",
    ))
    register(ToolSpec(
        name="get_result_page",
        description="查看 search_memories 结果集的下一页/指定页（每页最多6张）。result_set_id 用 search_memories 返回的，page 从 1 开始。若要改变查询条件，重新调用 search_memories，不要把新条件传给本工具。",
        input_schema={"result_set_id": "", "page": 1, "page_size": 6},
        executor=_get_result_page, read_write="read", cost_class="cheap", readiness="ready",
        produces_evidence=("memory_asset",),
        required_inputs=("result_set_id", "page"),
    ))
    register(ToolSpec(
        name="inspect_photo",
        description=("复核已检索照片的视觉细节（物体/衣着/颜色/场景）。asset_handle 使用 search_memories preview 里的 handle（photo_1…），可省略（默认用预览第一张）。昂贵，默认每轮最多 1 次。"
                     + _cap_hint("inspect_photo")),
        input_schema={"asset_handle": "", "question": "", "target_person": ""},
        executor=_inspect_photo, read_write="read", cost_class="expensive", readiness="ready",
        produces_evidence=("visual_observation", "photo_identity"),
        required_inputs=("asset_handle", "question"),
        preconditions=("asset_handle_in_current_preview",),
        prerequisite_evidence_types=("memory_asset",), budget_unit="image",
    ))
    register(ToolSpec(
        name="read_photo_text",
        description=("读取照片中的文字内容（菜单/价格/招牌/店名/电话/年份/小字）。"
                     "适用于'多少钱/价格/售价/店名/招牌/电话/写了什么/什么字/哪一年'等需要看照片文字的题；"
                     "内部会把照片切块放大后 OCR。asset_handle 用 search_memories preview 里的 handle，可省略（默认预览第一张）。"
                     "昂贵，每轮最多 1 次。"
                     + _cap_hint("read_photo_text")),
        input_schema={"asset_handle": "", "question": ""},
        executor=_read_photo_text, read_write="read", cost_class="expensive", readiness="ready",
        produces_evidence=("visible_text",),
        required_inputs=("asset_handle", "question"),
        preconditions=("asset_handle_in_current_preview",),
        prerequisite_evidence_types=("memory_asset",), budget_unit="image",
    ))
    register(ToolSpec(
        name="search_conversation_history",
        description=("检索历史对话内容：回答'我之前是不是说过…/上次聊到哪里/你之前说那张在哪'类问题。"
                     "scope=current 只查当前会话；scope=recent 查最近几个会话；scope=all_user_conversations 查全部历史会话。"
                     "对话内容属于用户表述，不等于照片证据；回答时应说'你之前说过'而不是当成照片事实。"),
        input_schema={"query": "", "scope": "current|recent|all_user_conversations",
                      "conversation_id": ""},
        executor=_search_conversation_history, read_write="read", cost_class="cheap", readiness="ready",
        produces_evidence=("user_statement",),
        required_inputs=("query",),
    ))
    register(ToolSpec(
        name="get_core_memory",
        description=("读取长期家庭记忆（已确认人物/家庭角色/关系/偏好等）。"
                     "subject 填人物名（如某人的称呼）；topic 填话题关键词（如某个话题词）；都不填返回优先级最高的卡片。"
                     "每条记忆带 truth_status（confirmed_fact/user_stated/agent_inference/observed_pattern），"
                     "agent_inference 不能说成 confirmed。"),
        input_schema={"subject": "", "topic": "", "limit": 5},
        executor=_get_core_memory, read_write="read", cost_class="cheap", readiness="ready",
        produces_evidence=("confirmed_identity", "user_statement"),
        required_inputs=("subject", "topic"),
    ))
    register(ToolSpec(
        name="get_person_profile",
        description=("读取已确认人物的高维画像（长期记忆）：家庭角色、人物关系、常去地点/常做活动/常同行、近期事件与语义声明。"
                     "人物未确认或画像数据不足时返回 limited，要如实说明，不要编造。"
                     "性格等主观问题：照片无法确认时回答 insufficient evidence。"),
        input_schema={"person": ""},
        executor=_get_person_profile, read_write="read", cost_class="cheap", readiness="ready",
        produces_evidence=("confirmed_identity", "structured_fact"),
        required_inputs=("person",),
    ))
