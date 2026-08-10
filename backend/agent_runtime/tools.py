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
import time
from pathlib import Path

from .tool_registry import ToolSpec, register

_RUNTIME: dict = {}


def bind_runtime(store, *, gamma=None, embedding_router=None, retrieval_config=None):
    from .result_set import ResultSetStore
    _RUNTIME["store"] = store
    _RUNTIME["gamma"] = gamma
    _RUNTIME["embedding_router"] = embedding_router
    _RUNTIME["retrieval_config"] = retrieval_config
    _RUNTIME["result_sets"] = ResultSetStore(store)


def _kernel():
    from ..evidence_retrieval import EvidenceRetrievalKernel
    if _RUNTIME.get("embedding_router") is not None:
        from ..retrieval import RetrievalConfig, build_default_retrievers
        config = _RUNTIME.get("retrieval_config") or RetrievalConfig()
        retrievers = build_default_retrievers(_RUNTIME["store"], embedding_router=_RUNTIME["embedding_router"], config=config)
        return EvidenceRetrievalKernel(_RUNTIME["store"], retrievers=retrievers,
                                       embedding_router=_RUNTIME["embedding_router"], config=config)
    return EvidenceRetrievalKernel(_RUNTIME["store"])


def _resolve_time_expression(value: str) -> str | None:
    """把相对时间（去年/今年/上个月/2023）解析成可被 parse_time_expression 接受的绝对时间。"""
    from datetime import datetime
    now = datetime.now()
    v = (value or "").strip()
    if not v:
        return None
    if "去年" in v:
        return f"{now.year - 1}年"
    if "今年" in v:
        return f"{now.year}年"
    if "前年" in v:
        return f"{now.year - 2}年"
    if "上个月" in v:
        y, m = (now.year, now.month - 1) if now.month > 1 else (now.year - 1, 12)
        return f"{y}年{m}月"
    import re
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
        for entity in store.list_entities(status="confirmed", scope_id=scope_id):
            if entity.get("canonical_name") == name:
                return entity.get("id")
    except Exception:
        pass
    return None


# ---- Tool 1: query_memory_facts ----
def _query_memory_facts(arguments: dict, *, context: dict | None = None) -> dict:
    operation = arguments.get("operation") or "count"
    filters = arguments.get("filters") or {}
    scope_id = (context or {}).get("scope_id") or "home-default"
    viewer_id = (context or {}).get("viewer_id") or "owner"
    if operation == "group":
        group_by = arguments.get("group_by") or "month"
        draft = _draft_from_filters(filters, answer_type="grouped_list", group_by=group_by)
    else:
        answer_type = {
            "count": "count", "exists": "exists", "first": "first_occurrence",
            "last": "last_occurrence", "date": "date", "media": "count",
        }.get(operation, "count")
        draft = _draft_from_filters(filters, answer_type=answer_type)
    from ..structured_memory import StructuredMemoryExecutor
    result = StructuredMemoryExecutor(_RUNTIME["store"]).execute(draft, _spec_for(draft, scope_id, viewer_id))
    return {
        "operation": operation,
        "answer_type": result.answer_type,
        "value": result.value,
        "total": result.total,
        "rows": result.rows if operation == "group" else None,
        "filters_applied": result.filters_applied,
        "coverage": {"complete": True},
    }


# ---- Tool 2: search_memories ----
def _search_memories(arguments: dict, *, context: dict | None = None) -> dict:
    query = arguments.get("query") or ""
    mode = arguments.get("mode") or "best"
    filters = arguments.get("filters") or {}
    scope_id = (context or {}).get("scope_id") or "home-default"
    viewer_id = (context or {}).get("viewer_id") or "owner"
    draft = _draft_from_filters({**filters, "query": query}, answer_type="asset_set")
    draft.result_requirement = {"mode": mode}
    spec = _spec_for(draft, scope_id, viewer_id)
    packet = _kernel().retrieve(spec)
    assets = packet.assets or []
    asset_ids = [item.get("asset_id") for item in assets if item.get("asset_id")]
    rs = _RUNTIME["result_sets"].new(
        scope_id=scope_id, query=query, asset_ids=asset_ids,
        unresolved=[g.get("reason") for g in (packet.gaps or [])],
    )
    preview = []
    handles = rs.handles()
    for i, item in enumerate(assets[:6]):
        handle = f"photo_{i + 1}"
        preview.append({
            "handle": handle,
            "captured_at": item.get("captured_at"),
            "level": item.get("level"),
            "condition_summary": _condition_summary(item),
        })
    _RUNTIME["last_handles"] = handles
    cond, satisfaction, answerability = _truth_contract(packet, rs.total)
    return {
        "result_set_id": rs.result_set_id,
        "query": query,
        "mode": mode,
        "total": rs.total,
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
    }


def _condition_summary(item: dict) -> dict:
    out = {}
    for key, cond in (item.get("condition_results") or {}).items():
        label = key.split(":", 1)[-1]
        out[label] = cond.get("status")
    return out


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
    asset_id = rs_store.resolve_handle(result_set_id, handle) if handle else None
    if handle and not asset_id:
        return {"summary": "无法解析选中的照片。", "delivered": 0, "blocked": ["bad_handle"]}
    return {
        "summary": f"已从结果集 {result_set_id} 授权原图交付。",
        "result_set_id": result_set_id,
        "handle": handle or "all",
        "delivered": 1 if asset_id else rs.total,
        "total": rs.total,
        "scope_id": rs.scope_id,
    }


# ---- Tool 4: inspect_photo ----
def _inspect_photo(arguments: dict, *, context: dict | None = None) -> dict:
    asset_handle = arguments.get("asset_handle") or ""
    question = arguments.get("question") or "请描述这张照片"
    scope_id = (context or {}).get("scope_id") or "home-default"
    task_state = (context or {}).get("task_state") or {}
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
    if row and row["scope_id"] != scope_id:
        return {"summary": "无法复核该照片（不在当前相册范围）。", "certainty": "uncertain",
                "persisted": False, "blocked": ["scope_mismatch"]}
    if not row or not row["path"] or not Path(row["path"]).is_file():
        return {"summary": "照片文件不可用。", "certainty": "uncertain", "persisted": False,
                "blocked": ["file_unavailable"]}
    row = store.connection.execute("SELECT path FROM assets WHERE id = ?", (asset_id,)).fetchone()
    if not row or not row["path"] or not Path(row["path"]).is_file():
        return {"summary": "照片文件不可用。", "certainty": "uncertain", "persisted": False}
    gamma = _RUNTIME.get("gamma")
    if gamma is None:
        return {"summary": "模型不可用。", "certainty": "uncertain", "persisted": False}
    try:
        image = {"base64": _base64_image(row["path"]), "mime_type": _mime_for(row["path"])}
        raw = gamma.chat(_INSPECT_PROMPT.format(question=question), images=[image],
                         json_mode=True, role="inspect")
    except Exception as exc:
        return {"summary": f"图片复核失败：{exc}", "certainty": "uncertain", "persisted": False}
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
        "source": "runtime_visual_inspection",
        "persisted": False,
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
问题：{question}"""


def register_tools():
    register(ToolSpec(
        name="query_memory_facts",
        description="查询结构化记忆事实：数量、是否存在、首次/最后一次出现、日期、月份/地点分组。",
        input_schema={"operation": "count|exists|first|last|date|group",
                      "filters": {"time": "", "person": "", "place": "", "media": ""}},
        executor=_query_memory_facts, read_write="read", cost_class="cheap", readiness="ready",
    ))
    register(ToolSpec(
        name="search_memories",
        description="检索家庭记忆：找照片、视觉语义（衣着/颜色/物体/场景）、混合查询。返回结果集摘要。",
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
        name="inspect_photo",
        description="复核已检索照片的视觉细节（物体/衣着/文字/场景）。asset_handle 必须使用 search_memories preview 里的 handle（photo_1…）。昂贵，默认每轮最多 1 次。",
        input_schema={"asset_handle": "", "question": ""},
        executor=_inspect_photo, read_write="read", cost_class="expensive", readiness="ready",
    ))
