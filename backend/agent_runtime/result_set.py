"""ResultSetStore + TaskState（v2 §10/§15/§16）。

- ResultSet：服务端持有完整结果集，模型只见 result_set_id / total / preview / has_more。
- handle 映射：photo_N -> asset_id（不向模型/用户泄漏内部 ID）；分页后 photo_N 为全局稳定序号。
- B3.1：TTL 过期、page(page_no) 分页、owner/revision 记录。
- TaskState：随 Tool-Loop 动态形成的任务状态（非固定前置 pipeline）。
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field

RESULT_SET_TTL_S = 24 * 60 * 60  # D7：ResultSet TTL 延长到 24h（多轮/跨会话稳定）


@dataclass
class ResultSet:
    result_set_id: str
    scope_id: str
    query: str
    asset_ids: list
    total: int = 0
    ordering: list = field(default_factory=list)
    unresolved: list = field(default_factory=list)
    created_at: str = ""
    owner: str = ""
    expires_at: float = 0.0
    revision: str = ""
    page_size: int = 6
    shown: int = 0

    def handles(self) -> dict[str, str]:
        """全量稳定 handle 映射（photo_N 跨页一致，N 为结果集中全局序号）。"""
        return {f"photo_{i + 1}": aid for i, aid in enumerate(self.asset_ids)}

    def preview(self, limit: int = 6) -> list[dict]:
        return [
            {"handle": f"photo_{i + 1}", "asset_id": aid}
            for i, aid in enumerate(self.asset_ids[:limit])
        ]

    def page(self, page_no: int, page_size: int | None = None) -> list[dict]:
        """返回第 page_no 页（1-based）的 handle 列表；handle 序号为全局序号。"""
        size = page_size or self.page_size
        start = max(0, (int(page_no) - 1) * size)
        end = start + size
        return [
            {"handle": f"photo_{start + i + 1}", "asset_id": aid}
            for i, aid in enumerate(self.asset_ids[start:end])
        ]


class ResultSetStore:
    def __init__(self, store, ttl_s: float = RESULT_SET_TTL_S):
        self.store = store
        self._memory: dict[str, ResultSet] = {}
        self.ttl_s = ttl_s
        self._ensure_table()

    def _ensure_table(self):
        try:
            self.store.connection.executescript(
                """CREATE TABLE IF NOT EXISTS agent_result_sets (
                    result_set_id TEXT PRIMARY KEY,
                    scope_id TEXT NOT NULL,
                    query TEXT NOT NULL DEFAULT '',
                    asset_ids_json TEXT NOT NULL DEFAULT '[]',
                    total INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL,
                    expires_at REAL NOT NULL DEFAULT 0
                );""")
            self.store.connection.commit()
        except Exception:
            pass

    def save(self, rs: ResultSet) -> ResultSet:
        if not rs.expires_at:
            rs.expires_at = time.time() + self.ttl_s
        self._memory[rs.result_set_id] = rs
        self._persist(rs)
        return rs

    def _persist(self, rs: ResultSet):
        """D7：ResultSet 落库（进程重启后仍可恢复）。"""
        try:
            import json as _json
            self.store.connection.execute(
                """INSERT OR REPLACE INTO agent_result_sets
                   (result_set_id, scope_id, query, asset_ids_json, total, created_at, expires_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (rs.result_set_id, rs.scope_id, rs.query or "",
                 _json.dumps(rs.asset_ids or [], ensure_ascii=False),
                 rs.total, rs.created_at or "", rs.expires_at))
            self.store.connection.commit()
        except Exception:
            pass

    def _load_from_db(self, result_set_id: str) -> ResultSet | None:
        """从 DB 恢复未过期结果集（内存丢失/重启后）。"""
        try:
            import json as _json
            row = self.store.connection.execute(
                "SELECT * FROM agent_result_sets WHERE result_set_id = ?",
                (result_set_id,)).fetchone()
            if row is None:
                return None
            if row["expires_at"] and time.time() > row["expires_at"]:
                return None
            asset_ids = _json.loads(row["asset_ids_json"] or "[]")
            rs = ResultSet(
                result_set_id=row["result_set_id"], scope_id=row["scope_id"],
                query=row["query"], asset_ids=list(asset_ids), total=row["total"],
                created_at=row["created_at"], expires_at=row["expires_at"],
                shown=min(6, len(asset_ids)),
            )
            return rs
        except Exception:
            return None

    def get(self, result_set_id: str) -> ResultSet | None:
        rs = self._memory.get(result_set_id)
        if rs is not None:
            if rs.expires_at and time.time() > rs.expires_at:
                self._memory.pop(result_set_id, None)
            else:
                return rs
        rs = self._load_from_db(result_set_id)
        if rs is not None:
            self._memory[result_set_id] = rs
        return rs

    def resolve_handle(self, result_set_id: str, handle: str) -> str | None:
        rs = self.get(result_set_id)
        if rs is None:
            return None
        return rs.handles().get(handle)

    def cleanup(self) -> int:
        """惰性清理过期结果集，返回清理数量。"""
        now = time.time()
        expired = [rid for rid, rs in self._memory.items()
                   if rs.expires_at and now > rs.expires_at]
        for rid in expired:
            self._memory.pop(rid, None)
        return len(expired)

    def new(self, *, scope_id, query, asset_ids, unresolved=None, owner="", revision="") -> ResultSet:
        rs = ResultSet(
            result_set_id=f"rs_{uuid.uuid4().hex[:10]}",
            scope_id=scope_id, query=query,
            asset_ids=list(asset_ids), total=len(asset_ids),
            unresolved=unresolved or [],
            owner=owner, revision=revision,
            shown=min(6, len(asset_ids)),
        )
        return self.save(rs)


@dataclass
class TaskState:
    user_goal: str = ""
    requested_actions: list = field(default_factory=list)
    requested_deliverables: list = field(default_factory=list)
    hard_constraints: list = field(default_factory=list)
    unresolved_conditions: list = field(default_factory=list)
    current_result_set: str | None = None
    selected_asset: str | None = None
    selected_asset_handle: str | None = None
    delivery_state: str = "not_requested"
    fulfillment: str = "pending"
    result_mode: str | None = None
    has_more: bool | None = None
    delivered_count: int | None = None
    fact_total: int | None = None
    fact_value: object = None
    fact_operation: str | None = None
    fact_rows: list | None = None
    fact_group_by: str | None = None
    last_tool: str | None = None
    write_proposal: dict | None = None
    tool_results: list = field(default_factory=list)
    search_satisfaction: str | None = None
    search_condition_summary: dict = field(default_factory=dict)
    result_total: int | None = None
    result_remaining: int | None = None
    result_preview: list = field(default_factory=list)
    # D3：active references（跨轮持续状态）
    active_person: str | None = None
    active_event: str | None = None
    open_questions: list = field(default_factory=list)
    last_user_goal: str = ""

    def record_tool_result(self, tool_call_id: str, tool_name: str, observation: dict):
        self.tool_results.append({
            "tool_call_id": tool_call_id,
            "tool": tool_name,
            "total": observation.get("total"),
            "satisfaction": observation.get("query_satisfaction"),
            "blocked": observation.get("blocked"),
            "inspect_text": observation.get("observation") or observation.get("summary"),
            "inspect_handle": observation.get("asset_handle"),
            "confirms_visual_only": observation.get("confirms_visual_only", False),
            "certainty": observation.get("certainty"),
            "ocr_text": observation.get("full_text") or "",
            "asset_ids": observation.get("asset_ids"),
            "operation": observation.get("operation"),
            "value": observation.get("value"),
            "rows": observation.get("rows"),
            "answer_type": observation.get("answer_type"),
            "filters_applied": observation.get("filters_applied"),
            "samples": observation.get("samples"),
            "recommended_resolution": observation.get("recommended_resolution"),
            "condition_summary": observation.get("condition_summary"),
        })

    def update_from_tool(self, tool_name: str, arguments: dict, observation: dict):
        person = (arguments.get("filters") or {}).get("person") or arguments.get("person") or ""
        if person:
            self.active_person = person
        if tool_name == "query_memory_facts":
            total = observation.get("total")
            self.fact_total = int(total) if total is not None else None
            self.fact_value = observation.get("value")
            self.fact_operation = observation.get("operation")
            self.fact_rows = observation.get("rows")
            self.fact_group_by = arguments.get("group_by") or "month"
            self.last_tool = "query_memory_facts"
            self.fulfillment = "empty" if total == 0 else ("fulfilled" if total else "pending")
        if tool_name == "search_memories" and observation.get("result_set_id"):
            self.current_result_set = observation["result_set_id"]
            self.result_mode = arguments.get("mode") or "best"
            self.has_more = bool(observation.get("has_more"))
            self.delivery_state = "available" if observation.get("has_more") else "complete"
            total = observation.get("total")
            self.result_total = int(total) if total is not None else None
            self.result_remaining = observation.get("remaining")
            self.result_preview = [p.get("handle") for p in (observation.get("preview") or [])][:20]
            if total == 0:
                self.fulfillment = "empty"
            else:
                self.fulfillment = "partial" if observation.get("gaps") else "fulfilled"
            self.search_satisfaction = observation.get("query_satisfaction")
            self.search_condition_summary = observation.get("condition_summary") or {}
        if tool_name == "get_original_photos":
            self.delivery_state = "delivered"
            self.delivered_count = observation.get("delivered")

    @classmethod
    def from_dict(cls, data: dict | None, *, user_goal: str = "") -> "TaskState":
        """B3.1：跨 turn 恢复结果集上下文（只恢复与结果集续接相关的字段）。"""
        data = data or {}
        task = cls(user_goal=user_goal or data.get("user_goal") or "")
        task.current_result_set = data.get("current_result_set")
        task.result_mode = data.get("result_mode")
        task.has_more = data.get("has_more")
        task.delivery_state = data.get("delivery_state") or "not_requested"
        task.fulfillment = data.get("fulfillment") or "pending"
        task.search_satisfaction = data.get("search_satisfaction")
        task.search_condition_summary = data.get("search_condition_summary") or {}
        # D12：跨轮续接时恢复结果集预览（显式要图/追问时 grounding 仍能展示证据网格）
        task.result_preview = data.get("result_preview") or []
        task.result_total = data.get("result_total")
        task.result_remaining = data.get("result_remaining")
        task.selected_asset_handle = data.get("selected_asset_handle")
        task.active_person = data.get("active_person")
        task.active_event = data.get("active_event")
        task.open_questions = data.get("open_questions") or []
        task.last_user_goal = data.get("last_user_goal") or task.user_goal
        return task

    def as_dict(self) -> dict:
        return {
            "user_goal": self.user_goal,
            "requested_actions": self.requested_actions,
            "hard_constraints": self.hard_constraints,
            "unresolved_conditions": self.unresolved_conditions,
            "current_result_set": self.current_result_set,
            "selected_asset": self.selected_asset,
            "selected_asset_handle": self.selected_asset_handle,
            "delivery_state": self.delivery_state,
            "fulfillment": self.fulfillment,
            "result_mode": self.result_mode,
            "has_more": self.has_more,
            "fact_total": self.fact_total,
            "fact_value": self.fact_value,
            "fact_operation": self.fact_operation,
            "fact_rows": self.fact_rows,
            "fact_group_by": self.fact_group_by,
            "last_tool": self.last_tool,
            "search_satisfaction": self.search_satisfaction,
            "search_condition_summary": self.search_condition_summary,
            "result_total": self.result_total,
            "result_remaining": self.result_remaining,
            "result_preview": self.result_preview,
            "tool_results": self.tool_results,
            "active_person": self.active_person,
            "active_event": self.active_event,
            "open_questions": self.open_questions,
            "last_user_goal": self.last_user_goal,
        }
