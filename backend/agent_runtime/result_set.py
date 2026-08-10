"""ResultSetStore + TaskState（v2 §10/§15/§16）。

- ResultSet：服务端持有完整结果集，模型只见 result_set_id / total / preview / has_more。
- handle 映射：photo_N -> asset_id（不向模型/用户泄漏内部 ID）。
- TaskState：随 Tool-Loop 动态形成的任务状态（非固定前置 pipeline）。
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field


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

    def handles(self) -> dict[str, str]:
        return {f"photo_{i + 1}": aid for i, aid in enumerate(self.asset_ids[:20])}

    def preview(self, limit: int = 6) -> list[dict]:
        return [
            {"handle": f"photo_{i + 1}", "asset_id": aid}
            for i, aid in enumerate(self.asset_ids[:limit])
        ]


class ResultSetStore:
    def __init__(self, store):
        self.store = store
        self._memory: dict[str, ResultSet] = {}

    def save(self, rs: ResultSet) -> ResultSet:
        self._memory[rs.result_set_id] = rs
        return rs

    def get(self, result_set_id: str) -> ResultSet | None:
        return self._memory.get(result_set_id)

    def resolve_handle(self, result_set_id: str, handle: str) -> str | None:
        rs = self.get(result_set_id)
        if rs is None:
            return None
        return rs.handles().get(handle)

    def new(self, *, scope_id, query, asset_ids, unresolved=None) -> ResultSet:
        rs = ResultSet(
            result_set_id=f"rs_{uuid.uuid4().hex[:10]}",
            scope_id=scope_id, query=query,
            asset_ids=list(asset_ids), total=len(asset_ids),
            unresolved=unresolved or [],
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

    def record_tool_result(self, tool_call_id: str, tool_name: str, observation: dict):
        self.tool_results.append({
            "tool_call_id": tool_call_id,
            "tool": tool_name,
            "total": observation.get("total"),
            "satisfaction": observation.get("query_satisfaction"),
            "blocked": observation.get("blocked"),
            "inspect_text": observation.get("observation"),
            "certainty": observation.get("certainty"),
        })

    def update_from_tool(self, tool_name: str, arguments: dict, observation: dict):
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
            if total == 0:
                self.fulfillment = "empty"
            else:
                self.fulfillment = "partial" if observation.get("gaps") else "fulfilled"
            self.search_satisfaction = observation.get("query_satisfaction")
            self.search_condition_summary = observation.get("condition_summary") or {}
        if tool_name == "get_original_photos":
            self.delivery_state = "delivered"
            self.delivered_count = observation.get("delivered")

    def as_dict(self) -> dict:
        return {
            "user_goal": self.user_goal,
            "requested_actions": self.requested_actions,
            "hard_constraints": self.hard_constraints,
            "unresolved_conditions": self.unresolved_conditions,
            "current_result_set": self.current_result_set,
            "selected_asset": self.selected_asset,
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
            "tool_results": self.tool_results,
        }
