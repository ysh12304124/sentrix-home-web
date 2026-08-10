"""ToolPolicy（v2 §11.2/§4.4）— 代码层唯一安全入口。

所有 Tool 调用统一经过：validate -> authorize -> budget check -> execute -> sanitize observation。
模型不能绕过。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class ToolDecision:
    allowed: bool
    reason: str = ""
    observation: dict | None = None
    error: str | None = None


class ToolPolicy:
    def __init__(self, *, scope_id="home-default", viewer_id="owner",
                 budget=None, inspect_allowed=True):
        self.scope_id = scope_id
        self.viewer_id = viewer_id
        self.budget = budget
        self.inspect_allowed = inspect_allowed

    def authorize(self, spec, tool_name: str, arguments: dict) -> ToolDecision:
        if spec.read_write != "read":
            return ToolDecision(False, f"write tool not allowed in read-only runtime: {tool_name}")
        if spec.readiness == "blocked":
            return ToolDecision(False, f"tool blocked: {tool_name}")
        if self.budget is not None and not self.budget.can_tool_call(
                inspection=spec.cost_class == "expensive"):
            return ToolDecision(False, "budget exhausted")
        return ToolDecision(True)

    def execute(self, spec, arguments: dict, *, context: dict | None = None) -> ToolDecision:
        decision = self.authorize(spec, spec.name, arguments)
        if not decision.allowed:
            return decision
        try:
            payload = spec.executor(arguments, context=context or {})
        except Exception as exc:
            return ToolDecision(False, "tool_execution_error", error=str(exc))
        if self.budget is not None:
            self.budget.record_tool_call(inspection=spec.cost_class == "expensive")
        observation = self._sanitize(payload, spec.name)
        return ToolDecision(True, "ok", observation=observation)

    _DEFAULT_ALLOWED = {
        "summary", "result_set_id", "handle", "total", "preview", "has_more",
        "remaining", "counts", "coverage", "facts", "items", "completeness",
        "unresolved", "delivered", "blocked", "observation", "certainty",
        "source", "persisted", "question", "asset_handle", "reason",
    }
    _TOOL_ALLOWED = {
        "query_memory_facts": _DEFAULT_ALLOWED | {
            "operation", "answer_type", "value", "rows", "filters_applied",
        },
        "search_memories": _DEFAULT_ALLOWED | {
            "query", "mode", "gaps", "query_satisfaction", "answerability",
            "condition_summary", "can_inspect", "inspect_hint",
        },
        "get_original_photos": _DEFAULT_ALLOWED | {"scope_id"},
        "inspect_photo": _DEFAULT_ALLOWED,
    }

    @classmethod
    def _sanitize(cls, payload: dict, tool_name: str = "") -> dict:
        """Tool observation 只保留模型可安全看到的部分（隐藏内部 ID 由各 Tool 负责）。"""
        if not isinstance(payload, dict):
            return {"raw": payload}
        allowed = cls._TOOL_ALLOWED.get(tool_name, cls._DEFAULT_ALLOWED)
        return {k: v for k, v in payload.items() if k in allowed}
