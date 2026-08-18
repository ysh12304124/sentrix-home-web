"""Tool Registry（v2 §11.1/§13）。

Tool 合同：name / description / input schema / output contract /
read/write risk / scope requirement / timeout / cost class / readiness / version。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from .task_state import EVIDENCE_TYPES


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    input_schema: dict
    executor: Callable[..., Any]
    read_write: str = "read"          # read | write | write_proposal
    scope_required: bool = True
    timeout_s: float = 15.0
    cost_class: str = "cheap"         # cheap | medium | expensive(visual)
    readiness: str = "ready"          # ready | limited | blocked
    version: str = "1"
    readiness_reason: str = ""
    produces_evidence: tuple[str, ...] = ()
    cannot_establish: tuple[str, ...] = ()
    budget_unit: str = "call"

    def __post_init__(self):
        produced = tuple(self.produces_evidence)
        prohibited = tuple(self.cannot_establish)
        unknown = (set(produced) | set(prohibited)) - EVIDENCE_TYPES
        if unknown:
            raise ValueError(f"unsupported evidence type: {sorted(unknown)[0]}")
        overlap = set(produced) & set(prohibited)
        if overlap:
            raise ValueError(f"evidence contract overlap: {sorted(overlap)[0]}")
        if self.budget_unit not in {"call", "image", "media_second"}:
            raise ValueError(f"unsupported budget unit: {self.budget_unit}")

    def can_satisfy(self, evidence_type: str) -> bool:
        return evidence_type in self.produces_evidence

    def as_contract(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "version": self.version,
            "read_write": self.read_write,
            "scope_required": self.scope_required,
            "cost_class": self.cost_class,
            "budget_unit": self.budget_unit,
            "readiness": self.readiness,
            "produces_evidence": list(self.produces_evidence),
            "cannot_establish": list(self.cannot_establish),
        }


_TOOL_SPECS: dict[str, ToolSpec] = {}


def register(spec: ToolSpec) -> None:
    _TOOL_SPECS[spec.name] = spec


def get_tool(name: str) -> ToolSpec | None:
    return _TOOL_SPECS.get(name)


def list_tools(readiness: str | None = None) -> list[ToolSpec]:
    tools = list(_TOOL_SPECS.values())
    if readiness:
        tools = [t for t in tools if t.readiness == readiness]
    return tools


def tool_readiness_matrix() -> dict[str, dict]:
    return {
        t.name: {
            "readiness": t.readiness,
            "reason": t.readiness_reason,
            "cost_class": t.cost_class,
            "read_write": t.read_write,
            "timeout_s": t.timeout_s,
        }
        for t in _TOOL_SPECS.values()
    }
