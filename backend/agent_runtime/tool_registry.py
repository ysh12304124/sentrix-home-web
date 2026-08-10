"""Tool Registry（v2 §11.1/§13）。

Tool 合同：name / description / input schema / output contract /
read/write risk / scope requirement / timeout / cost class / readiness / version。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable


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
