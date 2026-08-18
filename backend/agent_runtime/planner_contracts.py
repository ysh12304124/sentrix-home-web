"""Schema-level contracts for the open Agent 2 planner."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .task_state import TaskDeclaration


@dataclass(frozen=True)
class PlannerAction:
    kind: str
    declaration: TaskDeclaration | None = None
    tool: str = ""
    arguments: dict[str, Any] | None = None
    answer: str = ""
    missing_requirement_ids: tuple[str, ...] = ()
    candidate_refs: tuple[str, ...] = ()


def parse_planner_action(
    payload: dict[str, Any], *, known_requirement_ids: set[str] | None = None,
) -> PlannerAction:
    if not isinstance(payload, dict):
        raise ValueError("planner action must be an object")
    kind = str(payload.get("action") or "")
    if kind == "declare":
        return PlannerAction(
            kind=kind,
            declaration=TaskDeclaration.from_dict(dict(payload.get("declaration") or {})),
        )
    if kind == "tool_call":
        tool = str(payload.get("tool") or "")
        arguments = payload.get("arguments")
        if not tool or not isinstance(arguments, dict):
            raise ValueError("tool_call requires tool and arguments")
        return PlannerAction(kind=kind, tool=tool, arguments=dict(arguments))
    if kind == "final":
        answer = str(payload.get("answer") or "")
        if not answer:
            raise ValueError("final requires answer")
        return PlannerAction(kind=kind, answer=answer)
    if kind == "clarify":
        missing = tuple(str(value) for value in payload.get("missing_requirement_ids") or [])
        candidates = tuple(str(value) for value in payload.get("candidate_refs") or [])
        if not missing:
            raise ValueError("clarify requires missing requirements")
        known = known_requirement_ids if known_requirement_ids is not None else set()
        unknown = [requirement_id for requirement_id in missing if requirement_id not in known]
        if unknown:
            raise ValueError(f"unknown requirement: {unknown[0]}")
        return PlannerAction(
            kind=kind,
            missing_requirement_ids=missing,
            candidate_refs=candidates,
        )
    raise ValueError(f"unsupported planner action: {kind}")
