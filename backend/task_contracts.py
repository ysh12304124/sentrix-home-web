"""TaskContract — what the user is really asking the system to complete (TFPE2-1, first slice).

Derived deterministically from the model-judged ``QueryParseDraft`` and the
``QuerySpec``.  It carries the answer shape (count/date/exists/...), the primary
goal, the requested conditions and the completion criteria so the RetrievalStrategy
planner and the Structured Executor know exactly what "done" means for this turn.

Nothing here is a query-pattern rule: the model decides answer_type and the
strategy hint; code only maps and validates.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .query_contracts import QuerySpec, STRUCTURED_ANSWER_TYPES


@dataclass
class TaskContract:
    task_id: str
    primary_goal: str            # answer_fact | aggregate_memory | find_images | return_assets | person_summary
    answer_type: str
    requested_conditions: list[dict] = field(default_factory=list)
    result_requirement: dict = field(default_factory=dict)
    completion_criteria: dict = field(default_factory=dict)
    structured: dict = field(default_factory=dict)
    strategy_hint: str = ""

    def as_dict(self):
        return {
            "task_id": self.task_id, "primary_goal": self.primary_goal,
            "answer_type": self.answer_type,
            "requested_conditions": self.requested_conditions,
            "result_requirement": self.result_requirement,
            "completion_criteria": self.completion_criteria,
            "structured": self.structured, "strategy_hint": self.strategy_hint,
        }


def _primary_goal(draft, spec) -> str:
    if spec.answer_target == "person" or draft.answer_type in {"summary", "person_summary"}:
        return "person_summary"
    if draft.answer_type in STRUCTURED_ANSWER_TYPES:
        return "aggregate_memory" if draft.answer_type == "grouped_list" else "answer_fact"
    if spec.result_requirement.get("return_original_assets") or any(
            a.type == "return_assets" for a in (spec.actions or [])):
        return "return_assets"
    return "find_images"


def build_task_contract(draft, spec: QuerySpec, task_id: str = "task_runtime") -> TaskContract:
    conditions = [
        {
            "dimension": c.dimension, "value": c.value,
            "strictness": c.strictness, "negated": c.negated,
        }
        for c in spec.constraints
    ]
    structured = dict(draft.structured or {})
    if not structured.get("place") and draft.answer_target == "place":
        for c in spec.constraints_for("place"):
            structured["place"] = c.value
            break
    hard_required = any(c.strictness == "deterministic_hard" for c in spec.constraints)
    asset_required = draft.answer_type in {"asset_set", "summary"}
    return TaskContract(
        task_id=task_id,
        primary_goal=_primary_goal(draft, spec),
        answer_type=draft.answer_type,
        requested_conditions=conditions,
        result_requirement=dict(spec.result_requirement or {}),
        completion_criteria={
            "hard_conditions_required": hard_required,
            "asset_required": asset_required,
        },
        structured=structured,
        strategy_hint=draft.strategy_hint,
    )
