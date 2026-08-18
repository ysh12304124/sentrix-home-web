"""Counterfactual analysis tool for auditing Planner disagreements against Legacy execution."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class DisagreementItem:
    turn_id: int
    case_id: str
    legacy_action: str
    planner_action: str
    disagreement_kind: str  # tool_divergence | premature_final | over_planning | budget_divergence
    details: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "turn_id": self.turn_id,
            "case_id": self.case_id,
            "legacy_action": self.legacy_action,
            "planner_action": self.planner_action,
            "disagreement_kind": self.disagreement_kind,
            "details": self.details,
        }


def audit_counterfactual_turn(
    *,
    case_id: str,
    turn_id: int,
    legacy_action: str,
    planner_action: str,
) -> DisagreementItem | None:
    """Classify disagreement between legacy tool-loop action and shadow planner proposed action."""
    if legacy_action == planner_action:
        return None

    kind = "tool_divergence"
    if legacy_action == "final" and planner_action != "final":
        kind = "over_planning"
    elif legacy_action != "final" and planner_action == "final":
        kind = "premature_final"

    return DisagreementItem(
        turn_id=turn_id,
        case_id=case_id,
        legacy_action=legacy_action,
        planner_action=planner_action,
        disagreement_kind=kind,
    )
