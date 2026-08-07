"""ResponsePlan — describes the answer form, never the family facts (RX-2).

One ResponsePlan per user goal.  It controls text shape (answer-first, length),
image count (must equal the visible-asset count the answer mentions) and the
evidence entry behaviour.  It carries no household facts — those live in the
AnswerBrief only.
"""

from __future__ import annotations

from dataclasses import dataclass

from .answer_brief import AnswerBrief


@dataclass
class ResponsePlan:
    response_mode: str
    answer_first: bool = True
    max_paragraphs: int = 2
    image_count: int = 0
    include_uncertainty: bool = True
    evidence_entry: str = "collapsed"   # collapsed | expanded | hidden
    follow_up: str | None = None

    def as_dict(self):
        return {
            "response_mode": self.response_mode, "answer_first": self.answer_first,
            "max_paragraphs": self.max_paragraphs, "image_count": self.image_count,
            "include_uncertainty": self.include_uncertainty,
            "evidence_entry": self.evidence_entry, "follow_up": self.follow_up,
        }


def plan_response(brief: AnswerBrief) -> ResponsePlan:
    """Deterministic per-mode response plan (§5.1)."""
    mode = brief.response_mode
    visible = len(brief.visible_assets)
    has_facts = bool(brief.facts)
    if mode == "chat":
        return ResponsePlan("chat", answer_first=True, max_paragraphs=2, image_count=0,
                            include_uncertainty=False, evidence_entry="hidden")
    if mode == "asset_delivery":
        # Original delivery is the first priority: short, image-first, no analysis.
        return ResponsePlan("asset_delivery", answer_first=True, max_paragraphs=1,
                            image_count=visible, include_uncertainty=False,
                            evidence_entry="collapsed")
    if mode == "exact_result":
        return ResponsePlan("exact_result", answer_first=True, max_paragraphs=2,
                            image_count=min(visible, 3), include_uncertainty=True,
                            evidence_entry="collapsed")
    if mode == "approximate_result":
        return ResponsePlan("approximate_result", answer_first=True, max_paragraphs=2,
                            image_count=min(visible, 3), include_uncertainty=True,
                            evidence_entry="collapsed")
    if mode == "no_result":
        return ResponsePlan("no_result", answer_first=True, max_paragraphs=1, image_count=0,
                            include_uncertainty=False, evidence_entry="collapsed")
    if mode == "person_summary":
        return ResponsePlan("person_summary", answer_first=True, max_paragraphs=3,
                            image_count=0, include_uncertainty=has_facts,
                            evidence_entry="expanded" if not has_facts else "collapsed")
    if mode == "clarify":
        return ResponsePlan("clarify", answer_first=True, max_paragraphs=1, image_count=0,
                            include_uncertainty=False, evidence_entry="hidden")
    if mode == "structured_fact":
        return ResponsePlan("structured_fact", answer_first=True, max_paragraphs=1,
                            image_count=0, include_uncertainty=False,
                            evidence_entry="collapsed" if has_facts else "hidden")
    if mode == "aggregate_answer":
        return ResponsePlan("aggregate_answer", answer_first=True, max_paragraphs=2,
                            image_count=0, include_uncertainty=False,
                            evidence_entry="collapsed" if has_facts else "hidden")
    return ResponsePlan(mode, image_count=visible)
