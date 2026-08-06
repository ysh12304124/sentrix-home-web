"""Deterministic Final Router (Phase R9).

The Router owns the final route: none / contextual / evidence / clarify.  The
Parser's ``proposed_mode`` is advisory only — the Router combines it with
actions, facets, conditions, raw-message anchors, confirmed entities, the
current conversation focus and the NeutralProbe outcome.

The Parser never has a veto: ``draft.mode == none`` can no longer terminate a
household query.  A parser ``none`` with household anchors or a bare-noun
phrase routes to the NeutralProbe; a probe miss on an ambiguous phrase routes
to ``clarify`` (never to normal chat).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from .routing_rules import (
    has_general_verb,
    has_household_signal,
    is_contextual_follow_up,
    is_no_lookup,
    is_writing_compose,
    is_writing_request,
    message_anchored,
)


@dataclass(frozen=True)
class GateDecision:
    """Legacy routing decision contract (kept for the API envelope + tests)."""

    mode: str
    reason: str
    answer_target: str = "general"
    core_memory_reads: int = 0
    concrete_memory_reads: int = 0
    evidence_search_calls: int = 0
    query_parse_calls: int = 0
    original_image_allowed: bool = False
    proactivity_probe_performed: bool = False
    allow_probe: bool = False

    @property
    def proposed_mode(self):
        return self.mode

    def as_dict(self):
        data = self.__dict__.copy()
        data["proposed_mode"] = self.mode
        return data

# Explicit evidence operations: the user asked for assets, a person summary, a
# timeline, a comparison or a correction.
_EVIDENCE_ACTIONS = {"return_assets", "summarize_person", "summarize_event",
                     "timeline", "compare", "propose_correction"}
# answer_question is household only when it targets a strong dimension.
_STRONG_TARGETS = {"person", "event", "relationship"}
_BARE_NOUN_MAX_LEN = 20


@dataclass
class RouteDecision:
    mode: str                      # none | contextual | evidence | clarify | ambiguous
    reason: str
    probe_required: bool = False
    query_parse_calls: int = 0
    answer_target: str = "general"
    original_image_allowed: bool = False
    focus_ids: list[str] = field(default_factory=list)
    trace: dict[str, Any] = field(default_factory=dict)

    def as_gate_decision(self):
        """Legacy GateDecision projection for old callers/tests."""
        return GateDecision(
            mode="ambiguous" if self.mode == "clarify" else self.mode,
            reason=self.reason,
            answer_target=self.answer_target,
            core_memory_reads=1 if self.mode == "contextual" else 0,
            concrete_memory_reads=1 if self.mode == "evidence" else 0,
            evidence_search_calls=1 if self.mode == "evidence" else 0,
            query_parse_calls=self.query_parse_calls,
            original_image_allowed=self.original_image_allowed,
            allow_probe=self.probe_required or self.mode == "ambiguous",
        )


class ExplicitOperationDetector:
    """High-precision protocol fast paths (never a semantic classifier)."""

    @staticmethod
    def detect(message, *, api_signals=None) -> GateDecision | None:
        api_signals = api_signals or {}
        if api_signals.get("feedback"):
            return GateDecision("evidence", "explicit_feedback",
                                concrete_memory_reads=1, evidence_search_calls=1)
        if api_signals.get("selected_entity_id"):
            return GateDecision("evidence", "explicit_entity_selection",
                                answer_target="person", concrete_memory_reads=1,
                                evidence_search_calls=1)
        value = str(message or "").strip()
        if is_no_lookup(value):
            return GateDecision("none", "explicit_no_memory_lookup")
        # Writing fast-path is deliberately narrow: it fires only on a leading
        # writing/translation structure, never on mid-string "写".  Household
        # context exclusion happens in Router.route step 2 anyway.
        if is_writing_request(value):
            return GateDecision("none", "explicit_writing_task")
        return None


class Router:
    """Deterministic route decision combining every non-model signal."""

    def __init__(self, entity_resolver: Callable[[str], str | None] | None = None,
                 message_entity_resolver: Callable[[str], list[str]] | None = None):
        self._entity_resolver = entity_resolver
        self._message_entity_resolver = message_entity_resolver

    def route(self, message, draft, *, api_signals=None, conversation="",
              focus=None, entity_resolver=None,
              message_entity_resolver=None) -> RouteDecision:
        api_signals = api_signals or {}
        value = str(message or "").strip()
        resolver = entity_resolver or self._entity_resolver
        mention_resolver = message_entity_resolver or self._message_entity_resolver

        # 1. Explicit operations (protocol).
        op = ExplicitOperationDetector.detect(value, api_signals=api_signals)
        if op is not None:
            return RouteDecision(op.mode, op.reason, query_parse_calls=0,
                                 answer_target=op.answer_target)

        # 2. Writing/translation prefix — but only after family-context
        #    exclusion (no confirmed subject / facets / anchors / session focus).
        if is_writing_request(value) and not self._family_context(draft, value, focus):
            return RouteDecision("none", "explicit_writing_task")

        # 2.5 Structural writing request (mid-sentence composition verb) with a
        #     clean parser "none" — writing wins over a bare person mention.
        if is_writing_compose(value) and getattr(draft, "proposed_mode", None) == "none" \
                and not has_household_signal(draft):
            return RouteDecision("none", "writing_compose")

        # 3. Strong household signal -> evidence.
        strong = self._strong_household(draft)
        if strong["hit"]:
            return RouteDecision("evidence", strong["reason"], query_parse_calls=1,
                                 answer_target=strong["answer_target"],
                                 original_image_allowed=strong["original"])

        # 3.5 Parser-proposed contextual (a natural person mention without an
        #     explicit evidence ask) — checked BEFORE confirmed-person so
        #     "今晚回家时想起小黑了" stays contextual, not evidence.
        if getattr(draft, "proposed_mode", None) == "contextual" \
                and not self._has_evidence_ask(draft):
            return RouteDecision("contextual", "natural_person_mention",
                                 query_parse_calls=1)

        # 4. Confirmed person -> evidence (person intro must not fall to chat).
        #    Checked both in the parser draft and in the raw message, so a
        #    parser-none "介绍一下明哥" still reaches evidence.
        if self._confirmed_person(draft, resolver):
            return RouteDecision("evidence", "confirmed_person", query_parse_calls=1,
                                 answer_target="person")
        if mention_resolver is not None and mention_resolver(value):
            return RouteDecision("evidence", "confirmed_person_in_message",
                                 query_parse_calls=1, answer_target="person")

        # 5. Session follow-up reusing the persisted dialogue_states focus.
        followup_ids = self._session_follow_up(value, draft, focus)
        if followup_ids:
            return RouteDecision("evidence", "session_follow_up", query_parse_calls=1,
                                 focus_ids=followup_ids)

        # 5.5 Parser-down (timeout/failure) + general-intro verb: route to
        #     clarify, never to chat and never to a probe that may upgrade on
        #     visual noise.  With a healthy parser this branch is unreachable —
        #     "介绍一下明哥" then carries a person facet (evidence) and
        #     "解释一下量子纠缠" carries a clean none (step 6).
        if getattr(draft, "parser_failed", False) and has_general_verb(value) \
                and not has_household_signal(draft) and not message_anchored(value):
            return RouteDecision("clarify", "parser_down_general_ambiguous",
                                 query_parse_calls=1)

        # 6. Clear general question (family context already excluded: no facets /
        #    anchors / focus, and confirmed-entity failed above).  A parser that
        #    proposed evidence is contradictory — route to the probe, not none.
        if getattr(draft, "proposed_mode", "none") != "evidence" \
                and has_general_verb(value) and not has_household_signal(draft) \
                and not message_anchored(value):
            return RouteDecision("none", "general_concept_question",
                                 query_parse_calls=1)

        # 7. Weak household signal -> ambiguous, the NeutralProbe decides.
        weak = self._weak_household(draft, value)
        if weak["hit"]:
            return RouteDecision("ambiguous", weak["reason"], probe_required=True,
                                 query_parse_calls=1)

        # 8. No signal at all -> probe, then clarify (never fabricate).
        return RouteDecision("ambiguous", "no_signal", probe_required=True,
                             query_parse_calls=1)

    def resolve_after_probe(self, outcome, message, decision, draft) -> RouteDecision:
        """Finalize the route once the NeutralProbe has run."""
        value = str(message or "").strip()
        if outcome.decision == "upgrade":
            return RouteDecision("evidence", f"probe_upgrade:{outcome.reason}",
                                 query_parse_calls=decision.query_parse_calls,
                                 answer_target=decision.answer_target)
        if outcome.decision == "clarify":
            return RouteDecision("clarify", "weak_or_conflicting_probe",
                                 query_parse_calls=decision.query_parse_calls)
        # no_household_match: only a clear general intent may go to chat; an
        # ambiguous phrase goes to clarify instead of fabricating a reply.  A
        # parser failure (timeout) suppresses the general fallback too — the
        # message stays ambiguous and clarifies rather than risking chat.
        if not getattr(draft, "parser_failed", False) \
                and has_general_verb(value) and not has_household_signal(draft) \
                and not message_anchored(value):
            return RouteDecision("none", "general_concept_after_probe",
                                 query_parse_calls=decision.query_parse_calls)
        return RouteDecision("clarify", "no_household_match_ambiguous",
                             query_parse_calls=decision.query_parse_calls)

    @staticmethod
    def _family_context(draft, value, focus) -> bool:
        if has_household_signal(draft):
            return True
        if message_anchored(value):
            return True
        if focus and (focus.get("active_entity_ids") or focus.get("active_event_ids")):
            return True
        return False

    def _strong_household(self, draft) -> dict[str, Any]:
        actions = getattr(draft, "actions", []) or []
        if any(a.type in _EVIDENCE_ACTIONS for a in actions):
            return {"hit": True, "reason": "explicit_evidence_action",
                    "answer_target": next((a.target for a in actions if a.type in _EVIDENCE_ACTIONS), "general"),
                    "original": True}
        if any(a.type == "answer_question" and a.target in _STRONG_TARGETS for a in actions):
            return {"hit": True, "reason": "answer_question_strong_target",
                    "answer_target": next((a.target for a in actions if a.type == "answer_question"), "general"),
                    "original": False}
        if getattr(draft, "time_expression", None):
            return {"hit": True, "reason": "time_expression", "answer_target": "general", "original": False}
        if getattr(draft, "media_expressions", None):
            return {"hit": True, "reason": "media_expression", "answer_target": "general",
                    "original": True}
        if getattr(draft, "negative_conditions", None):
            return {"hit": True, "reason": "negation_condition", "answer_target": "general", "original": False}
        if getattr(draft, "entity_names", None):
            return {"hit": True, "reason": "entity_names", "answer_target": "person", "original": False}
        return {"hit": False, "reason": "", "answer_target": "general", "original": False}

    @staticmethod
    def _has_evidence_ask(draft) -> bool:
        actions = getattr(draft, "actions", []) or []
        if any(a.type in _EVIDENCE_ACTIONS for a in actions):
            return True
        if any(a.type == "answer_question" and a.target in _STRONG_TARGETS for a in actions):
            return True
        return False

    def _confirmed_person(self, draft, entity_resolver=None) -> bool:
        resolver = entity_resolver or self._entity_resolver
        if resolver is None:
            return False
        for name in getattr(draft, "entity_names", []) or []:
            if resolver(str(name)):
                return True
        for facet in getattr(draft, "facets", []) or []:
            if facet.dimension == "person" and resolver(facet.surface_text):
                return True
        return False

    def _weak_household(self, draft, value) -> dict[str, Any]:
        if message_anchored(value):
            return {"hit": True, "reason": "raw_message_anchor"}
        if getattr(draft, "facets", None):
            return {"hit": True, "reason": "weak_facets"}
        if getattr(draft, "semantic_conditions", None):
            return {"hit": True, "reason": "semantic_conditions"}
        if getattr(draft, "entity_names", None):
            return {"hit": True, "reason": "entity_names_unconfirmed"}
        if not has_general_verb(value) and not is_writing_request(value) \
                and len(value.strip()) <= _BARE_NOUN_MAX_LEN:
            return {"hit": True, "reason": "bare_noun_phrase"}
        return {"hit": False, "reason": ""}

    @staticmethod
    def _session_follow_up(value, draft, focus) -> list[str]:
        if not focus or not (focus.get("active_entity_ids") or focus.get("active_event_ids")):
            return []
        if has_household_signal(draft):
            return []
        if message_anchored(value):
            return []
        if is_contextual_follow_up(value):
            return list(focus.get("active_entity_ids", []))
        if not is_writing_request(value) and not has_general_verb(value) \
                and len(value.strip()) <= _BARE_NOUN_MAX_LEN:
            return list(focus.get("active_entity_ids", []))
        return []
