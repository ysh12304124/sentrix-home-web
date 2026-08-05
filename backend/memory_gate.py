"""Semantic routing gate for the Thin Agent.

Phase 2R rewrites the gate: keyword tables are no longer the final classifier.
Fast-paths are narrow structural rules (explicit writing instructions, explicit
API signals).  Otherwise the gate reads ``QueryParseDraft.mode`` produced by
``QueryParser``.  It may still upgrade/downgrade the model call based on later
evidence, but never falls back to keyword classification.
"""

from __future__ import annotations

from dataclasses import dataclass
import re


@dataclass(frozen=True)
class GateDecision:
    mode: str
    reason: str
    answer_target: str = "general"
    core_memory_reads: int = 0
    concrete_memory_reads: int = 0
    evidence_search_calls: int = 0
    query_parse_calls: int = 0
    original_image_allowed: bool = False
    proactivity_probe_performed: bool = False

    def as_dict(self):
        return self.__dict__.copy()


# Narrow structural fast-paths.  These are *not* semantic classifiers — every
# rule below has both positive and negative test coverage in test_semantic_
# routing.py to prove it does not silently swallow evidence requests.
_WRITING_PREFIX_RE = re.compile(r"^\s*(帮我写|请写|写一段|写一篇|生成一段|翻译|帮我起草|拟一份|写个|写篇)")


class MemoryGate:
    """Coordinator that maps a parser draft (or API signals) to a gate mode.

    The gate no longer owns any topic-word list.  All open-vocabulary
    classification lives in :class:`QueryParser`.
    """

    def fast_path(self, message, *, api_signals=None):
        """Return a :class:`GateDecision` when a narrow structural rule matches.

        Fast-paths never call the model.  Callers must run the query parser
        only when :meth:`fast_path` returns ``None``.
        """
        api_signals = api_signals or {}
        if api_signals.get("feedback"):
            return GateDecision(
                "evidence", "explicit_feedback",
                answer_target="general", concrete_memory_reads=1, evidence_search_calls=1,
            )
        if api_signals.get("selected_entity_id"):
            return GateDecision(
                "evidence", "explicit_entity_selection",
                answer_target="person", concrete_memory_reads=1, evidence_search_calls=1,
            )
        value = str(message or "").strip()
        if _WRITING_PREFIX_RE.search(value):
            return GateDecision("none", "explicit_writing_task")
        return None

    def classify(self, message, conversation="", *, draft=None, api_signals=None, proactive_enabled=False):
        """Return a :class:`GateDecision` for the current turn.

        Fast-paths run first (no model call).  When none match, ``draft``
        provides the semantic mode.
        """
        fast = self.fast_path(message, api_signals=api_signals)
        if fast is not None:
            return fast
        if draft is not None:
            mode = getattr(draft, "mode", None) or "none"
            actions = getattr(draft, "actions", None) or []
            answer_target = next((action.target for action in actions if action.type == "answer_question"), "general")
            original_allowed = any(action.type == "return_assets" for action in actions)
            reason = {
                "none": "general_chat",
                "contextual": "natural_person_mention",
                "evidence": "specific_household_question",
            }.get(mode, "general_chat")
            return GateDecision(
                mode=mode,
                reason=reason,
                answer_target=answer_target,
                core_memory_reads=1 if mode == "contextual" else 0,
                concrete_memory_reads=1 if mode == "evidence" else 0,
                evidence_search_calls=1 if mode == "evidence" else 0,
                query_parse_calls=1,
                original_image_allowed=original_allowed,
            )
        # No draft available and no fast-path match — safe default is none.
        # Callers that need evidence must run the parser first.
        return GateDecision("none", "no_parser_draft")
