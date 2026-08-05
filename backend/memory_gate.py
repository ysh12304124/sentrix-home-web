"""Semantic routing gate for the Thin Agent (Phase R R4).

Gate decisions are no longer a hard binary.  A decision carries
``proposed_mode`` and ``allow_probe``; the code routes:

  - explicit general-task structure (writing / translation / hypothesis /
    "don't look up my memory")        -> none, allow_probe=False
  - explicit household-evidence signal (feedback / selected_entity) -> evidence
  - parser says evidence/contextual    -> as the parser says
  - parser says none with no explicit  -> ``ambiguous``, allow_probe=True
    general-task structure                (the Neutral Probe decides)

P0-6: message length and model self-reported ``confidence`` are never routing
inputs.  Keyword tables are not the final classifier; structural signals only
*add* retrieval opportunities (probe), never block them.
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
    allow_probe: bool = False

    @property
    def proposed_mode(self):
        return self.mode

    def as_dict(self):
        data = self.__dict__.copy()
        data["proposed_mode"] = self.mode
        return data


# Narrow structural fast-paths.  These are *not* semantic classifiers — every
# rule below has both positive and negative test coverage in test_semantic_
# routing.py to prove it does not silently swallow evidence requests.
_WRITING_PREFIX_RE = re.compile(r"^\s*(帮我写|请写|写一段|写一篇|生成一段|翻译|帮我起草|拟一份|写个|写篇)")
# Mid-string writing markers ("以相册为主题写一篇短文") — still writing, but
# the prompt does not start with the prefix.
_WRITING_ANYWHERE_RE = re.compile(r"(写一篇|写一段|写个|写篇|拟一份|起草|生成一段|帮我写|请写)")
_NO_LOOKUP_RE = re.compile(r"^\s*(不用查|别查|别找|不用找|不需要查|不用看|不用搜|不看我的)")


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
        if _NO_LOOKUP_RE.search(value):
            return GateDecision("none", "explicit_no_memory_lookup")
        return None

    def classify(self, message, conversation="", *, draft=None, api_signals=None, proactive_enabled=False):
        """Return a :class:`GateDecision` for the current turn.

        Fast-paths run first (no model call).  When none match, ``draft``
        provides the semantic mode; a parser ``none`` without an explicit
        general-task structure is upgraded to ``ambiguous`` so the Neutral
        Probe can decide (R4).
        """
        fast = self.fast_path(message, api_signals=api_signals)
        if fast is not None:
            return fast
        if draft is not None:
            mode = getattr(draft, "mode", None) or "none"
            actions = getattr(draft, "actions", None) or []
            answer_target = next((action.target for action in actions if action.type == "answer_question"), "general")
            original_allowed = any(action.type == "return_assets" for action in actions)
            if mode == "evidence":
                return GateDecision(
                    mode="evidence", reason="specific_household_question",
                    answer_target=answer_target,
                    concrete_memory_reads=1, evidence_search_calls=1,
                    query_parse_calls=1, original_image_allowed=original_allowed,
                )
            if mode == "contextual":
                return GateDecision(
                    mode="contextual", reason="natural_person_mention",
                    answer_target=answer_target, core_memory_reads=1,
                    query_parse_calls=1, original_image_allowed=original_allowed,
                )
            # mode == none.  If the parser still surfaced household facets or
            # conditions, the message is ambiguous -> probe (P0-6: parser none
            # must not be a permanent dead end).  Otherwise an explicit
            # general-task structure is trusted as none.
            if self._has_household_signal(draft):
                return GateDecision(
                    "ambiguous", "parser_none_with_household_signal",
                    answer_target=answer_target,
                    query_parse_calls=1, allow_probe=True,
                )
            if self._explicit_general_task(message):
                return GateDecision("none", "general_task_with_parser_none",
                                    query_parse_calls=1)
            return GateDecision(
                "ambiguous", "parser_none_without_general_task",
                answer_target=answer_target,
                query_parse_calls=1, allow_probe=True,
            )
        # No draft available and no fast-path match — probe is the safe default.
        return GateDecision("ambiguous", "no_parser_draft", allow_probe=True)

    @staticmethod
    def _has_household_signal(draft):
        household_dimensions = {"person", "place", "activity", "clothing",
                                "object", "visual", "time", "relationship", "ocr"}
        if any(facet.dimension in household_dimensions for facet in getattr(draft, "facets", [])):
            return True
        if getattr(draft, "semantic_conditions", None):
            return True
        if getattr(draft, "negative_conditions", None):
            return True
        return False

    @staticmethod
    def _explicit_general_task(message):
        value = str(message or "").strip()
        if _WRITING_PREFIX_RE.search(value) or _WRITING_ANYWHERE_RE.search(value) or _NO_LOOKUP_RE.search(value):
            return True
        # General-intro task verbs.  A bare noun phrase (a short household
        # object reference) has no such verb and therefore routes to the probe;
        # a structured intro (e.g. "介绍一下家庭相册这个概念") is trusted as
        # general when the parser already said none.  Known tradeoff: a person
        # query that the parser hallucinates to none AND starts with an intro
        # verb is not probed — the parser prompt (R5) is the real fix.
        return bool(re.match(
            r"^(请)?(解释|解释一下|为什么|什么是|什么叫|假设|假如|编个|编一段|讲讲你怎么看|介绍一下|介绍下|说明一下|说说|讲讲)",
            value,
        ))
