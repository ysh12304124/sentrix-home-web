"""Neutral Retrieval Probe — decision support, never user facts (Phase R §4 / R9-2).

Multi-signal strong-candidate decision (P0-8), built on raw user text (P0-7).
The probe runs the shared retrievers under probe budgets and answers only
"should we upgrade to a formal retrieval?":

  - channel-calibrated scores (per-space thresholds)
  - multi-channel agreement (same Asset hit by >= N channels)
  - exact lexical whole-phrase hit
  - top-1 signal presence

R9-2 additions: the probe also reports ``channel_agreement``, ``top_candidates``,
``conflicts`` and ``index_health``, accepts session ``focus`` and a ``media_hint``,
and returns ``no_household_match`` when no channel produced any candidate — the
Router then decides clarify vs none.  The probe never produces an EvidencePacket,
never generates household facts, and its high-score signal alone is never treated
as a confirmed fact.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .config import RetrievalConfig


@dataclass
class ProbeOutcome:
    decision: str = "clarify"          # "upgrade" | "clarify" | "no_household_match"
    channel_counts: dict[str, int] = field(default_factory=dict)
    signals: dict[str, Any] = field(default_factory=dict)
    reason: str = ""
    channel_agreement: int = 0
    top_candidates: list[str] = field(default_factory=list)
    conflicts: list[str] = field(default_factory=list)
    index_health: dict[str, Any] = field(default_factory=dict)


class NeutralProbe:
    def __init__(self, config: RetrievalConfig | None = None):
        self.config = config or RetrievalConfig()

    def _per_space_minimum(self, retriever: str) -> float:
        space_key = {"visual_ann": "visual", "text_ann": "text", "lexical": "lexical"}.get(retriever)
        if not space_key:
            return 0.0
        return self.config.probe_min_for_space(space_key)

    def run(self, raw_text: str, channel_hits: dict[str, list[Any]], *,
            scope_id=None, viewer_id="owner", focus=None, media_hint=None,
            index_health=None) -> ProbeOutcome:
        counts = {name: len(hits) for name, hits in channel_hits.items() if hits}
        agreement = sum(1 for count in counts.values() if count > 0)
        shared = self._shared_assets(channel_hits)
        lexical_exact = self._lexical_exact(channel_hits.get("lexical", []), raw_text)
        top_signals = self._top_signals(channel_hits)
        min_channels = self.config.probe_min_channels
        top_candidates = self._top_candidates(channel_hits)
        conflicts = self._conflicts(channel_hits, shared)
        health = index_health or {}
        signals = {
            "agreement_channels": agreement,
            "shared_assets": sorted(shared),
            "lexical_exact": lexical_exact,
            "top_signals": top_signals,
            "focus_active": bool(focus and (focus.get("active_entity_ids") or focus.get("active_event_ids"))),
            "media_hint": media_hint,
        }
        if not counts and not lexical_exact:
            # R9-2: nothing matched at all — the Router decides clarify vs none.
            return ProbeOutcome("no_household_match", counts, signals,
                                "no channel candidates", channel_agreement=0,
                                top_candidates=top_candidates, index_health=health)
        if agreement >= min_channels:
            # Two or more independent channels found candidates — the message
            # has household relevance even if they don't agree on the same
            # asset yet.  The formal retrieval then reconciles them.
            return ProbeOutcome("upgrade", counts, signals,
                                f"{agreement} channels produced candidates",
                                channel_agreement=agreement,
                                top_candidates=top_candidates, conflicts=conflicts,
                                index_health=health)
        if lexical_exact:
            return ProbeOutcome("upgrade", counts, signals, "exact lexical phrase hit",
                                channel_agreement=agreement,
                                top_candidates=top_candidates, conflicts=conflicts,
                                index_health=health)
        above_min = [name for name, top in top_signals.items() if top is not None and top >= self._per_space_minimum(name)]
        if above_min and any(self._per_space_minimum(name) > 0 for name in above_min):
            return ProbeOutcome("upgrade", counts, signals,
                                f"per-space calibrated hit: {above_min}",
                                channel_agreement=agreement,
                                top_candidates=top_candidates, conflicts=conflicts,
                                index_health=health)
        return ProbeOutcome("clarify", counts, signals, "weak or conflicting candidates",
                            channel_agreement=agreement, top_candidates=top_candidates,
                            conflicts=conflicts, index_health=health)

    @staticmethod
    def _channel_assets(channel_hits):
        return {name: {hit.asset_id for hit in hits} for name, hits in channel_hits.items() if hits}

    @staticmethod
    def _top_candidates(channel_hits):
        seen, ordered = set(), []
        for name, hits in channel_hits.items():
            for hit in hits:
                if hit.asset_id not in seen:
                    seen.add(hit.asset_id)
                    ordered.append(hit.asset_id)
        return ordered

    @staticmethod
    def _conflicts(channel_hits, shared):
        assets = NeutralProbe._channel_assets(channel_hits)
        if shared:
            return [name for name, members in assets.items() if not members & set(shared)]
        return [name for name in assets]

    @staticmethod
    def _shared_assets(channel_hits):
        memberships: dict[str, set] = {}
        for name, hits in channel_hits.items():
            if not hits:
                continue
            memberships[name] = {hit.asset_id for hit in hits}
        if not memberships:
            return set()
        first = next(iter(memberships.values()))
        shared = first.copy()
        for assets in list(memberships.values())[1:]:
            shared &= assets
        return shared

    @staticmethod
    def _lexical_exact(lexical_hits, raw_text):
        normalized = "".join(str(raw_text or "").split()).lower()
        if not normalized:
            return False
        for hit in lexical_hits:
            matched = (hit.matched_text or "").lower()
            if matched and (normalized in matched or matched in normalized):
                return True
            if hit.raw_score >= 2.0:  # whole + at least one bigram matched
                return True
        return False

    @staticmethod
    def _top_signals(channel_hits):
        top = {}
        for name, hits in channel_hits.items():
            if not hits:
                continue
            best = max((hit.raw_score for hit in hits), default=None)
            top[name] = best
        return top
