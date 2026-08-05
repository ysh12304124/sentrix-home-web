"""Neutral Retrieval Probe — decision support, never user facts (Phase R §4).

Multi-signal strong-candidate decision (P0-8), built on raw user text (P0-7).
The probe runs the shared retrievers under probe budgets and answers only
"should we upgrade to a formal retrieval?":

  - channel-calibrated scores (per-space thresholds)
  - multi-channel agreement (same Asset hit by >= N channels)
  - exact lexical whole-phrase hit
  - top-1 signal presence

It never produces an EvidencePacket, never generates household facts, and its
high-score signal alone is never treated as a confirmed fact.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .config import RetrievalConfig


@dataclass
class ProbeOutcome:
    decision: str = "clarify"          # "upgrade" | "clarify" | "none"
    channel_counts: dict[str, int] = field(default_factory=dict)
    signals: dict[str, Any] = field(default_factory=dict)
    reason: str = ""


class NeutralProbe:
    def __init__(self, config: RetrievalConfig | None = None):
        self.config = config or RetrievalConfig()

    def _per_space_minimum(self, retriever: str) -> float:
        space_key = {"visual_ann": "visual", "text_ann": "text", "lexical": "lexical"}.get(retriever)
        if not space_key:
            return 0.0
        return self.config.probe_min_for_space(space_key)

    def run(self, raw_text: str, channel_hits: dict[str, list[Any]], *, scope_id=None, viewer_id="owner") -> ProbeOutcome:
        counts = {name: len(hits) for name, hits in channel_hits.items() if hits}
        agreement = sum(1 for count in counts.values() if count > 0)
        shared = self._shared_assets(channel_hits)
        lexical_exact = self._lexical_exact(channel_hits.get("lexical", []), raw_text)
        top_signals = self._top_signals(channel_hits)
        min_channels = self.config.probe_min_channels
        signals = {
            "agreement_channels": agreement,
            "shared_assets": sorted(shared),
            "lexical_exact": lexical_exact,
            "top_signals": top_signals,
        }
        if agreement >= min_channels and shared:
            return ProbeOutcome("upgrade", counts, signals,
                                f"{len(shared)} asset(s) hit by {agreement} channels")
        if lexical_exact:
            return ProbeOutcome("upgrade", counts, signals, "exact lexical phrase hit")
        above_min = [name for name, top in top_signals.items() if top is not None and top >= self._per_space_minimum(name)]
        if above_min and any(self._per_space_minimum(name) > 0 for name in above_min):
            return ProbeOutcome("upgrade", counts, signals, f"per-space calibrated hit: {above_min}")
        return ProbeOutcome("clarify", counts, signals, "weak or conflicting candidates")

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
