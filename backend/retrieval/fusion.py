"""Candidate fusion — RRF with evidence-class tiering (Phase R §5.4).

P1-1: channels are not all equal.
  - Hard constraints        : filtered before fusion, never fused.
  - Deterministic anchors   : metadata exact / confirmed entity — stable boost.
  - Lexical / Visual / Text : rank-fused with RRF (k=60, Cormack 2009 constant).
  - Adjacency               : inherits its seed's class, never independently
                              promoted (handled by the Kernel's expander pass).

Fusion only consumes ``rank`` (RRF) — raw cross-channel scores are never
compared directly.  Probe layer consumes ``calibrated_score`` separately.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .base import CandidateHit

RRF_K = 60
ANCHOR_BOOST = 1.0

ANCHOR_RETRIEVERS = ("metadata", "entity")
SEMANTIC_RETRIEVERS = ("lexical", "visual_ann", "text_ann")
EXPANDER_RETRIEVERS = ("adjacency",)


def rrf_score(ranks: dict[str, int], k: int = RRF_K) -> float:
    return sum(1.0 / (k + rank) for rank in ranks.values())


@dataclass
class FusedCandidate:
    asset_id: str
    channels: dict[str, int] = field(default_factory=dict)   # retriever -> rank
    rrf: float = 0.0
    evidence_class: str = "semantic"
    raw_scores: dict[str, float] = field(default_factory=dict)
    retriever_hits: list[CandidateHit] = field(default_factory=list)


def evidence_class_for(retriever: str) -> str:
    if retriever in ANCHOR_RETRIEVERS:
        return "anchor"
    if retriever in EXPANDER_RETRIEVERS:
        return "expander"
    return "semantic"


def fuse(
    channel_hits: dict[str, list[CandidateHit]],
    *,
    k: int = RRF_K,
    anchor_boost: float = ANCHOR_BOOST,
    include_classes: bool = True,
) -> list[FusedCandidate]:
    """Merge per-channel ranked hits into fusion-ranked candidates.

    ``channel_hits`` maps retriever name -> its ranked CandidateHits.  Returns
    candidates sorted by final score descending.
    """
    candidates: dict[str, FusedCandidate] = {}
    for retriever, hits in channel_hits.items():
        for rank, hit in enumerate(hits):
            candidate = candidates.setdefault(
                hit.asset_id,
                FusedCandidate(asset_id=hit.asset_id, channels={}, rrf=0.0,
                               evidence_class=evidence_class_for(retriever),
                               raw_scores={}, retriever_hits=[]),
            )
            candidate.channels[retriever] = rank + 1
            candidate.rrf = rrf_score(candidate.channels, k=k)
            candidate.raw_scores[retriever] = hit.raw_score
            candidate.retriever_hits.append(hit)
            # An asset that appears across multiple channels naturally gains
            # RRF; anchors also get a deterministic boost.
    ranked = []
    for candidate in candidates.values():
        score = candidate.rrf
        if candidate.evidence_class == "anchor":
            score += anchor_boost
        # Expanders inherit their seed; their own contribution is unboosted RRF.
        ranked.append((candidate, score))
    ranked.sort(key=lambda item: item[1], reverse=True)
    return [candidate for candidate, _ in ranked]
