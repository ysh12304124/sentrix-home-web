"""Ranking strategies (Phase R8-3).

R7 ablation showed the visual channel (Chinese-CLIP) is the strongest single
channel (Recall@10 0.887) and flat fusion diluted it.  Three configurable
strategies replace the single RRF path:

  visual_only      hard filters (scope/time/media) + pure visual ranking.
                   Default user-visible path; nothing may displace it.
  visual_backbone  visual keeps the top order; lexical/text only ADD
                   candidates that are NOT already in visual's top-K, appended
                   after the visual tail (recall extension, never re-ranking).
  late_fusion      per-channel calibrated-score normalization + weighted sum;
                   a weak channel below its threshold does not participate.
  weighted_rrf     rank-based weighted RRF across every enabled channel
                   (see fusion.fuse).  Best measured pure-code recall@K on
                   the album3-max 100QA sweep (top-50 asset recall 0.971);
                   used as the production ranking once candidate count is
                   bounded by the retrieval kernel's top-K convergence.

Strategy selected via ``SENTRIX_RETRIEVER_RANKING`` / RetrievalConfig.
``visual_backbone`` and ``late_fusion`` shadow-run alongside ``visual_only``
so the trace can show which GT each channel moved without affecting the
user-visible order (see EvidencePacket.channel_hits).
"""

from __future__ import annotations

from .base import CandidateHit
from .fusion import FusedCandidate

VISUAL_ONLY = "visual_only"
VISUAL_BACKBONE = "visual_backbone"
LATE_FUSION = "late_fusion"
WEIGHTED_RRF = "weighted_rrf"
STRATEGIES = (VISUAL_ONLY, VISUAL_BACKBONE, LATE_FUSION, WEIGHTED_RRF)

VISUAL_CHANNEL = "visual_ann"


def _candidate(asset_id, hits_by_channel) -> FusedCandidate:
    candidate = FusedCandidate(asset_id=asset_id)
    for channel, hits in hits_by_channel.items():
        for rank, hit in enumerate(hits):
            if hit.asset_id == asset_id:
                candidate.channels[channel] = rank + 1
                candidate.raw_scores[channel] = hit.raw_score
                candidate.retriever_hits.append(hit)
                break
    return candidate


def rank(
    channel_hits: dict[str, list[CandidateHit]],
    strategy: str,
    limit: int,
    *,
    fusion_weights: dict[str, float] | None = None,
) -> list[FusedCandidate]:
    """Order candidate assets according to the selected strategy."""
    if strategy == VISUAL_ONLY:
        return _rank_visual_only(channel_hits, limit)
    if strategy == VISUAL_BACKBONE:
        return _rank_visual_backbone(channel_hits, limit)
    if strategy == LATE_FUSION:
        return _rank_late_fusion(channel_hits, limit, fusion_weights)
    if strategy == WEIGHTED_RRF:
        return _rank_weighted_rrf(channel_hits, limit, fusion_weights)
    # Unknown strategy falls back to visual-only (the safest default).
    return _rank_visual_only(channel_hits, limit)


def _rank_visual_only(channel_hits, limit) -> list[FusedCandidate]:
    visual = channel_hits.get(VISUAL_CHANNEL, [])
    out = []
    for rank, hit in enumerate(visual[:limit]):
        candidate = FusedCandidate(asset_id=hit.asset_id, channels={VISUAL_CHANNEL: rank + 1},
                                   raw_scores={VISUAL_CHANNEL: hit.raw_score},
                                   retriever_hits=[hit])
        out.append(candidate)
    return out


def _rank_visual_backbone(channel_hits, limit) -> list[FusedCandidate]:
    visual = channel_hits.get(VISUAL_CHANNEL, [])
    primary = [(hit.asset_id, rank + 1, hit) for rank, hit in enumerate(visual)]
    primary_ids = {asset_id for asset_id, _, _ in primary}
    out = [_candidate(asset_id, channel_hits) for asset_id, _, _ in primary[:limit]]
    # Append recall from other channels, never displacing the visual order.
    for channel, hits in channel_hits.items():
        if channel == VISUAL_CHANNEL:
            continue
        for hit in hits:
            if hit.asset_id in primary_ids or any(c.asset_id == hit.asset_id for c in out):
                continue
            out.append(_candidate(hit.asset_id, channel_hits))
            if len(out) >= limit:
                break
    return out[:limit]


def _rank_weighted_rrf(channel_hits, limit, fusion_weights) -> list[FusedCandidate]:
    """Rank-based weighted RRF across all enabled channels (fusion.fuse).

    RRF is rank-based, so it is robust to the different raw-score scales of
    each channel; the weights express each channel's observed contribution.
    This is the strategy the pure-code sweep measured as the strongest
    recall@K, and the one the retrieval kernel bounds by candidate top-K.
    """
    from .fusion import fuse
    return fuse(channel_hits, channel_weights=fusion_weights)[:limit]


def _rank_late_fusion(channel_hits, limit, fusion_weights) -> list[FusedCandidate]:
    # Score-based fusion: normalize each channel's raw_score to [0,1] by its max,
    # weight it, sum across channels for each asset.  Weak channels with no
    # candidates simply contribute nothing.
    weights = fusion_weights or {}
    scored: dict[str, FusedCandidate] = {}
    for channel, hits in channel_hits.items():
        if not hits:
            continue
        weight = weights.get(channel, 1.0)
        channel_max = max(hit.raw_score for hit in hits) or 1.0
        for rank, hit in enumerate(hits):
            candidate = scored.setdefault(hit.asset_id, _candidate(hit.asset_id, channel_hits))
            candidate.rrf += weight * (hit.raw_score / channel_max)
    ranked = sorted(scored.values(), key=lambda c: c.rrf, reverse=True)
    return ranked[:limit]
