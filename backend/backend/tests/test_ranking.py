"""Phase R8-3 — ranking strategies: visual_only / visual_backbone / late_fusion."""

import unittest

from backend.retrieval.base import CandidateHit
from backend.retrieval.fusion import DEFAULT_CHANNEL_WEIGHTS
from backend.retrieval.ranking import LATE_FUSION, VISUAL_BACKBONE, VISUAL_ONLY, rank


def _hit(asset_id, retriever, score, rank_i):
    return CandidateHit(asset_id=asset_id, retriever=retriever, raw_score=score,
                        score_kind="cosine_similarity", higher_is_better=True, rank=rank_i)


class RankingStrategyTests(unittest.TestCase):
    def _channel_hits(self):
        return {
            "visual_ann": [_hit("gt_visual", "visual_ann", 0.90, 1),
                            _hit("other_visual", "visual_ann", 0.85, 2)],
            "lexical": [_hit("gt_lexical", "lexical", 3.0, 1)],
            "text_ann": [_hit("gt_text", "text_ann", 0.60, 1)],
        }

    def test_visual_only_returns_only_visual(self):
        ranked = rank(self._channel_hits(), VISUAL_ONLY, 10)
        self.assertEqual([c.asset_id for c in ranked], ["gt_visual", "other_visual"])

    def test_visual_backbone_keeps_visual_order_then_appends(self):
        ranked = rank(self._channel_hits(), VISUAL_BACKBONE, 10)
        ids = [c.asset_id for c in ranked]
        self.assertEqual(ids[:2], ["gt_visual", "other_visual"])  # visual order preserved
        self.assertIn("gt_lexical", ids[2:])                      # lexical appended, not displacing
        self.assertIn("gt_text", ids[2:])

    def test_visual_backbone_never_displaces_visual_top(self):
        ranked = rank(self._channel_hits(), VISUAL_BACKBONE, 10)
        self.assertEqual(ranked[0].asset_id, "gt_visual")

    def test_late_fusion_ranks_by_normalized_score(self):
        ranked = rank(self._channel_hits(), LATE_FUSION, 10, fusion_weights=DEFAULT_CHANNEL_WEIGHTS)
        ids = [c.asset_id for c in ranked]
        self.assertEqual(ids[0], "gt_visual")  # highest visual score
        self.assertIn("gt_lexical", ids)

    def test_visual_only_empty_without_visual_channel(self):
        hits = {"lexical": [_hit("a", "lexical", 1.0, 1)]}
        self.assertEqual(rank(hits, VISUAL_ONLY, 10), [])


if __name__ == "__main__":
    unittest.main()
