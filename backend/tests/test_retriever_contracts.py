"""Phase R R2 — retriever contracts: data shapes, filter context, fusion."""

import unittest
from datetime import datetime

from backend.query_contracts import Constraint, QueryAction, QueryFacet, QuerySpec
from backend.retrieval import HardFilterContext, RetrievalQuery
from backend.retrieval.base import CandidateHit
from backend.retrieval.fusion import fuse


class CandidateHitContractTests(unittest.TestCase):
    def test_score_direction_fields_present(self):
        hit = CandidateHit(asset_id="a", retriever="visual_ann", raw_score=0.2,
                           score_kind="cosine_distance", higher_is_better=False, rank=1)
        self.assertFalse(hit.higher_is_better)
        self.assertEqual(hit.score_kind, "cosine_distance")
        self.assertIsNone(hit.calibrated_score)


class HardFilterContextTests(unittest.TestCase):
    def _spec(self, constraints):
        return QuerySpec(query_id="q", scope_mode="single", scope_ids=["album1"],
                         viewer_id="owner", conversation_id="c", intent="answer",
                         answer_target="general", constraints=constraints)

    def test_media_and_negation_extracted(self):
        spec = self._spec([
            Constraint("media", "image", "deterministic_hard", "asset_metadata"),
            Constraint("media", "video", "deterministic_hard", "asset_metadata", negated=True),
        ])
        filters = HardFilterContext.from_spec(spec)
        self.assertEqual(filters.media_types, ("image",))
        self.assertIn("video", filters.negated_media)

    def test_time_bounds_parsed(self):
        spec = self._spec([Constraint("time", "2024 年 5 月", "deterministic_hard", "asset_metadata")])
        filters = HardFilterContext.from_spec(spec)
        self.assertIsNotNone(filters.time_bounds)
        self.assertEqual(filters.time_bounds[0], datetime(2024, 5, 1))
        self.assertEqual(filters.time_bounds[1], datetime(2024, 6, 1))


class RetrievalQueryTests(unittest.TestCase):
    def test_whole_query_from_constraint_source_text(self):
        spec = QuerySpec(query_id="q", scope_mode="single", scope_ids=["album1"], viewer_id="owner",
                         conversation_id="c", intent="answer", answer_target="general",
                         constraints=[Constraint("clothing", "毛绒睡衣", "semantic_required", "direct_or_possible", source_text="浅黄色毛绒睡衣")],
                         facets=[QueryFacet(dimension="visual", surface_text="自拍")])
        query = RetrievalQuery.from_spec(spec)
        self.assertEqual(query.whole_query, "浅黄色毛绒睡衣")
        self.assertEqual(len(query.facets), 1)
        self.assertEqual(query.facets[0].surface_text, "自拍")


class FusionTests(unittest.TestCase):
    def _hits(self, retriever, ids):
        return [CandidateHit(asset_id=item, retriever=retriever, raw_score=float(i + 1),
                             score_kind="discrete", higher_is_better=True, rank=i + 1)
                for i, item in enumerate(ids)]

    def test_multi_channel_agreement_rises(self):
        channel_hits = {
            "metadata": self._hits("metadata", ["a", "b"]),
            "lexical": self._hits("lexical", ["b"]),
        }
        ranked = fuse(channel_hits)
        # b is recalled by both channels (1/62 + 1/61), a only by one (1/61).
        self.assertEqual(ranked[0].asset_id, "b")
        self.assertGreater(ranked[0].rrf, ranked[1].rrf)

    def test_anchor_boost_promotes_anchor(self):
        channel_hits = {
            "metadata": self._hits("metadata", ["m"]),
            "visual_ann": self._hits("visual_ann", ["v"]),
        }
        ranked = fuse(channel_hits)
        self.assertEqual(ranked[0].asset_id, "m")
        self.assertEqual(ranked[0].evidence_class, "anchor")
        self.assertEqual(ranked[1].evidence_class, "semantic")

    def test_empty_channel_input(self):
        self.assertEqual(fuse({}), [])


if __name__ == "__main__":
    unittest.main()
