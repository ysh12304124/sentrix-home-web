"""Phase R R3 — fusion: RRF, evidence-class tiering, expander inheritance."""

import unittest

from backend.retrieval.base import CandidateHit
from backend.retrieval.fusion import (
    ANCHOR_RETRIEVERS,
    RRF_K,
    evidence_class_for,
    fuse,
    rrf_score,
)


def _hits(retriever, ids):
    return [CandidateHit(asset_id=item, retriever=retriever, raw_score=float(i + 1),
                         score_kind="discrete", higher_is_better=True, rank=i + 1)
            for i, item in enumerate(ids)]


class RrfTests(unittest.TestCase):
    def test_single_rank(self):
        self.assertAlmostEqual(rrf_score({"a": 1}), 1.0 / (RRF_K + 1))

    def test_two_channels_add(self):
        total = rrf_score({"a": 1, "b": 2})
        self.assertAlmostEqual(total, 1.0 / (RRF_K + 1) + 1.0 / (RRF_K + 2))


class EvidenceClassTests(unittest.TestCase):
    def test_anchor_retrievers(self):
        for name in ANCHOR_RETRIEVERS:
            self.assertEqual(evidence_class_for(name), "anchor")

    def test_semantic_retrievers(self):
        self.assertEqual(evidence_class_for("lexical"), "semantic")
        self.assertEqual(evidence_class_for("visual_ann"), "semantic")
        self.assertEqual(evidence_class_for("text_ann"), "semantic")

    def test_expander(self):
        self.assertEqual(evidence_class_for("adjacency"), "expander")


class FuseTests(unittest.TestCase):
    def test_anchor_beats_semantic_at_same_rank(self):
        ranked = fuse({
            "metadata": _hits("metadata", ["m"]),
            "visual_ann": _hits("visual_ann", ["v"]),
        })
        self.assertEqual(ranked[0].asset_id, "m")

    def test_multi_channel_agreement_beats_single(self):
        ranked = fuse({
            "lexical": _hits("lexical", ["x"]),
            "text_ann": _hits("text_ann", ["y", "x"]),
        })
        # x recalled by both channels -> higher RRF than y (single channel).
        self.assertEqual(ranked[0].asset_id, "x")

    def test_expander_inherits_and_never_promotes_alone(self):
        ranked = fuse({"adjacency": _hits("adjacency", ["e"])})
        self.assertEqual(ranked[0].evidence_class, "expander")
        # Alone, an expander carries no anchor boost — its RRF is small.
        self.assertLess(ranked[0].rrf, 0.02)

    def test_empty_channels(self):
        self.assertEqual(fuse({}), [])

    def test_custom_k(self):
        ranked = fuse({"metadata": _hits("metadata", ["m"])}, k=10)
        self.assertEqual(ranked[0].rrf, 1.0 / 11.0)


if __name__ == "__main__":
    unittest.main()
