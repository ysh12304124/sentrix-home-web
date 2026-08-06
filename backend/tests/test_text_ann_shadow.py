"""Phase R9-4 — bge-m3 text ANN shadow tests.

The bge shadow is a parallel experiment: it must never move the visual backbone
top candidates, and a dead sidecar must degrade to text_available=False (via the
client circuit breaker) instead of hanging or crashing the main API.
"""

import json
import unittest
from unittest import mock

import backend.embeddings.bge_text as bge_module
from backend.retrieval.base import CandidateHit
from backend.retrieval.ranking import rank


def _hit(asset_id, retriever, score=1.0):
    return CandidateHit(asset_id=asset_id, retriever=retriever, raw_score=score,
                        score_kind="cosine_similarity", higher_is_better=True, rank=1)


class SidecarClientCircuitBreakerTests(unittest.TestCase):
    def test_available_hits_health_endpoint(self):
        with mock.patch.object(bge_module.httpx, "get") as get:
            get.return_value = mock.Mock(status_code=200)
            embedder = bge_module.BgeM3TextQueryEmbedder(base_url="http://sidecar:8101")
            self.assertTrue(embedder.available)
            get.assert_called()

    def test_embed_roundtrip_parses_vector(self):
        with mock.patch.object(bge_module.httpx, "post") as post:
            post.return_value = mock.Mock()
            post.return_value.raise_for_status.return_value = None
            post.return_value.json.return_value = {"vector": [1.0, 0.0, 2.0]}
            embedder = bge_module.BgeM3TextQueryEmbedder(base_url="http://sidecar:8101")
            self.assertEqual(embedder.embed_query("厨房"), [1.0, 0.0, 2.0])

    def test_circuit_breaker_trips_after_three_failures(self):
        with mock.patch.object(bge_module.httpx, "post", side_effect=RuntimeError("down")):
            embedder = bge_module.BgeM3TextQueryEmbedder(base_url="http://sidecar:8101")
            for _ in range(3):
                self.assertEqual(embedder.embed_query("x"), [])
            # tripped -> further calls short-circuit to [] without HTTP.
            with mock.patch.object(bge_module.httpx, "post") as post:
                self.assertEqual(embedder.embed_query("y"), [])
                post.assert_not_called()

    def test_success_resets_breaker(self):
        with mock.patch.object(bge_module.httpx, "post", side_effect=RuntimeError("down")):
            embedder = bge_module.BgeM3TextQueryEmbedder(base_url="http://sidecar:8101")
            for _ in range(2):
                embedder.embed_query("x")
        with mock.patch.object(bge_module.httpx, "post") as post:
            post.return_value = mock.Mock()
            post.return_value.raise_for_status.return_value = None
            post.return_value.json.return_value = {"vector": [1.0]}
            self.assertEqual(embedder.embed_query("ok"), [1.0])


class TextAnnShadowRankingTests(unittest.TestCase):
    def test_bge_shadow_does_not_move_visual_top_k(self):
        channel_hits = {
            "visual_ann": [_hit("v1", "visual_ann", 0.9), _hit("v2", "visual_ann", 0.8)],
            # a strong bge text hit must not displace v1/v2
            "text_ann": [_hit("t1", "text_ann", 0.99)],
        }
        ranked = rank(channel_hits, "visual_backbone", limit=5)
        ids = [item.asset_id for item in ranked]
        self.assertEqual(ids[:2], ["v1", "v2"])
        self.assertIn("t1", ids[2:])

    def test_clip_text_absent_does_not_break_ranking(self):
        channel_hits = {"visual_ann": [_hit("v1", "visual_ann", 0.9)]}
        ranked = rank(channel_hits, "visual_backbone", limit=5)
        self.assertEqual([item.asset_id for item in ranked], ["v1"])


if __name__ == "__main__":
    unittest.main()
