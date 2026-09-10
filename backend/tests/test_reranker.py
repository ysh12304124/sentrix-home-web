import unittest

from backend.retrieval.reranker import (
    TemporalCluster,
    build_cluster_text,
    cluster_candidates,
    rerank_candidates,
)


class _Store:
    def __init__(self, assets, observations=None):
        self.assets = assets
        self.observations = observations or {}

    def get_asset(self, asset_id):
        return self.assets.get(asset_id)

    def list_observations(self, *, asset_id, limit=20):
        return list(self.observations.get(asset_id, []))[:limit]


class _Scorer:
    model_path = "fake-bge-reranker"
    device = "cpu"
    last_load_ms = 0.0

    def score(self, query, passages):
        return [0.9 if "red bicycle" in passage else 0.1 for passage in passages]


class _FailIfCalledScorer:
    def score(self, query, passages):
        raise AssertionError("whole-video candidates must not invoke BGE")


class TemporalClusterTests(unittest.TestCase):
    def test_rank_order_centres_cluster_only_same_video_within_one_second(self):
        store = _Store({
            "centre": {"parent_asset_id": "video-a", "source_timestamp_sec": 0.0},
            "near": {"parent_asset_id": "video-a", "source_timestamp_sec": 0.8},
            "not_transitive": {"parent_asset_id": "video-a", "source_timestamp_sec": 1.6},
            "other_video": {"parent_asset_id": "video-b", "source_timestamp_sec": 0.2},
            "still": {"media_type": "image"},
        })
        clusters = cluster_candidates(
            ["centre", "near", "not_transitive", "other_video", "still"],
            store,
            window_seconds=1.0,
        )
        self.assertEqual(
            [cluster.member_ids for cluster in clusters],
            [("centre", "near"), ("not_transitive",), ("other_video",), ("still",)],
        )

    def test_higher_ranked_frame_is_cluster_representative(self):
        store = _Store({
            "rank1": {"metadata_json": {"source_video_asset_id": "v", "timestamp_sec": 5.0}},
            "rank2": {"metadata_json": {"source_video_asset_id": "v", "timestamp_sec": 4.5}},
        })
        clusters = cluster_candidates(["rank1", "rank2"], store)
        self.assertEqual(clusters[0].center_id, "rank1")
        self.assertEqual(clusters[0].member_ids, ("rank1", "rank2"))


class ClusterTextTests(unittest.TestCase):
    def test_structured_fields_are_deduplicated_and_visual_text_is_not_repeated(self):
        observation = {
            "caption": "a person beside a red bicycle",
            "ocr_text": "BIKE SHOP",
            "objects": [{"label": "red bicycle", "details": ["metal basket"]}],
            "spatial_relations": [{"subject": "person", "relation": "beside", "object": "bicycle"}],
            "activity": "standing",
            "event_type": "shopping",
            "detail": {
                "caption": "a person beside a red bicycle",
                "ocr_text": "BIKE SHOP",
                "objects": [{"label": "red bicycle", "details": ["metal basket"]}],
                "visual_text": "a person beside a red bicycle BIKE SHOP red bicycle",
            },
        }
        store = _Store({"a": {}}, {"a": [observation]})
        text = build_cluster_text(TemporalCluster("a", ("a",)), store)
        self.assertEqual(text.count("a person beside a red bicycle"), 1)
        self.assertEqual(text.count("BIKE SHOP"), 1)
        self.assertEqual(text.count("red bicycle"), 2)  # caption + Object field
        self.assertIn("Object Description: metal basket", text)
        self.assertNotIn("Visual:", text)

    def test_visual_text_is_fallback_and_uniform_sample_is_removed(self):
        store = _Store(
            {"a": {}},
            {"a": [{"activity": "uniform_sample", "visual_text": "fallback description"}]},
        )
        text = build_cluster_text(TemporalCluster("a", ("a",)), store)
        self.assertEqual(text, "Visual: fallback description")
        self.assertNotIn("uniform_sample", text)


class RerankTests(unittest.TestCase):
    def test_cross_encoder_reorders_cluster_centres(self):
        store = _Store(
            {"first": {}, "target": {}},
            {
                "first": [{"caption": "blue boat"}],
                "target": [{"caption": "red bicycle"}],
            },
        )
        result = rerank_candidates(
            "where is the red bicycle", ["first", "target"], store, scorer=_Scorer(),
            coarse_rank_weight=0.0, bge_rank_weight=1.0,
        )
        self.assertEqual(result.asset_ids, ["target", "first"])
        self.assertEqual(result.telemetry["status"], "ok")
        self.assertEqual(result.telemetry["coarse_candidates"], 2)
        self.assertEqual(result.telemetry["temporal_clusters"], 2)

    def test_whole_video_summaries_preserve_rrf_until_keyframes_exist(self):
        store = _Store(
            {"v1": {"media_type": "video"}, "v2": {"media_type": "video"}},
            {"v1": [{"caption": "first"}], "v2": [{"caption": "second"}]},
        )
        result = rerank_candidates(
            "second", ["v1", "v2"], store, scorer=_FailIfCalledScorer(),
        )
        self.assertEqual(result.asset_ids, ["v1", "v2"])
        self.assertEqual(result.telemetry["status"], "skipped_whole_video_only")


if __name__ == "__main__":
    unittest.main()
