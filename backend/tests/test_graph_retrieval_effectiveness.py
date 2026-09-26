import os
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from backend.evidence_retrieval import (
    _filter_graph_candidates_by_source_score,
    _graph_head_quota,
    _merge_graph_head,
)
from backend.graph_memory.service import GraphMemoryService
from backend.retrieval import CandidateHit, HardFilterContext, RetrievalQuery
from backend.retrieval.graph_expander import GraphExpander
from backend.retrieval.graph_policy import graph_retrieval_policy


class _AssetStore:
    def __init__(self, assets):
        self.assets = assets

    def get_asset(self, asset_id):
        return self.assets.get(asset_id)


class GraphRetrievalEffectivenessTests(unittest.TestCase):
    def test_graph_candidates_cannot_globally_reorder_baseline_head(self):
        baseline = [{"asset_id": key} for key in "abcdef"]
        graph = [{"asset_id": "x"}, {"asset_id": "y"}]

        merged, effect = _merge_graph_head(baseline, graph, 4, 1)

        self.assertEqual([item["asset_id"] for item in merged[:4]], ["a", "b", "c", "x"])
        self.assertEqual(len(merged), 4)
        self.assertEqual(effect["displaced_ids"], ["d"])
        self.assertEqual(effect["promoted_ids"], ["x"])

    def test_graph_can_promote_verified_baseline_candidate_from_below_head(self):
        baseline = [{"asset_id": key} for key in "abcdef"]
        graph = [{"asset_id": "f"}]

        merged, effect = _merge_graph_head(baseline, graph, 4, 1)

        self.assertEqual([item["asset_id"] for item in merged[:4]], ["a", "b", "c", "f"])
        self.assertEqual([item["asset_id"] for item in merged].count("f"), 1)
        self.assertEqual(effect["displaced_ids"], ["d"])

    def test_all_relevant_keeps_baseline_and_adds_graph_items(self):
        baseline = [{"asset_id": key} for key in "abc"]
        graph = [{"asset_id": "d"}]

        merged, effect = _merge_graph_head(baseline, graph, 4, 1, all_relevant=True)

        self.assertEqual([item["asset_id"] for item in merged], ["a", "b", "c", "d"])
        self.assertEqual(effect["displaced_ids"], [])

    def test_relative_graph_score_floor_is_calibrated_per_source(self):
        candidates = [SimpleNamespace(asset_id="anchor"), SimpleNamespace(asset_id="global")]
        hits = {
            "anchor": SimpleNamespace(raw_score=0.75, metadata={"graph_source": "traversal"}),
            "global": SimpleNamespace(raw_score=0.22, metadata={"graph_source": "global_fallback"}),
        }
        kept = _filter_graph_candidates_by_source_score(candidates, hits, 0.55)
        self.assertEqual([candidate.asset_id for candidate in kept], ["anchor", "global"])

    def test_anchor_does_not_make_unrelated_event_summary_relevant(self):
        def node(node_id, kind, attributes, summary=""):
            return SimpleNamespace(
                node_id=node_id,
                node_type=SimpleNamespace(value=kind),
                attributes=attributes,
                summary=summary,
                event_node_ids=[],
                similarity_score=0.0,
            )

        summary = node("session-1", "SESSION", {
            "subtype": "event_summary", "scope_id": "album",
            "event_title": "birthday dinner", "event_summary": "family birthday dinner",
            "frame_ids": ["frame-1"],
        })
        frame = node("event-1", "EVENT", {"source_asset_id": "frame-1"})
        builder = SimpleNamespace(
            graph_db=SimpleNamespace(nodes={"session-1": summary, "event-1": frame}, links={})
        )

        class Engine:
            label_matcher = SimpleNamespace(match=lambda _q: {
                "objects": ["wedding"], "predicates": [], "all": ["wedding"],
            })

            @staticmethod
            def _extract_object_terms(_question):
                return []

            @staticmethod
            def detect_query_type(_question, _matched):
                return "ordinary"

        projected = GraphMemoryService._event_summary_projection(
            builder, Engine(), "wedding", "album", {"frame-1"},
        )
        self.assertEqual(projected, [])

    def test_explicit_photo_query_prefers_still_images_but_ambiguous_event_query_does_not(self):
        filters = HardFilterContext(scope_ids=("album",))
        photo_query = RetrievalQuery(whole_query="这张照片里有什么")
        video_query = RetrievalQuery(whole_query="这段视频里发生了什么")
        ambiguous_query = RetrievalQuery(whole_query="那天婚礼发生了什么")
        self.assertTrue(GraphExpander._prefers_still_images(photo_query, filters))
        self.assertFalse(GraphExpander._prefers_still_images(video_query, filters))
        self.assertFalse(GraphExpander._prefers_still_images(ambiguous_query, filters))

    def test_event_attribute_query_uses_event_graph_route_in_auto_mode(self):
        filters = HardFilterContext(scope_ids=("album",))
        query = RetrievalQuery(whole_query="天台婚礼迎宾展架")
        with patch.dict(os.environ, {"SENTRIX_GRAPH_RETRIEVAL_MODE": "auto"}):
            route = graph_retrieval_policy(query.whole_query, filters=filters, query=query)
        self.assertEqual(route["intent"], "event")
        self.assertTrue(route["enabled"])

    def test_long_natural_event_description_uses_event_graph_route(self):
        filters = HardFilterContext(scope_ids=("album",))
        question = "我记得参加亲友婚礼时，在天台迎宾展架旁拍了留影，那次婚礼在哪里办的？"
        query = RetrievalQuery(whole_query=question)
        with patch.dict(os.environ, {"SENTRIX_GRAPH_RETRIEVAL_MODE": "auto"}):
            route = graph_retrieval_policy(question, filters=filters, query=query)
        self.assertEqual(route["intent"], "event")
        self.assertTrue(route["enabled"])

    def test_event_members_are_added_as_bounded_graph_context_hits(self):
        filters = HardFilterContext(scope_ids=("album",))
        query = RetrievalQuery(whole_query="天台婚礼迎宾展架")
        route = {"intent": "event", "enabled": True, "mode": "on"}
        store = _AssetStore({"photo": {"media_type": "image", "metadata_json": {}}})
        expander = GraphExpander(store)
        anchor = {"event_id": "event-1", "event_title": "wedding", "asset_ids": ["photo"]}
        with patch("backend.agent_runtime.tools._event_keyword_anchor", return_value=anchor):
            hits = expander._event_context_candidates(query, filters, route, 12)
        self.assertEqual([hit.asset_id for hit in hits], ["photo"])
        self.assertEqual(hits[0].metadata["graph_source"], "event_context")
        self.assertEqual(hits[0].metadata["event_id"], "event-1")

    def test_explicit_image_filter_excludes_video_keyframes_and_parent_videos(self):
        store = _AssetStore({
            "photo": {"media_type": "image", "derived_kind": None, "metadata_json": {}},
            "frame": {"media_type": "image", "derived_kind": "video_keyframe_webp", "metadata_json": {}},
            "video": {"media_type": "video", "derived_kind": None, "metadata_json": {}},
        })
        expander = GraphExpander(store)
        self.assertFalse(expander._is_video_evidence("photo"))
        self.assertTrue(expander._is_video_evidence("frame"))
        self.assertTrue(expander._is_video_evidence("video"))

    def test_graph_head_is_capped_at_twenty_percent_and_scales_with_path_intent(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("SENTRIX_GRAPH_HEAD_QUOTA", None)
            self.assertEqual(_graph_head_quota(30, "ordinary", 20), 4)
            self.assertEqual(_graph_head_quota(30, "temporal", 20), 6)
            self.assertEqual(_graph_head_quota(10, "multi_hop", 20), 2)
            self.assertEqual(_graph_head_quota(30, "temporal", 3), 3)

    def test_explicit_graph_quota_override_is_respected_but_still_bounded(self):
        with patch.dict(os.environ, {"SENTRIX_GRAPH_HEAD_QUOTA": "2"}):
            self.assertEqual(_graph_head_quota(30, "multi_hop", 20), 2)
        with patch.dict(os.environ, {"SENTRIX_GRAPH_HEAD_QUOTA": "99"}):
            self.assertEqual(_graph_head_quota(30, "multi_hop", 20), 6)

    def test_graph_head_limits_repeated_assets_per_event_and_keeps_other_events(self):
        hits = [
            CandidateHit(f"a{i}", "graph", .9 - i * .01, "graph_path", True, i + 1,
                         metadata={"event_id": "event-a"})
            for i in range(5)
        ] + [CandidateHit("b1", "graph", .7, "graph_path", True, 6,
                          metadata={"event_id": "event-b"})]
        diversified = GraphExpander._diversify_by_event(hits, per_event_cap=3)
        self.assertEqual([hit.asset_id for hit in diversified], ["a0", "a1", "a2", "b1"])


if __name__ == "__main__":
    unittest.main()
