import unittest
from unittest.mock import patch

from backend.agent_runtime.semantic_slots import _normalize_slots
from backend.agent_runtime.tools import (
    _build_semantic_routes,
    _explicit_media_filter,
    _build_preview_entries,
    _is_synthetic_event_asset,
    _preview_indices,
    _preview_query_order,
    _recommended_handle,
    _retrieval_support_conditions,
    _needs_place_semantic_fallback,
)


class SemanticSlotTimeGroundingTests(unittest.TestCase):
    def test_single_leave_photo_promotes_semantic_single_person_frame(self):
        class Store:
            def __init__(self):
                self.rows = {
                    "group": {"caption": "四名男子在紫色纱幔前室内合影。",
                              "activity": "合影", "place": "室内空间",
                              "people": ["男子1", "男子2", "男子3", "男子4"]},
                    "target": {"caption": "男子站在紫色与白色帷幕前。",
                                "activity": "站立拍照", "place": "室内空间",
                                "people": ["男子"]},
                }

            def list_observations(self, asset_id=None, limit=1):
                return [self.rows[asset_id]]

        store = Store()
        question = "国庆节我在河北婚礼仪式舞台拍了留影，记得有紫色的布，地点是哪里？"
        order = _preview_query_order(["group", "target"], question, store)
        self.assertEqual(order, [1, 0])

    def test_explicit_user_media_type_survives_missing_tool_filter(self):
        self.assertEqual(
            _explicit_media_filter("我想找全家去正定游玩时拍的那段视频。"),
            "video",
        )
        self.assertEqual(_explicit_media_filter("帮我找那张婚礼照片。"), "image")
        self.assertEqual(_explicit_media_filter("把那张婚礼留影找出来。"), "image")
        self.assertEqual(_explicit_media_filter("找我和朋友的合影。"), "image")

    def test_mixed_media_question_is_not_hard_filtered(self):
        self.assertIsNone(
            _explicit_media_filter("比较这段视频和当时拍的照片。")
        )

    def test_synthetic_event_rendering_is_not_a_user_evidence_asset(self):
        self.assertTrue(_is_synthetic_event_asset({"file_name": "event_event_00001.webp"}))
        self.assertFalse(_is_synthetic_event_asset({"file_name": "2017-10-04 223147.jpg"}))

    def test_event_word_cannot_become_a_relative_time_filter(self):
        slots = _normalize_slots({
            "time": {"expr": "去年", "year": 2025, "kind": "relative"},
            "event": {"name": "婚礼", "certainty": "high"},
        }, "我记得参加亲友婚礼时，在天台迎宾展架旁拍了留影，那次婚礼在哪里办的？")
        self.assertIsNotNone(slots)
        self.assertEqual(slots["time"]["expr"], "")
        self.assertIsNone(slots["time"]["year"])
        self.assertEqual(slots["time"]["months"], [])

    def test_explicit_year_is_preserved(self):
        slots = _normalize_slots({
            "time": {"expr": "2017年", "year": 2017, "kind": "absolute"},
            "event": {"name": "婚礼", "certainty": "high"},
        }, "帮我找2017年河北易县的婚礼照片")
        self.assertEqual(slots["time"]["year"], 2017)
        self.assertEqual(slots["time"]["expr"], "2017年")

    def test_retrieval_keeps_planner_core_tool_and_visual_synonym_routes(self):
        routes = _build_semantic_routes(
            "规划目标：天台婚礼迎宾展架旁的留影",
            query_core="天台婚礼迎宾展架旁留影",
            tool_query="天台婚礼迎宾展架",
            object_terms=["展架", "留影"],
            user_query="我记得参加亲友婚礼时，在天台迎宾展架旁拍了留影，那次婚礼在哪里办的？",
        )
        self.assertEqual(
            [source for source, _ in routes],
            ["primary", "user_question", "slot_query_core", "tool_query_objects"],
        )
        self.assertIn("在哪里办", routes[1][1])
        self.assertIn("留影", routes[3][1])
        self.assertIn("展架", routes[3][1])
        self.assertNotIn("横幅", " ".join(text for _, text in routes))
        self.assertLessEqual(len(routes), 4)

    def test_retrieval_only_expands_genuine_lexical_equivalents(self):
        routes = _build_semantic_routes(
            "婚礼现场的迎宾牌和横幅", query_core="迎宾牌 横幅",
            tool_query="婚礼迎宾牌横幅", max_routes=4,
        )
        self.assertTrue(any(source == "object_synonym" and "迎宾标牌" in text
                            for source, text in routes))
        self.assertFalse(any("展架" in text for _, text in routes))

    def test_retrieval_routes_are_deduplicated_and_bounded(self):
        routes = _build_semantic_routes(
            "婚礼合影", query_core="婚礼合影", tool_query="婚礼合影",
            object_terms=["合影", "婚礼", "照片"], max_routes=3,
        )
        self.assertEqual(routes[0], ("primary", "婚礼合影"))
        self.assertLessEqual(len(routes), 3)
        self.assertEqual(len({text for _, text in routes}), len(routes))

    def test_scene_word_is_not_promoted_to_geographic_place(self):
        slots = _normalize_slots({
            "place": {"name": "天台", "hint": "天台", "certainty": "high"},
            "event": {"name": "婚礼", "certainty": "high"},
        }, "我记得参加亲友婚礼时，在天台迎宾展架旁拍了留影，那次婚礼在哪里办的？")
        self.assertEqual(slots["place"]["name"], "")
        self.assertEqual(slots["place"]["hint"], "")

    def test_full_user_visual_cues_drive_preview_and_recommended_handle(self):
        asset_ids = ["generic", "described_scene"]
        summaries = {
            "generic": "室内婚礼合影；舞台前站着几个人",
            "described_scene": "保定市夜间婚礼仪式舞台；紫色纱幔和灯光背景",
        }
        question = "我记得有次晚上在保定市婚礼仪式舞台前拍了紫色留影"
        preview = [
            {"handle": "photo_1", "evidence_summary": summaries["generic"]},
            {"handle": "photo_2", "evidence_summary": summaries["described_scene"]},
        ]
        with patch("backend.agent_runtime.tools._observation_summary",
                   side_effect=lambda _store, aid: summaries[aid]):
            order = _preview_query_order(asset_ids, question, None)
        self.assertEqual(order[0], 1)
        self.assertEqual(_recommended_handle(question, preview), "photo_2")

    def test_contextual_anchor_conjunction_beats_a_generic_scene_overlap(self):
        """Event-defining anchors must outweigh a lone generic cue.

        This is deliberately not tied to a benchmark filename or answer: any
        query whose event phrase is split differently in the caption exercises
        the same CJK context-coverage path.
        """
        asset_ids = ["generic_night_scene", "event_stage"]
        summaries = {
            "generic_night_scene": "夜间儿童在灯光和气球旁拍照。",
            "event_stage": "夜间婚礼仪式舞台前的留影，背景有帷幕。",
        }
        question = "帮我找晚上在婚礼仪式舞台前拍的留影"
        with patch("backend.agent_runtime.tools._observation_summary",
                   side_effect=lambda _store, aid: summaries[aid]):
            order = _preview_query_order(asset_ids, question, None)
        self.assertEqual(order[0], 1)

    def test_direct_time_and_place_can_use_contextually_matched_metadata(self):
        preview = [{
            "captured_at": "2017-10-04 22:31:47",
            "place": "保定市易县",
            "evidence_summary": "夜间婚礼仪式舞台前的留影，紫色帷幕背景。",
        }]
        conditions = _retrieval_support_conditions(
            "我在婚礼仪式舞台前拍的留影是哪一天、在哪里？", preview)
        self.assertEqual(conditions["semantic_context"], "matched")
        self.assertEqual(conditions["captured_at"], "matched")
        self.assertEqual(conditions["place"], "matched")

    def test_metadata_is_not_upgraded_for_a_generic_context_match(self):
        preview = [{
            "captured_at": "2017-10-04 22:31:47",
            "place": "保定市易县",
            "evidence_summary": "夜间儿童在气球旁拍照。",
        }]
        self.assertEqual(_retrieval_support_conditions(
            "婚礼仪式舞台前的留影是哪一天？", preview), {})

    def test_visible_preview_uses_query_selected_indices_and_stable_handles(self):
        def make_entry(_store, asset_id, handle, *, priority_rank=None,
                       selection_reason="", **_kwargs):
            return {"asset_id": asset_id, "handle": handle,
                    "priority_rank": priority_rank,
                    "selection_reason": selection_reason}

        with patch("backend.agent_runtime.tools._preview_entry", side_effect=make_entry):
            preview = _build_preview_entries(None, ["first", "second", "query_match"], [2, 0])

        self.assertEqual([item["asset_id"] for item in preview], ["query_match", "first"])
        self.assertEqual([item["handle"] for item in preview], ["photo_3", "photo_1"])
        self.assertEqual([item["priority_rank"] for item in preview], [1, 2])

    def test_semantic_search_preview_can_expose_more_than_a_result_page(self):
        asset_ids = [f"asset_{index}" for index in range(18)]
        with patch.dict("os.environ", {"SENTRIX_CANDIDATE_STRATEGY": "head_only"}):
            indices = _preview_indices(asset_ids, "best", None, query="礼物", limit=12)

        self.assertEqual(indices, list(range(12)))

    def test_explicit_named_place_slot_is_preserved(self):
        slots = _normalize_slots({
            "place": {"name": "易县", "hint": "河北省保定市易县", "certainty": "high"},
        }, "2017年国庆我在河北易县参加婚礼，帮我找照片")
        self.assertEqual(slots["place"]["name"], "易县")
        self.assertEqual(slots["place"]["hint"], "河北省保定市易县")

    def test_nonempty_but_unverified_place_results_trigger_soft_fallback(self):
        with patch("backend.agent_runtime.tools._place_matches", return_value=False):
            self.assertTrue(_needs_place_semantic_fallback(
                {"place": "易县沙岭"}, [{"asset_id": "indoor_photo"}], None,
            ))

    def test_verified_place_result_suppresses_soft_fallback(self):
        with patch("backend.agent_runtime.tools._place_matches", return_value=True):
            self.assertFalse(_needs_place_semantic_fallback(
                {"place": "易县沙岭"}, [{"asset_id": "verified_photo"}], None,
            ))


if __name__ == "__main__":
    unittest.main()
