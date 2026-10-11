"""Regression tests for bounded, ResultSet-scoped evidence delivery."""

import os
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

from PIL import Image

from backend.db import MemoryStore
from backend.model_clients import GammaClient
from backend.agent_runtime import tools as runtime_tools
from backend.agent_runtime.result_set import debug_asset_projection, TaskState as ResultTaskState
from backend.agent_runtime.intent import visual_intent
from backend.agent_runtime.completion import (
    CompletionState, RESOLVE_OCR, RESOLVE_VISUAL,
)
from backend.agent_runtime.runtime import (
    _model_visible_observation, _next_resolution_handle, _normalize_preview_handle,
    _pending_resolution,
)
from backend.agent_runtime.task_state import EvidenceRequirement, TaskDeclaration, TaskState


class ResultSetContractTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.store = MemoryStore(os.path.join(self.tmp, "test.db"))
        runtime_tools._RUNTIME.clear()
        runtime_tools.bind_runtime(self.store)
        runtime_tools.register_tools()

    def tearDown(self):
        self.store.close()

    def test_search_reference_hides_internal_asset_ids_and_bounds_preview(self):
        rs = runtime_tools._RUNTIME["result_sets"].new(
            scope_id="album", query="照片", asset_ids=[f"asset_{i}" for i in range(20)]
        )
        out = runtime_tools._search_from_prior_result_set(rs, "album")
        self.assertNotIn("asset_ids", out)
        self.assertLessEqual(len(out["preview"]), 6)
        self.assertEqual(out["preview"][0]["handle"], "photo_1")

    def test_result_page_rejects_new_query_and_caps_page_size(self):
        rs = runtime_tools._RUNTIME["result_sets"].new(
            scope_id="album", query="照片", asset_ids=[f"asset_{i}" for i in range(20)]
        )
        rejected = runtime_tools._get_result_page(
            {"result_set_id": rs.result_set_id, "query": "换成视频"},
            context={"scope_id": "album", "task_state": {}},
        )
        self.assertTrue(rejected["requires_new_search"])
        page = runtime_tools._get_result_page(
            {"result_set_id": rs.result_set_id, "page": 1, "page_size": 20},
            context={"scope_id": "album", "task_state": {}},
        )
        self.assertEqual(page["page_size"], 6)
        self.assertEqual(len(page["preview"]), 6)

    def test_preview_carries_bounded_observation_detail(self):
        asset = self.store.create_asset(
            "asset-detail", "detail.jpg", "image", "/tmp/detail.jpg",
            metadata={"captured_at": "2026-08-24T10:00:00"}, scope_id="album",
        )
        self.store.add_observation(asset["id"], {
            "caption": "户外聚会现场",
            "activity": "参加仪式",
            "objects": [{"label": "蓝色礼服"}],
            "detail": {"visible_details": [{"text": "两名伴娘穿蓝色长款礼服"}]},
        }, scope_id="album")
        rs = runtime_tools._RUNTIME["result_sets"].new(
            scope_id="album", query="伴娘", asset_ids=[asset["id"]]
        )
        out = runtime_tools._search_from_prior_result_set(rs, "album")
        self.assertIn("蓝色长款礼服", out["preview"][0]["evidence_summary"])
        self.assertNotIn("asset-detail", out["preview"][0]["evidence_summary"])
        self.assertEqual(out["preview"][0]["description_status"], "available")

    def test_preview_marks_missing_description_explicitly(self):
        asset = self.store.create_asset(
            "asset-no-detail", "no-detail.jpg", "image", "/tmp/no-detail.jpg",
            metadata={"captured_at": "2026-08-24T10:00:00"}, scope_id="album",
        )
        rs = runtime_tools._RUNTIME["result_sets"].new(
            scope_id="album", query="照片", asset_ids=[asset["id"]]
        )
        out = runtime_tools._search_from_prior_result_set(rs, "album")
        self.assertEqual(out["preview"][0]["evidence_summary"], "")
        self.assertEqual(out["preview"][0]["description_status"], "missing")

    def test_inspect_requires_current_result_set(self):
        out = runtime_tools._inspect_photo(
            {"asset_handle": "photo_1", "question": "有什么"},
            context={"scope_id": "album", "task_state": {}},
        )
        self.assertIn("no_current_result_set", out.get("blocked", []))

    def test_model_observation_hides_internal_asset_ids(self):
        visible = _model_visible_observation({
            "result_set_id": "rs_demo",
            "total": 20,
            "asset_ids": [f"asset_{i}" for i in range(20)],
            "preview": [{"handle": f"photo_{i}"} for i in range(8)],
        })
        self.assertNotIn("asset_ids", visible)
        self.assertEqual(len(visible["preview"]), 8)
        self.assertEqual(visible["recommended_handle"], "photo_0")

    def test_model_window_exposes_bounded_tail_without_full_descriptions(self):
        preview = [
            {"handle": f"photo_{i}", "evidence_summary": "细节" * 150,
             "people": [{"name": f"person_{n}"} for n in range(6)]}
            for i in range(18)
        ]
        visible = _model_visible_observation({
            "result_set_id": "rs_demo", "preview": preview,
            "recommended_handle": "photo_9",
        })
        self.assertEqual(len(visible["preview"]), 12)
        self.assertEqual(visible["preview"][0]["handle"], "photo_9")
        self.assertIn("photo_10", [item["handle"] for item in visible["preview"]])
        self.assertNotIn("photo_12", [item["handle"] for item in visible["preview"]])
        self.assertTrue(all(len(item["evidence_summary"]) <= 121
                            for item in visible["preview"]))
        self.assertTrue(all(len(item["people"]) <= 2 for item in visible["preview"]))

    def test_slot_score_cliff_keeps_recall_fallback_pool(self):
        candidates = [f"asset_{i}" for i in range(30)]
        scores = {candidate: (100.0 if i < 3 else 10.0 - i * 0.01)
                  for i, candidate in enumerate(candidates)}
        self.assertEqual(runtime_tools._slot_gap_bounded_ids(candidates, scores),
                         candidates[:12])
        self.assertEqual(runtime_tools._slot_gap_bounded_ids(candidates[:5], scores),
                         candidates[:5])
        flat_scores = {candidate: 10.0 for candidate in candidates}
        self.assertEqual(runtime_tools._slot_gap_bounded_ids(candidates, flat_scores),
                         candidates)

    def test_capture_neighbor_rescue_is_local_and_bounded(self):
        assets = {}
        for name, stamp in (
            ("anchor", "2019-06-03 12:00:00"),
            ("near", "2019-06-03 12:01:30"),
            ("far", "2019-06-03 13:00:00"),
        ):
            asset = self.store.create_asset(
                f"asset-{name}", f"{name}.jpg", "image", f"/tmp/{name}.jpg",
                metadata={"captured_at": stamp}, scope_id="album")
            assets[name] = asset["id"]
        outside = self.store.create_asset(
            "asset-outside", "outside.jpg", "image", "/tmp/outside.jpg",
            metadata={"captured_at": "2019-06-03 12:01:00"}, scope_id="other")
        video = self.store.create_asset(
            "asset-video", "near.mp4", "video", "/tmp/near.mp4",
            metadata={"captured_at": "2019-06-03 12:01:00"}, scope_id="album")
        scores = {assets["anchor"]: 1.0}
        added = runtime_tools._expand_capture_neighbors(
            scores, self.store, "album", media_constraint="image")
        self.assertEqual(added, {assets["near"]})
        self.assertAlmostEqual(scores[assets["near"]], 0.65)
        self.assertNotIn(assets["far"], scores)
        self.assertNotIn(outside["id"], scores)
        self.assertNotIn(video["id"], scores)

    def test_reference_keeps_original_visual_intent(self):
        rs = runtime_tools._RUNTIME["result_sets"].new(
            scope_id="album", query="照片", asset_ids=["asset_1"]
        )
        out = runtime_tools._search_from_prior_result_set(
            rs, "album", query="现场人物穿什么", user_goal="现场人物穿什么"
        )
        self.assertEqual(out["query"], "现场人物穿什么")
        self.assertEqual(out["recommended_resolution"]["tool"], "inspect_photo")

    def test_referent_with_new_grounded_place_reopens_full_search(self):
        rs = runtime_tools._RUNTIME["result_sets"].new(
            scope_id="album", query="孩子和花灯", asset_ids=["asset_1"]
        )
        self.assertTrue(runtime_tools._reference_has_new_grounded_anchor(
            rs, "正月初三去正定", "就是正月初三那次", {"place": "正定"}))
        self.assertFalse(runtime_tools._reference_has_new_grounded_anchor(
            rs, "孩子穿什么", "那次孩子穿什么", {}))

    def test_dining_scene_outranks_generic_group_photo(self):
        query = "我带明明和朋友在邯郸吃晚餐拍合影"
        dining = "三人在餐厅内合影，一人抱着孩子；餐饮空间"
        unrelated = "两个孩子在公园石雕前合影"
        weights = runtime_tools._preview_query_term_weights(query, [unrelated, dining])
        self.assertGreater(
            runtime_tools._preview_text_score(query, dining, weights),
            runtime_tools._preview_text_score(query, unrelated, weights))

    def test_inspect_retries_context_overflow_with_smaller_image(self):
        image_path = Path(self.tmp) / "large.jpg"
        Image.new("RGB", (1600, 1200), (100, 120, 140)).save(image_path)
        asset = self.store.create_asset(
            "asset-large", "large.jpg", "image", str(image_path), scope_id="album")
        gamma = GammaClient()
        runtime_tools._RUNTIME["gamma"] = gamma
        rs = runtime_tools._RUNTIME["result_sets"].new(
            scope_id="album", query="照片", asset_ids=[asset["id"]])
        with patch.object(gamma, "chat", side_effect=[
            ValueError("Input length (4326) exceeds model's maximum context length 4096"),
            '{"observation":"三人在餐厅内合影","certainty":"supported"}',
        ]) as chat:
            out = runtime_tools._inspect_photo(
                {"asset_handle": "photo_1", "question": "有谁"},
                context={"scope_id": "album", "task_state": {
                    "current_result_set": rs.result_set_id}})
        self.assertEqual(out["observation"], "三人在餐厅内合影")
        self.assertEqual(chat.call_count, 2)
        self.assertEqual(chat.call_args_list[1].kwargs["vision_options"],
                         {"num_predict": 192})

    def test_video_parent_preview_inspects_keyframe_and_delivers_video(self):
        video_path = Path(self.tmp) / "scene.mp4"
        video_path.write_bytes(b"not an image")
        frame_path = Path(self.tmp) / "frame.jpg"
        Image.new("RGB", (32, 32), (10, 80, 120)).save(frame_path)
        video = self.store.create_asset(
            "asset-video", "scene.mp4", "video", str(video_path), scope_id="album")
        self.store.create_asset(
            "asset-frame", "frame.jpg", "image", str(frame_path),
            metadata={"parent_asset_id": video["id"],
                      "derived_kind": "video_keyframe", "source_timestamp_sec": 4.5},
            scope_id="album")
        rs = runtime_tools._RUNTIME["result_sets"].new(
            scope_id="album", query="视频里的水灯", asset_ids=[video["id"]])
        preview = runtime_tools._search_from_prior_result_set(rs, "album")["preview"][0]
        self.assertEqual(preview["media_kind"], "video")
        self.assertEqual(preview["source_video_asset_id"], video["id"])
        gamma = GammaClient()
        runtime_tools._RUNTIME["gamma"] = gamma
        with patch.object(gamma, "chat", return_value=(
                '{"observation":"画面里有人展示水灯","certainty":"supported"}')):
            inspected = runtime_tools._inspect_photo(
                {"asset_handle": "photo_1", "question": "视频里做什么"},
                context={"scope_id": "album", "task_state": {
                    "current_result_set": rs.result_set_id}})
        self.assertEqual(inspected["certainty"], "supported")
        self.assertEqual(inspected["_source_asset_id"], video["id"])
        self.assertEqual(inspected["inspected_keyframe_asset_id"], "asset-frame")
        delivered = runtime_tools._get_original_photos(
            {"handle": "photo_1"}, context={"scope_id": "album", "task_state": {
                "current_result_set": rs.result_set_id}})
        self.assertEqual(delivered["media_type"], "video")
        self.assertEqual(delivered["url"], f"/api/assets/{video['id']}/file")

    def test_video_parent_without_frame_does_not_send_mp4_as_image(self):
        video_path = Path(self.tmp) / "empty.mp4"
        video_path.write_bytes(b"not an image")
        video = self.store.create_asset(
            "asset-empty-video", "empty.mp4", "video", str(video_path), scope_id="album")
        rs = runtime_tools._RUNTIME["result_sets"].new(
            scope_id="album", query="视频", asset_ids=[video["id"]])
        gamma = GammaClient()
        runtime_tools._RUNTIME["gamma"] = gamma
        with patch.object(gamma, "chat") as chat:
            out = runtime_tools._inspect_photo(
                {"asset_handle": "photo_1", "question": "有什么"},
                context={"scope_id": "album", "task_state": {
                    "current_result_set": rs.result_set_id}})
        self.assertIn("video_keyframe_unavailable", out["blocked"])
        chat.assert_not_called()

    def test_debug_projection_separates_full_candidates_from_preview(self):
        rs = runtime_tools._RUNTIME["result_sets"].new(
            scope_id="album", query="照片", asset_ids=[f"asset_{i}" for i in range(20)]
        )
        projection = debug_asset_projection(rs, [
            {"handle": "photo_7"}, {"handle": "photo_2"}, {"handle": "photo_7"},
        ])
        self.assertEqual(projection["debug_result_total"], 20)
        self.assertEqual(projection["debug_asset_ids"], [f"asset_{i}" for i in range(20)])
        self.assertEqual(projection["debug_preview_asset_ids"], ["asset_6", "asset_1"])
        self.assertEqual(projection["debug_preview_handles"], ["photo_7", "photo_2"])

    def test_preview_preserves_relevance_head_before_event_diversity(self):
        asset_ids = [f"asset_{i}" for i in range(10)]
        def groups(_store, asset_id):
            return "same-event" if asset_id in {"asset_0", "asset_1", "asset_2"} else asset_id
        with patch.object(runtime_tools, "_asset_group_key", side_effect=groups):
            indices = runtime_tools._preview_indices(asset_ids, "best", None)
        self.assertEqual(indices[:3], [0, 1, 2])
        self.assertEqual(len(indices), 6)

    def test_candidate_window_strategy_can_run_head_only_or_diversity_only(self):
        asset_ids = [f"asset_{i}" for i in range(12)]
        def groups(_store, asset_id):
            return "same-event" if asset_id in {"asset_0", "asset_1", "asset_2"} else asset_id
        with patch.object(runtime_tools, "_asset_group_key", side_effect=groups):
            with patch.dict("os.environ", {"SENTRIX_CANDIDATE_STRATEGY": "head_only"}):
                head = runtime_tools._preview_indices(asset_ids, "best", None, query="")
            with patch.dict("os.environ", {"SENTRIX_CANDIDATE_STRATEGY": "event_diversity"}):
                diverse = runtime_tools._preview_indices(asset_ids, "best", None, query="")
        self.assertEqual(head, list(range(6)))
        self.assertEqual(diverse[:4], [0, 3, 4, 5])

    def test_candidate_window_explains_bounded_preview_without_asset_ids(self):
        summary = runtime_tools._candidate_window_summary(
            ["a1", "a2", "a3"], [0, 2], None)
        self.assertEqual(summary["total_candidates"], 3)
        self.assertEqual(summary["visible_ranks"], [1, 3])
        self.assertNotIn("asset_ids", summary)

    def test_legacy_visual_arguments_are_bound_to_current_preview(self):
        normalized, requested = _normalize_preview_handle(
            {"image_id": "rs_private", "query": "照片里有几个人"},
            ["photo_3", "photo_4"],
        )
        self.assertEqual(normalized["asset_handle"], "photo_3")
        self.assertEqual(normalized["question"], "照片里有几个人")
        self.assertEqual(requested, "rs_private")

    def test_visual_intent_covers_decoration_and_display_board_questions(self):
        self.assertTrue(visual_intent("房间做了哪些主要的装饰布置"))
        self.assertTrue(visual_intent("展示牌上写了什么文字"))

    def test_preview_query_order_promotes_visual_detail_matches(self):
        asset_ids = ["noise_1", "answer", "noise_2"]
        summaries = {
            "noise_1": "户外现场；舞台；装饰灯光",
            "answer": "户外站立；展示牌；文字：这里的幸福",
            "noise_2": "户外现场；宾客",
        }
        with patch.object(runtime_tools, "_observation_summary",
                          side_effect=lambda _store, aid: summaries[aid]):
            order = runtime_tools._preview_query_order(
                asset_ids, "展示牌上写了什么文字", None)
        self.assertEqual(order[0], 1)

    def test_preview_query_order_uses_trusted_place_and_capture_time(self):
        class PreviewStore:
            def get_asset(self, asset_id):
                return {"captured_at": {
                    "wrong": "2018-04-01 14:00:00",
                    "answer": "2017-10-04 22:31:47",
                }[asset_id]}

            def list_observations(self, asset_id, limit=1):
                return [{"captured_at": self.get_asset(asset_id)["captured_at"]}]

        summaries = {
            "wrong": "婚礼现场；舞台灯光；宾客合影",
            "answer": "男子站立拍照；室内装饰布幔",
        }
        with patch.object(runtime_tools, "_observation_summary",
                          side_effect=lambda _store, aid: summaries[aid]), \
             patch.object(runtime_tools, "_place_matches",
                          side_effect=lambda item, place, _store:
                          item["asset_id"] == "answer" and place == "保定市"):
            order = runtime_tools._preview_query_order(
                ["wrong", "answer"],
                "2017年10月4日晚上在保定市婚礼舞台前拍照",
                PreviewStore(),
                trusted_constraints={"place": "保定市", "time": "2017年10月4日"},
            )
        self.assertEqual(order[0], 1)

    def test_query_order_applies_even_when_result_set_fits_preview(self):
        asset_ids = ["noise", "answer"]
        summaries = {"noise": "室内场景；文字：you", "answer": "装饰；文字：一起幸福"}
        with patch.object(runtime_tools, "_observation_summary",
                          side_effect=lambda _store, aid: summaries[aid]):
            order = runtime_tools._preview_indices(
                asset_ids, "best", None, query="装饰文字")
        self.assertEqual(order[0], 1)

    def test_inspection_handle_is_confined_to_visible_preview(self):
        arguments, requested = _normalize_preview_handle(
            {"asset_handle": "photo_1", "question": "文字"},
            ["photo_7", "photo_14"],
        )
        self.assertEqual(requested, "photo_1")
        self.assertEqual(arguments["asset_handle"], "photo_1")

    def test_model_time_filter_must_match_user_wording(self):
        sanitized = runtime_tools._sanitize_model_filters(
            {"time": "2026年8月24日", "place": "易县"},
            query="易县活动照片", user_goal="帮我找2017年的易县活动照片",
        )
        self.assertEqual(sanitized["time"], "2017年")
        no_time = runtime_tools._sanitize_model_filters(
            {"time": "2026年8月24日", "place": "易县"},
            query="易县活动照片", user_goal="帮我找易县活动照片",
        )
        self.assertNotIn("time", no_time)

    def test_model_scene_place_and_collective_person_do_not_become_hard_filters(self):
        sanitized = runtime_tools._sanitize_model_filters(
            {"place": "迎宾展架", "person": "我和同事", "time": "去年"},
            query="婚礼现场照片", user_goal="帮我找婚礼现场的照片",
        )
        self.assertNotIn("place", sanitized)
        self.assertNotIn("person", sanitized)
        self.assertNotIn("time", sanitized)

    def test_short_year_is_canonicalized_without_using_current_year(self):
        from backend.agent_runtime.canonical_intent import extract_time
        self.assertEqual(extract_time("我记得17年国庆拍的照片"), "2017年")

    def test_pending_resolution_reads_flattened_tool_recommendation(self):
        task = type("Task", (), {"tool_results": [{
            "tool": "search_memories",
            "recommended_resolution": {"needed": True, "tool": "read_photo_text"},
        }]})()
        self.assertEqual(_pending_resolution(task)["tool"], "read_photo_text")

    def test_pending_resolution_stays_open_after_failed_resolution_tool(self):
        task = type("Task", (), {"tool_results": [
            {"tool": "search_memories", "recommended_resolution": {
                "needed": True, "tool": "read_photo_text",
            }},
            {"tool": "read_photo_text", "ocr_text": "", "status": "partial",
             "reason": "ocr_failed"},
        ]})()
        self.assertEqual(_pending_resolution(task)["tool"], "read_photo_text")

    def test_pending_resolution_closes_after_usable_resolution_tool(self):
        task = type("Task", (), {"tool_results": [
            {"tool": "search_memories", "recommended_resolution": {
                "needed": True, "tool": "read_photo_text",
            }},
            {"tool": "read_photo_text", "ocr_text": "一起幸福", "status": "ok"},
        ]})()
        self.assertIsNone(_pending_resolution(task))

    def test_empty_search_does_not_overwrite_usable_result_set(self):
        state = ResultTaskState(current_result_set="rs_good")
        state.update_from_tool(
            "search_memories", {"mode": "best"},
            {"result_set_id": "rs_empty", "total": 0, "preview": [],
             "query_satisfaction": "no_match"},
        )
        self.assertEqual(state.current_result_set, "rs_good")

    def test_first_empty_search_keeps_empty_result_set_for_followup_state(self):
        state = ResultTaskState()
        state.update_from_tool(
            "search_memories", {"mode": "best"},
            {"result_set_id": "rs_empty", "total": 0, "preview": [],
             "query_satisfaction": "no_match"},
        )
        self.assertEqual(state.current_result_set, "rs_empty")

    def test_next_resolution_handle_skips_inspected_visual_candidate(self):
        task = type("Task", (), {
            "result_preview": ["photo_1", "photo_2"],
            "tool_results": [{"tool": "inspect_photo", "inspect_handle": "photo_1"}],
        })()
        self.assertEqual(_next_resolution_handle(task, "inspect_photo"), "photo_2")

    def test_completion_search_recommendation_reads_top_level_and_legacy_nested(self):
        top_level = [{
            "tool": "search_memories",
            "recommended_resolution": {"needed": True, "tool": "read_photo_text"},
        }]
        nested = [{
            "tool": "search_memories",
            "observation": {
                "recommended_resolution": {"needed": True, "tool": "read_photo_text"},
            },
        }]
        self.assertTrue(CompletionState._search_recommends(top_level, "read_photo_text"))
        self.assertTrue(CompletionState._search_recommends(nested, "read_photo_text"))

    def test_completion_uses_agent2_visual_requirement_before_regex(self):
        message = "这幅画面中呈现了哪些可辨识的物件"
        self.assertFalse(visual_intent(message))
        agent2 = TaskState.from_declaration(TaskDeclaration(
            goal=message,
            scope_id="album",
            requirements=(EvidenceRequirement(
                id="visual", evidence_type="visual_observation", description="画面内容",
            ),),
        ))
        completion = CompletionState(message)
        completion.update({"tool_results": [{
            "tool": "search_memories", "preview": [{"handle": "photo_1"}],
        }]}, agent2_task_state=agent2)
        self.assertIn(RESOLVE_VISUAL, [requirement.code for requirement in completion.blocking()])

    def test_completion_preserves_regex_visual_fallback_without_agent2_state(self):
        message = "这幅画面中有几个人"
        self.assertTrue(visual_intent(message))
        completion = CompletionState(message)
        completion.update({"tool_results": [{
            "tool": "search_memories", "preview": [{"handle": "photo_1"}],
        }]}, agent2_task_state=None)
        self.assertIn(RESOLVE_VISUAL, [requirement.code for requirement in completion.blocking()])

    def test_completion_keeps_ocr_blocked_after_partial_tool_result(self):
        message = "照片里的展架上写了什么"
        completion = CompletionState(message)
        completion.update({"tool_results": [
            {"tool": "search_memories", "preview": [{"handle": "photo_1"}]},
            {"tool": "read_photo_text", "ocr_text": "", "exact_values": [],
             "blocked": [], "recommended_resolution": None},
        ]})
        self.assertIn(RESOLVE_OCR, [r.code for r in completion.blocking()])

    def test_completion_satisfies_ocr_only_when_text_was_returned(self):
        message = "照片里的展架上写了什么"
        completion = CompletionState(message)
        completion.update({"tool_results": [
            {"tool": "search_memories", "preview": [{"handle": "photo_1"}]},
            {"tool": "read_photo_text", "ocr_text": "一起幸福",
             "exact_values": [], "blocked": []},
        ]})
        self.assertNotIn(RESOLVE_OCR, [r.code for r in completion.blocking()])

    def test_completion_keeps_visual_blocked_after_empty_inspection(self):
        message = "这张照片里有几个人"
        completion = CompletionState(message)
        completion.update({"tool_results": [
            {"tool": "search_memories", "preview": [{"handle": "photo_1"}]},
            {"tool": "inspect_photo", "inspect_text": "", "blocked": []},
        ]})
        self.assertIn(RESOLVE_VISUAL, [r.code for r in completion.blocking()])

    def test_completion_satisfies_visual_only_with_inspection_observation(self):
        message = "这张照片里有几个人"
        completion = CompletionState(message)
        completion.update({"tool_results": [
            {"tool": "search_memories", "preview": [{"handle": "photo_1"}]},
            {"tool": "inspect_photo", "inspect_text": "画面中有三个人",
             "blocked": []},
        ]})
        self.assertNotIn(RESOLVE_VISUAL, [r.code for r in completion.blocking()])


if __name__ == "__main__":
    unittest.main()
