"""Regression tests for bounded, ResultSet-scoped evidence delivery."""

import os
import tempfile
import unittest
from unittest.mock import patch

from backend.db import MemoryStore
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
            "caption": "婚礼现场",
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
        self.assertEqual(len(visible["preview"]), 5)
        self.assertEqual(visible["recommended_handle"], "photo_0")

    def test_reference_keeps_original_visual_intent(self):
        rs = runtime_tools._RUNTIME["result_sets"].new(
            scope_id="album", query="照片", asset_ids=["asset_1"]
        )
        out = runtime_tools._search_from_prior_result_set(
            rs, "album", query="婚礼伴娘穿什么", user_goal="婚礼伴娘穿什么"
        )
        self.assertEqual(out["query"], "婚礼伴娘穿什么")
        self.assertEqual(out["recommended_resolution"]["tool"], "inspect_photo")

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
        self.assertTrue(visual_intent("婚房做了哪些主要的婚庆布置"))
        self.assertTrue(visual_intent("迎宾展架上写了什么祝福文字"))

    def test_preview_query_order_promotes_visual_detail_matches(self):
        asset_ids = ["noise_1", "answer", "noise_2"]
        summaries = {
            "noise_1": "婚礼现场；舞台；装饰灯光",
            "answer": "户外站立；广告牌；文字：结婚这里的幸福",
            "noise_2": "婚礼现场；宾客",
        }
        with patch.object(runtime_tools, "_observation_summary",
                          side_effect=lambda _store, aid: summaries[aid]):
            order = runtime_tools._preview_query_order(
                asset_ids, "迎宾展架上写了什么祝福文字", None)
        self.assertEqual(order[0], 1)

    def test_query_order_applies_even_when_result_set_fits_preview(self):
        asset_ids = ["noise", "answer"]
        summaries = {"noise": "婚礼现场；文字：you", "answer": "广告牌；文字：一起幸福"}
        with patch.object(runtime_tools, "_observation_summary",
                          side_effect=lambda _store, aid: summaries[aid]):
            order = runtime_tools._preview_indices(
                asset_ids, "best", None, query="迎宾展架祝福文字")
        self.assertEqual(order[0], 1)

    def test_inspection_handle_is_confined_to_visible_preview(self):
        arguments, requested = _normalize_preview_handle(
            {"asset_handle": "photo_1", "question": "文字"},
            ["photo_7", "photo_14"],
        )
        self.assertEqual(requested, "photo_1")
        # Preserve an explicit stale handle so execution can reject it and
        # force a fresh search instead of silently inspecting another photo.
        self.assertEqual(arguments["asset_handle"], "photo_1")

    def test_model_time_filter_must_match_user_wording(self):
        sanitized = runtime_tools._sanitize_model_filters(
            {"time": "2026年8月24日", "place": "易县"},
            query="易县婚礼照片", user_goal="帮我找2017年的易县婚礼照片",
        )
        self.assertEqual(sanitized["time"], "2017年")
        no_time = runtime_tools._sanitize_model_filters(
            {"time": "2026年8月24日", "place": "易县"},
            query="易县婚礼照片", user_goal="帮我找易县婚礼照片",
        )
        self.assertNotIn("time", no_time)

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
