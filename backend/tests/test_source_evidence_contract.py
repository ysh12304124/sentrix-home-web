import unittest
from unittest.mock import patch

from backend.agent_runtime import tools as runtime_tools
from backend.agent_runtime.result_set import TaskState
from backend.agent_runtime.runtime import _build_answer_grounding
from services.photobench.backend.benchmark_orchestrator import _extract_image_sets


class SourceEvidenceContractTests(unittest.TestCase):
    def test_inspected_evidence_is_delivered_when_model_omits_image_handles(self):
        class FakeResultSet:
            scope_id = "scope_1"

            def visible_asset_ids(self):
                return ["asset_1", "asset_2"]

        class FakeResultSetStore:
            def get(self, result_set_id):
                return FakeResultSet() if result_set_id == "rs_1" else None

        task = TaskState(user_goal="找照片", current_result_set="rs_1")
        task.tool_results = [{
            "tool": "search_memories",
            "retrieved_asset_ids": ["asset_1", "asset_2"],
            "preview": [
                {"handle": "photo_1", "asset_id": "asset_1"},
                {"handle": "photo_2", "asset_id": "asset_2"},
            ],
        }, {
            "tool": "inspect_photo", "asset_id": "asset_2",
            "inspect_handle": "photo_2", "inspect_text": "与描述相符",
        }]
        with patch.dict(runtime_tools._RUNTIME, {"result_sets": FakeResultSetStore()}):
            grounding = _build_answer_grounding(
                message="帮我找一下这张照片", task=task,
            )

        self.assertEqual(grounding["display_mode"], "result_grid")
        self.assertEqual(grounding["selected_image_handles"], ["photo_2"])
        self.assertEqual(grounding["selected_asset_ids"], ["asset_2"])

    def test_uninspected_candidates_are_not_promoted_to_delivery(self):
        class FakeResultSet:
            scope_id = "scope_1"

            def visible_asset_ids(self):
                return ["asset_1", "asset_2"]

        class FakeResultSetStore:
            def get(self, result_set_id):
                return FakeResultSet() if result_set_id == "rs_1" else None

        task = TaskState(user_goal="找照片", current_result_set="rs_1")
        task.tool_results = [{
            "tool": "search_memories",
            "retrieved_asset_ids": ["asset_1", "asset_2"],
            "preview": [
                {"handle": "photo_1", "asset_id": "asset_1"},
                {"handle": "photo_2", "asset_id": "asset_2"},
            ],
        }]
        with patch.dict(runtime_tools._RUNTIME, {"result_sets": FakeResultSetStore()}):
            grounding = _build_answer_grounding(
                message="帮我找一下这张照片", task=task,
            )

        self.assertEqual(grounding["selected_image_handles"], [])
        self.assertEqual(grounding["selected_asset_ids"], [])

    def test_answer_grounding_keeps_retrieved_evidence_and_selected_sets_separate(self):
        task = TaskState(user_goal="看照片来源")
        task.current_result_set = "rs_1"
        task.result_preview = ["photo_1", "photo_2"]
        task.tool_results = [{
            "tool": "search_memories",
            "asset_ids": ["asset_1", "asset_2", "asset_3"],
            "preview": [
                {"handle": "photo_1", "asset_id": "asset_1", "place": "甲"},
                {"handle": "photo_2", "asset_id": "asset_2", "place": "乙"},
            ],
        }, {
            "tool": "query_memory_facts",
            "items": [{"asset_id": "asset_3", "captured_at": "2017-01-01"}],
        }, {
            "tool": "inspect_photo",
            "asset_id": "asset_1", "inspect_handle": "photo_1", "inspect_text": "两个人",
        }]
        class FakeResultSet:
            scope_id = "scope_1"

            def visible_asset_ids(self):
                return ["asset_1", "asset_2"]

        class FakeResultSetStore:
            def get(self, result_set_id):
                return FakeResultSet() if result_set_id == "rs_1" else None

        with patch.dict(runtime_tools._RUNTIME, {"result_sets": FakeResultSetStore()}):
            grounding = _build_answer_grounding(
                message="这张图在哪里？", task=task,
                selected_image_handles=["photo_1"], selected_image_ids=["asset_1"],
            )
        # search 候选与 facts 聚合来源都不注入可见证据；只有显式单张操作（inspect）产生证据。
        self.assertEqual(grounding["retrieved_asset_ids"], ["asset_1", "asset_2", "asset_3"])
        self.assertEqual(grounding["evidence_asset_ids"], ["asset_1"])
        # 交付仅保留模型显式选择的当前预览图片，不升级未检查候选。
        self.assertEqual(grounding["selected_asset_ids"], ["asset_1"])
        self.assertEqual(grounding["selected_image_handles"], ["photo_1"])

    def test_benchmark_extracts_candidates_without_promoting_them_to_delivery(self):
        result = {
            "tool_trace": [{
                "debug_asset_ids": ["asset_1", "asset_2"],
                "debug_preview_asset_ids": ["asset_1"],
                "debug_preview_handles": ["photo_1"],
            }],
            "answer_grounding": {
                "evidence_asset_ids": ["asset_1"],
                "selected_asset_ids": [],
            },
        }
        sets = _extract_image_sets(result)
        self.assertEqual(sets["retrieved_asset_ids"], ["asset_1", "asset_2"])
        self.assertEqual(sets["evidence_asset_ids"], ["asset_1"])
        self.assertEqual(sets["selected_asset_ids"], [])


if __name__ == "__main__":
    unittest.main()
