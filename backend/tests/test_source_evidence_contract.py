import unittest

from backend.agent_runtime.result_set import TaskState
from backend.agent_runtime.runtime import _build_answer_grounding
from services.photobench.backend.benchmark_orchestrator import _extract_image_sets


class SourceEvidenceContractTests(unittest.TestCase):
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
        grounding = _build_answer_grounding(
            message="这张图在哪里？", task=task,
            selected_image_handles=["photo_1"], selected_image_ids=["asset_1"],
        )
        # search 候选与 facts 聚合来源都不注入可见证据；只有显式单张操作（inspect）产生证据。
        self.assertEqual(grounding["retrieved_asset_ids"], ["asset_1", "asset_2", "asset_3"])
        self.assertEqual(grounding["evidence_asset_ids"], ["asset_1"])
        # 代码候选补齐（保 GT 不丢）会把 search 候选补进 selected 至 3 张，属既有行为。
        self.assertEqual(grounding["selected_asset_ids"], ["asset_1", "asset_2", "asset_3"])
        self.assertEqual(grounding["selected_image_handles"], [])

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
