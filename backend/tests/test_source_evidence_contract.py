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
        }]
        grounding = _build_answer_grounding(
            message="这张图在哪里？", task=task,
            selected_image_handles=["photo_1"], selected_image_ids=["asset_1"],
        )
        self.assertEqual(grounding["retrieved_asset_ids"], ["asset_1", "asset_2", "asset_3"])
        # A search preview is a candidate only; only the metadata source is evidence.
        self.assertEqual(grounding["evidence_asset_ids"], ["asset_3"])
        self.assertEqual(grounding["selected_asset_ids"], ["asset_3"])
        self.assertEqual(grounding["selected_delivery"], ["asset_3"])
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
