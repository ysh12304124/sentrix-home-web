import unittest
from unittest.mock import patch

from backend.agent_runtime import tools


class EventContractTests(unittest.TestCase):
    """event 自 query_memory_metadata 并入 query_memory_facts 后的契约。

    事件记录查询返回结构化清单与 source_asset_ids（供 grounding），
    但不携带 evidence_asset_ids——聚合工具绝不注入模型可见召回图片。
    """

    def test_event_operation_returns_structured_list_without_evidence(self):
        with patch.object(tools, "_query_event_evidence", return_value={
            "tool": "query_memory_facts", "metadata_operation": "event",
            "operation": "event", "answer_type": "event_list",
            "value": [{"event_id": "e1", "title": "婚礼"}],
            "items": [{"event_id": "e1", "title": "婚礼", "source_asset_ids": ["asset_1"]}],
            "total": 1, "source_asset_ids": ["asset_1"],
            "evidence_kind": "structured_event", "coverage": {"complete": True},
        }) as event_fn:
            result = tools._query_memory_facts({"operation": "event", "query": "婚礼"}, context={})
        event_fn.assert_called_once()
        self.assertEqual(result["operation"], "event")
        self.assertEqual(result["total"], 1)
        self.assertIn("source_asset_ids", result)
        # 不注入可见证据：聚合工具绝不能携带 evidence_asset_ids
        self.assertNotIn("evidence_asset_ids", result)

    def test_facts_operation_set_includes_event(self):
        self.assertIn("event", tools._FACT_OPERATIONS)


if __name__ == "__main__":
    unittest.main()
