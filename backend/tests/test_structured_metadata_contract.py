import unittest
from unittest.mock import patch

from backend.agent_runtime import tools


class StructuredMetadataContractTests(unittest.TestCase):
    def test_metadata_operation_is_separate_and_carries_sources(self):
        with patch.object(tools, "_query_memory_facts", return_value={
            "operation": "date", "value": "2017-11-05",
            "samples": [{"asset_id": "asset_1"}], "source_asset_ids": ["asset_1"],
        }) as facts:
            result = tools._query_memory_metadata({"operation": "date"}, context={})
        facts.assert_called_once()
        self.assertEqual(result["tool"], "query_memory_metadata")
        self.assertEqual(result["evidence_asset_ids"], ["asset_1"])


if __name__ == "__main__":
    unittest.main()
