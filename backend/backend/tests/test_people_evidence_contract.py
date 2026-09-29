import unittest
from unittest.mock import patch

from backend.agent_runtime import tools


class PeopleEvidenceContractTests(unittest.TestCase):
    def test_people_tool_binds_identity_to_requested_asset(self):
        class ResultSets:
            def resolve_handle(self, result_set_id, handle):
                return "asset_097" if result_set_id == "rs_1" and handle == "photo_1" else None
        class Store:
            def get_asset(self, asset_id):
                return {"id": asset_id, "scope_id": "scope_1"}
            class Conn:
                class Result:
                    def fetchone(self):
                        return None
                def execute(self, *_args):
                    return self.Result()
            connection = Conn()
        tools._RUNTIME["result_sets"] = ResultSets()
        tools._RUNTIME["store"] = Store()
        with patch.object(tools, "_confirmed_photo_identities", return_value=[
            {"person_name": "明明", "family_role": ""},
        ]):
            result = tools._query_photo_people(
                {"result_set_id": "rs_1", "asset_handle": "photo_1"},
                context={"scope_id": "scope_1", "task_state": {}},
            )
        self.assertEqual(result["evidence_asset_ids"], ["asset_097"])
        self.assertEqual(result["people"][0]["asset_id"], "asset_097")


if __name__ == "__main__":
    unittest.main()
