import unittest

from backend.agent_runtime.tools import _parse_search_validation_response


class SearchValidationContractTests(unittest.TestCase):
    def test_only_supported_rows_become_evidence_candidates(self):
        rows = _parse_search_validation_response({"candidates": [
            {"handle": "photo_1", "support_status": "supported", "time_match": True},
            {"handle": "photo_2", "support_status": "candidate_only"},
            {"handle": "photo_3", "support_status": "rejected"},
            {"handle": "photo_4", "support_status": "garbage"},
        ]})
        self.assertEqual([row["handle"] for row in rows], ["photo_1", "photo_2", "photo_3"])

    def test_malformed_model_output_is_not_promoted(self):
        self.assertEqual(_parse_search_validation_response("not json"), [])


if __name__ == "__main__":
    unittest.main()
