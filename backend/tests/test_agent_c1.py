import unittest

from scripts.benchmarks.evaluate_agent_c1 import CASES, _claim_summary


class AgentC1ReplayTests(unittest.TestCase):
    def test_c1_case_set_covers_required_real_memory_boundaries(self):
        case_ids = {item["id"] for item in CASES}
        self.assertTrue({
            "person_introduction", "clothing", "personality_boundary",
            "relationship_boundary", "preference_boundary", "follow_up",
            "original_evidence", "role_ambiguity", "no_evidence", "scope_switch",
        }.issubset(case_ids))

    def test_claim_summary_counts_unsupported_claims(self):
        summary = _claim_summary({
            "claims": [{"claim_id": "claim_1"}],
            "claim_verifications": [{"claim_id": "claim_1", "status": "unsupported"}],
            "claim_verification_status": "blocked",
            "repair_count": 1,
            "evidence_bundles": [{}],
            "claim_evidence_index": {"claim_1": {}},
        })
        self.assertEqual(summary["unsupported_count"], 1)
        self.assertEqual(summary["repair_count"], 1)
        self.assertEqual(summary["evidence_bundle_count"], 1)


if __name__ == "__main__":
    unittest.main()
