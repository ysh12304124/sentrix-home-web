"""RX-2 (D6): person summary with zero evidence must not emit family claims."""

import unittest

from backend.evidence_retrieval import EvidencePacket
from backend.query_contracts import Constraint, HARD, QuerySpec
from backend.thin_agent import ThinAgentRuntime


def _spec():
    return QuerySpec("q", "single", ["home"], "owner", "c", "answer", "person",
                     constraints=[Constraint("person", "明哥", HARD, "confirmed_bridge")])


def _asset(asset_id="asset-1"):
    return {"asset_id": asset_id, "file_name": "x.jpg", "media_type": "image",
            "observation_ids": ["obs-1"], "evidence_ids": [asset_id],
            "condition_results": {"person:明哥": {"status": "matched"}},
            "level": "exact", "observation_fields": {"place": "海边", "activity": "爬山"}}


class PersonGapTests(unittest.TestCase):
    def setUp(self):
        self.runtime = object.__new__(ThinAgentRuntime)

    def test_no_evidence_returns_gap_without_claims(self):
        packet = EvidencePacket("q", "home", "person")
        answer, statements = self.runtime._person_summary(_spec(), packet)
        self.assertIn("还没有足够", answer)
        self.assertNotIn("多次出现", answer)
        self.assertEqual(statements, [])

    def test_with_evidence_mentions_appearance_without_overstatement(self):
        packet = EvidencePacket("q", "home", "person",
                                assets=[_asset()], exact_results=[_asset()])
        answer, statements = self.runtime._person_summary(_spec(), packet)
        self.assertIn("出现在这些记录中", answer)
        self.assertNotIn("多次出现", answer)
        self.assertTrue(statements)
        self.assertIn("海边", answer)

    def test_evidence_zero_via_evidence_path_keeps_claims_empty(self):
        # Guard for the full path: person + confirmed entity but zero assets must
        # never produce a family-fact statement (the gap answer has none).
        packet = EvidencePacket("q", "home", "person", gaps=[
            {"condition": "confirmed_person", "reason": "没有找到当前范围内已确认的人物"}])
        answer, statements = self.runtime._person_summary(_spec(), packet)
        self.assertEqual(statements, [])
        self.assertNotIn("多次出现", answer)


if __name__ == "__main__":
    unittest.main()
