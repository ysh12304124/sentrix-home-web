"""Phase R R6 — answer dedup, human-readable templates, empty refusal."""

import unittest

from backend.answer_composer import compose_answer
from backend.thin_agent import ThinAgentRuntime


def _packet(assets, gaps=None):
    from backend.evidence_retrieval import EvidencePacket
    packet = EvidencePacket("q", "home", "general", gaps=gaps or [])
    packet.assets = assets
    return packet


class AllowedFactsDedupTests(unittest.TestCase):
    def test_same_condition_deduped_across_assets(self):
        packet = _packet([
            {"asset_id": "a1", "evidence_ids": ["a1", "obs1"],
             "condition_results": {"clothing:毛绒睡衣": {"status": "matched"}}},
            {"asset_id": "a2", "evidence_ids": ["a2", "obs2"],
             "condition_results": {"clothing:毛绒睡衣": {"status": "matched"}}},
        ])
        allowed = ThinAgentRuntime._allowed_facts(packet)
        matches = [item for item in allowed["allowed_answer_facts"]
                   if item["condition_key"] == "clothing:毛绒睡衣"]
        self.assertEqual(len(matches), 1)
        self.assertEqual(sorted(matches[0]["evidence_ids"]), ["a1", "a2", "obs1", "obs2"])

    def test_condition_text_is_human_readable(self):
        text = ThinAgentRuntime._human_condition_text("clothing:毛绒睡衣", "matched")
        self.assertIn("毛绒睡衣", text)
        self.assertNotIn("clothing:", text)      # internal key must not leak
        self.assertNotIn(":", text.split("「")[0])

    def test_possible_text_marks_uncertainty(self):
        text = ThinAgentRuntime._human_condition_text("place:厨房", "possible")
        self.assertIn("无法完全确认", text)

    def test_compose_answer_no_repeat_after_dedup(self):
        allowed = {
            "allowed_answer_facts": [
                {"text": "记录中有「毛绒睡衣」", "status": "matched",
                 "condition_key": "clothing:毛绒睡衣", "evidence_ids": ["a1", "a2"]},
            ],
            "allowed_possibilities": [],
            "required_unknowns": [],
        }
        composed = compose_answer({"answer": "找到结果", "statements": [
            {"text": "记录中有「毛绒睡衣」", "status": "matched", "evidence_ids": ["a1", "a2"],
             "condition_keys": ["clothing:毛绒睡衣"]}]}, allowed)
        self.assertTrue(composed["valid"])
        self.assertEqual(composed["answer"], "找到结果")


class EmptyPacketRefusalTests(unittest.TestCase):
    def test_human_text_no_internal_keys(self):
        packet = _packet([
            {"asset_id": "a1", "evidence_ids": ["a1"],
             "condition_results": {"visual:自拍": {"status": "possible"}}},
        ])
        allowed = ThinAgentRuntime._allowed_facts(packet)
        all_texts = " ".join(item["text"] for item in allowed["allowed_answer_facts"]
                             + allowed["allowed_possibilities"] + allowed["required_unknowns"])
        self.assertNotIn("visual:", all_texts)
        self.assertNotIn("condition_results", all_texts)


if __name__ == "__main__":
    unittest.main()
