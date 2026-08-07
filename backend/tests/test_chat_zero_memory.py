"""RX-5 chat zero-memory: a normal chat never reads family evidence."""

import unittest

from backend.query_contracts import QueryParseDraft
from backend.router import GateDecision
from backend.thin_agent import ThinAgentRuntime


class _Gamma:
    def chat(self, prompt, json_mode=True, role=None):
        return "我在听，今天感觉还不错。"


class ChatZeroMemoryTests(unittest.TestCase):
    def setUp(self):
        self.runtime = object.__new__(ThinAgentRuntime)
        self.runtime.gamma = _Gamma()
        self.runtime.router = None
        self.decision = GateDecision("none", "explicit_no_memory_lookup")

    def test_zero_evidence_zero_images_no_entry(self):
        result = self.runtime._normal_chat(
            "今天感觉怎么样", "", "conversation_1", "album2_e2b", "owner",
            self.decision, QueryParseDraft())
        self.assertIn("感觉还不错", result["answer"])
        self.assertEqual(result["evidence"], [])
        self.assertEqual(result["image_results"], [])
        self.assertEqual(result["evidence_status"], "not_applicable")
        self.assertFalse(result["memory_used"])
        self.assertFalse(result["evidence_required"])
        self.assertFalse(result["insufficient_evidence"] is True)  # chat: no gap badge
        self.assertEqual(result["evidence_layers"]["observations"], [])

    def test_no_evidence_language(self):
        result = self.runtime._normal_chat(
            "今天感觉怎么样", "", "conversation_1", "album2_e2b", "owner",
            self.decision, QueryParseDraft())
        self.assertNotIn("匹配", result["answer"])
        self.assertNotIn("证据", result["answer"])
        self.assertNotIn("数据库", result["answer"])


if __name__ == "__main__":
    unittest.main()
