"""Phase 12B-FC V2 — validation profile flags + response validation block."""

import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from backend.db import MemoryStore
from backend.thin_agent import ThinAgentRuntime
from backend.validation import full_chain_profile as prof


class _FakeGamma:
    def __init__(self):
        self.calls = []

    def chat(self, prompt, json_mode=False, role=None, **kwargs):
        self.calls.append((role, prompt))
        return '{"mode": "none"}' if json_mode else "我在听。"


class ValidationProfileTests(unittest.TestCase):
    def test_off_by_default(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            self.assertFalse(prof.validation_active())
            self.assertFalse(prof.no_fallback())

    def test_master_switch_required(self):
        with mock.patch.dict(os.environ, {"SENTRIX_AGENT_NO_FALLBACK": "1"}, clear=False):
            # master off -> no_fallback off even if flag set
            self.assertFalse(prof.no_fallback())
        with mock.patch.dict(os.environ, {"SENTRIX_12B_FULL_CHAIN_VALIDATION": "1",
                                          "SENTRIX_AGENT_NO_FALLBACK": "1"}, clear=False):
            self.assertTrue(prof.no_fallback())


class ResponseValidationBlockTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix="12b-val-")
        self.store = MemoryStore(str(Path(self.directory.name) / "memory.db"))
        self.gamma = _FakeGamma()
        self.runtime = ThinAgentRuntime(self.store, gamma=self.gamma)

    def tearDown(self):
        self.store.close()
        self.directory.cleanup()

    def test_validation_block_attached_with_ledger(self):
        env = {"SENTRIX_12B_FULL_CHAIN_VALIDATION": "1",
               "SENTRIX_AGENT_REQUIRE_MODEL_TRACE": "1",
               "SENTRIX_AGENT_NO_FALLBACK": "1",
               "SENTRIX_AGENT_REQUIRE_12B_ROLES": "1",
               "SENTRIX_AGENT_FAIL_ON_DEGRADATION": "1",
               "SENTRIX_AGENT_MODEL_PROFILE": "quality_12b"}
        with mock.patch.dict(os.environ, env, clear=False):
            result = self.runtime.answer_turn("帮我写一段生日祝福", scope_id="home")
        self.assertIn("validation", result)
        v = result["validation"]
        self.assertEqual(v["profile"], "12b_full_chain_no_fallback")
        self.assertIn("answer", v["actual_roles"])
        self.assertTrue(v["all_models_match"])
        self.assertIn("model_call_ledger", result)
        self.assertGreaterEqual(len(result["model_call_ledger"]), 1)

    def test_no_validation_block_when_off(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            result = self.runtime.answer_turn("帮我写一段生日祝福", scope_id="home")
        self.assertNotIn("validation", result)
        self.assertNotIn("model_call_ledger", result)


if __name__ == "__main__":
    unittest.main()
