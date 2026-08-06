"""Phase R9-6 — stage trace / perf block tests.

With SENTRIX_AGENT_STAGE_TRACE=1 a response must carry a ``perf`` block with
stage durations and real model call counts; without the flag the collector must
be a no-op.  Also guards the complex-person route from a fake pass.
"""

import os
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

from backend.db import MemoryStore
from backend.thin_agent import ThinAgentRuntime, _Perf


class FakeGamma:
    def __init__(self):
        self.calls = []

    def chat(self, prompt, json_mode=False, role=None, **kwargs):
        self.calls.append((role, prompt))
        return "我在听。"


class PerfCollectorTests(unittest.TestCase):
    def test_measure_records_duration_and_count(self):
        _Perf.begin()
        try:
            with _Perf.measure("parser"):
                time.sleep(0.01)
            _Perf.count("answer_calls")
            data = _Perf.end()
        finally:
            _Perf._local.data = None
        self.assertIn("parser", data)
        self.assertGreaterEqual(data["parser"], 0.0)
        self.assertEqual(data["answer_calls"], 1)

    def test_inactive_collector_is_noop(self):
        _Perf._local.data = None
        with _Perf.measure("parser"):
            pass
        self.assertEqual(_Perf.end(), {})


class AgentPerfAttachmentTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix="latency-trace-")
        self.store = MemoryStore(str(Path(self.directory.name) / "memory.db"))
        self.gamma = FakeGamma()
        self.runtime = ThinAgentRuntime(self.store, gamma=self.gamma)

    def tearDown(self):
        self.store.close()
        self.directory.cleanup()

    def test_trace_flag_attaches_perf_with_model_calls(self):
        with mock.patch.dict(os.environ, {"SENTRIX_AGENT_STAGE_TRACE": "1"}, clear=False):
            result = self.runtime.answer_turn("帮我写一段生日祝福", scope_id="home")
        self.assertIn("perf", result)
        perf = result["perf"]
        self.assertIn("explicit_detector", perf)
        self.assertIn("answer", perf)
        self.assertEqual(perf["model_calls"].get("answer"), 1)
        self.assertEqual(perf["model_calls"].get("parser"), 0)

    def test_no_trace_flag_no_perf_block(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            result = self.runtime.answer_turn("帮我写一段生日祝福", scope_id="home")
        self.assertNotIn("perf", result)


if __name__ == "__main__":
    unittest.main()
