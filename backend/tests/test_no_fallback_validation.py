"""Phase 12B-FC V2 — no-fallback validation profile behaviour.

Under the validation profile a model failure must NOT return fallback text; the
case must fail instead.  Off the profile, fallback works as before.
"""

import os
import unittest
from unittest import mock

from backend.model_routing import CircuitBreaker, ModelRouter


class _FailGamma:
    def chat(self, prompt, json_mode=True, role=None):
        raise RuntimeError("model down")


class NoFallbackValidationTests(unittest.TestCase):
    def _env(self, **overrides):
        base = {"SENTRIX_12B_FULL_CHAIN_VALIDATION": "1", "SENTRIX_AGENT_NO_FALLBACK": "1",
                "SENTRIX_AGENT_REQUIRE_MODEL_TRACE": "1"}
        base.update(overrides)
        return mock.patch.dict(os.environ, base, clear=False)

    def test_no_fallback_returns_none_when_model_fails(self):
        router = ModelRouter(gamma=_FailGamma())
        with self._env():
            out = router.chat("parser", "x", fallback=lambda: "FALLBACK")
        self.assertIsNone(out)

    def test_breaker_treated_closed_under_validation(self):
        router = ModelRouter(gamma=_FailGamma(), breaker=CircuitBreaker(threshold=1))
        router.breaker.record_failure("parser")  # would trip normally
        with self._env():
            out = router.chat("parser", "x", fallback=lambda: "FALLBACK")
        # breaker ignored -> attempts the call -> fails -> None (no fallback)
        self.assertIsNone(out)

    def test_off_profile_fallback_still_works(self):
        router = ModelRouter(gamma=_FailGamma())
        out = router.chat("parser", "x", fallback=lambda: "FALLBACK")
        self.assertEqual(out, "FALLBACK")


if __name__ == "__main__":
    unittest.main()
