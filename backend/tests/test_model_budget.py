"""Phase R R5 — model routing, unified deadline and call budgets (P0-11/P0-12)."""

import time
import unittest

from backend.model_clients import GammaClient
from backend.model_routing import (
    CircuitBreaker,
    ModelRouter,
    RequestDeadline,
    resolve_specs,
)


class RoleModelResolutionTests(unittest.TestCase):
    def test_default_all_roles_main_model(self):
        specs = resolve_specs({"OLLAMA_MODEL": "gemma4:12b"})
        self.assertEqual(specs["parser"].model, "gemma4:12b")
        self.assertEqual(specs["answer"].model, "gemma4:12b")
        self.assertEqual(specs["verify"].model, "gemma4:12b")

    def test_split_roles_respected(self):
        specs = resolve_specs({
            "OLLAMA_MODEL": "gemma4:12b",
            "SENTRIX_PARSE_MODEL": "gemma4:e2b-it",
            "SENTRIX_ANSWER_MODEL": "gemma4:12b",
            "SENTRIX_PARSE_BACKEND": "e2b",
            "SENTRIX_PARSE_BASE_URL": "http://127.0.0.1:8100",
        })
        self.assertEqual(specs["parser"].backend, "e2b")
        self.assertEqual(specs["parser"].base_url, "http://127.0.0.1:8100")
        self.assertEqual(specs["parser"].model, "gemma4:e2b-it")
        self.assertEqual(specs["answer"].backend, "ollama_local")

    def test_gamma_client_role_endpoint(self):
        client = GammaClient(
            model="gemma4:12b", parse_model="gemma4:e2b-it",
            parse_backend="e2b", parse_base_url="http://127.0.0.1:8100",
        )
        # SENTRIX_LLM_BACKEND defaults to vllm (openai-compatible), so every
        # role endpoint is normalized to a /v1 suffix on the vLLM base URL.
        self.assertEqual(client._endpoint_for("parser"), ("http://127.0.0.1:8100/v1", "gemma4:e2b-it"))
        self.assertEqual(client._endpoint_for("answer"), ("http://127.0.0.1:8100/v1", "gemma4:12b"))


class RequestDeadlineTests(unittest.TestCase):
    def test_remaining_decreases(self):
        deadline = RequestDeadline(deadline_seconds=1.0, phase_budgets={"parser": 0.5})
        time.sleep(0.05)
        self.assertLess(deadline.remaining(), 1.0)
        self.assertGreater(deadline.remaining(), 0.0)

    def test_phase_budget_capped_by_remaining(self):
        deadline = RequestDeadline(deadline_seconds=0.01, phase_budgets={"parser": 5.0})
        self.assertLess(deadline.phase_available("parser"), 5.0)


class CircuitBreakerTests(unittest.TestCase):
    def test_trips_after_threshold(self):
        breaker = CircuitBreaker(threshold=2)
        self.assertFalse(breaker.is_tripped("parser"))
        breaker.record_failure("parser")
        breaker.record_failure("parser")
        self.assertTrue(breaker.is_tripped("parser"))

    def test_success_resets(self):
        breaker = CircuitBreaker(threshold=2)
        breaker.record_failure("parser")
        breaker.record_success("parser")
        breaker.record_failure("parser")
        self.assertFalse(breaker.is_tripped("parser"))


class ModelRouterBudgetTests(unittest.TestCase):
    def test_tripped_role_uses_fallback(self):
        router = ModelRouter(gamma=None, breaker=CircuitBreaker(threshold=1))
        router.breaker.record_failure("parser")
        result = router.chat("parser", "hello", fallback=lambda: "fallback")
        self.assertEqual(result, "fallback")

    def test_no_gamma_uses_fallback(self):
        router = ModelRouter(gamma=None)
        self.assertEqual(router.chat("answer", "hi", fallback=lambda: "none"), "none")

    def test_parser_role_forwarded_to_gamma(self):
        class FakeGamma:
            def __init__(self):
                self.calls = []

            def chat(self, prompt, json_mode=True, role=None):
                self.calls.append((prompt, role))
                return "ok"

        gamma = FakeGamma()
        router = ModelRouter(gamma=gamma)
        router.chat("parser", "分类", json_mode=True)
        self.assertEqual(gamma.calls[0][1], "parser")


if __name__ == "__main__":
    unittest.main()
