import unittest
from unittest.mock import Mock, patch

import httpx

from backend.runtime_providers import (
    ManagerLifecycleProvider,
    ManagerTelemetryProvider,
    OpenAICompatibleInferenceProvider,
    UnavailableLifecycleProvider,
    UnavailableTelemetryProvider,
    create_runtime_providers,
    normalize_openai_base_url,
)


def response(method, url, payload, status=200):
    request = httpx.Request(method, url)
    return httpx.Response(status, request=request, json=payload)


class RuntimeProviderTests(unittest.TestCase):
    def test_normalize_openai_base_url_accepts_host_port_and_v1(self):
        self.assertEqual(normalize_openai_base_url("127.0.0.1:8100"), "http://127.0.0.1:8100/v1")
        self.assertEqual(normalize_openai_base_url("http://host:8080/v1/"), "http://host:8080/v1")

    @patch("backend.runtime_providers.httpx.post")
    def test_generic_chat_removes_vllm_extension(self, post):
        post.return_value = response("POST", "http://host:8080/v1/chat/completions", {"choices": []})
        provider = OpenAICompatibleInferenceProvider("host:8080", api_mode="generic")
        provider.chat({
            "model": "m", "messages": [], "stream_options": None,
            "chat_template_kwargs": {"enable_thinking": False},
        })
        self.assertNotIn("chat_template_kwargs", post.call_args.kwargs["json"])
        self.assertNotIn("stream_options", post.call_args.kwargs["json"])

    @patch("backend.runtime_providers.httpx.post")
    def test_vllm_chat_keeps_vllm_extension(self, post):
        post.return_value = response("POST", "http://host:8080/v1/chat/completions", {"choices": []})
        provider = OpenAICompatibleInferenceProvider("host:8080", api_mode="vllm")
        provider.chat({"model": "m", "messages": [], "chat_template_kwargs": {"enable_thinking": False}})
        self.assertIn("chat_template_kwargs", post.call_args.kwargs["json"])

    @patch("backend.runtime_providers.httpx.get")
    def test_list_models_is_openai_compatible(self, get):
        get.return_value = response("GET", "http://host:8080/v1/models", {"data": [{"id": "model-a"}]})
        result = OpenAICompatibleInferenceProvider("host:8080").list_models()
        self.assertEqual(result["models"], ["model-a"])

    @patch("backend.runtime_providers.httpx.post")
    def test_manager_token_count_is_optional(self, post):
        unmanaged = OpenAICompatibleInferenceProvider("host:8080")
        self.assertIsNone(unmanaged.token_count([{"role": "user", "content": "hi"}]))
        post.return_value = response("POST", "http://manager:8500/tokenize-current", {
            "prompt_tokens": 12, "max_model_len": 4096,
        })
        managed = OpenAICompatibleInferenceProvider("host:8080", manager_url="manager:8500")
        self.assertEqual(managed.token_count([{"role": "user", "content": "hi"}])["prompt_tokens"], 12)

    @patch("backend.runtime_providers.httpx.request")
    def test_manager_lifecycle_contract(self, request):
        request.side_effect = [
            response("GET", "http://manager:8500/profiles", [{"id": "a"}]),
            response("POST", "http://manager:8500/start", {"accepted": True}),
            response("GET", "http://manager:8500/state", {"profile": "a"}),
            response("POST", "http://manager:8500/stop", {"stopped": True}),
        ]
        provider = ManagerLifecycleProvider("manager:8500")
        self.assertEqual(provider.profiles()["profiles"][0]["id"], "a")
        self.assertTrue(provider.start({"profile": "a"})["accepted"])
        self.assertEqual(provider.state()["profile"], "a")
        self.assertTrue(provider.stop()["stopped"])

    @patch("backend.runtime_providers.httpx.get", side_effect=httpx.ConnectError("offline"))
    def test_manager_telemetry_failure_is_nonfatal(self, _get):
        result = ManagerTelemetryProvider("manager:8500").gpu_stats()
        self.assertEqual(result["status"], "unavailable")
        self.assertIn("offline", result["error"])

    def test_unmanaged_runtime_has_explicit_optional_provider_status(self):
        providers = create_runtime_providers("host:8080")
        self.assertIsInstance(providers.lifecycle, UnavailableLifecycleProvider)
        self.assertIsInstance(providers.telemetry, UnavailableTelemetryProvider)
        self.assertEqual(providers.lifecycle.state()["status"], "not_applicable")
        self.assertEqual(providers.telemetry.gpu_stats()["status"], "not_applicable")

    def test_managed_runtime_factory_selects_manager_implementations(self):
        providers = create_runtime_providers("host:8080", manager_url="manager:8500", api_mode="vllm")
        self.assertIsInstance(providers.lifecycle, ManagerLifecycleProvider)
        self.assertIsInstance(providers.telemetry, ManagerTelemetryProvider)
        self.assertTrue(providers.inference.capabilities()["vllm_extensions"])


if __name__ == "__main__":
    unittest.main()
