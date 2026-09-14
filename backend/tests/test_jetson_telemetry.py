import unittest
from unittest.mock import patch

from backend.jetson_telemetry import LocalJetsonLlamaCppTelemetryProvider


class JetsonTelemetryTests(unittest.TestCase):
    def test_local_pss_and_device_are_labeled_without_inventing_kv_bytes(self):
        provider = LocalJetsonLlamaCppTelemetryProvider("http://127.0.0.1:8100/v1")
        with patch.object(provider, "_pid", return_value=42), \
             patch.object(provider, "_pss_mib", return_value=512.25), \
             patch.object(provider, "_device", return_value={"gpu_utilization_pct": 20}), \
             patch.object(provider, "_metrics", return_value={}):
            self.assertEqual(provider.gpu_stats()["data"]["gpus"][0]["gpu_utilization_pct"], 20)
        self.assertEqual(provider.process_memory()["data"]["process_memory_used_mib"], 512.25)
        self.assertEqual(provider.process_memory()["data"]["memory_unit"], "process_pss_uma_mib")
        self.assertEqual(provider.kv_cache()["status"], "unavailable")

    def test_stale_pid_does_not_reuse_old_pss_as_zero_or_current(self):
        provider = LocalJetsonLlamaCppTelemetryProvider("http://127.0.0.1:8100/v1", pss_interval=0)
        with patch.object(provider, "_device", return_value={}), \
             patch.object(provider, "_metrics", return_value={}), \
             patch.object(provider, "_pid", side_effect=[42, ValueError("stale PID")]), \
             patch.object(provider, "_pss_mib", return_value=100):
            provider.gpu_stats()
            provider.gpu_stats()
        self.assertNotIn("process_memory_used_mib", provider.process_memory()["data"])
        self.assertIn("stale PID", provider.process_memory()["data"]["pss_error"])

    @patch("backend.jetson_telemetry.httpx.get")
    def test_missing_llamacpp_kv_gauges_stay_missing(self, get):
        provider = LocalJetsonLlamaCppTelemetryProvider("http://127.0.0.1:8100/v1")
        get.return_value.text = "llamacpp:prompt_tokens_total 12\nllamacpp:n_tokens_max 12\n"
        self.assertEqual(provider._metrics(), {})
        get.return_value.text = "llamacpp:kv_cache_tokens 3\nllamacpp:kv_cache_usage_ratio 0.25\n"
        self.assertEqual(provider._metrics()["kv_cache_used_tokens"], 3)


if __name__ == "__main__":
    unittest.main()
