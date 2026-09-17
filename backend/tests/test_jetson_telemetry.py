import unittest
from pathlib import Path
from unittest.mock import patch

from backend.jetson_telemetry import LocalJetsonLlamaCppTelemetryProvider, _parse_tegrastats_line


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

    def test_default_sample_interval_matches_gpu_sampler(self):
        provider = LocalJetsonLlamaCppTelemetryProvider("http://127.0.0.1:8100/v1")
        self.assertEqual(provider.pss_interval, 0.5)
        self.assertEqual(provider.device_interval, 0.5)

    def test_pid_falls_back_to_port_scan_when_pid_file_stale(self):
        from tempfile import TemporaryDirectory
        with TemporaryDirectory() as tmp:
            pid_file = Path(tmp) / "server.pid"
            pid_file.write_text("1\n", encoding="ascii")
            provider = LocalJetsonLlamaCppTelemetryProvider(
                "http://127.0.0.1:8100/v1", pid_file=str(pid_file),
            )
            with patch.object(provider, "_pid_from_proc", side_effect=[ValueError("stale"), 1801352]), \
                 patch.object(provider, "_pid_from_port", return_value=1801352):
                self.assertEqual(provider._pid(), 1801352)

    def test_parse_tegrastats_line(self):
        sample = _parse_tegrastats_line("RAM 8192/16384MB GR3D_FREQ 41% VDD_GPU_SOC 1234mW gpu@52.5C")
        self.assertEqual(sample["system_memory_used_mib"], 8192.0)
        self.assertEqual(sample["gpu_utilization_pct"], 41.0)
        self.assertEqual(sample["power_draw_w"], 1.234)
        self.assertEqual(sample["temperature_c"], 52.5)

    @patch.object(LocalJetsonLlamaCppTelemetryProvider, "_device", return_value={
        "system_memory_used_mib": 6000,
        "system_memory_total_mib": 16000,
        "system_memory_available_mib": 10000,
        "system_memory_scope": "host_all_processes",
    })
    def test_system_memory_scope_is_host(self, _device):
        provider = LocalJetsonLlamaCppTelemetryProvider("http://127.0.0.1:8100/v1")
        provider._device_sample = _device.return_value
        result = provider.system_memory()
        self.assertEqual(result["data"]["system_memory_used_mib"], 6000)
        self.assertEqual(result["data"]["system_memory_scope"], "host_all_processes")


if __name__ == "__main__":
    unittest.main()
