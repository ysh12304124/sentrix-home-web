#!/usr/bin/env python3
from __future__ import annotations

import unittest
from unittest.mock import patch

from backend import runtime_providers as providers


class RuntimeProviderSystemMemoryTests(unittest.TestCase):
    def test_augments_system_related_ram_and_gpu_fields(self):
        compute = [
            {"pid": 100, "process_name": "python", "used_memory_mib": 11.0},
            {"pid": 200, "process_name": "VLLM::EngineCore", "used_memory_mib": 22.0},
            {"pid": 999, "process_name": "unrelated", "used_memory_mib": 33.0},
        ]
        processes = [
            {"pid": 100, "process_name": "python", "rss_mib": 100.0, "listening_ports": ["8771"], "identity": "python benchmark_orchestrator.py"},
            {"pid": 200, "process_name": "VLLM::EngineCore", "rss_mib": 200.0, "listening_ports": ["8100"], "identity": "vllm serve --port 8100"},
            {"pid": 999, "process_name": "other", "rss_mib": 999.0, "listening_ports": [], "identity": "other"},
        ]
        with patch.object(providers, "_host_process_rows", return_value=processes), \
             patch.object(providers, "_process_pss_mib", side_effect=[50.0, 150.0]), \
             patch.object(providers, "_process_smaps_field_mib", side_effect=[100.0, 200.0]):
            data = providers._augment_host_process_memory(
                {"root_pid": 200}, process_hint="vllm", model_pids={200}, compute_apps=compute,
            )
        self.assertEqual(data["model_process_system_memory_used_mib"], 200.0)
        self.assertEqual(data["benchmark_process_memory_used_mib"], 200.0)
        self.assertEqual(data["benchmark_process_memory_scope"], "all_related_pss_plus_model_uma_extra")
        self.assertEqual(data["benchmark_process_gpu_memory_mib"], 33.0)
        self.assertEqual(data["all_processes_memory_mib"], 66.0)

    def test_missing_pss_does_not_fallback_to_rss_or_write_product_total(self):
        processes = [
            {"pid": 100, "process_name": "python", "rss_mib": 100.0, "listening_ports": ["8771"], "identity": "python benchmark_orchestrator.py"},
            {"pid": 200, "process_name": "llama-server", "rss_mib": 200.0, "listening_ports": ["8100"], "identity": "llama-server --port 8100"},
        ]
        with patch.object(providers, "_host_process_rows", return_value=processes), \
             patch.object(providers, "_process_pss_mib", side_effect=[None, 150.0]):
            data = providers._augment_host_process_memory(
                {"root_pid": 200, "product_stack_memory_mib": 999.0},
                process_hint="llama-server", endpoint_port="8100", model_pids={200}, compute_apps=[],
            )
        self.assertNotIn("product_stack_memory_mib", data)
        self.assertNotIn("benchmark_process_memory_used_mib", data)
        self.assertEqual(data["model_process_system_memory_used_mib"], 200.0)

    def test_missing_model_smaps_rss_does_not_write_product_total(self):
        processes = [
            {"pid": 100, "process_name": "python", "rss_mib": 100.0, "listening_ports": ["8771"], "identity": "python benchmark_orchestrator.py"},
            {"pid": 200, "process_name": "llama-server", "rss_mib": 200.0, "listening_ports": ["8100"], "identity": "llama-server --port 8100"},
        ]
        with patch.object(providers, "_host_process_rows", return_value=processes), \
             patch.object(providers, "_process_pss_mib", return_value=150.0), \
             patch.object(providers, "_process_smaps_field_mib", return_value=None):
            data = providers._augment_host_process_memory(
                {"root_pid": 200}, process_hint="llama-server", endpoint_port="8100", model_pids={200}, compute_apps=[],
            )
        self.assertNotIn("product_stack_memory_mib", data)
        self.assertNotIn("benchmark_process_memory_used_mib", data)


if __name__ == "__main__":
    unittest.main()
