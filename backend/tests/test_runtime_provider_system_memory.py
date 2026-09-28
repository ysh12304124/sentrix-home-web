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
        with patch.object(providers, "_host_process_rows", return_value=processes):
            data = providers._augment_host_process_memory(
                {"root_pid": 200}, process_hint="vllm", model_pids={200}, compute_apps=compute,
            )
        self.assertEqual(data["model_process_system_memory_used_mib"], 200.0)
        self.assertEqual(data["benchmark_process_memory_used_mib"], 300.0)
        self.assertEqual(data["benchmark_process_gpu_memory_mib"], 33.0)
        self.assertEqual(data["all_processes_memory_mib"], 66.0)


if __name__ == "__main__":
    unittest.main()
