#!/usr/bin/env python3
"""Regression coverage for complete live telemetry history responses."""

from __future__ import annotations

import tempfile
import threading
import unittest
from pathlib import Path

from services.photobench.backend import benchmark_orchestrator as orchestrator


class TelemetryHistoryTests(unittest.TestCase):
    def test_get_run_returns_complete_jsonl_history_not_recent_tail(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run = object.__new__(orchestrator.BenchmarkRun)
            run.lock = threading.RLock()
            run.state = {}
            run.telemetry_source = "host_nvidia_smi"
            run._current_phase = "qa_eval"
            run.results_root = root
            run.run_id = "run-full-history"
            run.persist = lambda wait=False: None
            run_dir = root / run.run_id
            run_dir.mkdir(parents=True)

            for index in range(300):
                run._persist_gpu_sample({
                    "_t": float(index),
                    "memory_used_mib": 1000.0 + index,
                    "model_process_memory_used_mib": 500.0 + index,
                    "gpu_utilization_pct": float(index % 100),
                    "source": "host_nvidia_smi",
                })

            # The persisted run snapshot remains bounded for I/O safety.
            self.assertEqual(len(run.state["telemetry_live"]["history"]), 240)
            samples_path = run_dir / "gpu_samples.jsonl"
            self.assertEqual(len(samples_path.read_text(encoding="utf-8").splitlines()), 300)

            # Simulate an in-flight append that leaves a partial final line.
            with samples_path.open("a", encoding="utf-8") as handle:
                handle.write('{"_t":')

            repository = orchestrator.OrchestratorRepository(root)
            repository.runs[run.run_id] = run
            payload = repository.get_run(run.run_id)
            history = payload["telemetry_live"]["history"]
            self.assertEqual(len(history), 300)
            self.assertEqual(history[0]["t"], 0.0)
            self.assertEqual(history[-1]["t"], 299.0)
            self.assertEqual(history[0]["phase"], "qa_eval")
            self.assertEqual(payload["telemetry_live"]["samples_count"], 300)


if __name__ == "__main__":
    unittest.main()
