import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


MODULE_PATH = Path(__file__).resolve().parents[1] / "backend" / "benchmark_orchestrator.py"
SPEC = importlib.util.spec_from_file_location("benchmark_orchestrator_suite", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class SuiteControlTests(unittest.TestCase):
    def setUp(self):
        manifest = json.loads(
            (MODULE.BENCHMARK_DATA_ROOT / "album3-14" / "manifest.json").read_text(encoding="utf-8")
        )
        self.results = tempfile.TemporaryDirectory()
        self.run = MODULE.BenchmarkRun(
            run_id="test-pending", album_id="album3-14", manifest=manifest,
            model_profile="qwen3.5-4b", qa_set="compact-10q",
            sentrix_url="http://sentrix.invalid", judge_url="http://judge.invalid",
            vllm_api_url="http://manager.invalid", vllm_target_id="test",
            vllm_model_base_url="http://model.invalid/v1",
            results_root=Path(self.results.name),
        )

    def tearDown(self):
        self.results.cleanup()

    def test_cancelled_pending_run_never_records_started_at(self):
        self.run.cancel(source="test")
        self.run.execute()

        self.assertEqual(self.run.state["status"], "cancelled")
        self.assertIsNone(self.run.state["started_at"])
        self.assertIsNotNone(self.run.state["created_at"])

    def test_cancelled_processing_poll_exits_without_request(self):
        self.run.state["scope_id"] = "scope-test"
        self.run._cancel.set()

        with patch.object(MODULE, "request_json") as request:
            self.run._phase_processing()

        requested_urls = [call.args[0] for call in request.call_args_list]
        self.assertFalse(any("/api/assets" in url for url in requested_urls))
        self.assertEqual(self.run.state["phases"]["pipeline_processing"]["poll_iterations"], 0)

    def test_start_suite_rejects_existing_active_run(self):
        repository = MODULE.OrchestratorRepository(Path(self.results.name) / "repository")
        repository.runs[self.run.run_id] = self.run

        with self.assertRaisesRegex(ValueError, "another benchmark suite is still active"):
            repository.start_suite({
                "album_id": "album3-14",
                "qa_set": "compact-10q",
                "models": ["qwen3.5-4b"],
            })

        self.assertEqual(list(repository.runs), [self.run.run_id])

    def test_memory_profile_rejects_existing_active_run(self):
        repository = MODULE.OrchestratorRepository(Path(self.results.name) / "repository")
        repository.runs[self.run.run_id] = self.run

        with self.assertRaisesRegex(ValueError, "benchmark suite is active"):
            repository.start_memory_profile({"run_ids": [self.run.run_id]})

    def test_gpu_sampler_derives_comparable_memory_from_absolute_kv_capacity(self):
        sampler = MODULE.GpuSampler("http://manager.invalid")
        sampler.samples = [
            {
                "model_process_memory_used_mib": 10240.0,
                "kv_cache_usage_pct": 0.0,
                "kv_cache_capacity_gib": 2.0,
                "kv_cache_capacity_tokens": 32000,
                "weight_gib": 6.0,
                "peak_activation_gib": 0.2,
                "non_torch_gib": 0.1,
                "cuda_graph_gib": 0.1,
            },
            {
                "model_process_memory_used_mib": 10240.0,
                "kv_cache_usage_pct": 25.0,
                "kv_cache_capacity_gib": 2.0,
                "kv_cache_capacity_tokens": 32000,
                "weight_gib": 6.0,
                "peak_activation_gib": 0.2,
                "non_torch_gib": 0.1,
                "cuda_graph_gib": 0.1,
            },
        ]

        result = sampler.aggregate()["memory_profile"]

        self.assertEqual(result["fixed_base_memory_gib"], 8.0)
        self.assertEqual(result["kv_cache_used_peak_gib"], 0.5)
        self.assertEqual(result["comparable_workload_memory_gib"], 8.5)
        self.assertEqual(result["kv_cache_capacity_tokens"], 32000)


if __name__ == "__main__":
    unittest.main()
