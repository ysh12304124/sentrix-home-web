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

    def test_reuse_bases_group_spaces_by_album_and_model(self):
        spaces = [
            {"id": "scope-gemma", "name": "PhotoBench-20260825-album3-max-gemma4-12b-it", "created_at": "2026-08-25T10:00:00Z"},
            {"id": "scope-qwen", "name": "PhotoBench-20260825-album3-max-qwen3.5-0.8-lora-v2", "created_at": "2026-08-25T09:00:00Z"},
        ]
        runs = [
            {"run_id": "run-gemma", "scope_id": "scope-gemma", "album_id": "album3-max", "model_profile": "gemma4-12b-it", "scope_source": "created"},
            {"run_id": "run-qwen", "scope_id": "scope-qwen", "album_id": "album3-max", "model_profile": "qwen3.5-0.8-lora-v2", "scope_source": "created"},
        ]

        groups = MODULE._build_reuse_bases(spaces, runs)

        self.assertEqual({(item["album_id"], item["model_profile"]) for item in groups}, {
            ("album3-max", "gemma4-12b-it"), ("album3-max", "qwen3.5-0.8-lora-v2"),
        })
        gemma = next(item for item in groups if item["model_profile"] == "gemma4-12b-it")
        self.assertEqual(gemma["scope_id"], "scope-gemma")
        self.assertEqual(gemma["source_run_ids"], ["run-gemma"])

    def test_reuse_base_uses_scope_name_when_scope_has_cross_model_reuse_history(self):
        groups = MODULE._build_reuse_bases(
            [{"id": "scope", "name": "PhotoBench-20260825-album3-max-gemma4-e2b-it", "created_at": "2026-08-25"}],
            [
                {"run_id": "wrong", "scope_id": "scope", "album_id": "album3-max", "model_profile": "gemma4-12b-it"},
                {"run_id": "right", "scope_id": "scope", "album_id": "album3-max", "model_profile": "gemma4-e2b-it"},
            ],
        )
        self.assertEqual(len(groups), 1)
        self.assertEqual(groups[0]["model_profile"], "gemma4-e2b-it")
        self.assertEqual(groups[0]["source_run_ids"], ["right"])

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

    def test_query_current_model_verifies_live_served_name(self):
        repository = MODULE.OrchestratorRepository(Path(self.results.name) / "current-query")

        def request(url, **_kwargs):
            if url.endswith("/state"):
                return {
                    "profile": "qwen3.5-4b",
                    "served_model_name": "qwen3.5-4b",
                    "max_num_seqs": 12,
                }
            if url.endswith("/models"):
                return {"data": [{"id": "qwen3.5-4b"}]}
            raise AssertionError(url)

        with patch.object(MODULE, "request_json", side_effect=request):
            snapshot = repository.query_current_model(
                "http://manager.invalid", "http://model.invalid/v1",
            )

        self.assertEqual(snapshot["model_id"], "qwen3.5-4b")
        self.assertEqual(snapshot["served_model_name"], "qwen3.5-4b")
        self.assertEqual(snapshot["state"]["max_num_seqs"], 12)

    def test_start_suite_pins_current_model_without_switching(self):
        repository = MODULE.OrchestratorRepository(Path(self.results.name) / "current-suite")
        requested_urls = []

        def request(url, *_args, **_kwargs):
            requested_urls.append(url)
            if url.endswith("/state"):
                return {
                    "profile": "qwen3.5-4b",
                    "served_model_name": "qwen3.5-4b",
                    "max_num_seqs": 12,
                }
            if url.endswith("/models"):
                return {"data": [{"id": "qwen3.5-4b"}]}
            if url.endswith("/api/model-profiles/bind-runtime"):
                return {"status": "ok"}
            raise AssertionError(url)

        target = {
            "manager_url": "http://manager.invalid",
            "model_base_url": "http://model.invalid/v1",
        }
        with (
            patch.object(MODULE, "resolve_vllm_target", return_value=("test", target)),
            patch.object(MODULE, "request_json", side_effect=request),
            patch.object(MODULE.threading.Thread, "start"),
        ):
            result = repository.start_suite({
                "album_id": "album3-14",
                "qa_set": "compact-10q",
                "models": [MODULE.CURRENT_MODEL_SELECTION],
                "sentrix_url": "http://sentrix.invalid",
            })

        run = repository.runs[result["run_ids"][0]]
        self.assertEqual(result["models"], ["qwen3.5-4b"])
        self.assertEqual(run.model_profile, "qwen3.5-4b")
        self.assertTrue(run.use_current_model)
        self.assertNotIn("model_deploy", run._selected_phase_names())
        self.assertFalse(any(url.endswith(("/start", "/stop")) for url in requested_urls))

    def test_current_model_run_skips_deploy_and_is_never_reclaimed(self):
        manifest = json.loads(
            (MODULE.BENCHMARK_DATA_ROOT / "album3-14" / "manifest.json").read_text(encoding="utf-8")
        )
        run = MODULE.BenchmarkRun(
            run_id="test-current", album_id="album3-14", manifest=manifest,
            model_profile="qwen3.5-4b", qa_set="compact-10q",
            sentrix_url="http://sentrix.invalid", judge_url="http://judge.invalid",
            vllm_api_url="http://manager.invalid", vllm_target_id="test",
            vllm_model_base_url="http://model.invalid/v1",
            results_root=Path(self.results.name), use_current_model=True,
            current_model_snapshot={"state": {"max_num_seqs": 12}},
        )
        self.addCleanup(run._stop_persist_writer)

        self.assertNotIn("model_deploy", run._selected_phase_names())
        self.assertEqual(run._resolve_qa_concurrency(), 12)
        with patch.object(MODULE, "request_json") as request:
            run._reclaim_vllm_after_cancel()
        request.assert_not_called()

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
