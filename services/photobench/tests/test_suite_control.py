import importlib.util
import json
import sqlite3
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

    def test_live_metric_preview_uses_completed_item_fields_only(self):
        self.run.state["items"] = [
            {"answerability": "answerable", "media_retrieval_counts": {"gt": 2, "matched": 1},
             "retrieval_recall": 0.5, "judge": {"score": 2}},
            {"answerability": "unanswerable", "media_retrieval_counts": {"gt": 0, "matched": 0},
             "retrieval_recall": None, "judge": {"score": 0}},
        ]
        self.run._qa_judge_completed = 1

        self.run._refresh_live_metric_preview()

        summary = self.run.state["summary"]
        self.assertEqual(summary["retrieval_recall_micro"], 0.5)
        self.assertEqual(summary["media_retrieval_recall_micro"], 0.5)
        self.assertEqual(summary["answer_quality_mean"], 1)
        self.assertEqual(summary["judge_valid_count"], 2)
        self.assertTrue(summary["live_preview"])

    def test_face_quality_distinguishes_missing_assets_from_zero_or_multiple_faces(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            album_dir = root / "album" / "identity"
            album_dir.mkdir(parents=True)
            (album_dir / "face_info_cn.json").write_text(json.dumps({
                "image_to_face_ids": {
                    "a.jpg": ["person-1"],
                    "b.jpg": ["person-1"],
                    "multi.jpg": ["person-2"],
                    "zero.jpg": ["person-2"],
                    "absent.jpg": ["person-3"],
                    "multiple-gt.jpg": ["person-1", "person-2"],
                }
            }), encoding="utf-8")
            data_dir = root / "data"
            data_dir.mkdir()
            db_path = data_dir / "sentrix.db"
            conn = sqlite3.connect(db_path)
            try:
                conn.execute("CREATE TABLE assets (id TEXT, file_name TEXT, scope_id TEXT)")
                conn.execute("CREATE TABLE face_instances (id TEXT, asset_id TEXT, cluster_id TEXT)")
                conn.executemany(
                    "INSERT INTO assets VALUES (?, ?, ?)",
                    [(f"asset-{name}", name, "scope") for name in (
                        "a.jpg", "b.jpg", "multi.jpg", "zero.jpg", "multiple-gt.jpg",
                    )],
                )
                conn.executemany(
                    "INSERT INTO face_instances VALUES (?, ?, ?)",
                    [
                        ("face-a", "asset-a.jpg", "cluster-a"),
                        ("face-b", "asset-b.jpg", "cluster-b"),
                        ("face-m1", "asset-multi.jpg", "cluster-m1"),
                        ("face-m2", "asset-multi.jpg", "cluster-m2"),
                        ("face-multiple-gt", "asset-multiple-gt.jpg", "cluster-other"),
                    ],
                )
                conn.commit()
            finally:
                conn.close()
            with patch.object(MODULE, "REPOSITORY_ROOT", root):
                quality = MODULE.benchmark_face_clustering_quality("scope", album_dir.parent)

        self.assertTrue(quality["available"])
        self.assertEqual(quality["gt_single_image_count"], 5)
        self.assertEqual(quality["asset_name_coverage_count"], 4)
        self.assertEqual(quality["gt_assets_missing_from_scope_count"], 1)
        self.assertEqual(quality["detected_zero_face_count"], 1)
        self.assertEqual(quality["detected_single_face_count"], 2)
        self.assertEqual(quality["detected_multiple_faces_count"], 1)
        self.assertEqual(quality["evaluable_face_count"], 2)
        self.assertEqual(quality["gt_multi_identity_image_count"], 1)

    def test_face_quality_excludes_uncertain_evidence_from_strict_cluster_metric(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            album_dir = root / "album" / "identity"
            album_dir.mkdir(parents=True)
            (album_dir / "face_info_cn.json").write_text(json.dumps({
                "image_to_face_ids": {
                    "a.jpg": ["person-1"],
                    "b.jpg": ["person-1"],
                }
            }), encoding="utf-8")
            data_dir = root / "data"
            data_dir.mkdir()
            conn = sqlite3.connect(data_dir / "sentrix.db")
            try:
                conn.execute("CREATE TABLE assets (id TEXT, file_name TEXT, scope_id TEXT)")
                conn.execute(
                    "CREATE TABLE face_instances (id TEXT, asset_id TEXT, cluster_id TEXT, validity TEXT)"
                )
                conn.executemany(
                    "INSERT INTO assets VALUES (?, ?, ?)",
                    [("asset-a", "a.jpg", "scope"), ("asset-b", "b.jpg", "scope")],
                )
                conn.executemany(
                    "INSERT INTO face_instances VALUES (?, ?, ?, ?)",
                    [
                        ("face-a", "asset-a", "cluster-1", "verified"),
                        ("noise-a", "asset-a", None, "uncertain"),
                        ("face-b", "asset-b", "cluster-1", "verified"),
                    ],
                )
                conn.commit()
            finally:
                conn.close()
            with patch.object(MODULE, "REPOSITORY_ROOT", root):
                quality = MODULE.benchmark_face_clustering_quality("scope", album_dir.parent)

        self.assertTrue(quality["available"])
        self.assertEqual(quality["detected_single_face_count"], 2)
        self.assertEqual(quality["detected_multiple_faces_count"], 0)
        self.assertEqual(quality["raw_detected_multiple_faces_count"], 1)
        self.assertEqual(quality["same_person_pair_f1"], 1.0)

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

    def test_reuse_base_accepts_legacy_run_list_model_name(self):
        groups = MODULE._build_reuse_bases(
            [{
                "id": "scope-qwen",
                "name": "PhotoBench-20260918-142618-album3-max-video10-qwen3-vl-4b-instruct",
                "created_at": "2026-09-18T06:26:18Z",
                "status": "active",
            }],
            [{
                "run_id": "run-qwen",
                "scope_id": "scope-qwen",
                "album_id": "album3-max-video10",
                "model_name": "qwen3-vl-4b-instruct",
            }],
        )

        self.assertEqual(len(groups), 1)
        self.assertEqual(groups[0]["base_id"], "album3-max-video10::qwen3-vl-4b-instruct")
        self.assertEqual(groups[0]["source_run_ids"], ["run-qwen"])

    def test_run_list_omits_heavy_detail_fields_but_keeps_rejudge_status(self):
        repository = MODULE.OrchestratorRepository.__new__(MODULE.OrchestratorRepository)
        repository.lock = MODULE.threading.RLock()
        repository.runs = {
            "run-large": {
                "run_id": "run-large", "status": "completed", "qa_count": 1,
                "model_profile": "qwen3.5-4b",
                "items": [{"qa_id": "q1", "judge_status": "completed"}],
                "phases": {"qa_eval": {"debug_trace": ["large payload"]}},
                "rejudge": {"status": "completed", "task_id": "judge-1", "results": ["large payload"]},
                "summary": {"total": 1, "completed": 1, "answer_quality_mean": 0.5},
            }
        }

        listed = repository.list_runs()[0]

        self.assertNotIn("phases", listed)
        self.assertEqual(listed["model_profile"], "qwen3.5-4b")
        self.assertEqual(listed["rejudge"], {"status": "completed", "task_id": "judge-1"})
        self.assertEqual(listed["summary"]["answer_quality_mean"], 0.5)

    def test_live_run_list_uses_progress_counters_without_scanning_items(self):
        class UnscannableItems:
            def __len__(self):
                return 1000

            def __iter__(self):
                raise AssertionError("live progress must not walk stored QA traces")

        repository = MODULE.OrchestratorRepository.__new__(MODULE.OrchestratorRepository)
        repository.lock = MODULE.threading.RLock()
        repository.runs = {
            "run-live": {
                "run_id": "run-live", "status": "running", "qa_count": 1000,
                "items": UnscannableItems(),
                "phases": {"qa_eval": {"progress": {"completed": 321}}},
            }
        }

        listed = repository.list_runs()[0]

        self.assertEqual(listed["summary"]["completed"], 321)
        self.assertEqual(listed["summary"]["total"], 1000)

    def test_graph_quality_persist_does_not_block_run_list_and_is_single_flight(self):
        repository = MODULE.OrchestratorRepository.__new__(MODULE.OrchestratorRepository)
        repository.results_root = Path(self.results.name)
        repository.lock = MODULE.threading.RLock()
        repository.graph_quality_locks = {}
        repository.runs = {
            "run-large": {
                "run_id": "run-large", "status": "completed", "qa_count": 1,
                "album_id": "album3-14", "scope_id": "scope-test", "items": [],
                "summary": {"graph_quality": {
                    "available": True, "status": "ready", "scope_id": "scope-test",
                    "face_clustering": {"available": True, "gt_single_image_count": 202},
                }},
            }
        }
        snapshot_started = MODULE.threading.Event()
        allow_snapshot = MODULE.threading.Event()
        persist_started = MODULE.threading.Event()
        allow_persist = MODULE.threading.Event()
        persist_finished = MODULE.threading.Event()

        def slow_snapshot(*_args, **_kwargs):
            snapshot_started.set()
            if not allow_snapshot.wait(timeout=3):
                raise TimeoutError("test did not release graph-quality snapshot")
            return {"available": True, "scope_id": "scope-test", "nodes": 12}

        def slow_persist(*_args, **_kwargs):
            persist_started.set()
            if not allow_persist.wait(timeout=3):
                raise TimeoutError("test did not release graph-quality persistence")
            persist_finished.set()

        with (
            patch.object(MODULE, "benchmark_graph_quality_snapshot", side_effect=slow_snapshot),
            patch.object(MODULE, "atomic_json", side_effect=slow_persist),
        ):
            first = repository.get_graph_quality("run-large")
            self.assertEqual(first.get("status"), "computing")
            self.assertTrue(snapshot_started.wait(timeout=2))

            # A second poll must not start another expensive graph build.
            self.assertEqual(repository.get_graph_quality("run-large").get("status"), "computing")
            self.assertEqual(repository.list_runs()[0]["run_id"], "run-large")

            allow_snapshot.set()
            self.assertTrue(persist_started.wait(timeout=2))
            # The sidecar write is deliberately blocked; unrelated progress
            # and list requests must still be served immediately.
            self.assertEqual(repository.list_runs()[0]["run_id"], "run-large")
            allow_persist.set()
            self.assertTrue(persist_finished.wait(timeout=2))

    def test_get_run_uses_saved_summary_without_rewalking_completed_traces(self):
        repository = MODULE.OrchestratorRepository.__new__(MODULE.OrchestratorRepository)
        repository.lock = MODULE.threading.RLock()
        repository.runs = {
            "run-complete": {
                "run_id": "run-complete", "status": "completed", "qa_count": 2,
                "items": [{"trace": "large"}, {"trace": "large"}],
                "summary": {"total": 2, "completed": 2, "graph_qa_by_type": []},
            }
        }
        repository._effective_summary = lambda _state: self.fail(
            "a completed run with a saved summary must not be recomputed on page refresh"
        )

        result = repository.get_run("run-complete")

        self.assertEqual(result["summary"]["completed"], 2)
        self.assertEqual(result["item_count"], 2)
        self.assertNotIn("items", result)

    def test_get_run_uses_lightweight_progress_for_active_run(self):
        repository = MODULE.OrchestratorRepository.__new__(MODULE.OrchestratorRepository)
        repository.lock = MODULE.threading.RLock()
        repository.runs = {
            "run-live": {
                "run_id": "run-live", "status": "running", "qa_count": 1000,
                "items": [],
                "phases": {"qa_eval": {"progress": {"completed": 321}}},
            }
        }
        repository._effective_summary = lambda _state: self.fail(
            "live polling must not recompute summaries from QA traces"
        )

        result = repository.get_run("run-live")

        self.assertEqual(result["summary"]["completed"], 321)
        self.assertEqual(result["summary"]["total"], 1000)

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

    def test_gpu_sampler_keeps_apple_footprint_out_of_vllm_formula(self):
        sampler = MODULE.GpuSampler("http://manager.invalid")
        sampler.samples = [
            {"memory_unit": "apple_phys_footprint_mib", "model_process_memory_used_mib": 2400.0,
             "model_process_system_memory_used_mib": 2400.0, "system_memory_used_mib": 12000.0,
             "gpu_in_use_unified_memory_mib": 4000.0},
            {"memory_unit": "apple_phys_footprint_mib", "model_process_memory_used_mib": 2800.0,
             "model_process_system_memory_used_mib": 2800.0, "system_memory_used_mib": 13000.0,
             "gpu_in_use_unified_memory_mib": 5000.0},
        ]

        result = sampler.aggregate()

        self.assertEqual(result["source"], "apple_unified")
        self.assertEqual(result["model_process_system_memory_used_mib"]["peak"], 2800.0)
        self.assertEqual(result["gpu_in_use_unified_memory_mib"]["peak"], 5000.0)
        self.assertEqual(result["memory_profile"]["method"], "apple_phys_footprint_v1")
        self.assertIsNone(result["memory_profile"]["comparable_workload_memory_gib"])
        self.assertNotIn("fixed_base_memory_gib", result["memory_profile"])


if __name__ == "__main__":
    unittest.main()
