"""Phase 1 Evidence Retrieval Kernel benchmark — unit test wrapper.

Wraps ``scripts/benchmarks/evaluate_evidence_retrieval.py`` so the ten canonical
cases (B-01 to B-10) can be exercised inside CI without a live Ollama.  This
test records the current baseline; strict expectations are added in later
phases.
"""

import importlib.util
import os
import sys
import tempfile
import unittest
from pathlib import Path


def _load_benchmark_module():
    root = Path(__file__).resolve().parents[2]
    path = root / "scripts" / "benchmarks" / "evaluate_evidence_retrieval.py"
    spec = importlib.util.spec_from_file_location("_phase1_benchmark", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["_phase1_benchmark"] = module
    spec.loader.exec_module(module)
    return module


class PhaseR1ARetrievalRunnerTests(unittest.TestCase):
    """Phase R R1A retrieval-only runner shape over a synthetic fixture.

    These tests exercise the measurement framework (report shape, metrics,
    hidden-set exclusion, channel forward-compat).  Strict per-channel Recall
    thresholds are added once Phase R2 wires the real retrievers.
    """

    @classmethod
    def setUpClass(cls):
        root = Path(__file__).resolve().parents[2]
        path = root / "scripts" / "benchmarks" / "evaluate_retrieval_kernel.py"
        spec = importlib.util.spec_from_file_location("_r1a_runner", path)
        cls.runner = importlib.util.module_from_spec(spec)
        sys.modules["_r1a_runner"] = cls.runner
        spec.loader.exec_module(cls.runner)

    def _store(self, directory):
        from backend.db import MemoryStore
        return MemoryStore(str(Path(directory) / "r1a.db"))

    def _seed(self, directory):
        from backend.db import MemoryStore
        from scripts.benchmarks.fixture import seed_fixture
        store = MemoryStore(str(Path(directory) / "r1a.db"))
        seed_fixture(store)
        return store

    def _synthetic_cases(self):
        # Shape tests must not depend on the external benchmark samples
        # directory (which lives on 153 as /home/asus/samples but not on every
        # developer machine).  A synthetic case exercises the same code path.
        return [{"key": "album1-01", "album": "album1", "query_cn": "睡衣自拍",
                 "ground_truth": ["IMG_DEMO_01.JPG"]}]

    def test_runner_produces_report_shape(self):
        with tempfile.TemporaryDirectory(prefix="r1a-shape-") as directory:
            store = self._seed(directory)
            try:
                results = self.runner.run(store, self._synthetic_cases(),
                                          "cached", top_k=20, include="album1-01")
                self.assertEqual(len(results), 1)
                for key in ("key", "recall_at", "mrr", "precision_at_5", "all_relevant",
                            "empty_gt_fp", "hard_violation", "ranked_ids", "latency_s"):
                    self.assertIn(key, results[0])
                for k in (1, 5, 10, 20):
                    self.assertIn(k, results[0]["recall_at"])
            finally:
                store.close()

    def test_hidden_exclusion_skips_keys(self):
        import json
        with tempfile.TemporaryDirectory(prefix="r1a-hidden-") as directory:
            manifest = Path(directory) / "hidden.json"
            manifest.write_text(json.dumps({"hidden_keys": [{"key": "album1-01", "category": "object"}]}), encoding="utf-8")
            store = self._seed(directory)
            try:
                results = self.runner.run(store, self._synthetic_cases(),
                                          "cached", top_k=20, include="album1-01", exclude_hidden=str(manifest))
                self.assertEqual(results, [])
            finally:
                store.close()

    def test_aggregate_covers_metrics(self):
        with tempfile.TemporaryDirectory(prefix="r1a-agg-") as directory:
            store = self._seed(directory)
            try:
                sample = [{"key": "k", "recall_at": {1: 1.0, 5: 1.0, 10: 1.0, 20: 1.0},
                           "mrr": 1.0, "precision_at_5": 1.0, "all_relevant": True,
                           "empty_gt_fp": False, "hard_violation": 0, "latency_s": 0.1}]
                summary = self.runner._aggregate(sample)
                self.assertEqual(summary["total"], 1)
                self.assertEqual(summary["recall_at_10"], 1.0)
                self.assertEqual(summary["all_relevant_count"], 1)
                self.assertEqual(summary["empty_gt_fp_count"], 0)
            finally:
                store.close()


class Phase1BenchmarkStructureTests(unittest.TestCase):
    """Sanity-check the benchmark shape.

    The suite does not enforce case-level passes here — Phase 2R adds the
    strict acceptance tests in ``test_semantic_routing.py`` and friends.
    """

    @classmethod
    def setUpClass(cls):
        cls.module = _load_benchmark_module()

    def test_ten_cases_are_defined(self):
        ids = [case["id"] for case in self.module.CASES]
        self.assertEqual(ids, [f"B-{index:02d}" for index in range(1, 11)])

    def test_fixture_seeds_expected_rows(self):
        from backend.db import MemoryStore

        with tempfile.TemporaryDirectory(prefix="phase1-seed-check-") as directory:
            store = MemoryStore(str(Path(directory) / "seed.db"))
            try:
                self.module._seed_fixture(store)
                assets = {row["id"] for row in store.list_assets(limit=100)}
                expected = {row["asset_id"] for row in self.module.FIXTURE_ROWS}
                self.assertEqual(assets, expected)
                self.assertTrue(store.is_confirmed_person_name("明哥", scope_id="album3"))
                self.assertTrue(store.is_confirmed_person_name("妈妈", scope_id="album2"))
            finally:
                store.close()

    def test_thin_agent_v1_on_report_is_well_formed(self):
        report = self.module._run_configuration("thin_agent_v1_on", "1")
        self.assertEqual(report["configuration"], "thin_agent_v1_on")
        self.assertEqual(report["total_cases"], 10)
        self.assertIn("passed", report)
        for case in report["cases"]:
            self.assertIn("id", case)
            self.assertIn("passed", case)
            self.assertIn("checks", case)
            self.assertIn("detected_mode", case["checks"])

    def test_thin_agent_v1_off_report_is_well_formed(self):
        report = self.module._run_configuration("thin_agent_v1_off", None)
        self.assertEqual(report["configuration"], "thin_agent_v1_off")
        self.assertEqual(report["total_cases"], 10)


if __name__ == "__main__":
    unittest.main()
