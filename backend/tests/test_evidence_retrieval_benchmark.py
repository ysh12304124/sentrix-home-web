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
