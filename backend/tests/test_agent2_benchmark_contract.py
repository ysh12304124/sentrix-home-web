import importlib.util
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = ROOT / "scripts" / "benchmarks" / "evaluate_agent2_shadow.py"
SPEC = importlib.util.spec_from_file_location("evaluate_agent2_shadow", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class Agent2BenchmarkContractTests(unittest.TestCase):
    def test_case_manifest_requires_all_generalization_strata_and_no_answers(self):
        cases = MODULE.load_case_manifest(ROOT / "scripts" / "benchmarks" / "agent2_shadow_cases.json")
        self.assertEqual(set(cases), {
            "baseline_regression", "multi_hop_composition", "ambiguity_recovery",
            "evidence_boundary", "safety_provenance",
        })

    def test_case_manifest_rejects_expected_answers(self):
        with self.assertRaisesRegex(ValueError, "must not contain answer"):
            MODULE.validate_case_manifest({"strata": {
                name: [{"case_id": "x", "answer": "leak"}]
                for name in MODULE.REQUIRED_STRATA
            }})


if __name__ == "__main__":
    unittest.main()
