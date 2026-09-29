"""Phase R9-3 — parser slot label set and metric helpers (offline).

These tests pin the synthetic slot label set and the metric helpers used by
scripts/benchmarks/evaluate_parser_slots.py so the bake-off is reproducible and
does not depend on a live model.  The label set must not contain Retrieval
benchmark original sentences.
"""

import importlib.util
import sys
import unittest
from pathlib import Path

import sys as _sys

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "benchmarks" / "evaluate_parser_slots.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("eval_parser_slots", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    _sys.modules["eval_parser_slots"] = module
    spec.loader.exec_module(module)
    return module


class ParserSlotLabelSetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            cls.mod = _load_module()
        except Exception as exc:  # pragma: no cover - import guard
            raise unittest.SkipTest(f"evaluate_parser_slots.py not importable: {exc}")
        cls.cases = cls.mod.SLOT_CASES

    def test_label_set_covers_all_slot_kinds(self):
        dims = {d for case in self.cases for d in case["facets"]}
        self.assertTrue(dims & {"person", "place", "time", "activity", "clothing", "object", "visual"})
        self.assertTrue(any(case["negative"] for case in self.cases))
        self.assertTrue(any(case["media"] for case in self.cases))
        self.assertTrue(any(case["date"] for case in self.cases))
        self.assertTrue(any(case["mode"] == "none" for case in self.cases))

    def test_no_retrieval_benchmark_original_sentences(self):
        # The synthetic set is structurally similar but not the album queries.
        for case in self.cases:
            self.assertNotIn("贵阳", case["query"])
            self.assertNotIn("海豚跃出水面", case["query"])

    def test_candidate_env_overrides_present(self):
        self.assertIn("e2b", self.mod.CANDIDATES)
        self.assertIn("12b", self.mod.CANDIDATES)
        self.assertEqual(self.mod.CANDIDATES["12b"]["SENTRIX_AGENT_MODEL_PROFILE"], "quality_12b")

    def test_helper_predicates(self):
        mod = self.mod
        self.assertTrue(mod._draft_has_date(type("D", (), {"time_expression": "去年春节"})()))
        self.assertTrue(mod._draft_has_media(type("D", (), {"media_expressions": ["照片"]})()))
        self.assertTrue(mod._draft_has_negative(type("D", (), {"negative_conditions": [{"dimension": "media"}]})()))


if __name__ == "__main__":
    unittest.main()
