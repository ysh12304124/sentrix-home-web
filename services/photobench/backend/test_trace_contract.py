import importlib.util
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).with_name("benchmark_orchestrator.py")
SPEC = importlib.util.spec_from_file_location("benchmark_orchestrator_trace", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class Agent2TraceContractTests(unittest.TestCase):
    def test_aggregates_optional_agent2_trace_without_rewriting_legacy_runs(self):
        summary = MODULE.summarize_agent2_trace([
            {"agent2_trace": {}},
            {"agent2_trace": {
                "planner_decisions": [{"status": "accepted"}, {"status": "fallback"}],
                "requirement_status_counts": {"satisfied": 2, "blocked_budget": 1},
                "evidence_coverage": {"entries": 3, "partial_entries": 1},
                "terminal_reason": "shadow_only",
                "budget_outcome": {"tool_calls": 2},
            }},
        ])

        self.assertEqual(summary["planner_decision_count"], 2)
        self.assertEqual(summary["planner_fallback_count"], 1)
        self.assertEqual(summary["requirement_status_counts"], {"satisfied": 2, "blocked_budget": 1})
        self.assertEqual(summary["evidence_coverage"]["partial_entries"], 1)

    def test_returns_empty_summary_when_trace_is_absent(self):
        self.assertEqual(MODULE.summarize_agent2_trace([]), {
            "available": False,
            "planner_decision_count": 0,
            "planner_fallback_count": 0,
            "requirement_status_counts": {},
            "evidence_coverage": {"entries": 0, "partial_entries": 0},
            "terminal_reasons": {},
            "budget_outcomes": [],
        })

    def test_aggregates_full_debug_trace_requirements_and_ledger_entries(self):
        summary = MODULE.summarize_agent2_trace([{
            "agent2_trace": {
                "task_state": {"requirements": [
                    {"status": "satisfied"}, {"status": "blocked_budget"},
                ]},
                "evidence_ledger": {"entries": [
                    {"coverage": {"requested": 2, "processed": 1}},
                    {"coverage": {"requested": 1, "processed": 1}},
                ]},
            },
        }])

        self.assertEqual(summary["requirement_status_counts"], {
            "satisfied": 1, "blocked_budget": 1,
        })
        self.assertEqual(summary["evidence_coverage"], {
            "entries": 2, "partial_entries": 1,
        })


if __name__ == "__main__":
    unittest.main()
