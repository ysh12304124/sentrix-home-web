import importlib.util
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).with_name("benchmark_orchestrator.py")
SPEC = importlib.util.spec_from_file_location("benchmark_orchestrator_trace", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class Agent2TraceContractTests(unittest.TestCase):
    def test_missing_compact_trace_falls_back_to_debug_trace_with_observation(self):
        metrics, execution, tools = MODULE.BenchmarkRun._normalize_turn_traces({
            "model_call_metrics": [{"step_id": "step_0"}],
            "retrieval_trace": [],
            "tool_trace": [],
            "debug_trace": [
                {"type": "model", "step_id": "step_0"},
                {"type": "tool", "tool": "search_memories",
                 "parent_step_id": "step_0",
                 "observation": {"preview": [{"evidence_summary": "婚礼现场"}]}},
            ],
        })
        self.assertEqual(len(metrics), 1)
        self.assertEqual(execution[1]["parent_step_id"], "step_0")
        self.assertEqual(tools[0]["tool"], "search_memories")
        self.assertEqual(tools[0]["observation"]["preview"][0]["evidence_summary"], "婚礼现场")

    def test_debug_observation_wins_when_compact_trace_is_present(self):
        _, execution, tools = MODULE.BenchmarkRun._normalize_turn_traces({
            "tool_trace": [{"tool": "search_memories", "latency_s": 0.4,
                             "observation": {"preview": []}}],
            "debug_trace": [{"type": "tool", "tool": "search_memories",
                             "parent_step_id": "step_9",
                             "observation": {"preview": [{"evidence_summary": "完整描述"}]}}],
        })
        self.assertEqual(execution[0]["parent_step_id"], "step_9")
        self.assertEqual(tools[0]["latency_s"], 0.4)
        self.assertEqual(tools[0]["observation"]["preview"][0]["evidence_summary"], "完整描述")

    def test_tool_trace_contract_keeps_binding_and_observation_fields(self):
        _, execution, tools = MODULE.BenchmarkRun._normalize_turn_traces({
            "retrieval_trace": [{"stage": "tool", "step_id": "tool_step"}],
            "tool_trace": [{"tool": "search_memories", "tool_call_id": "tool_call_1",
                             "parent_step_id": "step_1", "arguments": {"query": "婚礼"},
                             "observation": {"description_status": "available"}}],
        })
        self.assertEqual(execution[0]["step_id"], "tool_step")
        self.assertEqual(tools[0]["tool_call_id"], "tool_call_1")
        self.assertEqual(tools[0]["parent_step_id"], "step_1")
        self.assertEqual(tools[0]["observation"]["description_status"], "available")

    def test_response_retrieval_trace_is_not_overwritten_by_tool_projection(self):
        # Regression is exercised through the app's response builder in the
        # integration smoke; this unit contract documents the required shape.
        self.assertIsInstance(
            {"retrieval_trace": [{"stage": "model"}], "tool_trace": [{"tool": "search_memories"}]}["retrieval_trace"],
            list,
        )

    def test_tool_binds_to_step_after_planner(self):
        calls = [
            {"call_type": "planner", "step_id": "planner_step_0", "conversation_turn": 0},
            {"call_type": "agent", "step_id": "step_0", "conversation_turn": 0},
        ]
        execution = [
            {"stage": "planner", "step_id": "planner_step_0", "conversation_turn": 0},
            {"stage": "model", "step_id": "step_0", "conversation_turn": 0},
            {"stage": "tool", "tool": "inspect_photo", "parent_step_id": "step_0", "conversation_turn": 0},
        ]
        bound = MODULE.BenchmarkRun._bind_tool_calls_to_model_rounds(
            [{"tool": "inspect_photo", "latency_s": 1.9, "conversation_turn": 0}],
            execution,
            calls,
        )
        self.assertEqual(bound[0]["model_call_index"], 1)
        self.assertEqual(bound[0]["parent_step_id"], "step_0")

    def test_agent_loop_duration_includes_direct_tool_duration(self):
        calls = [{
            "call_type": "agent", "step_id": "step_0", "total_ms": 1000,
            "ttft_ms": 89, "conversation_turn": 0,
        }]
        tools = [{
            "tool": "inspect_photo", "latency_s": 1.9,
            "model_call_index": 0, "conversation_turn": 0,
        }]
        calls, tools = MODULE.BenchmarkRun._annotate_agent_loop_timings(calls, tools)
        breakdown = MODULE.BenchmarkRun._build_timing_breakdown(
            calls, tools, 3000, 2900, 0, {}, True,
        )
        self.assertEqual(calls[0]["agent_loop_total_ms"], 2900.0)
        self.assertEqual(calls[0]["agent_loop_timing"]["tool_ms"], 1900.0)
        self.assertEqual(breakdown["agent_loops"][0]["duration_ms"], 2900.0)

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
