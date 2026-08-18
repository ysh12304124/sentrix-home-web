import unittest

from backend.agent_runtime.budget_manager import BudgetState
from backend.agent_runtime.runtime import RuntimeTurn, public_agent2_trace


class AgentRuntimeTraceContractTests(unittest.TestCase):
    def test_turn_keeps_agent2_trace_separate_from_legacy_task_state(self):
        turn = RuntimeTurn(profile="tool_loop", budget=BudgetState())
        turn.task_state = {"current_result_set": "legacy_result_set"}

        self.assertEqual(turn.agent2_trace, {})
        self.assertEqual(turn.task_state, {"current_result_set": "legacy_result_set"})

    def test_public_agent2_trace_redacts_scope_and_evidence_references(self):
        trace = public_agent2_trace({
            "task_declaration": {"goal": "read sign", "scope_id": "album1"},
            "task_state": {"requirements": [{"id": "text", "status": "satisfied"}]},
            "evidence_ledger": {"scope_id": "album1", "entries": [
                {"input_refs": ["photo_1"], "provenance_refs": ["asset_private"]},
            ]},
            "planner_decisions": [{"kind": "declare", "status": "accepted"}],
            "terminal_reason": "shadow_only",
            "budget_outcome": {"model_steps": 1},
        })

        self.assertNotIn("scope_id", str(trace))
        self.assertNotIn("asset_private", str(trace))
        self.assertEqual(trace["requirement_status_counts"], {"satisfied": 1})
        self.assertEqual(trace["planner_decisions"][0]["status"], "accepted")


if __name__ == "__main__":
    unittest.main()
