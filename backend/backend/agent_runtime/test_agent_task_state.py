import unittest

from backend.agent_runtime.task_state import (
    EvidenceRequirement,
    TaskDeclaration,
    TaskState,
    UNMET_REASONS,
    TASK_TERMINAL_STATES,
)


class TaskStateTests(unittest.TestCase):
    def test_requirement_transitions_from_open_to_running_to_satisfied(self):
        declaration = TaskDeclaration(
            goal="read the visible text in a selected photo",
            scope_id="album1",
            requirements=(
                EvidenceRequirement(id="sign_text", evidence_type="visible_text"),
            ),
        )
        state = TaskState.from_declaration(declaration)

        state.mark_running("sign_text")
        state.mark_satisfied("sign_text", evidence_refs=("tool_call_1",))

        self.assertEqual(state.requirement("sign_text").status, "satisfied")
        self.assertEqual(state.requirement("sign_text").evidence_refs, ("tool_call_1",))
        self.assertEqual(state.requirement("sign_text").coverage_status, "confirmed")

    def test_requirement_partially_supported_and_unmet_reason(self):
        declaration = TaskDeclaration(
            goal="read the visible text in a selected photo",
            scope_id="album1",
            requirements=(
                EvidenceRequirement(id="sign_text", evidence_type="visible_text"),
                EvidenceRequirement(id="price", evidence_type="visible_text", parent_id="sign_text", lineage_reason="refined"),
            ),
        )
        state = TaskState.from_declaration(declaration)
        state.mark_partially_supported("sign_text", evidence_refs=("call_1",))
        self.assertEqual(state.requirement("sign_text").status, "partially_supported")
        self.assertEqual(state.requirement("sign_text").coverage_status, "supported")

        state.mark_unmet("price", reason="budget_exhausted", status="unresolved")
        self.assertEqual(state.requirement("price").status, "unresolved")
        self.assertEqual(state.requirement("price").unmet_reason, "budget_exhausted")

    def test_failed_evidence_stays_active_for_bounded_recovery(self):
        declaration = TaskDeclaration(
            goal="read visible text",
            scope_id="album1",
            requirements=(EvidenceRequirement(id="text", evidence_type="visible_text"),),
        )
        state = TaskState.from_declaration(declaration)
        state.mark_evidence_failed("text", reason="ocr_failed", evidence_refs=("call_1",))

        requirement = state.requirement("text")
        self.assertEqual(requirement.status, "running")
        self.assertEqual(requirement.coverage_status, "failed")
        self.assertEqual(requirement.failure_reason, "ocr_failed")
        self.assertEqual(requirement.evidence_refs, ("call_1",))

    def test_task_terminal_outcome_decoupled(self):
        declaration = TaskDeclaration(
            goal="find a photo",
            scope_id="album1",
            requirements=(EvidenceRequirement(id="asset", evidence_type="memory_asset"),),
        )
        state = TaskState.from_declaration(declaration)
        state.set_terminal_outcome("unsupported_clarified")
        self.assertEqual(state.terminal_outcome, "unsupported_clarified")

        with self.assertRaisesRegex(ValueError, "unsupported terminal outcome"):
            state.set_terminal_outcome("invalid_outcome")

    def test_rejects_unknown_evidence_type(self):
        with self.assertRaisesRegex(ValueError, "unsupported evidence type"):
            EvidenceRequirement(id="unsupported", evidence_type="question_kind")

    def test_rejects_transition_that_skips_running(self):
        state = TaskState.from_declaration(TaskDeclaration(
            goal="find a photo",
            scope_id="album1",
            requirements=(EvidenceRequirement(id="asset", evidence_type="memory_asset"),),
        ))

        # Directly calling _transition from open to satisfied without running should fail
        with self.assertRaisesRegex(ValueError, "invalid transition"):
            state._transition("asset", "satisfied")

    def test_round_trips_declaration_and_requirement_state(self):
        declaration = TaskDeclaration(
            goal="find a family photo",
            scope_id="album1",
            constraints={"time": "2025"},
            requirements=(
                EvidenceRequirement(id="asset", evidence_type="memory_asset", parent_id="p1", lineage_reason="root"),
            ),
        )
        state = TaskState.from_declaration(declaration)
        state.mark_running("asset")
        state.set_terminal_outcome("answered")

        restored = TaskState.from_dict(state.as_dict())

        self.assertEqual(restored.as_dict(), state.as_dict())


if __name__ == "__main__":
    unittest.main()
