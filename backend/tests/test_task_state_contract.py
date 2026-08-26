import unittest

from backend.agent_runtime.evidence_ledger import EvidenceLedger, LedgerEntry
from backend.agent_runtime.requirement_completion import RequirementCompletion
from backend.agent_runtime.task_state import EvidenceRequirement, TaskDeclaration, TaskState


class TaskStateContractTests(unittest.TestCase):
    def _state(self):
        return TaskState.from_declaration(TaskDeclaration(
            goal="count people in a photo", scope_id="album1",
            requirements=(EvidenceRequirement("people", "visual_observation", "人数"),),
        ))

    def test_unbound_same_type_evidence_cannot_satisfy_requirement(self):
        state = self._state()
        ledger = EvidenceLedger(scope_id="album1")
        ledger.append(LedgerEntry(
            "call_1", "inspect_photo", "visual_observation", ("photo_1",),
            ("photo_1",), extracted_value="红色雨伞", asset_id="photo_1",
            provenance_scope_id="album1",
        ))
        self.assertEqual(RequirementCompletion(state, ledger)
                         .auto_match_all_open_requirements(), 0)
        self.assertEqual(state.requirement("people").status, "open")

    def test_bound_evidence_is_the_only_path_to_complete(self):
        state = self._state()
        ledger = EvidenceLedger(scope_id="album1")
        ledger.append(LedgerEntry(
            "call_1", "inspect_photo", "visual_observation", ("photo_1",),
            ("photo_1",), extracted_value="照片中有 3 个人", asset_id="photo_1",
            requirement_refs=("people",), provenance_scope_id="album1",
        ))
        self.assertEqual(RequirementCompletion(state, ledger)
                         .auto_match_all_open_requirements(), 1)
        self.assertEqual(state.requirement("people").status, "satisfied")

    def test_unavailable_is_terminal_requirement_state(self):
        state = self._state()
        state.mark_unavailable("people")
        self.assertEqual(state.requirement("people").status, "unavailable")
        self.assertEqual(state.recompute_status(has_available_tools=False), "insufficient_evidence")


if __name__ == "__main__":
    unittest.main()
