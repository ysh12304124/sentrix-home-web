import unittest

from backend.agent_runtime.evidence_ledger import EvidenceLedger
from backend.agent_runtime.task_state import EvidenceRequirement, TaskDeclaration, TaskState
from backend.agent_runtime.tool_registry import ToolSpec
from backend.agent_runtime.runtime import record_agent2_tool_evidence


def _execute(arguments, *, context=None):
    return {}


class Agent2EvidenceRecordingTests(unittest.TestCase):
    def test_records_matching_tool_evidence_and_satisfies_requirement(self):
        task = TaskState.from_declaration(TaskDeclaration(
            goal="read sign", scope_id="album1",
            requirements=(EvidenceRequirement(id="text", evidence_type="visible_text"),),
        ))
        ledger = EvidenceLedger(scope_id="album1")
        spec = ToolSpec(name="read_photo_text", description="ocr", input_schema={},
                        executor=_execute, produces_evidence=("visible_text",))

        recorded = record_agent2_tool_evidence(
            task, ledger, spec, tool_call_id="tool_call_1", input_refs=("photo_1",))

        self.assertTrue(recorded)
        self.assertEqual(task.requirement("text").status, "satisfied")
        self.assertEqual(ledger.entries[0].evidence_type, "visible_text")

    def test_ignores_incompatible_capability_without_changing_requirement(self):
        task = TaskState.from_declaration(TaskDeclaration(
            goal="read sign", scope_id="album1",
            requirements=(EvidenceRequirement(id="text", evidence_type="visible_text"),),
        ))
        ledger = EvidenceLedger(scope_id="album1")
        spec = ToolSpec(name="inspect_photo", description="visual", input_schema={},
                        executor=_execute, produces_evidence=("visual_observation",))

        self.assertFalse(record_agent2_tool_evidence(
            task, ledger, spec, tool_call_id="tool_call_1", input_refs=("photo_1",)))
        self.assertEqual(task.requirement("text").status, "open")
        self.assertEqual(ledger.entries, [])


if __name__ == "__main__":
    unittest.main()
