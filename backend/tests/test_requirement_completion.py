import unittest

from backend.agent_runtime.evidence_ledger import Coverage, EvidenceLedger, LedgerEntry
from backend.agent_runtime.requirement_completion import RequirementCompletion
from backend.agent_runtime.task_state import (
    EvidenceRequirement,
    TaskDeclaration,
    TaskState,
)
from backend.agent_runtime.tool_registry import ToolSpec


class RequirementCompletionTests(unittest.TestCase):
    def test_returns_capabilities_that_match_open_requirement_evidence(self):
        state = TaskState.from_declaration(TaskDeclaration(
            goal="read a menu price",
            scope_id="album1",
            requirements=(
                EvidenceRequirement(id="price", evidence_type="visible_text"),
            ),
        ))
        ledger = EvidenceLedger(scope_id="album1")
        completion = RequirementCompletion(state, ledger)
        specs = (
            ToolSpec(
                name="search_memories",
                description="search",
                input_schema={},
                executor=lambda **_: {},
                produces_evidence=("memory_asset",),
            ),
            ToolSpec(
                name="read_photo_text",
                description="ocr",
                input_schema={},
                executor=lambda **_: {},
                produces_evidence=("visible_text",),
            ),
        )

        allowed = completion.allowed_capabilities(specs)

        self.assertEqual([spec.name for spec in allowed], ["read_photo_text"])

    def test_marks_requirement_satisfied_only_with_compatible_ledger_evidence(self):
        state = TaskState.from_declaration(TaskDeclaration(
            goal="read a menu price",
            scope_id="album1",
            requirements=(
                EvidenceRequirement(id="price", evidence_type="visible_text"),
            ),
        ))
        ledger = EvidenceLedger(scope_id="album1")
        ledger.append(LedgerEntry(
            tool_call_id="call_ocr_1",
            capability="read_photo_text",
            evidence_type="visible_text",
            input_refs=("photo_1",),
            provenance_refs=("asset_1",),
            asset_id="photo_1",
            subject="steak_price",
            requirement_refs=("price",),
        ))
        completion = RequirementCompletion(state, ledger)
        state.mark_running("price")

        # Wrong asset filter should not satisfy
        satisfied_wrong_asset = completion.satisfy_from_entry("price", "call_ocr_1", asset_id="photo_2")
        self.assertFalse(satisfied_wrong_asset)

        # Correct asset filter satisfies
        satisfied = completion.satisfy_from_entry("price", "call_ocr_1", asset_id="photo_1")

        self.assertTrue(satisfied)
        self.assertEqual(state.requirement("price").status, "satisfied")
        self.assertEqual(state.requirement("price").evidence_refs, ("call_ocr_1",))

    def test_many_to_many_and_partial_matching(self):
        state = TaskState.from_declaration(TaskDeclaration(
            goal="find steak restaurant price and photo",
            scope_id="album1",
            requirements=(
                EvidenceRequirement(id="photo_req", evidence_type="memory_asset"),
                EvidenceRequirement(id="price_req", evidence_type="visible_text"),
            ),
        ))
        ledger = EvidenceLedger(scope_id="album1")
        ledger.append(LedgerEntry(
            tool_call_id="call_search",
            capability="search_memories",
            evidence_type="memory_asset",
            input_refs=("steak",),
            provenance_refs=("photo_1",),
            asset_id="photo_1",
            requirement_refs=("photo_req",),
        ))
        ledger.append(LedgerEntry(
            tool_call_id="call_ocr",
            capability="read_photo_text",
            evidence_type="visible_text",
            input_refs=("photo_1",),
            provenance_refs=("photo_1",),
            coverage=Coverage(requested=2, processed=1, skipped_budget=1),
            asset_id="photo_1",
            requirement_refs=("price_req",),
        ))

        completion = RequirementCompletion(state, ledger)
        matched = completion.auto_match_all_open_requirements()
        self.assertEqual(matched, 2)
        self.assertEqual(state.requirement("photo_req").status, "satisfied")
        self.assertEqual(state.requirement("price_req").status, "partially_supported")


if __name__ == "__main__":
    unittest.main()
