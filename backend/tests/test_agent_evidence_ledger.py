import unittest

from backend.agent_runtime.evidence_ledger import Coverage, EvidenceLedger, LedgerEntry


class EvidenceLedgerTests(unittest.TestCase):
    def test_records_typed_evidence_with_partial_coverage(self):
        ledger = EvidenceLedger(scope_id="album1")
        ledger.append(LedgerEntry(
            tool_call_id="tool_call_1",
            capability="read_photo_text",
            evidence_type="visible_text",
            input_refs=("photo_1",),
            provenance_refs=("asset_1",),
            certainty="confirmed",
            coverage=Coverage(requested=2, processed=1, skipped_budget=1),
            asset_id="asset_1",
            subject="menu_price",
            extracted_value="188",
        ))

        entry = ledger.entries[0]
        self.assertTrue(entry.coverage.is_partial)
        self.assertEqual(entry.asset_id, "asset_1")
        self.assertEqual(entry.subject, "menu_price")
        self.assertEqual(entry.extracted_value, "188")
        self.assertEqual(entry.coverage.as_dict(), {
            "requested": 2,
            "processed": 1,
            "skipped_budget": 1,
            "failed": 0,
        })

    def test_rejects_duplicate_tool_call_id(self):
        ledger = EvidenceLedger(scope_id="album1")
        entry = LedgerEntry(
            tool_call_id="tool_call_1",
            capability="search_memories",
            evidence_type="memory_asset",
            input_refs=(),
            provenance_refs=("asset_1",),
        )
        ledger.append(entry)

        with self.assertRaisesRegex(ValueError, "duplicate tool call"):
            ledger.append(entry)

    def test_rejects_provenance_from_another_memory_space(self):
        ledger = EvidenceLedger(scope_id="album1")

        with self.assertRaisesRegex(ValueError, "scope mismatch"):
            ledger.append(LedgerEntry(
                tool_call_id="tool_call_1",
                capability="search_memories",
                evidence_type="memory_asset",
                input_refs=(),
                provenance_refs=("asset_1",),
                provenance_scope_id="album2",
            ))

    def test_round_trips_entries(self):
        ledger = EvidenceLedger(scope_id="album1")
        ledger.append(LedgerEntry(
            tool_call_id="tool_call_1",
            capability="inspect_photo",
            evidence_type="visual_observation",
            input_refs=("photo_1",),
            provenance_refs=("asset_1",),
            certainty="uncertain",
            failure_reason="partial_visibility",
            asset_id="photo_1",
            region_bbox=(0.1, 0.2, 0.5, 0.6),
        ))

        restored = EvidenceLedger.from_dict(ledger.as_dict())

        self.assertEqual(restored.as_dict(), ledger.as_dict())

    def test_build_answer_context_keeps_bound_facts_and_unknowns(self):
        from backend.agent_runtime.task_state import EvidenceRequirement, TaskDeclaration, TaskState

        task = TaskState.from_declaration(TaskDeclaration(
            goal="find the place and sign",
            scope_id="album1",
            requirements=(
                EvidenceRequirement(id="place", evidence_type="location_metadata", description="place"),
                EvidenceRequirement(id="sign", evidence_type="visible_text", description="sign"),
            ),
        ))
        ledger = EvidenceLedger(scope_id="album1")
        ledger.append(LedgerEntry(
            tool_call_id="call_search",
            capability="search_memories",
            evidence_type="location_metadata",
            input_refs=("photo_1",),
            provenance_refs=("asset_1",),
            extracted_value="秦皇岛如是海度假村",
            confidence=0.96,
            requirement_refs=("place",),
            provenance_scope_id="album1",
        ))

        context = ledger.build_answer_context("where and what sign", task)

        self.assertEqual(context["facts"][0]["value"], "秦皇岛如是海度假村")
        self.assertEqual(context["facts"][0]["requirement_refs"], ["place"])
        self.assertEqual(context["unknowns"][0]["requirement_id"], "sign")
        self.assertEqual(context["conflicts"], [])

    def test_build_answer_context_preserves_same_asset_conflict(self):
        ledger = EvidenceLedger(scope_id="album1")
        for call_id, value in (("call_1", "ABC"), ("call_2", "ABD")):
            ledger.append(LedgerEntry(
                tool_call_id=call_id,
                capability="read_photo_text",
                evidence_type="visible_text",
                input_refs=("photo_1",),
                provenance_refs=("photo_1",),
                subject="sign",
                asset_id="photo_1",
                extracted_value=value,
                requirement_refs=("sign",),
                provenance_scope_id="album1",
            ))

        context = ledger.build_answer_context("what does the sign say", {
            "requirements": [{"id": "sign", "evidence_type": "visible_text", "status": "open"}],
        })

        self.assertEqual(len(context["conflicts"]), 1)
        self.assertEqual(set(context["conflicts"][0]["values"]), {"ABC", "ABD"})


if __name__ == "__main__":
    unittest.main()
