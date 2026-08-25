import unittest

from backend.agent_runtime.evidence_contract import PUBLIC_EVIDENCE_TYPES
from backend.agent_runtime.evidence_ledger import EvidenceLedger, LedgerEntry
from backend.agent_runtime.task_state import EvidenceRequirement


class EvidenceContractTests(unittest.TestCase):
    def test_public_types_are_frozen_and_memory_reference_is_internal_only(self):
        self.assertEqual(len(PUBLIC_EVIDENCE_TYPES), 9)
        self.assertNotIn("memory_reference", PUBLIC_EVIDENCE_TYPES)
        with self.assertRaises(ValueError):
            EvidenceRequirement("bad", "memory_reference")
        with self.assertRaises(ValueError):
            LedgerEntry("call", "search_memories", "memory_reference", (), ())

    def test_ledger_rejects_cross_scope_and_duplicate_evidence(self):
        ledger = EvidenceLedger(scope_id="album1")
        entry = LedgerEntry(
            "call_1", "search_memories", "memory_asset", (), (),
            provenance_scope_id="album1", requirement_refs=("asset",),
        )
        ledger.append(entry)
        with self.assertRaisesRegex(ValueError, "duplicate"):
            ledger.append(entry)
        with self.assertRaisesRegex(ValueError, "scope"):
            ledger.append(LedgerEntry(
                "call_2", "search_memories", "memory_asset", (), (),
                provenance_scope_id="album2", requirement_refs=("asset",),
            ))


if __name__ == "__main__":
    unittest.main()
