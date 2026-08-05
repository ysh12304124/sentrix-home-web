"""Phase 6 memory correction tests."""

import os
import tempfile
import time
import unittest

from backend.db import MemoryStore
from backend.memory_corrections import MemoryCorrections, MemoryCorrectionError


class MemoryCorrectionsTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix="corrections-")
        self.store = MemoryStore(os.path.join(self.directory.name, "memory.db"))
        self.entity = self.store.create_entity("明哥", "person", "confirmed", scope_id="home")
        self.corrections = MemoryCorrections(self.store)

    def tearDown(self):
        self.store.close()
        self.directory.cleanup()

    def _propose(self, request_id=None):
        return self.corrections.propose(
            scope_id="home", actor="owner",
            target_type="entity", target_id=self.entity["id"],
            changed_fields={"canonical_name": "明先生"},
            evidence_ids=["obs-1"], request_id=request_id,
        )

    def test_propose_stores_pending_row_without_writing_canonical(self):
        proposal = self._propose()
        # Canonical remains untouched — canonical_name unchanged.
        entity = self.store.get_entity(self.entity["id"])
        self.assertEqual(entity["canonical_name"], "明哥")
        # Proposal recorded with a confirmation token and TTL.
        self.assertTrue(proposal["confirmation_token"])
        self.assertIn(":", proposal["expires_at"])
        # Previous revision is the logical baseline (1) — no canonical writes.
        self.assertEqual(proposal["previous_revision"], 1)

    def test_apply_requires_matching_confirmation_token(self):
        proposal = self._propose()
        with self.assertRaises(MemoryCorrectionError):
            self.corrections.apply(proposal_id=proposal["proposal_id"],
                                    confirmation_token="wrong-token", actor="owner")

    def test_store_apply_authorized_revision_matches_service(self):
        """The MemoryStore wrapper must be equivalent to calling the service."""
        proposal = self.store.propose_memory_correction(
            scope_id="home", actor="owner",
            target_type="entity", target_id=self.entity["id"],
            changed_fields={"canonical_name": "明先生"},
        )
        result = self.store.apply_authorized_revision(
            proposal_id=proposal["proposal_id"],
            confirmation_token=proposal["confirmation_token"],
            actor="owner",
        )
        self.assertEqual(result["new_revision"], 2)
        # Canonical row still untouched.
        self.assertEqual(self.store.get_entity(self.entity["id"])["canonical_name"], "明哥")

    def test_apply_bumps_revision_and_preserves_canonical(self):
        entity_before = self.store.get_entity(self.entity["id"])
        proposal = self._propose()
        result = self.corrections.apply(
            proposal_id=proposal["proposal_id"],
            confirmation_token=proposal["confirmation_token"],
            actor="owner",
        )
        after = self.store.get_entity(self.entity["id"])
        # Canonical row is byte-for-byte preserved — corrections live in the
        # agent-owned revisions log only.
        self.assertEqual(after["canonical_name"], entity_before["canonical_name"])
        self.assertEqual(after.get("raw_json") or "{}", entity_before.get("raw_json") or "{}")
        self.assertEqual(result["new_revision"], 2)

    def test_request_id_is_idempotent(self):
        first = self._propose(request_id="req-1")
        second = self._propose(request_id="req-1")
        self.assertEqual(first["proposal_id"], second["proposal_id"])

    def test_apply_is_idempotent_after_success(self):
        proposal = self._propose()
        applied_once = self.corrections.apply(
            proposal_id=proposal["proposal_id"],
            confirmation_token=proposal["confirmation_token"],
            actor="owner",
        )
        applied_again = self.corrections.apply(
            proposal_id=proposal["proposal_id"],
            confirmation_token=proposal["confirmation_token"],
            actor="owner",
        )
        self.assertEqual(applied_once["new_revision"], applied_again["new_revision"])

    def test_expired_token_is_rejected(self):
        corrections = MemoryCorrections(self.store, token_ttl_seconds=0)
        proposal = corrections.propose(
            scope_id="home", actor="owner",
            target_type="entity", target_id=self.entity["id"],
            changed_fields={"canonical_name": "临时"},
        )
        time.sleep(1)
        with self.assertRaises(MemoryCorrectionError):
            corrections.apply(
                proposal_id=proposal["proposal_id"],
                confirmation_token=proposal["confirmation_token"],
                actor="owner",
            )


if __name__ == "__main__":
    unittest.main()
