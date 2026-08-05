"""Phase 5 Core Memory Card lifecycle tests."""

import os
import tempfile
import unittest
from datetime import datetime, timedelta

from backend.db import MemoryStore
from backend.core_memory import CoreMemoryStore


class CoreMemoryTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix="core-memory-")
        self.store = MemoryStore(os.path.join(self.directory.name, "memory.db"))
        self.cms = CoreMemoryStore(self.store)

    def tearDown(self):
        self.store.close()
        self.directory.cleanup()

    def _card(self, subject_id="entity-ming", display="明哥"):
        return self.cms.upsert_card(scope_id="home", subject_type="person",
                                     subject_id=subject_id, display_name=display)

    def _item(self, card_id, **overrides):
        return self.cms.upsert_item(
            card_id=card_id,
            text=overrides.get("text", "明哥是家人"),
            epistemic_type=overrides.get("epistemic_type", "confirmed_fact"),
            source_type=overrides.get("source_type", "entity"),
            source_ids=overrides.get("source_ids", ["entity-ming"]),
            source_revisions=overrides.get("source_revisions", {"entity-ming": 1}),
        )

    def test_upsert_card_is_idempotent_per_scope_subject(self):
        first = self._card()
        second = self._card()
        self.assertEqual(first, second)

    def test_list_cards_omits_cards_without_active_items(self):
        empty = self._card(subject_id="entity-empty", display="空卡片")
        active_card = self._card(subject_id="entity-active", display="活卡片")
        self._item(active_card)
        cards = self.cms.list_cards(scope_id="home")
        subject_ids = [card["subject_id"] for card in cards]
        self.assertIn("entity-active", subject_ids)
        self.assertNotIn("entity-empty", subject_ids)

    def test_promotion_needs_three_queries_from_two_conversations(self):
        card = self._card()
        self._item(card)
        # Two queries in the same conversation — do not promote.
        for _ in range(2):
            self.cms.record_access(card_id=card, conversation_id="conv-A", viewer_id="owner")
        cards = self.cms.list_cards(scope_id="home")
        self.assertEqual(cards[0]["priority"], 0.0)
        # Third query from a distinct conversation reaches the threshold.
        self.cms.record_access(card_id=card, conversation_id="conv-B", viewer_id="owner")
        cards = self.cms.list_cards(scope_id="home")
        self.assertEqual(cards[0]["priority"], 1.0)

    def test_invalidation_only_targets_matching_item(self):
        card = self._card()
        target_item = self._item(card, text="旧事实", source_ids=["entity-ming"], source_revisions={"entity-ming": 1})
        keeper_item = self._item(card, text="其它事实", source_ids=["entity-other"], source_revisions={"entity-other": 1})
        self.cms.invalidate_by_source_revision("entity", "entity-ming", revision=2)
        items = self.cms.list_cards(scope_id="home")[0]["items"]
        item_ids = [item["item_id"] for item in items]
        self.assertNotIn(target_item, item_ids)
        self.assertIn(keeper_item, item_ids)

    def test_epistemic_type_is_validated(self):
        card = self._card()
        with self.assertRaises(ValueError):
            self._item(card, epistemic_type="opinion")

    def test_demote_stale_cards_zeros_priority_beyond_threshold(self):
        card = self._card()
        self._item(card)
        # Backdate the access timestamp.
        cutoff = (datetime.utcnow() - timedelta(days=120)).isoformat(timespec="seconds")
        self.store.connection.execute(
            "UPDATE agent_core_memory_cards SET priority = 1.0, last_accessed_at = ? WHERE card_id = ?",
            (cutoff, card),
        )
        self.store.connection.commit()
        self.cms.demote_stale_cards(days_threshold=90)
        cards = self.cms.list_cards(scope_id="home")
        self.assertEqual(cards[0]["priority"], 0.0)


if __name__ == "__main__":
    unittest.main()
