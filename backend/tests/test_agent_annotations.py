import os
import sqlite3
import tempfile
import unittest

from backend.agent_annotations import AnnotationStore
from backend.agent import MemoryAgent
from backend.db import MemoryStore


class AgentAnnotationStoreTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.database = f"{self.temp_dir.name}/sentrix.db"
        self.store = MemoryStore(self.database)

    def tearDown(self):
        self.store.close()
        self.temp_dir.cleanup()

    def test_versioned_migration_is_idempotent_and_preserves_canonical_tables(self):
        canonical_before = set(row["name"] for row in self.store.connection.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table'"
        ))

        annotations = AnnotationStore(self.store.connection)
        again = AnnotationStore(self.store.connection)

        self.assertTrue(annotations.available)
        self.assertTrue(again.available)
        self.assertEqual(
            self.store.connection.execute("SELECT COUNT(*) FROM agent_schema_migrations").fetchone()[0],
            1,
        )
        canonical_after = set(row["name"] for row in self.store.connection.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table'"
        ))
        self.assertEqual(canonical_before, canonical_after - {
            "agent_schema_migrations", "agent_user_assertions", "agent_impressions",
            "agent_proactivity_preferences", "agent_scene_cooldowns", "agent_claim_conflicts",
            "agent_annotation_visibility",
        })

    def test_disabled_feature_does_not_create_agent_tables(self):
        connection = sqlite3.connect(":memory:")
        try:
            annotations = AnnotationStore(connection, enabled=False)
            self.assertFalse(annotations.available)
            self.assertEqual(
                connection.execute("SELECT COUNT(*) FROM sqlite_master WHERE name LIKE 'agent_%'").fetchone()[0],
                0,
            )
        finally:
            connection.close()

    def test_assertion_is_idempotent_and_never_writes_canonical_fact(self):
        person = self.store.create_entity("明哥", "person", "confirmed", "孩子", 1.0)
        annotations = AnnotationStore(self.store.connection)
        before = self.store.connection.execute("SELECT COUNT(*) FROM semantic_claims").fetchone()[0]

        first = annotations.record_user_assertion(
            scope_id="home-default", actor_id="viewer-1", viewer_id="viewer-1",
            conversation_id="conversation-1", assertion_text="那次是在姥姥家",
            subject_entity_id=person["id"], idempotency_key="request-1",
        )
        second = annotations.record_user_assertion(
            scope_id="home-default", actor_id="viewer-1", viewer_id="viewer-1",
            conversation_id="conversation-1", assertion_text="那次是在姥姥家",
            subject_entity_id=person["id"], idempotency_key="request-1",
        )

        self.assertEqual(first["id"], second["id"])
        self.assertEqual(self.store.connection.execute("SELECT COUNT(*) FROM agent_user_assertions").fetchone()[0], 1)
        self.assertEqual(self.store.connection.execute("SELECT COUNT(*) FROM semantic_claims").fetchone()[0], before)
        self.assertEqual(first["status"], "pending")

    def test_orphaned_assertion_is_kept_for_audit(self):
        annotations = AnnotationStore(self.store.connection)

        assertion = annotations.record_user_assertion(
            scope_id="home-default", actor_id="viewer-1", viewer_id="viewer-1",
            conversation_id="conversation-1", assertion_text="这不是原来的地点",
            subject_entity_id="deleted-entity", idempotency_key="request-orphan",
        )

        self.assertEqual(assertion["status"], "orphaned")
        self.assertEqual(self.store.connection.execute("SELECT COUNT(*) FROM agent_user_assertions").fetchone()[0], 1)

    def test_preference_and_scene_cooldown_use_upsert_keys(self):
        annotations = AnnotationStore(self.store.connection)

        annotations.upsert_preference("home-default", "viewer-1", enabled=True, level=2)
        annotations.upsert_preference("home-default", "viewer-1", enabled=False, level=0)
        first = annotations.upsert_scene_cooldown("home-default", "viewer-1", "scene-1", "2026-08-04", "2026-08-11", "ignored")
        second = annotations.upsert_scene_cooldown("home-default", "viewer-1", "scene-1", "2026-08-04", "2026-09-04", "accepted")

        self.assertEqual(self.store.connection.execute("SELECT COUNT(*) FROM agent_proactivity_preferences").fetchone()[0], 1)
        self.assertFalse(self.store.connection.execute(
            "SELECT enabled FROM agent_proactivity_preferences WHERE scope_id = ? AND viewer_id = ?",
            ("home-default", "viewer-1"),
        ).fetchone()[0])
        self.assertEqual(first["id"], second["id"])
        self.assertEqual(self.store.connection.execute("SELECT COUNT(*) FROM agent_scene_cooldowns").fetchone()[0], 1)
        self.assertEqual(second["outcome"], "accepted")

    def test_checksum_mismatch_disables_store_without_raising(self):
        connection = sqlite3.connect(":memory:")
        connection.execute("CREATE TABLE agent_schema_migrations (version INTEGER PRIMARY KEY, applied_at TEXT, checksum TEXT)")
        connection.execute("INSERT INTO agent_schema_migrations VALUES (1, '2026-08-04', 'wrong')")
        connection.commit()
        try:
            annotations = AnnotationStore(connection)
            self.assertFalse(annotations.available)
            self.assertIsNotNone(annotations.error)
        finally:
            connection.close()

    def test_memory_agent_owns_annotation_store_without_changing_memory_store_api(self):
        agent = MemoryAgent(self.store, gamma=object(), clip=object())

        self.assertTrue(agent.annotation_store.available)
        self.assertIs(agent.annotation_store.connection, self.store.connection)


if __name__ == "__main__":
    unittest.main()
