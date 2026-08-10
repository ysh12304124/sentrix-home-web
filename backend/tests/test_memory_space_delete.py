import os
import tempfile
import unittest
from pathlib import Path

from backend.db import MemoryStore


class DeleteMemorySpaceTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = f"{self.temp_dir.name}/sentrix.db"
        self.media_dir = Path(self.temp_dir.name) / "media"
        self.media_dir.mkdir(parents=True, exist_ok=True)
        self.store = MemoryStore(self.db_path)
        self.store.create_memory_space("scope-a", "测试相册 A", kind="benchmark")

    def tearDown(self):
        self.store.close()
        self.temp_dir.cleanup()

    def _seed_scope(self, scope_id, suffix="a"):
        """Populate a scope with asset+observation+event+face+entity+vector data."""
        img_path = self.media_dir / f"asset_{scope_id}_{suffix}.jpg"
        img_path.write_bytes(b"jpeg-bytes-" + suffix.encode())
        sha = f"sha-{scope_id}-{suffix}"
        self.store.create_asset(
            f"asset-{scope_id}-{suffix}", f"pic-{suffix}.jpg", "image",
            str(img_path), "image/jpeg",
            metadata={"content_sha256": sha, "captured_at": "2025-04-01T10:00:00Z"},
            scope_id=scope_id,
        )
        obs = self.store.add_observation(
            f"asset-{scope_id}-{suffix}",
            {"caption": "客厅家人", "people": [], "activity": "聚餐", "event_type": "gathering"},
            scope_id=scope_id,
        )
        face = self.store.add_face_instance(
            f"asset-{scope_id}-{suffix}", obs["id"],
            {"bbox": [1, 2, 3, 4], "confidence": 0.9, "embedding": [1, 0, 0]},
        )
        # trigger event merge (needs the observation dict, not just id)
        self.store.merge_observation_into_event(obs)
        return img_path, sha, obs["id"], face["id"]

    def test_delete_home_default_forbidden(self):
        with self.assertRaises(ValueError):
            self.store.delete_memory_space("home-default")

    def test_delete_empty_scope_removes_space(self):
        stats = self.store.delete_memory_space("scope-a")
        self.assertEqual(stats["assets"], 0)
        self.assertIsNone(self.store._row("SELECT id FROM memory_spaces WHERE id = ?", ("scope-a",)))

    def test_delete_removes_all_scope_rows(self):
        img_path, sha, obs_id, face_id = self._seed_scope("scope-a", "x")
        stats = self.store.delete_memory_space("scope-a")

        self.assertGreaterEqual(stats["assets"], 1)
        scoped_tables = [
            "assets", "observations", "events", "memory_vectors",
            "entities", "face_clusters", "ingest_batches", "relationships",
            "facts", "semantic_profiles", "semantic_claims",
            "person_event_memory", "person_patterns", "query_gaps",
            "dialogue_states", "trips", "entity_merge_candidates",
        ]
        for tbl in scoped_tables:
            row = self.store._row(f"SELECT COUNT(*) AS c FROM {tbl} WHERE scope_id = ?", ("scope-a",))
            self.assertEqual(row["c"], 0, f"table {tbl} still has scope-a rows")

        # rebuild_runs uses `scope` field
        row = self.store._row("SELECT COUNT(*) AS c FROM rebuild_runs WHERE scope = ?", ("scope-a",))
        self.assertEqual(row["c"], 0)

        # JOIN-dependent tables (via observations/events/entities/face_clusters)
        # After scope delete, observations table is empty for that scope, so these
        # should also be empty for anything referencing the deleted rows
        self.assertIsNone(self.store._row("SELECT id FROM face_instances WHERE id = ?", (face_id,)))
        self.assertIsNone(self.store._row("SELECT id FROM entity_mentions WHERE observation_id = ?", (obs_id,)))
        self.assertIsNone(self.store._row("SELECT observation_id FROM event_observations WHERE observation_id = ?", (obs_id,)))
        self.assertIsNone(self.store._row("SELECT id FROM memory_spaces WHERE id = ?", ("scope-a",)))

        # Physical file removed (sha not referenced anymore)
        self.assertFalse(img_path.exists(), "physical file should be removed")
        self.assertEqual(stats["files_removed"], 1)

    def test_delete_preserves_other_scopes(self):
        self.store.create_memory_space("scope-b", "测试相册 B", kind="benchmark")
        img_a, sha_a, obs_a, _ = self._seed_scope("scope-a", "a")
        img_b, sha_b, obs_b, _ = self._seed_scope("scope-b", "b")

        self.store.delete_memory_space("scope-a")

        # scope-b intact
        self.assertIsNotNone(self.store._row("SELECT id FROM memory_spaces WHERE id = ?", ("scope-b",)))
        self.assertIsNotNone(self.store._row("SELECT id FROM observations WHERE id = ?", (obs_b,)))
        self.assertTrue(img_b.exists())

        # scope-a gone
        self.assertIsNone(self.store._row("SELECT id FROM observations WHERE id = ?", (obs_a,)))
        self.assertFalse(img_a.exists())

    def test_delete_preserves_shared_sha256_file(self):
        """Two scopes reference the same physical file (same sha256).
        Deleting one scope must not remove the file."""
        self.store.create_memory_space("scope-b", "测试相册 B", kind="benchmark")
        # Both scopes share the same physical file and sha256
        img_path = self.media_dir / "shared.jpg"
        img_path.write_bytes(b"shared-bytes")
        shared_sha = "shared-sha-256"
        self.store.create_asset(
            "asset-shared-a", "shared.jpg", "image", str(img_path), "image/jpeg",
            metadata={"content_sha256": shared_sha}, scope_id="scope-a",
        )
        self.store.create_asset(
            "asset-shared-b", "shared.jpg", "image", str(img_path), "image/jpeg",
            metadata={"content_sha256": shared_sha}, scope_id="scope-b",
        )

        stats = self.store.delete_memory_space("scope-a")

        # File must survive because scope-b still references the same sha
        self.assertTrue(img_path.exists(), "shared file must be preserved")
        self.assertEqual(stats.get("files_removed", 0), 0)
        self.assertIsNotNone(self.store._row("SELECT id FROM assets WHERE id = ?", ("asset-shared-b",)))


if __name__ == "__main__":
    unittest.main()
