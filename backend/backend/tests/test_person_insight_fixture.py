import os
import tempfile
import unittest
from unittest import mock

from backend.db import MemoryStore
from scripts.benchmarks.person_insight_fixture import (
    backup_sqlite,
    build_missing_events,
    ensure_sqlite_backend,
)


class PersonInsightFixtureTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.source_path = f"{self.temp_dir.name}/source.db"
        self.store = MemoryStore(self.source_path)
        self.store.create_memory_space("album-a", "相册 A")
        self.person = self._seed_source()

    def tearDown(self):
        self.store.close()
        self.temp_dir.cleanup()

    def _seed_source(self):
        asset = self.store.create_asset(
            "a1", "one.jpg", "image", "/tmp/one.jpg", "image/jpeg", scope_id="album-a"
        )
        obs = self.store.add_observation(
            "a1", {"caption": "家人合影", "people": []}, scope_id="album-a"
        )
        self.store.add_face_instance(
            "a1", obs["id"],
            {"bbox": [1, 2, 3, 4], "confidence": 0.95, "quality": 0.9, "embedding": [1, 0, 0]},
        )
        self.store.merge_observation_into_event(obs)
        return self.store.create_entity(
            "爸爸", "person", "confirmed", family_role="父亲", scope_id="album-a"
        )

    def test_backup_isolates_source_writes(self):
        before_assets = self.store.count("assets")
        before_events = self.store.count("events")
        before_vectors = self.store.count("memory_vectors")
        before_name = self.store.get_entity(self.person["id"])["canonical_name"]
        before_role = self.store.get_entity(self.person["id"])["family_role"]
        self.store.close()

        work_path = backup_sqlite(self.source_path, f"{self.temp_dir.name}/work.db")
        work = MemoryStore(work_path)
        work.update_person_identity_state(self.person["id"], role="母亲")
        work.create_asset(
            "a2", "two.jpg", "image", "/tmp/two.jpg", "image/jpeg", scope_id="album-a"
        )
        work.close()

        self.store = MemoryStore(self.source_path)
        self.assertEqual(self.store.count("assets"), before_assets)
        self.assertEqual(self.store.count("events"), before_events)
        self.assertEqual(self.store.count("memory_vectors"), before_vectors)
        person = self.store.get_entity(self.person["id"])
        self.assertEqual(person["canonical_name"], before_name)
        self.assertEqual(person["family_role"], before_role)
        self.assertEqual(person["family_role"], "父亲")

    def test_backup_preserves_integrity_and_content(self):
        self.store.close()
        work_path = backup_sqlite(self.source_path, f"{self.temp_dir.name}/work.db")
        reopened = MemoryStore(work_path)
        self.assertEqual(reopened.count("assets"), 1)
        self.assertEqual(reopened.count("memory_vectors"), 1)
        reopened.close()

    def test_script_rejects_non_sqlite_vector_backend(self):
        with mock.patch.dict(os.environ, {"SENTRIX_VECTOR_BACKEND": "qdrant"}, clear=False):
            with self.assertRaisesRegex(RuntimeError, "SQLite vector backend"):
                ensure_sqlite_backend()

    def test_script_accepts_default_sqlite_backend(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            ensure_sqlite_backend()

    def test_build_missing_events_fails_when_events_exist(self):
        with self.assertRaisesRegex(RuntimeError, "already has .+ events"):
            build_missing_events(self.store, "album-a")

    def test_build_missing_events_builds_when_empty(self):
        empty = MemoryStore(f"{self.temp_dir.name}/empty.db")
        empty.create_memory_space("album-empty", "空相册")
        empty.create_asset(
            "e1", "e1.jpg", "image", "/tmp/e1.jpg", "image/jpeg", scope_id="album-empty"
        )
        obs = empty.add_observation(
            "e1", {"caption": "空相册里的照片", "captured_at": "2026-01-01T10:00:00"},
            scope_id="album-empty",
        )
        result = build_missing_events(empty, "album-empty")
        self.assertGreater(result["events"], 0)
        empty.close()


if __name__ == "__main__":
    unittest.main()
