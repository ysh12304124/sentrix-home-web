import os
import tempfile
import unittest

_APP_TMP = tempfile.mkdtemp(prefix="sentrix-family-graph-api-")
os.environ["SENTRIX_DB_PATH"] = os.path.join(_APP_TMP, "sentrix.db")
os.environ["SENTRIX_DATA_DIR"] = os.path.join(_APP_TMP, "data")

from fastapi.testclient import TestClient  # noqa: E402

from backend import app as app_module  # noqa: E402
from backend.db import MemoryStore  # noqa: E402


class FamilyGraphApiTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.store = MemoryStore(f"{self.temp_dir.name}/sentrix.db")
        self.store.create_memory_space("album-a", "相册 A")
        self.father = self.store.create_entity("甲", "person", "confirmed", scope_id="album-a")
        self.daughter = self.store.create_entity("乙", "person", "confirmed", scope_id="album-a")
        self.store.set_family_relationship(
            "album-a", self.father["id"], "父亲", self.daughter["id"], "女儿",
            source="user_override", confidence=1.0,
        )
        app_module.store = self.store
        self.client = TestClient(app_module.app)

    def tearDown(self):
        self.client.close()
        self.store.close()
        self.temp_dir.cleanup()

    def test_delete_family_relationship_retracts_both_directions(self):
        response = self.client.request(
            "DELETE", "/api/family-graph/relationships", json={
                "scope_id": "album-a",
                "subject_entity_id": self.father["id"],
                "object_entity_id": self.daughter["id"],
            },
        )

        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.json()["ok"])
        self.assertEqual(self.store.list_effective_family_relationships("album-a"), [])

    def test_graph_people_include_rendering_projection(self):
        response = self.client.get("/api/family-graph", params={"scope_id": "album-a"})

        self.assertEqual(response.status_code, 200)
        person = next(item for item in response.json()["people"] if item["id"] == self.father["id"])
        self.assertIn("avatar_face_instance_id", person)
        self.assertIn("appearance_count", person)
        self.assertEqual(person["representative_media"], [])

    def test_graph_hides_pending_cluster_internal_name(self):
        pending = self.store.create_entity("待确认人物簇 · cluster_private", "person", "pending", scope_id="album-a")

        response = self.client.get("/api/family-graph", params={"scope_id": "album-a"})

        person = next(item for item in response.json()["people"] if item["id"] == pending["id"])
        self.assertEqual(person["display_name"], "待命名人物")


if __name__ == "__main__":
    unittest.main()
