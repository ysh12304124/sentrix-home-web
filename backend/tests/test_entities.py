import tempfile
import unittest

from backend.db import MemoryStore


class NativeEntityMemoryTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.store = MemoryStore(f"{self.temp_dir.name}/sentrix.db")
        self.store.create_asset("a1", "one.jpg", "image", "/tmp/one.jpg", "image/jpeg")
        self.store.create_asset("a2", "two.jpg", "image", "/tmp/two.jpg", "image/jpeg")
        self.obs1 = self.store.add_observation("a1", {"caption": "客厅里的家人", "people": []})
        self.obs2 = self.store.add_observation("a2", {"caption": "客厅里的家人", "people": []})

    def tearDown(self):
        self.store.close()
        self.temp_dir.cleanup()

    def test_similar_faces_share_cluster_and_confirmation_updates_memory(self):
        first = self.store.add_face_instance("a1", self.obs1["id"], {"bbox": [1, 2, 3, 4], "confidence": 0.95, "embedding": [1, 0, 0]})
        second = self.store.add_face_instance("a2", self.obs2["id"], {"bbox": [2, 3, 4, 5], "confidence": 0.94, "embedding": [0.99, 0.01, 0]})

        self.assertEqual(first["cluster_id"], second["cluster_id"])
        detail = self.store.confirm_face_cluster(first["cluster_id"], "妈妈", "母亲")

        self.assertEqual(detail["entity"]["canonical_name"], "妈妈")
        self.assertEqual(detail["entity"]["status"], "confirmed")
        self.assertEqual(self.store.get_observation(self.obs1["id"])["people"][0]["name"], "妈妈")
        self.assertEqual(self.store.get_observation(self.obs2["id"])["people"][0]["name"], "妈妈")
        self.assertTrue(any(fact["predicate"] == "家庭角色" and fact["status"] == "active" for fact in detail["facts"]))
        self.assertGreaterEqual(len(self.store.search_vectors("visual", [1, 0, 0])), 2)

    def test_dissimilar_faces_do_not_share_cluster(self):
        first = self.store.add_face_instance("a1", self.obs1["id"], {"bbox": [1, 2, 3, 4], "confidence": 0.95, "embedding": [1, 0, 0]})
        second = self.store.add_face_instance("a2", self.obs2["id"], {"bbox": [2, 3, 4, 5], "confidence": 0.94, "embedding": [0, 1, 0]})
        self.assertNotEqual(first["cluster_id"], second["cluster_id"])

    def test_person_entity_exposes_a_face_instance_for_avatar_rendering(self):
        face = self.store.add_face_instance("a1", self.obs1["id"], {"bbox": [1, 2, 30, 40], "confidence": 0.95, "embedding": [1, 0, 0]})
        entity_id = self.store._row("SELECT entity_id FROM face_clusters WHERE id = ?", (face["cluster_id"],))["entity_id"]

        entity = next(item for item in self.store.list_entities() if item["id"] == entity_id)
        avatar = self.store.get_face_instance(face["id"])

        self.assertEqual(entity["avatar_face_instance_id"], face["id"])
        self.assertEqual(avatar["bbox_json"], [1, 2, 30, 40])

    def test_confirmation_rebuilds_event_roles_and_person_knowledge(self):
        event_one = self.store.merge_observation_into_event(self.obs1)
        event_two = self.store.merge_observation_into_event(self.obs2)
        first = self.store.add_face_instance("a1", self.obs1["id"], {"bbox": [1, 2, 3, 4], "confidence": 0.95, "embedding": [1, 0, 0]})
        self.store.add_face_instance("a2", self.obs2["id"], {"bbox": [2, 3, 4, 5], "confidence": 0.94, "embedding": [0.99, 0.01, 0]})
        cluster = self.store._row("SELECT * FROM face_clusters WHERE id = ?", (first["cluster_id"],))
        self.store.update_asset("a1", "processed", {"source_owner_id": cluster["entity_id"], "source_confidence": 1.0})

        detail = self.store.confirm_face_cluster(first["cluster_id"], "妈妈", "母亲")
        event = self.store.get_event(event_one["id"])

        roles = {(item["person_name"], item["role"]) for item in event["participant_roles"]}
        self.assertIn(("妈妈", "visible_subject"), roles)
        self.assertIn(("妈妈", "captured_by"), roles)
        self.assertIn("妈妈", event["summary"])
        self.assertEqual(detail["semantic_profile"]["person_id"], cluster["entity_id"])
        self.assertTrue(any(claim["dimension"] == "identity" for claim in detail["semantic_claims"]))


if __name__ == "__main__":
    unittest.main()
