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


if __name__ == "__main__":
    unittest.main()
