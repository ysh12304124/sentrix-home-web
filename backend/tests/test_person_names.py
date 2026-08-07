import tempfile
import unittest

from backend.db import MemoryStore


class PersonNamesAndMergeTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.store = MemoryStore(f"{self.temp_dir.name}/sentrix.db")
        self.store.create_asset("a1", "one.jpg", "image", "/tmp/one.jpg", "image/jpeg")
        self.store.create_asset("a2", "two.jpg", "image", "/tmp/two.jpg", "image/jpeg")
        self.store.create_asset("a3", "three.jpg", "image", "/tmp/three.jpg", "image/jpeg")
        self.store.create_asset("a4", "four.jpg", "image", "/tmp/four.jpg", "image/jpeg")
        self.obs1 = self.store.add_observation("a1", {"caption": "客厅里的家人", "people": []})
        self.obs2 = self.store.add_observation("a2", {"caption": "客厅里的家人", "people": []})
        self.obs3 = self.store.add_observation("a3", {"caption": "客厅里的家人", "people": []})
        self.obs4 = self.store.add_observation("a4", {"caption": "客厅里的家人", "people": []})

    def tearDown(self):
        self.store.close()
        self.temp_dir.cleanup()

    def _face(self, asset_id, obs_id, embedding):
        return self.store.add_face_instance(
            asset_id, obs_id,
            {"bbox": [1, 2, 3, 4], "confidence": 0.95, "embedding": embedding},
        )

    def test_confirm_two_clusters_with_same_name_merges_into_one_person(self):
        first = self._face("a1", self.obs1["id"], [1, 0, 0])
        second = self._face("a2", self.obs2["id"], [0, 1, 0])
        self.assertNotEqual(first["cluster_id"], second["cluster_id"])

        result = self.store.confirm_face_cluster(first["cluster_id"], "明哥")
        self.assertFalse(result.get("merged_into"))
        merged = self.store.confirm_face_cluster(second["cluster_id"], "明哥")
        self.assertTrue(merged.get("merged_into"))
        self.assertEqual(merged["entity"]["canonical_name"], "明哥")
        # Both face instances now belong to the same person.
        people = [p for p in self.store.list_entities() if p["entity_type"] == "person"]
        confirmed = [p for p in people if p["status"] == "confirmed"]
        self.assertEqual(len(confirmed), 1)
        self.assertEqual(confirmed[0]["canonical_name"], "明哥")

    def test_alias_match_triggers_merge(self):
        first = self._face("a1", self.obs1["id"], [1, 0, 0])
        second = self._face("a2", self.obs2["id"], [0, 1, 0])
        detail = self.store.confirm_face_cluster(first["cluster_id"], "明哥")
        self.store.set_person_aliases(detail["entity"]["id"], ["小明"])
        target_entity = detail["entity"]["id"]
        merged = self.store.confirm_face_cluster(second["cluster_id"], "小明")
        self.assertTrue(merged.get("merged_into"))
        self.assertEqual(merged["entity"]["id"], target_entity)

    def test_rename_updates_canonical_and_folds_old_name_into_aliases(self):
        face = self._face("a1", self.obs1["id"], [1, 0, 0])
        detail = self.store.confirm_face_cluster(face["cluster_id"], "明哥")
        person_id = detail["entity"]["id"]
        renamed = self.store.rename_person(person_id, "大明")
        self.assertEqual(renamed["entity"]["canonical_name"], "大明")
        aliases = self.store.person_aliases(person_id)
        self.assertIn("明哥", aliases)
        revisions = self.store._rows(
            "SELECT * FROM entity_revisions WHERE entity_id = ? AND field_name = 'canonical_name'",
            (person_id,),
        )
        self.assertEqual(len(revisions), 1)
        self.assertEqual(revisions[0]["old_value"], "明哥")
        self.assertEqual(revisions[0]["new_value"], "大明")

    def test_cross_scope_same_name_does_not_merge(self):
        first = self._face("a1", self.obs1["id"], [1, 0, 0])
        second = self._face("a2", self.obs2["id"], [0, 1, 0])
        # Put the second cluster into another scope via its entity.
        self.store.connection.execute(
            "UPDATE entities SET scope_id = 'album_other' WHERE id = ?",
            (self.store._row("SELECT entity_id FROM face_clusters WHERE id = ?", (second["cluster_id"],))["entity_id"],),
        )
        self.store.connection.execute(
            "UPDATE face_clusters SET scope_id = 'album_other' WHERE id = ?",
            (second["cluster_id"],),
        )
        self.store.connection.commit()
        result = self.store.confirm_face_cluster(first["cluster_id"], "明哥")
        self.assertFalse(result.get("merged_into"))
        # Same name but different scope: no merge, a new confirmed person appears.
        merged = self.store.confirm_face_cluster(second["cluster_id"], "明哥")
        self.assertFalse(merged.get("merged_into"))
        people = [p for p in self.store.list_entities() if p["entity_type"] == "person" and p["status"] == "confirmed"]
        self.assertEqual(len(people), 2)

    def test_reject_person_candidate_deletes_entity_and_associations(self):
        face = self._face("a1", self.obs1["id"], [1, 0, 0])
        cluster = self.store._row("SELECT * FROM face_clusters WHERE id = ?", (face["cluster_id"],))
        entity_id = cluster["entity_id"]
        # Create a relationship and a semantic claim for the candidate.
        other = self.store.create_entity("另一个人", "person", "confirmed")
        self.store.create_relationship(entity_id, "朋友", other["id"], [], 1.0, "active")
        self.store.maintain_semantic_claim(entity_id, "identity", "称呼", "明哥", confidence_source="derived")
        deleted = self.store.delete_person_candidate(entity_id)
        self.assertTrue(deleted["deleted"])
        self.assertIsNone(self.store.get_entity(entity_id))
        self.assertEqual(self.store.list_relationships(entity_id), [])
        # Face instance survives as evidence.
        rows = self.store._rows("SELECT * FROM face_instances WHERE cluster_id IS NULL")
        self.assertGreaterEqual(len(rows), 1)

    def test_reject_confirmed_person_is_refused(self):
        face = self._face("a1", self.obs1["id"], [1, 0, 0])
        detail = self.store.confirm_face_cluster(face["cluster_id"], "明哥")
        with self.assertRaises(ValueError):
            self.store.delete_person_candidate(detail["entity"]["id"])


if __name__ == "__main__":
    unittest.main()
