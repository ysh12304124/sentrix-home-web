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

    def test_single_sample_pending_face_is_reviewable(self):
        face = self.store.add_face_instance("a1", self.obs1["id"], {"bbox": [1, 2, 30, 40], "confidence": 0.55, "embedding": [1, 0, 0]})
        entity_id = self.store._row("SELECT entity_id FROM face_clusters WHERE id = ?", (face["cluster_id"],))["entity_id"]
        entity = next(item for item in self.store.list_entities() if item["id"] == entity_id)
        cluster = next(item for item in self.store.list_face_clusters() if item["id"] == face["cluster_id"])

        self.assertTrue(entity["reviewable"])
        self.assertTrue(entity["single_sample"])
        self.assertTrue(cluster["reviewable"])
        self.assertTrue(cluster["single_sample"])

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

    def test_confirmation_resegments_split_events_using_confirmed_person_evidence(self):
        for asset_id in ("a1", "a2"):
            self.store.update_asset(asset_id, "queued", {
                "captured_at": "2026-07-01T18:00:00+08:00",
                "captured_location": "家中餐厅",
            })
        self.store.connection.execute(
            "UPDATE observations SET captured_at = ?, place = ?, activity = ?, event_type = ? WHERE id = ?",
            ("2026-07-01T18:00:00+08:00", "家中餐厅", "准备晚餐", "用餐", self.obs1["id"]),
        )
        self.store.connection.execute(
            "UPDATE observations SET captured_at = ?, place = ?, activity = ?, event_type = ? WHERE id = ?",
            ("2026-07-01T18:00:00+08:00", "家中餐厅", "公开演讲", "演讲", self.obs2["id"]),
        )
        self.store.connection.commit()
        self.store.upsert_vector("visual", "asset", "a1", [1.0, 0.0], "test-clip")
        self.store.upsert_vector("visual", "asset", "a2", [0.0, 1.0], "test-clip")
        first_event = self.store.merge_observation_into_event(self.store.get_observation(self.obs1["id"]))
        second_event = self.store.merge_observation_into_event(self.store.get_observation(self.obs2["id"]))
        self.assertNotEqual(first_event["id"], second_event["id"])

        first = self.store.add_face_instance("a1", self.obs1["id"], {"bbox": [1, 2, 30, 40], "confidence": 0.95, "embedding": [1, 0, 0]})
        self.store.add_face_instance("a2", self.obs2["id"], {"bbox": [1, 2, 30, 40], "confidence": 0.94, "embedding": [0.99, 0.01, 0]})
        self.store.confirm_face_cluster(first["cluster_id"], "妈妈", "母亲")

        self.assertEqual(self.store.count("events"), 1)
        event = self.store.list_events()[0]
        self.assertEqual(len(event["observation_ids"]), 2)
        self.assertTrue(any(item["person_name"] == "妈妈" for item in event["participant_roles"]))

    def test_person_memory_does_not_inherit_scene_clothing_as_person_attribute(self):
        self.store.connection.execute(
            "UPDATE observations SET clothing_json = ? WHERE id = ?",
            ('["红色外套"]', self.obs1["id"]),
        )
        self.store.connection.execute(
            "UPDATE observations SET clothing_json = ? WHERE id = ?",
            ('["蓝色制服"]', self.obs2["id"]),
        )
        self.store.connection.commit()
        event = self.store.merge_observation_into_event(self.obs1)
        self.store.connection.execute(
            "INSERT OR IGNORE INTO event_observations(event_id, observation_id) VALUES (?, ?)",
            (event["id"], self.obs2["id"]),
        )
        self.store.connection.commit()
        face = self.store.add_face_instance("a1", self.obs1["id"], {"bbox": [1, 2, 30, 40], "confidence": 0.95, "embedding": [1, 0, 0]})

        detail = self.store.confirm_face_cluster(face["cluster_id"], "妈妈", "母亲")

        clothing = [claim["value_text"] for claim in detail["semantic_claims"] if claim["dimension"] == "clothing"]
        self.assertEqual(clothing, [])

    def test_face_scoped_appearance_evidence_becomes_person_clothing_claim(self):
        event = self.store.merge_observation_into_event(self.obs1)
        face = self.store.add_face_instance(
            "a1", self.obs1["id"],
            {"bbox": [10, 20, 40, 60], "confidence": 0.95, "embedding": [1, 0, 0]},
        )
        detail = self.store.confirm_face_cluster(face["cluster_id"], "妈妈", "母亲")

        appearance = self.store.record_person_appearance_evidence(
            detail["entity"]["id"], face["id"], [0, 0, 100, 180], ["红色针织衫"], 0.88, "test-vision",
        )
        memory = self.store.rebuild_person_memory(detail["entity"]["id"])

        clothing_claims = [claim for claim in memory["claims"] if claim["dimension"] == "clothing"]
        self.assertEqual([claim["value_text"] for claim in clothing_claims], ["红色针织衫"])
        self.assertEqual(clothing_claims[0]["evidence_ids_json"], [appearance["id"]])
        self.assertEqual(clothing_claims[0]["supporting_event_ids_json"], [event["id"]])
        evidence = self.store.list_person_appearance_evidence(detail["entity"]["id"])
        self.assertEqual(evidence[0]["face_instance_id"], face["id"])
        self.assertEqual(evidence[0]["asset_id"], "a1")

    def test_legacy_person_confirmation_updates_native_entity(self):
        person = self.store.upsert_person("旧人物", confidence=0.7)

        updated = self.store.update_person(person["id"], "妈妈", "confirmed")
        entity = self.store.get_entity(f"entity_{person['id']}")

        self.assertEqual(updated["name"], "妈妈")
        self.assertEqual(entity["canonical_name"], "妈妈")
        self.assertEqual(entity["status"], "confirmed")

    def test_native_entity_confirmation_resolves_active_face_cluster(self):
        face = self.store.add_face_instance(
            "a1", self.obs1["id"],
            {"bbox": [1, 2, 30, 40], "confidence": 0.95, "embedding": [1, 0, 0]},
        )
        entity_id = self.store._row("SELECT entity_id FROM face_clusters WHERE id = ?", (face["cluster_id"],))["entity_id"]

        detail = self.store.confirm_person_entity(entity_id, "妈妈", "母亲")

        self.assertEqual(detail["entity"]["id"], entity_id)
        self.assertEqual(detail["entity"]["canonical_name"], "妈妈")
        self.assertEqual(detail["entity"]["status"], "confirmed")
        self.assertEqual(self.store._row("SELECT status FROM face_clusters WHERE id = ?", (face["cluster_id"],))["status"], "confirmed")

    def test_observation_entities_link_place_objects_emotion_and_event_evidence(self):
        self.store.connection.execute(
            "UPDATE observations SET place = ?, objects_json = ?, raw_json = ? WHERE id = ?",
            ("家中餐厅", '["生日蛋糕"]', '{"emotions": ["喜悦"]}', self.obs1["id"]),
        )
        self.store.connection.commit()
        event = self.store.merge_observation_into_event(self.store.get_observation(self.obs1["id"]))

        entities = self.store.maintain_observation_entities(self.obs1["id"], event["id"])

        self.assertEqual({item["entity_type"] for item in entities}, {"place", "object", "emotion"})
        cake = next(item for item in entities if item["canonical_name"] == "生日蛋糕")
        detail = self.store.get_entity_detail(cake["id"])
        self.assertEqual(detail["events"][0]["id"], event["id"])
        self.assertEqual(detail["observations"][0]["id"], self.obs1["id"])
        self.assertTrue(any(item["predicate"] == "出现在" for item in detail["relationships"]))

    def test_entity_index_adds_capture_day_and_can_be_rebuilt(self):
        self.store.connection.execute(
            "UPDATE observations SET captured_at = ?, place = ?, objects_json = ?, raw_json = ? WHERE id = ?",
            ("2026-08-03T10:30:00+08:00", "家中餐厅", '["生日蛋糕"]', '{"emotions": ["喜悦"]}', self.obs1["id"]),
        )
        self.store.connection.commit()
        result = self.store.reindex_observation_entities()
        entities = self.store.list_entities()
        names_by_type = {(item["entity_type"], item["canonical_name"]) for item in entities}

        self.assertEqual(result["observations"], 2)
        self.assertIn(("time", "2026-08-03"), names_by_type)
        self.assertTrue(all(item["reviewable"] for item in entities if item["entity_type"] != "person"))


if __name__ == "__main__":
    unittest.main()
