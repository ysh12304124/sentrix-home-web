import json
import tempfile
import unittest
from pathlib import Path
import importlib.util

from backend.db import MemoryStore


def load_cover_maintenance():
    path = Path(__file__).resolve().parents[2] / "scripts" / "maintenance" / "backfill_event_covers.py"
    spec = importlib.util.spec_from_file_location("backfill_event_covers", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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

    def test_user_entity_property_supersedes_derived_value_and_blocks_later_derived_update(self):
        place = self.store.create_entity("家中餐厅", "place", confidence=0.7)
        derived = self.store.maintain_entity_property(place["id"], "alias", "餐厅", 0.7, [self.obs1["id"]])
        user = self.store.set_entity_property(place["id"], "alias", "我们的饭桌", [self.obs2["id"]])
        retained = self.store.maintain_entity_property(place["id"], "alias", "模型餐厅", 0.95, [self.obs1["id"]])
        detail = self.store.get_entity_detail(place["id"])

        self.assertEqual(derived["status"], "active")
        self.assertEqual(user["status"], "active")
        self.assertEqual(retained["id"], user["id"])
        self.assertEqual(detail["properties"][0]["value"], "我们的饭桌")
        self.assertEqual(detail["properties"][0]["source"], "user")
        self.assertEqual([item["status"] for item in detail["property_history"]], ["active", "superseded"])

    def test_conflicting_derived_property_stays_pending_for_review(self):
        place = self.store.create_entity("家中餐厅", "place", confidence=0.7)
        self.store.maintain_entity_property(place["id"], "scene_type", "餐厅", 0.7, [self.obs1["id"]])
        candidate = self.store.maintain_entity_property(place["id"], "scene_type", "厨房", 0.8, [self.obs2["id"]])

        self.assertEqual(candidate["status"], "pending")

    def test_equivalent_relationships_merge_evidence_instead_of_creating_duplicates(self):
        cake = self.store.create_entity("生日蛋糕", "object", confidence=0.7)
        place = self.store.create_entity("家中餐厅", "place", confidence=0.7)

        first = self.store.create_relationship(cake["id"], "出现在", place["id"], [self.obs1["id"]], 0.6)
        merged = self.store.create_relationship(cake["id"], "出现在", place["id"], [self.obs2["id"]], 0.8, "active")

        relationships = self.store.list_relationships(cake["id"])
        self.assertEqual(len(relationships), 1)
        self.assertEqual(merged["id"], first["id"])
        self.assertEqual(merged["status"], "active")
        self.assertEqual(merged["evidence_ids_json"], [self.obs1["id"], self.obs2["id"]])
        self.assertEqual(merged["confidence"], 0.8)
        self.assertEqual(merged["revision"], 2)

    def test_semantic_consolidation_creates_review_candidate_without_merging_entities(self):
        lakeside = self.store.create_entity("湖边", "place", confidence=0.8)
        waterside = self.store.create_entity("水边", "place", confidence=0.8)
        self.store.connection.execute(
            "INSERT INTO entity_observations(entity_id, observation_id, confidence, source, created_at) VALUES (?, ?, 0.8, 'test', datetime('now'))",
            (lakeside["id"], self.obs1["id"]),
        )
        self.store.connection.execute(
            "INSERT INTO entity_observations(entity_id, observation_id, confidence, source, created_at) VALUES (?, ?, 0.8, 'test', datetime('now'))",
            (waterside["id"], self.obs2["id"]),
        )
        self.store.connection.commit()

        candidates = self.store.derive_entity_merge_candidates()

        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0]["status"], "pending")
        self.assertEqual(candidates[0]["entity_type"], "place")
        self.assertEqual(set(candidates[0]["entity_ids"]), {lakeside["id"], waterside["id"]})
        self.assertEqual(set(candidates[0]["evidence_ids"]), {self.obs1["id"], self.obs2["id"]})
        self.assertEqual(self.store.get_entity(lakeside["id"])["status"], "pending")
        self.assertEqual(self.store.get_entity(waterside["id"])["status"], "pending")

    def test_semantic_consolidation_never_groups_people_or_memory_spaces(self):
        self.store.create_entity("湖边", "person", scope_id="album_a")
        self.store.create_entity("水边", "person", scope_id="album_a")
        self.store.create_entity("湖边", "place", scope_id="album_a")
        self.store.create_entity("水边", "place", scope_id="album_b")

        candidates = self.store.derive_entity_merge_candidates()

        self.assertEqual(candidates, [])

    def test_user_confirmation_merges_candidate_into_selected_stable_entity_with_audit(self):
        lakeside = self.store.create_entity("湖边", "place", confidence=0.8)
        waterside = self.store.create_entity("水边", "place", confidence=0.8)
        for entity, observation in ((lakeside, self.obs1), (waterside, self.obs2)):
            self.store.connection.execute(
                "INSERT INTO entity_observations(entity_id, observation_id, confidence, source, created_at) VALUES (?, ?, 0.8, 'test', datetime('now'))",
                (entity["id"], observation["id"]),
            )
        self.store.connection.commit()
        candidate = self.store.derive_entity_merge_candidates()[0]

        merged = self.store.confirm_entity_merge_candidate(candidate["id"], lakeside["id"])

        self.assertEqual(merged["status"], "confirmed")
        self.assertEqual(merged["target_entity_id"], lakeside["id"])
        self.assertEqual(self.store.get_entity(lakeside["id"])["status"], "pending")
        self.assertEqual(self.store.get_entity(waterside["id"])["status"], "superseded")
        target_evidence = self.store.get_entity_detail(lakeside["id"])["evidence_files"]
        self.assertEqual({item["evidence_id"] for item in target_evidence}, {self.obs1["id"], self.obs2["id"]})
        revisions = self.store._rows("SELECT * FROM entity_revisions WHERE entity_id = ?", (waterside["id"],))
        self.assertTrue(any(item["field_name"] == "semantic_merge_target" for item in revisions))

    def test_observation_entities_derive_explainable_place_and_time_properties(self):
        self.store.update_asset("a1", "queued", {"captured_at": "2026-08-03T19:30:00+08:00"})
        self.store.connection.execute(
            "UPDATE observations SET captured_at = ?, place = ? WHERE id = ?",
            ("2026-08-03T19:30:00+08:00", "家中餐厅", self.obs1["id"]),
        )
        self.store.connection.commit()

        self.store.maintain_observation_entities(self.obs1["id"])
        entities = {(item["entity_type"], item["canonical_name"]): item for item in self.store.list_entities()}
        place_detail = self.store.get_entity_detail(entities[("place", "家中餐厅")]["id"])
        time_detail = self.store.get_entity_detail(entities[("time", "2026-08-03")]["id"])
        place_properties = {item["property_key"]: item for item in place_detail["properties"]}
        time_properties = {item["property_key"]: item for item in time_detail["properties"]}

        self.assertEqual(place_properties["scene_type"]["value"], "家中餐厅")
        self.assertEqual(place_properties["scene_type"]["source"], "observation_extraction")
        self.assertEqual(place_properties["scene_type"]["evidence_ids"], [self.obs1["id"]])
        self.assertEqual(time_properties["date"]["value"], "2026-08-03")
        self.assertEqual(time_properties["year"]["value"], 2026)
        self.assertEqual(time_properties["month"]["value"], 8)
        self.assertEqual(time_properties["season"]["value"], "夏")
        self.assertEqual(time_properties["part_of_day"]["value"], ["傍晚"])

    def test_gps_place_keeps_coordinates_separate_from_visual_place_name(self):
        self.store.update_asset("a1", "queued", {"captured_location": "30.274100,120.155100"})
        self.store.connection.execute(
            "UPDATE observations SET place = ? WHERE id = ?", ("西湖湖畔", self.obs1["id"])
        )
        self.store.connection.commit()

        self.store.maintain_observation_entities(self.obs1["id"])
        place = next(item for item in self.store.list_entities() if item["entity_type"] == "place")
        properties = {item["property_key"]: item for item in self.store.get_entity_detail(place["id"])["properties"]}

        self.assertEqual(place["canonical_name"], "西湖湖畔")
        self.assertEqual(properties["geo"]["value"], {"latitude": 30.2741, "longitude": 120.1551})
        self.assertEqual(properties["geo"]["source"], "asset_gps")
        self.assertEqual(properties["scene_type"]["value"], "西湖湖畔")

    def test_semantic_place_primary_wins_over_gps_as_entity_name(self):
        self.store.update_asset("a1", "queued", {"captured_location": "30.274100,120.155100"})
        self.store.connection.execute(
            "UPDATE observations SET place = ?, canonical_json = ? WHERE id = ?",
            (
                "湖边餐厅",
                json.dumps({
                    "place": "湖边餐厅",
                    "scene_type": "餐饮空间",
                    "semantic": {"place": {"primary": "餐饮空间", "details": ["室内", "有餐桌"]}},
                }, ensure_ascii=False),
                self.obs1["id"],
            ),
        )
        self.store.connection.commit()

        self.store.maintain_observation_entities(self.obs1["id"])
        place = next(item for item in self.store.list_entities() if item["entity_type"] == "place")
        properties = {item["property_key"]: item for item in self.store.get_entity_detail(place["id"])["properties"]}

        self.assertEqual(place["canonical_name"], "餐饮空间")
        self.assertEqual(properties["geo"]["value"], {"latitude": 30.2741, "longitude": 120.1551})
        self.assertEqual(properties["semantic_primary"]["value"], "餐饮空间")
        self.assertEqual(properties["semantic_details"]["value"], ["室内", "有餐桌"])
        self.assertEqual(properties["visual_place_descriptions"]["value"], ["湖边餐厅"])

    def test_private_place_has_an_alias_for_standard_entity_lists(self):
        place = self.store.create_entity("家中餐厅", "place", confidence=1.0)
        self.store.set_entity_property(place["id"], "alias", "我们的饭桌", [self.obs1["id"]])
        self.store.set_entity_property(place["id"], "private_flag", True, [self.obs1["id"]])

        listed = self.store.public_entity(self.store.get_entity(place["id"]))

        self.assertEqual(listed["canonical_name"], "我们的饭桌")
        self.assertTrue(listed["private"])
        self.assertEqual(self.store.get_entity(place["id"])["canonical_name"], "家中餐厅")

    def test_person_user_properties_are_versioned_and_not_overwritten_by_derivation(self):
        person = self.store.create_entity("妈妈", "person", "confirmed", confidence=1.0)
        self.store.set_entity_property(person["id"], "is_self", True, [self.obs1["id"]])
        self.store.set_entity_property(person["id"], "relation_to_user", "本人", [self.obs1["id"]])
        self.store.set_entity_property(person["id"], "groups", ["家人", "旅行伙伴"], [self.obs2["id"]])
        retained = self.store.maintain_entity_property(person["id"], "relation_to_user", "同事", 0.9, [self.obs2["id"]])
        properties = {item["property_key"]: item for item in self.store.get_entity_detail(person["id"])["properties"]}

        self.assertTrue(properties["is_self"]["value"])
        self.assertEqual(properties["relation_to_user"]["value"], "本人")
        self.assertEqual(properties["groups"]["value"], ["家人", "旅行伙伴"])
        self.assertEqual(retained["id"], properties["relation_to_user"]["id"])

    def test_confirmed_people_in_one_event_create_a_pending_cooccurrence_candidate(self):
        event = self.store.merge_observation_into_event(self.obs1)
        first = self.store.create_entity("妈妈", "person", "confirmed", confidence=1.0)
        second = self.store.create_entity("明哥", "person", "confirmed", confidence=1.0)
        self.store.upsert_event_participant(event["id"], first["id"], "visible_subject", [self.obs1["id"]], 0.9)
        self.store.upsert_event_participant(event["id"], second["id"], "visible_subject", [self.obs1["id"]], 0.9)

        candidates = [item for item in self.store.list_relationships() if item["predicate"] == "共同出现"]

        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0]["status"], "pending")
        self.assertEqual(candidates[0]["evidence_ids_json"], [self.obs1["id"]])
        self.assertGreaterEqual(candidates[0]["confidence"], 0.5)

    def test_trip_candidate_requires_cross_day_material_gps_displacement(self):
        first = self.store.create_event({
            "id": "event_one", "title": "第一天抵达", "place": "30.274100,120.155100", "time_start": "2025-05-01T10:00:00+08:00",
        })
        second = self.store.create_event({
            "id": "event_two", "title": "第二天游览", "place": "31.230400,121.473700", "time_start": "2025-05-02T10:00:00+08:00",
        })
        ordinary = self.store.create_event({
            "id": "event_three", "title": "晚餐", "place": "家中餐厅", "time_start": "2025-08-01T18:00:00+08:00",
        })
        distant = self.store.create_event({
            "id": "event_four", "title": "下一周的日常", "place": "办公室", "time_start": "2025-05-09T10:00:00+08:00",
        })
        self.store.connection.executemany(
            "INSERT INTO event_observations(event_id, observation_id) VALUES (?, ?)",
            [(first["id"], self.obs1["id"]), (second["id"], self.obs2["id"])],
        )
        self.store.connection.commit()

        trips = self.store.derive_trip_candidates()

        self.assertEqual(len(trips), 1)
        self.assertEqual(trips[0]["status"], "pending")
        self.assertEqual(trips[0]["event_ids_json"], [first["id"], second["id"]])
        self.assertEqual(trips[0]["place_names_json"], ["30.274100,120.155100", "31.230400,121.473700"])
        self.assertEqual(trips[0]["evidence_ids_json"], [self.obs1["id"], self.obs2["id"]])
        self.assertNotIn(ordinary["id"], trips[0]["event_ids_json"])
        self.assertNotIn(distant["id"], trips[0]["event_ids_json"])

    def test_trip_candidate_rejects_nearby_gps_changes_across_days(self):
        first = self.store.create_event({
            "id": "event_one", "title": "第一天晚餐", "place": "30.256200,120.159700", "time_start": "2025-05-01T18:00:00+08:00",
        })
        second = self.store.create_event({
            "id": "event_two", "title": "第二天散步", "place": "30.286200,120.129000", "time_start": "2025-05-02T10:00:00+08:00",
        })
        self.store.connection.executemany(
            "INSERT INTO event_observations(event_id, observation_id) VALUES (?, ?)",
            [(first["id"], self.obs1["id"]), (second["id"], self.obs2["id"])],
        )
        self.store.connection.commit()

        self.assertEqual(self.store.derive_trip_candidates(), [])

    def test_user_can_confirm_trip_without_losing_evidence_or_stable_identity(self):
        first = self.store.create_event({
            "id": "trip_event_one", "title": "出发", "place": "30.274100,120.155100", "time_start": "2025-05-01T10:00:00+08:00",
        })
        second = self.store.create_event({
            "id": "trip_event_two", "title": "抵达", "place": "31.230400,121.473700", "time_start": "2025-05-02T10:00:00+08:00",
        })
        self.store.connection.executemany(
            "INSERT INTO event_observations(event_id, observation_id) VALUES (?, ?)",
            [(first["id"], self.obs1["id"]), (second["id"], self.obs2["id"])],
        )
        self.store.connection.commit()
        candidate = self.store.derive_trip_candidates()[0]

        confirmed = self.store.confirm_trip(candidate["id"], "五一沪杭行", "旅行")
        detail = self.store.get_trip_detail(candidate["id"])

        self.assertEqual(confirmed["id"], candidate["id"])
        self.assertEqual(confirmed["status"], "active")
        self.assertEqual(confirmed["name"], "五一沪杭行")
        self.assertEqual(confirmed["trip_type"], "旅行")
        self.assertEqual(confirmed["evidence_ids_json"], [self.obs1["id"], self.obs2["id"]])
        self.assertEqual(detail["events"][0]["id"], first["id"])
        self.assertEqual(detail["revisions"][0]["action"], "confirmed")
        self.assertEqual(self.store.derive_trip_candidates(), [])

    def test_user_can_reject_trip_and_candidate_does_not_reappear(self):
        first = self.store.create_event({
            "id": "trip_event_one", "title": "出发", "place": "30.274100,120.155100", "time_start": "2025-05-01T10:00:00+08:00",
        })
        second = self.store.create_event({
            "id": "trip_event_two", "title": "抵达", "place": "31.230400,121.473700", "time_start": "2025-05-02T10:00:00+08:00",
        })
        self.store.connection.executemany(
            "INSERT INTO event_observations(event_id, observation_id) VALUES (?, ?)",
            [(first["id"], self.obs1["id"]), (second["id"], self.obs2["id"])],
        )
        self.store.connection.commit()
        candidate = self.store.derive_trip_candidates()[0]

        rejected = self.store.reject_trip(candidate["id"])

        self.assertEqual(rejected["status"], "rejected")
        self.assertEqual(self.store.derive_trip_candidates(), [])
        self.assertEqual(self.store.get_trip_detail(candidate["id"])["revisions"][0]["action"], "rejected")

    def test_user_event_edit_preserves_evidence_and_records_cover_revision(self):
        event = self.store.create_event({
            "id": "editable_event", "title": "待修正事件", "event_type": "日常", "time_start": "2025-05-01T10:00:00+08:00",
        })
        self.store.connection.executemany(
            "INSERT INTO event_observations(event_id, observation_id) VALUES (?, ?)",
            [(event["id"], self.obs1["id"]), (event["id"], self.obs2["id"])],
        )
        self.store.connection.execute("UPDATE events SET cover_asset_id = ? WHERE id = ?", ("a1", event["id"]))
        self.store.connection.commit()

        updated = self.store.update_event(event["id"], {
            "event_type": "旅行", "time_end": "2025-05-02T18:00:00+08:00", "cover_asset_id": "a2",
        })
        detail = self.store.get_event_detail(event["id"])

        self.assertEqual(updated["event_type"], "旅行")
        self.assertEqual(updated["time_end"], "2025-05-02T18:00:00+08:00")
        self.assertEqual(updated["cover_asset_id"], "a2")
        self.assertEqual(set(updated["asset_ids"]), {"a1", "a2"})
        cover_revision = next(item for item in detail["event_revisions"] if item["field_name"] == "cover_asset_id")
        self.assertEqual(cover_revision["new_value"], "a2")

    def test_event_merge_rehomes_all_foreign_key_projections_before_delete(self):
        target = self.store.create_event({"id": "merge-target", "title": "目标事件"})
        source = self.store.create_event({"id": "merge-source", "title": "来源事件"})
        entity = self.store.create_entity("餐饮空间", "place", confidence=0.8)
        self.store.upsert_event_entity(source["id"], entity["id"], "地点", [self.obs1["id"]], 0.8)
        self.store.update_event(source["id"], {"summary": "来源修订"})

        merged = self.store._merge_events(target["id"], source["id"])

        self.assertEqual(merged["id"], target["id"])
        self.assertIsNone(self.store.get_event(source["id"]))
        self.assertTrue(any(item["canonical_name"] == "餐饮空间" for item in self.store.list_event_entities(target["id"])))
        self.assertFalse(self.store._rows("SELECT 1 FROM event_revisions WHERE event_id = ?", (source["id"],)))

    def test_event_cover_must_be_evidence_asset_from_that_event(self):
        event = self.store.create_event({"id": "editable_event", "title": "待修正事件"})
        self.store.connection.execute("INSERT INTO event_observations(event_id, observation_id) VALUES (?, ?)", (event["id"], self.obs1["id"]))
        self.store.connection.commit()

        with self.assertRaises(ValueError):
            self.store.update_event(event["id"], {"cover_asset_id": "missing-asset"})

    def test_derived_event_cover_uses_event_image_evidence_and_preserves_user_choice(self):
        event = self.store.create_event({"id": "cover_event", "title": "封面选择"})
        self.store.connection.execute("UPDATE observations SET confidence = ? WHERE id = ?", (0.45, self.obs1["id"]))
        self.store.connection.execute("UPDATE observations SET confidence = ? WHERE id = ?", (0.9, self.obs2["id"]))
        self.store.connection.executemany(
            "INSERT INTO event_observations(event_id, observation_id) VALUES (?, ?)",
            [(event["id"], self.obs1["id"]), (event["id"], self.obs2["id"])],
        )
        self.store.connection.commit()

        selected = self.store.select_event_cover(event["id"])

        self.assertEqual(selected["cover_asset_id"], "a2")
        self.assertEqual(selected["cover_selection"]["source"], "derived")
        self.assertEqual(selected["cover_selection"]["asset_id"], "a2")
        self.assertEqual(selected["cover_selection"]["evidence_observation_id"], self.obs2["id"])
        self.assertEqual(selected["cover_selection"]["criteria"]["media_type"], "image")

        self.store.update_event(event["id"], {"cover_asset_id": "a1"})
        retained = self.store.select_event_cover(event["id"])

        self.assertEqual(retained["cover_asset_id"], "a1")
        self.assertEqual(retained["cover_selection"]["source"], "user")

    def test_cover_backfill_reports_before_apply_and_preserves_user_selection(self):
        event = self.store.create_event({"id": "backfill_event", "title": "待回填封面"})
        self.store.connection.execute("INSERT INTO event_observations(event_id, observation_id) VALUES (?, ?)", (event["id"], self.obs1["id"]))
        self.store.connection.commit()
        module = load_cover_maintenance()

        report = module.backfill_event_covers(self.store, apply=False)
        self.assertEqual(report["eligible"], 1)
        self.assertEqual(self.store.get_event(event["id"])["cover_selection"], {})

        applied = module.backfill_event_covers(self.store, apply=True)
        self.assertEqual(applied["updated"], 1)
        self.assertEqual(self.store.get_event(event["id"])["cover_selection"]["source"], "derived")

        self.store.update_event(event["id"], {"cover_asset_id": "a1"})
        self.assertEqual(module.backfill_event_covers(self.store, apply=True)["updated"], 0)

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

    def test_person_profile_summarizes_evidence_backed_places_and_activities(self):
        self.store.connection.execute(
            "UPDATE observations SET place = ?, activity = ? WHERE id = ?",
            ("家中餐厅", "准备晚餐", self.obs1["id"]),
        )
        self.store.connection.commit()
        self.store.merge_observation_into_event(self.store.get_observation(self.obs1["id"]))
        face = self.store.add_face_instance("a1", self.obs1["id"], {"bbox": [1, 2, 30, 40], "confidence": 0.95, "embedding": [1, 0, 0]})

        detail = self.store.confirm_face_cluster(face["cluster_id"], "妈妈", "母亲")

        self.assertIn("常见地点：家中餐厅", detail["semantic_profile"]["summary_zh"])
        self.assertIn("常见活动：准备晚餐", detail["semantic_profile"]["summary_zh"])

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

    def test_native_person_entity_reject_retires_candidate_and_clusters(self):
        face = self.store.add_face_instance(
            "a1", self.obs1["id"],
            {"bbox": [1, 2, 30, 40], "confidence": 0.95, "embedding": [0, 1, 0]},
        )
        entity_id = self.store._row("SELECT entity_id FROM face_clusters WHERE id = ?", (face["cluster_id"],))["entity_id"]
        other = self.store.add_face_instance(
            "a2", self.obs2["id"],
            {"bbox": [2, 3, 40, 50], "confidence": 0.95, "embedding": [0, 0, 1]},
        )
        other_entity = self.store._row("SELECT entity_id FROM face_clusters WHERE id = ?", (other["cluster_id"],))["entity_id"]
        relationship = self.store.create_relationship(entity_id, "可能同框", other_entity, [self.obs1["id"]], 0.6, "pending")

        rejected = self.store.reject_person_entity(entity_id)

        self.assertEqual(rejected["id"], entity_id)
        self.assertEqual(rejected["status"], "rejected")
        self.assertEqual(self.store._row("SELECT status FROM face_clusters WHERE id = ?", (face["cluster_id"],))["status"], "rejected")
        self.assertEqual(self.store._row("SELECT status FROM relationships WHERE id = ?", (relationship["id"],))["status"], "retracted")
        self.assertNotIn(entity_id, {item["id"] for item in self.store.list_entities(scope_id="home-default")})
        with self.assertRaises(ValueError):
            self.store.reject_person_entity(self.store.confirm_person_entity(other_entity, "爸爸", "父亲")["entity"]["id"])

    def test_observation_entities_link_place_objects_atmosphere_and_event_evidence(self):
        self.store.connection.execute(
            "UPDATE observations SET place = ?, objects_json = ?, raw_json = ?, canonical_json = ? WHERE id = ?",
            (
                "家中餐厅", '["生日蛋糕"]', '{"emotions": ["温馨"]}',
                json.dumps({
                    "semantic": {
                        "place": {"primary": "餐饮空间", "details": ["室内"]},
                        "objects": [{"primary": "食品与饮品", "label": "生日蛋糕", "details": ["桌面"]}],
                        "atmosphere": {"labels": ["温馨"], "details": ["暖色光线"]},
                    }
                }, ensure_ascii=False),
                self.obs1["id"],
            ),
        )
        self.store.connection.commit()
        event = self.store.merge_observation_into_event(self.store.get_observation(self.obs1["id"]))

        entities = self.store.maintain_observation_entities(self.obs1["id"], event["id"])

        self.assertEqual({item["entity_type"] for item in entities}, {"place", "object", "atmosphere"})
        cake = next(item for item in entities if item["canonical_name"] == "生日蛋糕")
        detail = self.store.get_entity_detail(cake["id"])
        self.assertEqual(detail["events"][0]["id"], event["id"])
        self.assertEqual(detail["observations"][0]["id"], self.obs1["id"])
        self.assertTrue(any(item["predicate"] == "出现在" for item in detail["relationships"]))
        self.assertEqual(detail["evidence_files"][0]["file_name"], "one.jpg")
        listed = next(item for item in self.store.list_entities() if item["id"] == cake["id"])
        self.assertEqual(listed["preview_asset_id"], "a1")
        self.assertEqual(listed["preview_file_name"], "one.jpg")

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

    def test_entity_index_reads_atmosphere_from_persisted_model_payload(self):
        self.store.connection.execute(
            "UPDATE observations SET canonical_json = ? WHERE id = ?",
            ('{"semantic": {"atmosphere": {"labels": ["温馨"], "details": ["暖色光线"]}}}', self.obs1["id"]),
        )
        self.store.connection.commit()
        self.store.reindex_observation_entities()

        self.assertTrue(any(item["entity_type"] == "atmosphere" and item["canonical_name"] == "温馨" for item in self.store.list_entities()))

    def test_atmosphere_entities_preserve_labels_and_evidence(self):
        self.store.connection.execute(
            "UPDATE observations SET canonical_json = ? WHERE id = ?",
            ('{"semantic": {"atmosphere": {"labels": ["温馨", "轻松"], "details": ["暖色光线"]}}}', self.obs1["id"]),
        )
        self.store.connection.commit()

        entities = self.store.maintain_observation_entities(self.obs1["id"])
        atmospheres = {item["canonical_name"]: item for item in entities if item["entity_type"] == "atmosphere"}
        warm = self.store.get_entity_detail(atmospheres["温馨"]["id"])

        self.assertEqual(set(atmospheres), {"温馨", "轻松"})
        properties = {item["property_key"]: item for item in warm["properties"]}
        self.assertEqual(properties["atmosphere_label"]["value"], "温馨")
        self.assertEqual(properties["semantic_details"]["value"], ["暖色光线"])
        self.assertEqual(properties["semantic_details"]["evidence_ids"], [self.obs1["id"]])

    def test_legacy_emotion_is_exposed_as_atmosphere_group(self):
        legacy = self.store.create_entity("平静", "emotion", confidence=0.6)
        self.store.connection.execute(
            "INSERT INTO entity_observations(entity_id, observation_id, confidence, source, created_at) VALUES (?, ?, ?, ?, ?)",
            (legacy["id"], self.obs1["id"], 0.6, "observation_extraction", "2026-08-03T00:00:00+00:00"),
        )
        self.store.connection.commit()

        groups = self.store.list_semantic_entity_groups("home-default")

        self.assertTrue(any(group["entity_type"] == "atmosphere" and group["canonical_name"] == "平静" for group in groups))
        self.assertFalse(any(group["entity_type"] == "emotion" for group in groups))

    def test_reindex_migrates_legacy_raw_mood_entity_to_normalized_entity(self):
        legacy = self.store.create_entity("面带微笑", "emotion", confidence=0.7)
        place = self.store.create_entity("家中餐厅", "place", confidence=0.8)
        self.store.create_relationship(legacy["id"], "出现在", place["id"], [self.obs1["id"]], 0.7)
        self.store.connection.execute(
            "UPDATE observations SET raw_json = ? WHERE id = ?",
            ('{"gamma": {"emotions": ["面带微笑"]}}', self.obs1["id"]),
        )
        self.store.connection.commit()

        result = self.store.reindex_observation_entities()
        atmospheres = [item for item in self.store.list_entities(public=False) if item["entity_type"] == "atmosphere"]
        joyful = next(item for item in atmospheres if item["canonical_name"] == "热闹")
        properties = {item["property_key"]: item for item in self.store.list_entity_properties(joyful["id"])}

        self.assertEqual(result["normalized_moods"], 1)
        self.assertFalse(any(item["id"] == legacy["id"] for item in self.store.list_entities(public=False)))
        self.assertEqual(joyful["evidence_count"], 1)
        self.assertEqual(properties["raw_atmosphere_labels"]["value"], ["面带微笑"])
        self.assertTrue(any(
            relationship["subject_entity_id"] == joyful["id"] and relationship["object_entity_id"] == place["id"]
            for relationship in self.store.list_relationships(joyful["id"])
        ))

    def test_reindex_retires_unclassified_legacy_mood_without_erasing_raw_observation(self):
        legacy = self.store.create_entity("可爱", "emotion", confidence=0.6)
        self.store.connection.execute(
            "UPDATE observations SET raw_json = ? WHERE id = ?",
            ('{"gamma": {"emotions": ["可爱"]}}', self.obs1["id"]),
        )
        self.store.connection.execute(
            "INSERT INTO entity_observations(entity_id, observation_id, confidence, source, created_at) VALUES (?, ?, ?, ?, ?)",
            (legacy["id"], self.obs1["id"], 0.6, "observation_extraction", "2026-08-03T00:00:00+00:00"),
        )
        self.store.connection.commit()

        result = self.store.reindex_observation_entities()

        self.assertEqual(result["retired_unclassified_moods"], 1)
        self.assertEqual(self.store.get_entity(legacy["id"])["status"], "rejected")
        self.assertEqual(self.store.get_observation(self.obs1["id"])["raw"]["gamma"]["emotions"], ["可爱"])

    def test_event_projects_linked_entities_with_observation_evidence(self):
        self.store.connection.execute(
            "UPDATE observations SET captured_at = ?, place = ?, objects_json = ?, raw_json = ? WHERE id = ?",
            ("2026-08-03T10:30:00+08:00", "家中餐厅", '["生日蛋糕"]', '{"emotions": ["喜悦"]}', self.obs1["id"]),
        )
        self.store.connection.commit()
        event = self.store.merge_observation_into_event(self.store.get_observation(self.obs1["id"]))
        person = self.store.create_entity("妈妈", "person", "confirmed", confidence=1.0)
        self.store.upsert_event_participant(event["id"], person["id"], "visible_subject", [self.obs1["id"]], 0.9)
        self.store.maintain_observation_entities(self.obs1["id"], event["id"])

        detail = self.store.get_event_detail(event["id"])
        by_type = {item["entity_type"]: item for item in detail["entities"]}

        self.assertEqual(set(by_type), {"person", "place", "object", "atmosphere", "time"})
        self.assertEqual(by_type["person"]["relation"], "参与")
        self.assertEqual(by_type["place"]["relation"], "地点")
        self.assertEqual(by_type["object"]["relation"], "包含物件")
        self.assertEqual(by_type["atmosphere"]["relation"], "画面氛围")
        self.assertEqual(by_type["time"]["relation"], "时间")
        self.assertEqual(by_type["object"]["evidence_ids_json"], [self.obs1["id"]])
        self.assertEqual(by_type["object"]["evidence_count"], 1)

    def test_object_entities_preserve_raw_label_and_add_controlled_category(self):
        self.store.connection.execute(
            "UPDATE observations SET objects_json = ? WHERE id = ?",
            ('["生日蛋糕", "自行车"]', self.obs1["id"]),
        )
        self.store.connection.commit()

        self.store.maintain_observation_entities(self.obs1["id"])
        objects = {
            item["canonical_name"]: item for item in self.store.list_entities()
            if item["entity_type"] == "object"
        }
        cake = {item["property_key"]: item for item in self.store.get_entity_detail(objects["生日蛋糕"]["id"])["properties"]}
        bicycle = {item["property_key"]: item for item in self.store.get_entity_detail(objects["自行车"]["id"])["properties"]}

        self.assertEqual(cake["label"]["value"], "生日蛋糕")
        self.assertEqual(cake["category"]["value"], "食物")
        self.assertEqual(bicycle["category"]["value"], "交通工具")
        self.assertEqual(cake["label"]["evidence_ids"], [self.obs1["id"]])
        self.assertNotIn("salience", cake)

    def test_semantic_entity_groups_keep_members_and_evidence_without_physical_merge(self):
        self.store.connection.execute("UPDATE observations SET place = ? WHERE id = ?", ("湖边", self.obs1["id"]))
        self.store.connection.execute("UPDATE observations SET place = ? WHERE id = ?", ("水边", self.obs2["id"]))
        self.store.connection.commit()
        self.store.maintain_observation_entities(self.obs1["id"])
        self.store.maintain_observation_entities(self.obs2["id"])

        groups = self.store.list_semantic_entity_groups()
        waterfront = next(item for item in groups if item["canonical_name"] == "滨水区域")

        self.assertTrue(waterfront["is_semantic_cluster"])
        self.assertEqual(waterfront["source_labels"], ["水边", "湖边"])
        self.assertEqual(len(waterfront["member_entity_ids"]), 2)
        self.assertEqual(waterfront["evidence_count"], 2)
        detail = self.store.get_semantic_entity_group(waterfront["id"])
        self.assertEqual({item["id"] for item in detail["observations"]}, {self.obs1["id"], self.obs2["id"]})
        self.assertEqual(len([item for item in self.store.list_entities() if item["entity_type"] == "place"]), 2)

    def test_semantic_entity_groups_auto_cluster_realistic_scene_descriptions(self):
        labels = ("城市河流岸边", "户外湖泊", "海滩、海岸线")
        for index, label in enumerate(labels, 1):
            entity = self.store.create_entity(label, "place", confidence=0.7)
            observation = self.obs1 if index == 1 else self.obs2
            self.store.connection.execute(
                "INSERT INTO entity_observations(entity_id, observation_id, confidence, source, created_at) VALUES (?, ?, 0.7, 'test', datetime('now'))",
                (entity["id"], observation["id"]),
            )
        self.store.connection.commit()

        waterfront = next(item for item in self.store.list_semantic_entity_groups() if item["canonical_name"] == "滨水空间")

        self.assertEqual(set(waterfront["source_labels"]), set(labels))
        self.assertTrue(waterfront["is_semantic_cluster"])
        self.assertEqual(waterfront["rationale"]["strategy"], "semantic_concept")

    def test_model_selected_scene_type_is_the_stable_place_entity(self):
        self.store.enrich_observation(self.obs1["id"], {
            "place": "城市河流岸边", "scene_type": "滨水空间", "caption": "河边散步",
        })

        entities = self.store.maintain_observation_entities(self.obs1["id"])
        place = next(item for item in entities if item["entity_type"] == "place")
        properties = {item["property_key"]: item for item in self.store.get_entity_detail(place["id"])["properties"]}

        self.assertEqual(place["canonical_name"], "滨水空间")
        self.assertEqual(properties["scene_type"]["value"], "滨水空间")
        self.assertEqual(properties["visual_place_descriptions"]["value"], ["城市河流岸边"])

    def test_semantic_group_uses_primary_property_and_aggregates_details(self):
        restaurant = self.store.create_entity("餐厅", "place", confidence=0.8)
        cafe = self.store.create_entity("咖啡馆", "place", confidence=0.8)
        for entity, observation, details in (
            (restaurant, self.obs1, ["室内", "有餐桌"]),
            (cafe, self.obs2, ["室内", "咖啡或茶"]),
        ):
            self.store.connection.execute(
                "INSERT INTO entity_observations(entity_id, observation_id, confidence, source, created_at) VALUES (?, ?, 0.8, 'test', datetime('now'))",
                (entity["id"], observation["id"]),
            )
            self.store.maintain_entity_property(entity["id"], "semantic_primary", "餐饮空间", 0.8, [observation["id"]], "semantic_taxonomy_v1")
            self.store.maintain_entity_property_values(entity["id"], "semantic_details", details, 0.8, [observation["id"]], "semantic_taxonomy_v1")
        self.store.connection.commit()

        group = next(item for item in self.store.list_semantic_entity_groups() if item["canonical_name"] == "餐饮空间")

        self.assertEqual(set(group["source_labels"]), {"餐厅", "咖啡馆"})
        self.assertEqual(set(group["semantic_details"]), {"室内", "有餐桌", "咖啡或茶"})

    def test_scene_type_replaces_old_free_text_place_projection_for_same_observation(self):
        self.store.enrich_observation(self.obs1["id"], {"place": "湖边", "caption": "湖边散步"})
        old_place = next(item for item in self.store.maintain_observation_entities(self.obs1["id"]) if item["entity_type"] == "place")
        self.store.enrich_observation(self.obs1["id"], {"place": "湖边", "scene_type": "滨水空间"})
        current_place = next(item for item in self.store.maintain_observation_entities(self.obs1["id"]) if item["entity_type"] == "place")

        self.assertEqual(current_place["canonical_name"], "滨水空间")
        old_links = self.store._rows("SELECT * FROM entity_observations WHERE entity_id = ?", (old_place["id"],))
        self.assertEqual(old_links, [])

    def test_gps_places_do_not_create_a_semantic_group_without_a_primary(self):
        first = self.store.create_entity("30.091900,120.496900", "place", confidence=0.8)
        second = self.store.create_entity("30.092200,120.501000", "place", confidence=0.8)
        for entity, observation in ((first, self.obs1), (second, self.obs2)):
            self.store.connection.execute(
                "INSERT INTO entity_observations(entity_id, observation_id, confidence, source, created_at) VALUES (?, ?, 0.8, 'gps', datetime('now'))",
                (entity["id"], observation["id"]),
            )
        self.store.connection.commit()

        groups = self.store.list_semantic_entity_groups()
        self.assertFalse(any(item["rationale"]["strategy"] == "nearby_gps_grid" for item in groups))
        self.assertEqual(
            {item["canonical_name"] for item in groups if item["entity_type"] == "place"},
            {"30.091900,120.496900", "30.092200,120.501000"},
        )


if __name__ == "__main__":
    unittest.main()
