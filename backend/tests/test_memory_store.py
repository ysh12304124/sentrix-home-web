import tempfile
from concurrent.futures import ThreadPoolExecutor
import unittest
from datetime import timezone

from backend.db import MemoryStore, parse_time


class MemoryStoreTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.store = MemoryStore(f"{self.temp_dir.name}/sentrix.db")

    def tearDown(self):
        self.store.close()
        self.temp_dir.cleanup()

    def test_observations_merge_into_same_event_by_time_and_place(self):
        asset_one = self.store.create_asset("a1", "one.jpg", "image", "/tmp/one.jpg")
        asset_two = self.store.create_asset("a2", "two.jpg", "image", "/tmp/two.jpg")
        first = self.store.add_observation(asset_one["id"], {
            "captured_at": "2026-07-01T10:00:00+08:00",
            "place": "城市公园",
            "activity": "散步",
            "event_type": "日常",
            "caption": "一家人在公园散步",
            "people": ["妈妈"],
        })
        second = self.store.add_observation(asset_two["id"], {
            "captured_at": "2026-07-01T11:30:00+08:00",
            "place": "城市公园",
            "activity": "散步",
            "event_type": "日常",
            "caption": "孩子在公园玩耍",
            "people": ["宝宝"],
        })

        event_one = self.store.merge_observation_into_event(first)
        event_two = self.store.merge_observation_into_event(second)

        self.assertEqual(event_one["id"], event_two["id"])
        self.assertEqual(self.store.count("events"), 1)
        self.assertEqual(self.store.count("event_observations"), 2)

    def test_event_place_never_uses_asset_gps_when_observation_has_semantic_place(self):
        asset = self.store.create_asset(
            "gps_asset", "meal.jpg", "image", "/tmp/meal.jpg",
            metadata={"captured_at": "2026-07-01T18:00:00+08:00", "captured_location": "30.2458,120.2989"},
        )
        observation = self.store.add_observation(asset["id"], {
            "captured_at": "2026-07-01T18:00:00+08:00", "place": "餐厅", "activity": "用餐",
            "event_type": "餐饮", "caption": "桌上有蛋糕",
            "canonical": {"semantic": {"available": True, "place": {"primary": "餐饮空间", "details": []}}},
        })

        event = self.store.merge_observation_into_event(observation)

        self.assertEqual(event["place"], "餐厅")
        self.assertNotIn("30.2458", event["place"])

    def test_naive_capture_time_is_normalized_before_event_comparison(self):
        naive = parse_time("2018-06-28T12:16:02")
        aware = parse_time("2018-06-28T12:16:02+00:00")

        self.assertEqual(naive.tzinfo, timezone.utc)
        self.assertEqual((naive - aware).total_seconds(), 0)

    def test_event_does_not_promote_structured_people_descriptions_to_participants(self):
        asset = self.store.create_asset("structured_people", "people.jpg", "image", "/tmp/people.jpg")
        person = {"description": "一位成年人", "appearance": "短发"}
        observation_one = self.store.add_observation(asset["id"], {
            "captured_at": "2026-07-01T10:00:00+08:00",
            "place": "客厅",
            "activity": "家庭活动",
            "event_type": "日常",
            "people": [person],
        })
        observation_two = self.store.add_observation(asset["id"], {
            "captured_at": "2026-07-01T10:30:00+08:00",
            "place": "客厅",
            "activity": "家庭活动",
            "event_type": "日常",
            "people": [dict(person)],
        })

        event = self.store.merge_observation_into_event(observation_one)
        merged = self.store.merge_observation_into_event(observation_two)

        self.assertEqual(event["id"], merged["id"])
        self.assertEqual(merged["participants"], [])

    def test_event_does_not_use_album_provenance_to_cancel_semantic_conflict(self):
        metadata = {
            "captured_location": "家中餐厅",
            "source_album_id": "birthday-2026",
            "captured_at": "2026-07-01T18:00:00+08:00",
        }
        first_asset = self.store.create_asset("birthday_a", "a.jpg", "image", "/tmp/a.jpg", metadata=metadata)
        second_asset = self.store.create_asset("birthday_b", "b.jpg", "image", "/tmp/b.jpg", metadata={
            **metadata,
            "captured_at": "2026-07-01T18:16:00+08:00",
        })
        first = self.store.add_observation(first_asset["id"], {
            "captured_at": metadata["captured_at"],
            "place": "室内餐桌旁",
            "activity": "点蜡烛",
            "event_type": "生日庆祝",
            "caption": "桌上有生日蛋糕",
        })
        second = self.store.add_observation(second_asset["id"], {
            "captured_at": "2026-07-01T18:16:00+08:00",
            "place": "人物近景",
            "activity": "面对镜头微笑",
            "event_type": "人像摄影",
            "caption": "一位成员面对镜头微笑",
        })

        event_one = self.store.merge_observation_into_event(first)
        event_two = self.store.merge_observation_into_event(second)

        self.assertNotEqual(event_one["id"], event_two["id"])
        self.assertEqual(self.store.count("events"), 2)

    def test_event_seed_never_uses_imported_event_hint_as_title(self):
        asset = self.store.create_asset("unlabeled", "photo.jpg", "image", "/tmp/photo.jpg", metadata={
            "captured_at": "2026-07-01T18:00:00+08:00",
            "captured_location": "家中餐厅",
            "event_hint": "家庭生日",
        })
        observation = self.store.add_observation(asset["id"], {
            "captured_at": "2026-07-01T18:00:00+08:00",
            "caption": "餐桌上的物品",
            "activity": "室内活动",
            "event_type": "日常记录",
        })

        event = self.store.merge_observation_into_event(observation)

        self.assertEqual(event["title"], "待总结事件")
        self.assertEqual(event["event_type"], "待判断")

    def test_same_location_different_activities_do_not_merge_without_shared_evidence(self):
        first_asset = self.store.create_asset("activity_a", "a.jpg", "image", "/tmp/a.jpg", metadata={
            "captured_at": "2026-07-01T10:00:00+08:00",
            "captured_location": "城市公园",
            "source_device_id": "phone-a",
        })
        second_asset = self.store.create_asset("activity_b", "b.jpg", "image", "/tmp/b.jpg", metadata={
            "captured_at": "2026-07-01T11:00:00+08:00",
            "captured_location": "城市公园",
            "source_device_id": "phone-b",
        })
        first = self.store.add_observation(first_asset["id"], {
            "captured_at": "2026-07-01T10:00:00+08:00",
            "place": "城市公园", "activity": "打太极", "event_type": "运动",
            "objects": ["太极剑"], "people": [{"entity_id": "person-a"}],
        })
        second = self.store.add_observation(second_asset["id"], {
            "captured_at": "2026-07-01T11:00:00+08:00",
            "place": "城市公园", "activity": "咖啡聊天", "event_type": "社交",
            "objects": ["咖啡杯"], "people": [{"entity_id": "person-b"}],
        })

        first_event = self.store.merge_observation_into_event(first)
        second_event = self.store.merge_observation_into_event(second)

        self.assertNotEqual(first_event["id"], second_event["id"])
        self.assertEqual(self.store.count("events"), 2)

    def test_event_candidate_score_handles_missing_capture_time(self):
        first_asset = self.store.create_asset("no_time_a", "a.jpg", "image", "/tmp/a.jpg", metadata={"captured_location": "客厅"})
        second_asset = self.store.create_asset("no_time_b", "b.jpg", "image", "/tmp/b.jpg", metadata={"captured_location": "客厅"})
        first = self.store.add_observation(first_asset["id"], {"place": "客厅", "activity": "聊天", "event_type": "日常"})
        second = self.store.add_observation(second_asset["id"], {"place": "客厅", "activity": "聊天", "event_type": "日常"})

        first_event = self.store.merge_observation_into_event(first)
        second_event = self.store.merge_observation_into_event(second)

        self.assertEqual(first_event["id"], second_event["id"])

    def test_existing_database_schema_receives_new_face_and_event_columns(self):
        self.store.connection.execute("CREATE TABLE legacy_events (id TEXT PRIMARY KEY)")
        self.store.connection.execute("CREATE TABLE legacy_face_instances (id TEXT PRIMARY KEY)")
        self.store.connection.commit()
        self.assertIn("aggregation_score", {row[1] for row in self.store.connection.execute("PRAGMA table_info(events)")})
        self.assertIn("quality", {row[1] for row in self.store.connection.execute("PRAGMA table_info(face_instances)")})

    def test_legacy_events_table_is_migrated_before_time_index_creation(self):
        import sqlite3

        with tempfile.NamedTemporaryFile(suffix=".db") as file:
            connection = sqlite3.connect(file.name)
            connection.execute("CREATE TABLE events (id TEXT PRIMARY KEY, title TEXT NOT NULL)")
            connection.commit()
            connection.close()
            migrated = MemoryStore(file.name)
            columns = {row[1] for row in migrated.connection.execute("PRAGMA table_info(events)")}
            self.assertIn("time_start", columns)
            migrated.close()

    def test_cross_device_same_activity_merges_with_candidate_score(self):
        first_asset = self.store.create_asset("device_a", "a.jpg", "image", "/tmp/a.jpg", metadata={
            "captured_at": "2026-07-01T18:00:00+08:00", "captured_location": "家中餐厅", "source_device_id": "phone-a",
        })
        second_asset = self.store.create_asset("device_b", "b.jpg", "image", "/tmp/b.jpg", metadata={
            "captured_at": "2026-07-01T18:04:00+08:00", "captured_location": "家中餐厅", "source_device_id": "phone-b",
        })
        first = self.store.add_observation(first_asset["id"], {"captured_at": "2026-07-01T18:00:00+08:00", "place": "家中餐厅", "activity": "生日庆祝", "event_type": "庆祝", "objects": ["蛋糕"]})
        second = self.store.add_observation(second_asset["id"], {"captured_at": "2026-07-01T18:04:00+08:00", "place": "家中餐厅", "activity": "生日庆祝", "event_type": "庆祝", "objects": ["蜡烛", "蛋糕"]})

        first_event = self.store.merge_observation_into_event(first)
        second_event = self.store.merge_observation_into_event(second)

        self.assertEqual(first_event["id"], second_event["id"])
        self.assertGreaterEqual(second_event["aggregation_score"], 0.7)
        self.assertIsInstance(second_event["aggregation_breakdown"], dict)

    def test_asset_processing_update_keeps_import_provenance_metadata(self):
        asset = self.store.create_asset("provenance", "cake.jpg", "image", "/tmp/cake.jpg", metadata={
            "captured_at": "2026-07-01T18:00:00+08:00",
            "captured_location": "家中餐厅",
            "source_album_id": "birthday-2026",
            "event_hint": "家庭生日",
        })

        updated = self.store.update_asset(asset["id"], "processed", {"observation_id": "obs_1"})

        self.assertEqual(updated["metadata_json"]["event_hint"], "家庭生日")
        self.assertEqual(updated["captured_location"], "家中餐厅")

    def test_conflicting_fact_is_pending_until_confirmed(self):
        first = self.store.maintain_fact("宝宝", "喜欢", "积木", ["obs_1"], confidence=0.8)
        conflict = self.store.maintain_fact("宝宝", "喜欢", "画画", ["obs_2"], confidence=0.7)

        self.assertEqual(first["status"], "active")
        self.assertEqual(conflict["status"], "pending")
        self.assertEqual(self.store.get_fact(first["id"])["status"], "active")

        self.store.confirm_fact(conflict["id"])
        self.assertEqual(self.store.get_fact(first["id"])["status"], "superseded")
        self.assertEqual(self.store.get_fact(conflict["id"])["status"], "active")

    def test_rebuilding_same_semantic_claim_does_not_create_duplicate_pending_versions(self):
        person = self.store.create_entity("儿子", "person", "confirmed", "孩子", 1.0)
        first = self.store.maintain_semantic_claim(person["id"], "capture", "拍摄", "家庭生日", ["obs_1"], ["event_1"], 0.8)
        pending = self.store.maintain_semantic_claim(person["id"], "capture", "拍摄", "家庭出游", ["obs_2"], ["event_2"], 0.8)
        repeated = self.store.maintain_semantic_claim(person["id"], "capture", "拍摄", "家庭出游", ["obs_3"], ["event_2"], 0.9)

        claims = self.store.list_semantic_claims(person["id"])
        self.assertEqual(len(claims), 2)
        self.assertEqual(repeated["id"], pending["id"])
        self.assertEqual(set(repeated["evidence_ids_json"]), {"obs_2", "obs_3"})
        self.assertEqual(first["status"], "active")
        self.assertEqual(pending["status"], "active")

    def test_semantic_claims_normalize_common_chinese_predicates_and_keep_confidence_source(self):
        person = self.store.create_entity("妈妈", "person", "confirmed", "母亲", 1.0)

        claim = self.store.maintain_semantic_claim(person["id"], "activity", "参加过", "公园散步", ["obs_1"], ["event_1"], 0.8, confidence_source="vision_model")

        self.assertEqual(claim["predicate"], "参与")
        self.assertEqual(claim["confidence_source"], "vision_model")

    def test_rebuild_run_has_a_versioned_audit_record(self):
        run = self.store.start_rebuild("test-rebuild", "benchmark-db")
        self.assertEqual(run["status"], "running")

        finished = self.store.finish_rebuild(run["id"], "completed", {"assets": 2})

        self.assertEqual(finished["status"], "completed")
        self.assertEqual(finished["stats_json"]["assets"], 2)

    def test_event_detail_exposes_original_assets_and_observations(self):
        asset = self.store.create_asset("asset_evidence", "family.jpg", "image", "/tmp/family.jpg", "image/jpeg", 12)
        observation = self.store.add_observation(asset["id"], {
            "captured_at": "2026-07-01T10:00:00+08:00",
            "place": "客厅",
            "activity": "庆祝",
            "event_type": "生日",
            "caption": "桌上有蛋糕",
            "raw": {"model": "gemma4:12b", "objects": ["蛋糕"]},
        })
        event = self.store.merge_observation_into_event(observation)

        detail = self.store.get_event_detail(event["id"])

        self.assertEqual(detail["observations"][0]["id"], observation["id"])
        self.assertEqual(detail["observations"][0]["asset"]["file_name"], "family.jpg")
        self.assertEqual(detail["observations"][0]["raw_json"]["objects"], ["蛋糕"])

    def test_event_can_be_edited_and_story_is_persisted(self):
        event = self.store.create_event({"title": "手工事件", "summary": "原始记录", "place": "家里"})
        updated = self.store.update_event(event["id"], {"title": "修正后的事件", "summary": "人工补充"})
        story = self.store.create_story({"title": "家庭故事", "event_ids": [event["id"]]})

        self.assertEqual(updated["title"], "修正后的事件")
        self.assertEqual(self.store.get_story(story["id"])["event_ids"], [event["id"]])

    def test_asset_content_hash_is_persisted_and_duplicate_lookup_is_available(self):
        asset = self.store.create_asset("hash_a", "a.jpg", "image", "/tmp/a.jpg", metadata={"sha256": "abc123"})

        self.assertEqual(asset["content_sha256"], "abc123")
        self.assertEqual(self.store.find_asset_by_hash("abc123")["id"], asset["id"])

    def test_asset_hash_and_exif_metadata_are_preserved_on_update(self):
        asset = self.store.create_asset("metadata_a", "a.jpg", "image", "/tmp/a.jpg", metadata={
            "sha256": "digest", "exif": {"captured_at": "2026-07-01T10:00:00+08:00", "gps": "31,121", "device": "phone"}
        })
        updated = self.store.update_asset(asset["id"], "processed", {"observation_id": "obs_1"})

        self.assertEqual(updated["content_sha256"], "digest")
        self.assertEqual(updated["metadata_json"]["exif"]["device"], "phone")

    def test_independent_task_connections_can_write_same_database(self):
        database = f"{self.temp_dir.name}/concurrent.db"

        def write(index):
            store = MemoryStore(database)
            try:
                asset = store.create_asset(f"asset_{index}", f"{index}.jpg", "image", f"/tmp/{index}.jpg")
                return store.update_asset(asset["id"], "processing")["status"]
            finally:
                store.close()

        with ThreadPoolExecutor(max_workers=3) as executor:
            statuses = list(executor.map(write, range(3)))

        verifier = MemoryStore(database)
        try:
            self.assertEqual(statuses, ["processing", "processing", "processing"])
            self.assertEqual(verifier.count("assets"), 3)
        finally:
            verifier.close()


if __name__ == "__main__":
    unittest.main()
