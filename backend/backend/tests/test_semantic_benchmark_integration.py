import json
import tempfile
import unittest
from pathlib import Path

from backend.db import MemoryStore
from scripts.benchmarks.prepare_household_benchmark import prepare_benchmark


class SemanticBenchmarkIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.store = MemoryStore(f"{self.temp_dir.name}/sentrix.db")

    def tearDown(self):
        self.store.close()
        self.temp_dir.cleanup()

    def _photo(self, asset_id, file_name, scope_id, captured_at, activity, embedding):
        asset = self.store.create_asset(
            asset_id,
            file_name,
            "image",
            f"/tmp/{file_name}",
            metadata={
                "captured_at": captured_at,
                "captured_location": "家庭客厅",
                "scope_id": scope_id,
            },
            scope_id=scope_id,
        )
        observation = self.store.add_observation(
            asset["id"],
            {
                "captured_at": captured_at,
                "place": "家庭客厅",
                "activity": activity,
                "event_type": "家庭活动",
                "caption": activity,
                "people": [],
            },
            scope_id=scope_id,
        )
        event = self.store.merge_observation_into_event(observation)
        face = self.store.add_face_instance(
            asset["id"],
            observation["id"],
            {"bbox": [1, 2, 30, 40], "confidence": 0.95, "embedding": embedding},
        )
        return asset, observation, event, face

    def test_confirmed_person_has_event_memory_and_cross_event_patterns(self):
        self.store.create_memory_space("album-a", "家庭相册 A")
        first = self._photo("a1", "a1.jpg", "album-a", "2025-01-01T10:00:00+08:00", "准备早餐", [1, 0, 0])
        second = self._photo("a2", "a2.jpg", "album-a", "2025-01-02T10:00:00+08:00", "准备早餐", [0.99, 0.01, 0])

        detail = self.store.confirm_face_cluster(first[3]["cluster_id"], "妈妈", "母亲")
        memory = self.store.get_person_memory(detail["entity"]["id"])

        self.assertEqual(len(memory["event_memory"]), 2)
        self.assertTrue(any(item["pattern_type"] == "activity" for item in memory["patterns"]))
        breakfast = next(item for item in memory["patterns"] if item["pattern_type"] == "activity")
        self.assertEqual(breakfast["support_count"], 2)
        self.assertEqual({item["scope_id"] for item in memory["event_memory"]}, {"album-a"})

    def test_clothing_normalization_keeps_raw_evidence_and_aggregates_pattern(self):
        self.store.create_memory_space("album-a", "家庭相册 A")
        first = self._photo("a1", "a1.jpg", "album-a", "2025-01-01T10:00:00+08:00", "拍照", [1, 0, 0])
        second = self._photo("a2", "a2.jpg", "album-a", "2025-01-02T10:00:00+08:00", "拍照", [0.99, 0.01, 0])
        detail = self.store.confirm_face_cluster(first[3]["cluster_id"], "妈妈", "母亲")
        person_id = detail["entity"]["id"]
        self.store.record_person_appearance_evidence(person_id, first[3]["id"], [0, 0, 100, 100], ["深色西装外套"], 0.9, "test")
        self.store.record_person_appearance_evidence(person_id, second[3]["id"], [0, 0, 100, 100], ["黑色西装外套"], 0.9, "test")
        memory = self.store.rebuild_person_memory(person_id)

        appearance = self.store.list_person_appearance_evidence(person_id)
        self.assertEqual({"深色西装外套", "黑色西装外套"}, {appearance[0]["clothing_json"][0], appearance[1]["clothing_json"][0]})
        clothing = [item for item in memory["patterns"] if item["pattern_type"] == "clothing"]
        self.assertEqual(len(clothing), 1)
        self.assertEqual(clothing[0]["value_text"], "西装外套")
        self.assertEqual(clothing[0]["support_count"], 2)

    def test_scope_filter_prevents_cross_album_people_and_events(self):
        self.store.create_memory_space("album-a", "家庭相册 A")
        self.store.create_memory_space("album-b", "家庭相册 B")
        first = self._photo("a1", "a1.jpg", "album-a", "2025-01-01T10:00:00+08:00", "散步", [1, 0, 0])
        second = self._photo("b1", "b1.jpg", "album-b", "2025-01-01T10:00:00+08:00", "聚餐", [0, 1, 0])
        self.store.confirm_face_cluster(first[3]["cluster_id"], "妈妈", "母亲")

        self.assertEqual({item["id"] for item in self.store.list_events(scope_id="album-a")}, {first[2]["id"]})
        self.assertEqual({item["id"] for item in self.store.list_events(scope_id="album-b")}, {second[2]["id"]})
        self.assertTrue(all(item["scope_id"] == "album-a" for item in self.store.list_entities(scope_id="album-a")))
        self.assertTrue(all(item["scope_id"] == "album-b" for item in self.store.list_assets(scope_id="album-b")))

    def test_vector_search_is_scoped_to_the_memory_space(self):
        self.store.create_memory_space("album-a", "家庭相册 A")
        self.store.create_memory_space("album-b", "家庭相册 B")
        first = self.store.create_asset("a1", "a1.jpg", "image", "/tmp/a1.jpg", metadata={"scope_id": "album-a"}, scope_id="album-a")
        second = self.store.create_asset("b1", "b1.jpg", "image", "/tmp/b1.jpg", metadata={"scope_id": "album-b"}, scope_id="album-b")
        self.store.upsert_vector("visual", "asset", first["id"], [1, 0], "test-clip")
        self.store.upsert_vector("visual", "asset", second["id"], [1, 0], "test-clip")

        self.assertEqual({item["scope_id"] for item in self.store.search_vectors("visual", [1, 0], scope_id="album-a")}, {"album-a"})
        self.assertEqual({item["scope_id"] for item in self.store.search_vectors("visual", [1, 0], scope_id="album-b")}, {"album-b"})

    def test_benchmark_manifest_intersects_images_and_keeps_labels_out_of_import(self):
        root = Path(self.temp_dir.name) / "samples"
        album = root / "album1"
        (album / "images").mkdir(parents=True)
        (album / "faceid").mkdir()
        (album / "images" / "keep.jpg").write_bytes(b"image")
        (album / "metadata.json").write_text(json.dumps({
            "keep.jpg": {"time": "2025-01-01T10:00:00", "latitude": 1, "longitude": 2},
            "missing.jpg": {"time": "2025-01-01T10:01:00", "latitude": 3, "longitude": 4},
        }), encoding="utf-8")
        (album / "faceid" / "face_info_cn.json").write_text(json.dumps({
            "face_id_to_nicknames": {"1": ["不要写入记忆"]},
            "image_to_face_ids": {"keep.jpg": ["1"], "missing.jpg": ["1"]},
        }), encoding="utf-8")
        (album / "query.json").write_text(json.dumps([{
            "query_cn": "测试图片", "ground_truth": ["keep.jpg"],
        }]), encoding="utf-8")

        manifest = prepare_benchmark(root)
        record = manifest["spaces"][0]
        self.assertEqual(record["import"]["files"][0]["file_name"], "keep.jpg")
        self.assertEqual(record["diagnostics"]["metadata_missing_images"], ["missing.jpg"])
        self.assertNotIn("face_id_to_nicknames", record["import"])
        self.assertEqual(record["evaluation"]["queries"][0]["ground_truth"], ["keep.jpg"])

    def test_person_evidence_contract_contains_face_samples_assets_and_events(self):
        self.store.create_memory_space("album-a", "家庭相册 A")
        asset, observation, event, face = self._photo(
            "a1", "a1.jpg", "album-a", "2025-01-01T10:00:00+08:00", "家庭活动", [1, 0, 0]
        )
        confirmed = self.store.confirm_face_cluster(face["cluster_id"], "妈妈", "母亲")

        evidence = self.store.get_person_evidence(confirmed["entity"]["id"])

        self.assertEqual(evidence["entity"]["canonical_name"], "妈妈")
        self.assertTrue(evidence["face_samples"])
        self.assertEqual(evidence["face_samples"][0]["asset_id"], asset["id"])
        self.assertTrue(evidence["assets"])
        self.assertEqual(evidence["assets"][0]["file_name"], "a1.jpg")
        self.assertEqual(evidence["events"][0]["id"], event["id"])
        self.assertEqual(evidence["scope_id"], "album-a")

    def test_confirmation_without_events_returns_refresh_counts(self):
        self.store.create_memory_space("album-a", "家庭相册 A")
        asset = self.store.create_asset(
            "a1", "a1.jpg", "image", "/tmp/a1.jpg", metadata={"scope_id": "album-a"}, scope_id="album-a"
        )
        observation = self.store.add_observation(asset["id"], {"scope_id": "album-a"}, scope_id="album-a")
        face = self.store.add_face_instance(
            asset["id"], observation["id"], {"bbox": [1, 2, 30, 40], "confidence": 0.95, "embedding": [1, 0, 0]}
        )
        self.store.connection.execute("DELETE FROM event_observations")
        self.store.connection.execute("DELETE FROM events")
        self.store.connection.commit()

        result = self.store.confirm_face_cluster(face["cluster_id"], "爸爸", "父亲")

        self.assertIsInstance(result["refresh_counts"], dict)
        self.assertEqual(result["refresh_counts"]["events"], 0)

    def test_split_and_merge_keep_the_memory_space(self):
        self.store.create_memory_space("album-a", "家庭相册 A")
        first = self._photo("a1", "a1.jpg", "album-a", "2025-01-01T10:00:00+08:00", "活动", [1, 0, 0])
        second = self._photo("a2", "a2.jpg", "album-a", "2025-01-01T11:00:00+08:00", "活动", [0.99, 0.01, 0])

        split = self.store.split_face_instance(first[3]["cluster_id"], first[3]["id"])

        self.assertEqual(split["scope_id"], "album-a")
        self.assertEqual(self.store.get_face_instance(first[3]["id"])["scope_id"], "album-a")

        merged = self.store.merge_face_clusters(first[3]["cluster_id"], split["id"])
        self.assertEqual(merged["scope_id"], "album-a")
