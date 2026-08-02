import tempfile
import unittest
from pathlib import Path

from backend.db import MemoryStore
from backend.pipeline import IngestionPipeline


class FakeGamma:
    model = "test-gamma"

    def analyze_image(self, path, metadata=None):
        return {"caption": "一张带文字的家庭照片", "activity": "聚会", "place": "家里", "people": [], "objects": ["蛋糕"], "ocr_text": "生日快乐", "event_type": "聚会", "facts": [], "confidence": 0.8, "model": self.model}

    def summarize_event(self, event, observations):
        return {"title": "生日庆祝", "event_type": "庆祝活动", "activity": "围绕蛋糕庆祝", "summary": "一组照片记录了围绕蛋糕的庆祝活动。", "confidence": 0.88, "model": self.model}


class GenericGamma(FakeGamma):
    def analyze_image(self, path, metadata=None):
        return {"caption": "一张家庭照片", "activity": "家庭活动", "place": "家里", "people": [], "objects": [], "ocr_text": "", "event_type": "家庭记录", "facts": [], "confidence": 0.8, "model": self.model}


class ConflictingGamma(FakeGamma):
    def analyze_image(self, path, metadata=None):
        if Path(path).name == "first.jpg":
            return {"caption": "餐桌旁准备晚餐", "activity": "准备晚餐", "place": "家里", "people": [], "objects": [], "ocr_text": "", "event_type": "用餐", "facts": [], "confidence": 0.8, "model": self.model}
        return {"caption": "讲台上的公开发言", "activity": "公开演讲", "place": "家里", "people": [], "objects": [], "ocr_text": "", "event_type": "演讲", "facts": [], "confidence": 0.8, "model": self.model}


class FakeClip:
    model_name = "test-clip"
    error = None

    def embed_image(self, path):
        return [1.0, 0.0, 0.0]

    def embed_text(self, text):
        return [0.0, 1.0, 0.0]


class BrokenClip(FakeClip):
    def embed_text(self, text):
        raise RuntimeError("embedding failed")


class SequencedClip(FakeClip):
    def __init__(self, image_embeddings):
        self.image_embeddings = list(image_embeddings)

    def embed_image(self, path):
        return self.image_embeddings.pop(0)


class FakeFace:
    enabled = True
    error = None
    identity_model = "test-face"
    identity_ready = True

    def detect(self, path):
        return [{"bbox": [0, 0, 10, 10], "confidence": 0.95, "embedding": [1.0, 0.0, 0.0], "embedding_model": "test-face", "embedding_version": "v1", "identity_ready": True}]


class PipelineTests(unittest.TestCase):
    def test_image_runs_native_observation_event_entity_and_vectors(self):
        with tempfile.TemporaryDirectory() as directory:
            image = Path(directory) / "family.jpg"
            image.write_bytes(b"test")
            store = MemoryStore(f"{directory}/memory.db")
            pipeline = IngestionPipeline(store, gamma=FakeGamma(), face=FakeFace(), clip=FakeClip())
            asset = pipeline.create_asset(image)
            result = pipeline.process(asset["id"])

            self.assertEqual(result["status"], "processed")
            observation = store.list_observations()[0]
            self.assertEqual(observation["ocr_text"], "生日快乐")
            self.assertEqual(store.count("events"), 1)
            self.assertEqual(store.count("face_clusters"), 1)
            self.assertEqual({item["entity_type"] for item in store.list_entities()}, {"person", "place", "object"})
            self.assertGreaterEqual(store.count("memory_vectors"), 3)
            self.assertEqual(store.get_asset(asset["id"])["metadata_json"]["faces"][0]["embedding_model"], "test-face")
            self.assertGreaterEqual(store.get_asset(asset["id"])["metadata_json"]["processing_seconds"], 0)
            self.assertIn("analysis_wall_seconds", store.get_asset(asset["id"])["metadata_json"]["processing_timings"])
            self.assertTrue(store.get_asset(asset["id"])["metadata_json"]["processing_timings"]["parallel"])

    def test_source_member_builds_semantic_profile_without_being_visible_in_photo(self):
        with tempfile.TemporaryDirectory() as directory:
            image = Path(directory) / "cake.jpg"
            image.write_bytes(b"test")
            store = MemoryStore(f"{directory}/memory.db")
            son = store.create_entity("儿子", "person", "confirmed", "孩子", 1.0)
            pipeline = IngestionPipeline(store, gamma=FakeGamma(), face=FakeFace(), clip=FakeClip())
            asset = pipeline.create_asset(image, metadata={
                "source_owner_id": son["id"],
                "source_owner_label": "儿子",
                "source_confidence": 1.0,
                "captured_at": "2026-07-01T18:00:00+08:00",
                "captured_location": "家中餐厅",
            })

            pipeline.process(asset["id"])

            profile = store.get_semantic_profile(son["id"])
            claims = store.list_semantic_claims(son["id"])
            self.assertIsNotNone(profile)
            self.assertTrue(any(claim["predicate"] == "拍摄" for claim in claims))

    def test_event_summary_is_generated_after_observations_are_clustered(self):
        with tempfile.TemporaryDirectory() as directory:
            image = Path(directory) / "cake.jpg"
            image.write_bytes(b"test")
            store = MemoryStore(f"{directory}/memory.db")
            pipeline = IngestionPipeline(store, gamma=FakeGamma(), face=FakeFace(), clip=FakeClip())
            asset = pipeline.create_asset(image, metadata={"captured_at": "2026-07-01T18:00:00+08:00", "captured_location": "家中餐厅"})
            pipeline.process(asset["id"])
            event = store.list_events()[0]

            updated = pipeline.summarize_event(event["id"])

            self.assertEqual(updated["title"], "生日庆祝")
            self.assertEqual(updated["event_type"], "庆祝活动")
            self.assertEqual(updated["summary"], "一组照片记录了围绕蛋糕的庆祝活动。")

    def test_dissimilar_images_at_same_time_and_place_split_before_summary(self):
        with tempfile.TemporaryDirectory() as directory:
            first_image = Path(directory) / "first.jpg"
            second_image = Path(directory) / "second.jpg"
            first_image.write_bytes(b"first")
            second_image.write_bytes(b"second")
            store = MemoryStore(f"{directory}/memory.db")
            pipeline = IngestionPipeline(
                store, gamma=ConflictingGamma(), face=FakeFace(),
                clip=SequencedClip([[1.0, 0.0], [0.0, 1.0]]),
            )
            metadata = {"captured_at": "2026-07-01T18:00:00+08:00", "captured_location": "家中餐厅", "source_album_id": "shared-album"}

            pipeline.process(pipeline.create_asset(first_image, metadata=metadata)["id"], summarize_event=False)
            pipeline.process(pipeline.create_asset(second_image, metadata=metadata)["id"], summarize_event=False)

            self.assertEqual(store.count("events"), 2)
            events = store.list_events()
            self.assertTrue(any(event["aggregation_breakdown"].get("split_guard") == "semantic_visual_conflict" for event in events))

    def test_confirmed_face_is_available_to_event_scoring_before_merge(self):
        with tempfile.TemporaryDirectory() as directory:
            first_image = Path(directory) / "first.jpg"
            second_image = Path(directory) / "second.jpg"
            first_image.write_bytes(b"first")
            second_image.write_bytes(b"second")
            store = MemoryStore(f"{directory}/memory.db")
            metadata = {"captured_at": "2026-07-01T18:00:00+08:00", "captured_location": "家中餐厅"}
            first_asset = store.create_asset("seed", first_image.name, "image", str(first_image), metadata=metadata)
            first = store.add_observation(first_asset["id"], {
                "captured_at": metadata["captured_at"], "place": "厨房近景", "activity": "准备晚餐", "event_type": "用餐",
            })
            store.upsert_vector("visual", "asset", first_asset["id"], [1.0, 0.0], "test-clip")
            store.merge_observation_into_event(first)
            face = store.add_face_instance(first_asset["id"], first["id"], {
                "bbox": [0, 0, 10, 10], "confidence": 0.95, "embedding": [1.0, 0.0, 0.0],
                "embedding_model": "test-face", "embedding_version": "v1",
            })
            store.confirm_face_cluster(face["cluster_id"], "妈妈", "母亲")
            pipeline = IngestionPipeline(
                store, gamma=ConflictingGamma(), face=FakeFace(), clip=SequencedClip([[0.0, 1.0]]),
            )

            result = pipeline.process(pipeline.create_asset(second_image, metadata=metadata)["id"], summarize_event=False)

            self.assertEqual(result["status"], "processed")
            self.assertEqual(store.count("events"), 1)
            observation = store.get_observation(result["metadata_json"]["observation_id"])
            self.assertTrue(any(person.get("name") == "妈妈" for person in observation["people"] if isinstance(person, dict)))

    def test_processing_an_asset_twice_is_idempotent(self):
        with tempfile.TemporaryDirectory() as directory:
            image = Path(directory) / "family.jpg"
            image.write_bytes(b"test")
            store = MemoryStore(f"{directory}/memory.db")
            pipeline = IngestionPipeline(store, gamma=FakeGamma(), face=FakeFace(), clip=FakeClip())
            asset = pipeline.create_asset(image)

            first = pipeline.process(asset["id"])
            second = pipeline.process(asset["id"])

            self.assertEqual(first["status"], "processed")
            self.assertEqual(second["status"], "processed")
            self.assertEqual(store.count("observations"), 1)
            self.assertEqual(store.count("face_instances"), 1)
            self.assertEqual(store.count("events"), 1)

    def test_fast_image_processing_exposes_evidence_before_semantic_enrichment(self):
        with tempfile.TemporaryDirectory() as directory:
            image = Path(directory) / "family.jpg"
            image.write_bytes(b"test")
            store = MemoryStore(f"{directory}/memory.db")
            pipeline = IngestionPipeline(store, gamma=FakeGamma(), face=FakeFace(), clip=FakeClip())
            asset = pipeline.create_asset(image, metadata={"captured_location": "家里"})

            fast = pipeline.process_fast_image(asset["id"])

            self.assertEqual(fast["status"], "semantic_enriching")
            self.assertEqual(store.count("face_instances"), 1)
            self.assertEqual(store.count("events"), 1)
            observation = store.get_observation(fast["metadata_json"]["observation_id"])
            self.assertEqual(observation["caption"], "")
            self.assertEqual(observation["canonical"]["semantic_status"], "pending")

            complete = pipeline.enrich_fast_image(asset["id"], summarize_event=False)

            self.assertEqual(complete["status"], "processed")
            self.assertEqual(store.get_observation(observation["id"])["caption"], "一张带文字的家庭照片")
            self.assertTrue(any(item["canonical_name"] == "蛋糕" for item in store.list_entities()))

    def test_asset_import_persists_sha256_and_exif_boundary(self):
        with tempfile.TemporaryDirectory() as directory:
            image = Path(directory) / "family.jpg"
            image.write_bytes(b"test")
            store = MemoryStore(f"{directory}/memory.db")
            pipeline = IngestionPipeline(store, gamma=FakeGamma(), face=FakeFace(), clip=FakeClip())
            asset = pipeline.create_asset(image, metadata={"captured_at": "2026-07-01T18:00:00+08:00"})

            self.assertEqual(len(asset["content_sha256"]), 64)
            self.assertEqual(asset["captured_at"], "2026-07-01T18:00:00+08:00")
            self.assertIn("exif", asset["metadata_json"])

    def test_asset_import_discards_event_and_activity_hints(self):
        with tempfile.TemporaryDirectory() as directory:
            image = Path(directory) / "family.jpg"
            image.write_bytes(b"test")
            store = MemoryStore(f"{directory}/memory.db")
            pipeline = IngestionPipeline(store, gamma=FakeGamma(), face=FakeFace(), clip=FakeClip())
            asset = pipeline.create_asset(image, metadata={
                "captured_at": "2026-07-01T18:00:00+08:00",
                "captured_location": "家中餐厅",
                "event_id": "birthday",
                "activity_hint": "生日聚会",
                "source_identity": "not-a-provenance-field",
            })

            imported = asset["metadata_json"]
            self.assertNotIn("event_id", imported)
            self.assertNotIn("activity_hint", imported)
            self.assertNotIn("source_identity", imported)
            self.assertEqual(asset["captured_location"], "家中餐厅")

    def test_same_content_reuses_existing_asset(self):
        with tempfile.TemporaryDirectory() as directory:
            first_path = Path(directory) / "first.jpg"
            second_path = Path(directory) / "second.jpg"
            first_path.write_bytes(b"same-content")
            second_path.write_bytes(b"same-content")
            store = MemoryStore(f"{directory}/memory.db")
            pipeline = IngestionPipeline(store, gamma=FakeGamma(), face=FakeFace(), clip=FakeClip())

            first = pipeline.create_asset(first_path)
            second = pipeline.create_asset(second_path)

            self.assertEqual(first["id"], second["id"])
            self.assertEqual(store.count("assets"), 1)

    def test_failed_processing_cleans_up_partial_derived_memory(self):
        with tempfile.TemporaryDirectory() as directory:
            image = Path(directory) / "broken.jpg"
            image.write_bytes(b"test")
            store = MemoryStore(f"{directory}/memory.db")
            pipeline = IngestionPipeline(store, gamma=FakeGamma(), face=FakeFace(), clip=BrokenClip())
            asset = pipeline.create_asset(image)

            result = pipeline.process(asset["id"])

            self.assertEqual(result["status"], "failed")
            self.assertEqual(store.count("observations"), 0)
            self.assertEqual(store.count("face_instances"), 0)
            self.assertEqual(store.count("events"), 0)
            self.assertEqual(store.count("memory_vectors"), 0)


if __name__ == "__main__":
    unittest.main()
