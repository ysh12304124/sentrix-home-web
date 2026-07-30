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
            self.assertEqual(store.count("entities"), 1)
            self.assertGreaterEqual(store.count("memory_vectors"), 3)
            self.assertEqual(store.get_asset(asset["id"])["metadata_json"]["faces"][0]["embedding_model"], "test-face")

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
