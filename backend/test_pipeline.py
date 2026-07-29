import tempfile
import unittest
from pathlib import Path

from backend.db import MemoryStore
from backend.pipeline import IngestionPipeline


class FakeGamma:
    model = "test-gamma"

    def analyze_image(self, path, metadata=None):
        return {"caption": "一张带文字的家庭照片", "activity": "聚会", "place": "家里", "people": [], "objects": ["蛋糕"], "ocr_text": "生日快乐", "event_type": "聚会", "facts": [], "confidence": 0.8, "model": self.model}


class FakeClip:
    model_name = "test-clip"
    error = None

    def embed_image(self, path):
        return [1.0, 0.0, 0.0]

    def embed_text(self, text):
        return [0.0, 1.0, 0.0]


class FakeFace:
    enabled = True
    error = None

    def detect(self, path):
        return [{"bbox": [0, 0, 10, 10], "confidence": 0.95, "embedding": [1.0, 0.0, 0.0]}]


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


if __name__ == "__main__":
    unittest.main()
