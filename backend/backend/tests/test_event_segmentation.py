import tempfile
import unittest

from backend.db import MemoryStore


class EventSegmentationTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.store = MemoryStore(f"{self.temp_dir.name}/memory.db")

    def tearDown(self):
        self.store.close()
        self.temp_dir.cleanup()

    def _observation(self, asset_id, filename, vector, activity="家庭活动", event_type="家庭记录", objects=None, people=None, vector_model="test-clip", source_album_id="shared-album"):
        asset = self.store.create_asset(asset_id, filename, "image", f"/tmp/{filename}", metadata={
            "captured_at": "2026-07-01T18:00:00+08:00", "captured_location": "家中餐厅", "source_album_id": source_album_id,
        })
        observation = self.store.add_observation(asset["id"], {
            "captured_at": "2026-07-01T18:00:00+08:00", "place": "家中餐厅",
            "activity": activity, "event_type": event_type, "objects": objects or [], "people": people or [],
        })
        self.store.upsert_vector("visual", "asset", asset["id"], vector, vector_model, {"observation_id": observation["id"]})
        return observation

    def test_similar_visual_evidence_merges_when_caption_activity_is_generic(self):
        first = self._observation("asset_a", "a.jpg", [1.0, 0.0])
        second = self._observation("asset_b", "b.jpg", [0.98, 0.02])

        first_event = self.store.merge_observation_into_event(first)
        second_event = self.store.merge_observation_into_event(second)

        self.assertEqual(first_event["id"], second_event["id"])
        self.assertGreaterEqual(second_event["aggregation_breakdown"]["visual_similarity"], 0.9)
        self.assertTrue(second_event["aggregation_breakdown"]["visual_available"])

    def test_dissimilar_visual_evidence_splits_with_conflicting_activity_and_type(self):
        first = self._observation("asset_a", "a.jpg", [1.0, 0.0], "准备晚餐", "用餐")
        second = self._observation("asset_b", "b.jpg", [0.0, 1.0], "公开演讲", "演讲")

        first_event = self.store.merge_observation_into_event(first)
        second_event = self.store.merge_observation_into_event(second)

        self.assertNotEqual(first_event["id"], second_event["id"])
        self.assertEqual(second_event["aggregation_breakdown"]["split_guard"], "semantic_visual_conflict")
        self.assertEqual(second_event["aggregation_breakdown"]["candidates"][0]["breakdown"]["confirmed_people"], 0.0)

    def test_album_provenance_does_not_override_semantic_conflict_without_vectors(self):
        first = self._observation(
            "asset_a", "a.jpg", [], "准备晚餐", "用餐", source_album_id="evt_birthday",
        )
        second = self._observation(
            "asset_b", "b.jpg", [], "公开演讲", "演讲", source_album_id="evt_birthday",
        )

        first_event = self.store.merge_observation_into_event(first)
        second_event = self.store.merge_observation_into_event(second)

        self.assertNotEqual(first_event["id"], second_event["id"])
        breakdown = second_event["aggregation_breakdown"]["candidates"][0]["breakdown"]
        self.assertNotIn("album", breakdown)
        self.assertEqual(second_event["aggregation_breakdown"]["candidates"][0]["breakdown"]["people"], 0.0)

    def test_album_provenance_does_not_override_semantic_conflict_without_vectors(self):
        first = self._observation(
            "asset_a", "a.jpg", [], "准备晚餐", "用餐", source_album_id="evt_birthday",
        )
        second = self._observation(
            "asset_b", "b.jpg", [], "公开演讲", "演讲", source_album_id="evt_birthday",
        )

        first_event = self.store.merge_observation_into_event(first)
        second_event = self.store.merge_observation_into_event(second)

        self.assertNotEqual(first_event["id"], second_event["id"])
        breakdown = second_event["aggregation_breakdown"]["candidates"][0]["breakdown"]
        self.assertNotIn("album", breakdown)

    def test_dissimilar_visual_evidence_alone_does_not_split_an_event(self):
        first = self._observation("asset_a", "a.jpg", [1.0, 0.0])
        second = self._observation("asset_b", "b.jpg", [0.0, 1.0])

        first_event = self.store.merge_observation_into_event(first)
        second_event = self.store.merge_observation_into_event(second)

        self.assertEqual(first_event["id"], second_event["id"])
        self.assertTrue(second_event["aggregation_breakdown"]["visual_available"])
        self.assertIsNone(second_event["aggregation_breakdown"]["split_guard"])

    def test_visual_scoring_does_not_compare_different_embedding_models(self):
        first = self._observation("asset_a", "a.jpg", [1.0, 0.0], vector_model="clip-v1")
        second = self._observation("asset_b", "b.jpg", [1.0, 0.0], vector_model="clip-v2")

        first_event = self.store.merge_observation_into_event(first)
        second_event = self.store.merge_observation_into_event(second)

        self.assertEqual(first_event["id"], second_event["id"])
        self.assertFalse(second_event["aggregation_breakdown"]["visual_available"])

    def test_visual_scoring_ignores_mismatched_vector_dimensions(self):
        first = self._observation("asset_a", "a.jpg", [1.0, 0.0])
        second = self._observation("asset_b", "b.jpg", [1.0, 0.0, 0.0])

        first_event = self.store.merge_observation_into_event(first)
        second_event = self.store.merge_observation_into_event(second)

        self.assertEqual(first_event["id"], second_event["id"])
        self.assertFalse(second_event["aggregation_breakdown"]["visual_available"])

    def test_model_people_descriptions_do_not_bridge_a_conflicting_event(self):
        first = self._observation(
            "asset_a", "a.jpg", [1.0, 0.0], "准备晚餐", "用餐",
            people=[{"description": "一位成年人"}],
        )
        second = self._observation(
            "asset_b", "b.jpg", [0.0, 1.0], "公开演讲", "演讲",
            people=[{"description": "一位成年人"}],
        )

        first_event = self.store.merge_observation_into_event(first)
        second_event = self.store.merge_observation_into_event(second)

        self.assertNotEqual(first_event["id"], second_event["id"])
        self.assertEqual(second_event["aggregation_breakdown"]["split_guard"], "semantic_visual_conflict")

    def test_event_lookup_indexes_exist_after_schema_creation(self):
        indexes = {
            row["name"] for row in self.store.connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'index'"
            ).fetchall()
        }

        self.assertIn("idx_event_observations_observation", indexes)
        self.assertIn("idx_memory_vectors_visual_asset", indexes)


if __name__ == "__main__":
    unittest.main()
