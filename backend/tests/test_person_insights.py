import tempfile
import unittest

from backend.db import MemoryStore
from backend.person_insights import core_person_score, cosine, rank_core_people, select_representatives


class CorePersonRankerTests(unittest.TestCase):
    def test_core_ranker_prefers_cross_date_recurrence_over_one_day_volume(self):
        rows = [
            {"person_id": "repeat", "date_count": 6, "event_count": 5,
             "member_count": 8, "co_person_count": 3, "scene_count": 3,
             "quality": 0.82, "confirmed": False},
            {"person_id": "burst", "date_count": 1, "event_count": 1,
             "member_count": 40, "co_person_count": 1, "scene_count": 1,
             "quality": 0.90, "confirmed": False},
        ]
        ranked = rank_core_people(rows, limit=10)
        self.assertEqual(ranked[0]["person_id"], "repeat")
        self.assertEqual(ranked[1]["tier"], "incidental")

    def test_confirmed_people_sort_first(self):
        rows = [
            {"person_id": "high_volume", "date_count": 1, "event_count": 1,
             "member_count": 40, "co_person_count": 1, "scene_count": 1,
             "quality": 0.90, "confirmed": False},
            {"person_id": "confirmed_low", "date_count": 1, "event_count": 1,
             "member_count": 2, "co_person_count": 1, "scene_count": 1,
             "quality": 0.60, "confirmed": True},
        ]
        ranked = rank_core_people(rows, limit=10)
        self.assertEqual(ranked[0]["person_id"], "confirmed_low")
        self.assertEqual(ranked[0]["tier"], "core")

    def test_rank_core_never_exceeds_ten(self):
        rows = [
            {"person_id": f"p{i}", "date_count": 3, "event_count": 2,
             "member_count": 5, "co_person_count": 2, "scene_count": 2,
             "quality": 0.8, "confirmed": False}
            for i in range(15)
        ]
        ranked = rank_core_people(rows, limit=10)
        self.assertEqual(len(ranked), 15)
        core = [item for item in ranked if item["tier"] == "core"]
        self.assertLessEqual(len(core), 10)

    def test_low_evidence_people_become_incidental(self):
        rows = [
            {"person_id": "low", "date_count": 1, "event_count": 1,
             "member_count": 1, "co_person_count": 0, "scene_count": 0,
             "quality": 0.5, "confirmed": False},
            {"person_id": "normal", "date_count": 3, "event_count": 2,
             "member_count": 4, "co_person_count": 2, "scene_count": 2,
             "quality": 0.8, "confirmed": False},
        ]
        ranked = rank_core_people(rows, limit=10)
        by_id = {item["person_id"]: item for item in ranked}
        self.assertEqual(by_id["low"]["tier"], "incidental")

    def test_rank_is_stable_for_equal_inputs(self):
        rows = [
            {"person_id": "a", "date_count": 2, "event_count": 2,
             "member_count": 4, "co_person_count": 2, "scene_count": 2,
             "quality": 0.8, "confirmed": False},
            {"person_id": "b", "date_count": 2, "event_count": 2,
             "member_count": 4, "co_person_count": 2, "scene_count": 2,
             "quality": 0.8, "confirmed": False},
        ]
        first = [item["person_id"] for item in rank_core_people(rows)]
        second = [item["person_id"] for item in rank_core_people(rows)]
        self.assertEqual(first, second)

    def test_core_score_is_bounded_and_breaks_down(self):
        row = {"person_id": "p", "date_count": 6, "event_count": 5,
               "member_count": 8, "co_person_count": 3, "scene_count": 3,
               "quality": 0.82, "confirmed": True}
        score = core_person_score(row)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)
        self.assertIn("score_breakdown", row)
        self.assertEqual(row["score_breakdown"]["single_day_penalty"], 0.0)


class RepresentativeSelectorTests(unittest.TestCase):
    def _candidate(self, i, event_id, quality=0.9, body_visibility=0.8,
                   captured_at="2026-01-01T10:00:00"):
        return {
            "face_instance_id": f"f{i}",
            "asset_id": f"a{i}",
            "event_id": event_id,
            "quality": quality,
            "body_visibility": body_visibility,
            "captured_at": captured_at,
        }

    def test_per_event_cap_is_two(self):
        candidates = [
            self._candidate(i, event_id=f"e{i % 2}", captured_at=f"2026-01-0{i + 1}T10:00:00")
            for i in range(6)
        ]
        selected = select_representatives(candidates, {}, limit=12, per_event=2)
        self.assertEqual(len(selected), 4)

    def test_degrades_to_quality_then_date_without_vectors(self):
        candidates = [
            self._candidate(0, "e1", quality=0.5, captured_at="2026-02-01T10:00:00"),
            self._candidate(1, "e1", quality=0.9, captured_at="2026-01-01T10:00:00"),
            self._candidate(2, "e1", quality=0.9, captured_at="2026-01-02T10:00:00"),
        ]
        selected = select_representatives(candidates, {}, limit=12, per_event=2)
        # 每事件最多 2 张；优先质量，其次日期。
        self.assertEqual(len(selected), 2)
        self.assertEqual(selected[0]["face_instance_id"], "f1")
        self.assertEqual(selected[1]["face_instance_id"], "f2")

    def test_near_duplicate_asset_is_dropped(self):
        candidates = [
            self._candidate(0, "e1"),
            self._candidate(1, "e1", captured_at="2026-01-02T10:00:00"),
        ]
        vectors = {"a0": [1.0, 0.0], "a1": [0.999, 0.001]}
        selected = select_representatives(candidates, vectors, limit=12, per_event=2,
                                          duplicate_threshold=0.94)
        self.assertEqual(len(selected), 1)
        self.assertEqual(selected[0]["asset_id"], "a0")

    def test_selection_records_breakdown_version(self):
        candidate = self._candidate(0, "e1")
        selected = select_representatives([candidate], {}, limit=12, per_event=2)
        self.assertIn("selection_json", selected[0])
        self.assertEqual(selected[0]["selection_json"]["version"], "representative-v1")


class CosineTests(unittest.TestCase):
    def test_cosine_identical_vectors_is_one(self):
        self.assertAlmostEqual(cosine([1.0, 0.0], [1.0, 0.0]), 1.0, places=6)

    def test_cosine_orthogonal_vectors_is_zero(self):
        self.assertAlmostEqual(cosine([1.0, 0.0], [0.0, 1.0]), 0.0, places=6)

    def test_cosine_handles_zero_vector(self):
        self.assertEqual(cosine([0.0, 0.0], [1.0, 0.0]), 0.0)


class PersonCandidateFeaturesTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.store = MemoryStore(f"{self.temp_dir.name}/sentrix.db")
        self.store.create_memory_space("album-a", "相册 A")
        self.store.create_memory_space("album-b", "相册 B")

    def tearDown(self):
        self.store.close()
        self.temp_dir.cleanup()

    def _person_fixture(self, scope_id, prefix):
        asset_1 = self.store.create_asset(
            f"{prefix}1", f"{prefix}1.jpg", "image", f"/tmp/{prefix}1.jpg", "image/jpeg", scope_id=scope_id
        )
        asset_2 = self.store.create_asset(
            f"{prefix}2", f"{prefix}2.jpg", "image", f"/tmp/{prefix}2.jpg", "image/jpeg", scope_id=scope_id
        )
        obs_1 = self.store.add_observation(
            f"{prefix}1", {"caption": "合影", "captured_at": "2026-01-01T10:00:00"}, scope_id=scope_id
        )
        obs_2 = self.store.add_observation(
            f"{prefix}2", {"caption": "再见面", "captured_at": "2026-01-02T10:00:00"}, scope_id=scope_id
        )
        face_1 = self.store.add_face_instance(
            f"{prefix}1", obs_1["id"],
            {"bbox": [1, 2, 3, 4], "confidence": 0.95, "quality": 0.9, "embedding": [1, 0, 0]},
        )
        self.store.add_face_instance(
            f"{prefix}2", obs_2["id"],
            {"bbox": [1, 2, 3, 4], "confidence": 0.95, "quality": 0.8, "embedding": [1, 0, 0]},
        )
        self.store.merge_observation_into_event(obs_1)
        self.store.merge_observation_into_event(obs_2)
        return face_1["cluster_id"]

    def test_features_are_scope_isolated_and_complete(self):
        cluster_a = self._person_fixture("album-a", "pa")
        cluster_b = self._person_fixture("album-b", "pb")
        features = self.store.person_candidate_features("album-a")
        cluster_ids = {feature["cluster_id"] for feature in features}
        self.assertIn(cluster_a, cluster_ids)
        self.assertNotIn(cluster_b, cluster_ids)
        for feature in features:
            for key in ("person_id", "cluster_id", "date_count", "event_count",
                        "member_count", "co_person_count", "scene_count",
                        "quality", "confirmed"):
                self.assertIn(key, feature)
            self.assertEqual(feature["date_count"], 2)


if __name__ == "__main__":
    unittest.main()
