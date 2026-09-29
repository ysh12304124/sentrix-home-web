"""TFPE v2: StructuredMemoryExecutor — exact SQL count/first/last/group/place/entity."""

import unittest

from backend.db import MemoryStore
from backend.query_contracts import Constraint, HARD, QueryParseDraft, QuerySpec
from backend.structured_memory import StructuredMemoryExecutor

_SEED = [
    # asset_id, captured_at, media_type, captured_location, observation_id, place
    ("a1", "2024-01-15T10:00:00", "image", "上海", "obs1", "上海"),
    ("a2", "2024-03-20T09:00:00", "image", "杭州", "obs2", "杭州"),
    ("a3", "2024-10-05T08:00:00", "image", "", "obs3", "上海"),
    ("a4", "2024-10-20T11:00:00", "video", "", "obs4", ""),
    ("a5", "2023-12-01T10:00:00", "image", "贵阳", "obs5", "贵阳"),
]


def _store():
    store = MemoryStore(":memory:")
    for asset_id, captured, media, location, obs_id, place in _SEED:
        store.create_asset(asset_id, f"{asset_id}.jpg", media, f"/x/{asset_id}.jpg",
                           metadata={"captured_at": captured, "captured_location": location},
                           scope_id="home")
        store.add_observation(asset_id, {"id": obs_id, "captured_at": captured, "place": place},
                              scope_id="home")
    # confirmed person 明哥 -> observation obs3 (asset a3)
    store.connection.execute(
        "INSERT INTO entities(id, scope_id, entity_type, canonical_name, status, created_at, updated_at) "
        "VALUES ('entity-1', 'home', 'person', '明哥', 'confirmed', '2024-01-01T00:00:00', '2024-01-01T00:00:00')")
    store.connection.execute(
        "INSERT INTO entity_mentions(id, entity_id, observation_id, confidence, created_at) "
        "VALUES ('em-1', 'entity-1', 'obs3', 0.9, '2024-01-01T00:00:00')")
    store.connection.commit()
    return store


def _draft(answer_type, structured=None):
    return QueryParseDraft(answer_type=answer_type, structured=structured or {})


def _spec(*constraints):
    return QuerySpec("q", "single", ["home"], "owner", "c", "answer", "general",
                     constraints=list(constraints))


_TIME_2024 = {"time_range": {"start": "2024-01-01", "end": "2025-01-01"}}


class StructuredMemoryTests(unittest.TestCase):
    def setUp(self):
        self.executor = StructuredMemoryExecutor(_store())

    def test_count_images_in_2024(self):
        draft = _draft("count", {"time_range": _TIME_2024["time_range"], "media_type": "image"})
        result = self.executor.execute(draft, _spec(), "structured_fact")
        self.assertEqual(result.total, 3)
        self.assertTrue(result.exact)

    def test_count_all_in_2024_includes_video(self):
        draft = _draft("count", _TIME_2024)
        result = self.executor.execute(draft, _spec(), "structured_fact")
        self.assertEqual(result.total, 4)

    def test_count_video_only(self):
        draft = _draft("count", {"time_range": _TIME_2024["time_range"], "media_type": "video"})
        result = self.executor.execute(draft, _spec(), "structured_fact")
        self.assertEqual(result.total, 1)

    def test_exists(self):
        draft = _draft("exists", _TIME_2024)
        result = self.executor.execute(draft, _spec(), "structured_fact")
        self.assertTrue(result.value)

    def test_exists_false_in_empty_period(self):
        draft = _draft("exists", {"time_range": {"start": "2030-01-01", "end": "2031-01-01"}})
        result = self.executor.execute(draft, _spec(), "structured_fact")
        self.assertFalse(result.value)

    def test_first_and_last(self):
        first = self.executor.execute(_draft("first_occurrence", _TIME_2024), _spec(), "structured_fact")
        last = self.executor.execute(_draft("last_occurrence", _TIME_2024), _spec(), "structured_fact")
        self.assertEqual(first.value, "2024-01-15T10:00:00")
        self.assertEqual(last.value, "2024-10-20T11:00:00")

    def test_group_by_month(self):
        draft = _draft("grouped_list", {
            **_TIME_2024,
            "aggregation": {"op": "group_by", "group_by": "month"},
        })
        result = self.executor.execute(draft, _spec(), "aggregation")
        counts = {row["group"]: row["count"] for row in result.rows}
        self.assertEqual(counts, {"2024-01": 1, "2024-03": 1, "2024-10": 2})

    def test_group_by_place(self):
        draft = _draft("grouped_list", {"aggregation": {"op": "group_by", "group_by": "place"}})
        result = self.executor.execute(draft, _spec(), "aggregation")
        counts = {row["group"]: row["count"] for row in result.rows}
        self.assertEqual(counts, {"上海": 2, "杭州": 1, "贵阳": 1, "未知": 1})

    def test_place_filter(self):
        draft = _draft("count", {"time_range": _TIME_2024["time_range"], "place": "上海"})
        result = self.executor.execute(draft, _spec(), "structured_fact")
        self.assertEqual(result.total, 2)

    def test_scope_authorization_not_bypassed(self):
        draft = _draft("count", _TIME_2024)
        spec = QuerySpec("q", "single", ["other-scope"], "owner", "c", "answer", "general", constraints=[])
        result = self.executor.execute(draft, spec, "structured_fact")
        self.assertEqual(result.total, 0)

    def test_person_last_occurrence(self):
        draft = _draft("last_occurrence", None)
        spec = _spec(Constraint("person", "明哥", HARD, "confirmed_bridge"))
        spec.entity_ids = ["entity-1"]
        result = self.executor.execute(draft, spec, "entity_fact")
        self.assertEqual(result.value, "2024-10-05T08:00:00")

    def test_person_count(self):
        draft = _draft("count", None)
        spec = _spec(Constraint("person", "明哥", HARD, "confirmed_bridge"))
        spec.entity_ids = ["entity-1"]
        result = self.executor.execute(draft, spec, "entity_fact")
        self.assertEqual(result.total, 1)

    def test_media_constraint_from_spec(self):
        draft = _draft("count", _TIME_2024)
        spec = _spec(Constraint("media", "video", HARD, "asset_metadata"))
        result = self.executor.execute(draft, spec, "structured_fact")
        self.assertEqual(result.total, 1)

    def test_inclusive_end_includes_last_day(self):
        store = MemoryStore(":memory:")
        store.create_asset("b1", "b1.jpg", "image", "/x/b1.jpg",
                           metadata={"captured_at": "2024-01-31T23:00:00"}, scope_id="home")
        store.add_observation("b1", {"id": "obsb1", "captured_at": "2024-01-31T23:00:00"}, scope_id="home")
        executor = StructuredMemoryExecutor(store)
        draft = _draft("count", {"time_range": {"start": "2024-01-01", "end": "2024-01-31"}})
        result = executor.execute(draft, _spec(), "structured_fact")
        self.assertEqual(result.total, 1)


if __name__ == "__main__":
    unittest.main()
