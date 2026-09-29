"""Retrieval index derived projection (Phase 3)."""

import os
import tempfile
import unittest

from backend.db import MemoryStore
from backend.retrieval_indexes import RetrievalIndex


class RetrievalIndexTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix="retrieval-index-")
        self.store = MemoryStore(os.path.join(self.directory.name, "memory.db"))
        self.index = RetrievalIndex(self.store)
        self.asset = self.store.create_asset(
            "asset-1", "a.jpg", "image", "/tmp/a.jpg",
            metadata={"captured_at": "2024-05-12T10:00:00"}, scope_id="home",
        )

    def tearDown(self):
        self.store.close()
        self.directory.cleanup()

    def _add_observation(self, **overrides):
        payload = {
            "id": overrides.get("id", "obs-1"),
            "scope_id": overrides.get("scope_id", "home"),
            "captured_at": "2024-05-12T10:00:00",
            "caption": overrides.get("caption", "厨房拿碗"),
            "place": overrides.get("place", "厨房"),
            "activity": overrides.get("activity", "拿碗"),
            "people": overrides.get("people", ["明哥"]),
            "clothing": overrides.get("clothing", ["红色外套"]),
            "objects": overrides.get("objects", ["碗"]),
            "confidence": 0.85,
        }
        return self.store.add_observation(self.asset["id"], payload, scope_id=payload["scope_id"])

    def test_refresh_populates_derived_rows_per_field(self):
        observation = self._add_observation()
        self.index.refresh_from_observation(observation)
        places = self.index.search("home", "place", "厨房")
        self.assertTrue(places)
        self.assertEqual(places[0]["observation_id"], "obs-1")
        clothing = self.index.search("home", "clothing", "红色")
        self.assertTrue(clothing)
        objects = self.index.search("home", "object", "碗")
        self.assertTrue(objects)

    def test_refresh_replaces_prior_rows_on_revision_change(self):
        first = self._add_observation()
        self.index.refresh_from_observation(first)
        # Simulate a revision update: the same observation is refreshed with a
        # different place; the old row must be gone.
        updated = dict(first, place="餐厅", revision=2)
        self.index.refresh_from_observation(updated)
        self.assertFalse(self.index.search("home", "place", "厨房"))
        self.assertTrue(self.index.search("home", "place", "餐厅"))

    def test_rebuild_all_regenerates_from_canonical(self):
        observation = self._add_observation()
        self.index.refresh_from_observation(observation)
        self.store.connection.execute("DELETE FROM observation_search_terms")
        self.store.connection.commit()
        rebuilt = self.index.rebuild_all()
        self.assertEqual(rebuilt, 1)
        self.assertTrue(self.index.search("home", "place", "厨房"))

    def test_scope_filter_isolates_terms(self):
        self.store.create_asset("asset-other", "b.jpg", "image", "/tmp/b.jpg", scope_id="other")
        obs_home = self._add_observation()
        self.index.refresh_from_observation(obs_home)
        obs_other = self.store.add_observation(
            "asset-other",
            {"id": "obs-other", "scope_id": "other", "captured_at": "2024-05-12T10:00:00",
             "caption": "户外风景", "place": "公园", "activity": "散步", "confidence": 0.8},
            scope_id="other",
        )
        self.index.refresh_from_observation(obs_other)
        self.assertTrue(self.index.search("other", "place", "公园"))
        self.assertFalse(self.index.search("home", "place", "公园"))


if __name__ == "__main__":
    unittest.main()
