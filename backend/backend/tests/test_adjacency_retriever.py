"""Phase R R3B — seed-based adjacency expansion and budgets."""

import unittest

from backend.db import MemoryStore
from backend.retrieval import HardFilterContext, RetrievalConfig
from backend.retrieval.adjacency import AdjacencyRetriever


class AdjacencyRetrieverTests(unittest.TestCase):
    def _store(self):
        store = MemoryStore(":memory:")
        now = "2026-08-06T00:00:00Z"
        assets = [
            ("asset_1", "album1", "2023-11-16T14:00:00+00:00"),
            ("asset_2", "album1", "2023-11-16T14:05:00+00:00"),
            ("asset_3", "album1", "2023-11-16T15:00:00+00:00"),
            ("asset_4", "album2", "2023-11-16T14:02:00+00:00"),  # other scope
        ]
        for asset_id, scope, captured in assets:
            store.create_asset(asset_id, f"{asset_id}.JPG", "image", f"/tmp/{asset_id}", "image/jpeg", 1,
                               {"scope_id": scope, "captured_at": captured})
        for asset_id, captured in [("asset_1", "2023-11-16T14:00:00+00:00"),
                                   ("asset_2", "2023-11-16T14:05:00+00:00"),
                                   ("asset_3", "2023-11-16T15:00:00+00:00")]:
            store.connection.execute(
                """INSERT INTO observations (id, scope_id, asset_id, captured_at, source_type, caption,
                    people_json, objects_json, ocr_text, confidence, raw_json, canonical_json,
                    clothing_json, spatial_relations_json, revision, created_at, updated_at)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (f"obs_{asset_id}", "album1", asset_id, captured, "vision", "x",
                 "[]", "[]", "", 0.9, "{}", "{}", "[]", "[]", 1, now, now),
            )
        # event_1 links obs_asset_1 and obs_asset_3 (same event).
        store.connection.execute("INSERT INTO events (id, scope_id, title, event_type, created_at, updated_at) VALUES (?,?,?,?,?,?)",
                                 ("event_1", "album1", "同一事件", "gathering", now, now))
        store.connection.execute("INSERT INTO event_observations (event_id, observation_id) VALUES (?,?)",
                                 ("event_1", "obs_asset_1"))
        store.connection.execute("INSERT INTO event_observations (event_id, observation_id) VALUES (?,?)",
                                 ("event_1", "obs_asset_3"))
        store.connection.commit()
        return store

    def _retriever(self, store):
        return AdjacencyRetriever(store, config=RetrievalConfig())

    def test_event_expansion_finds_sibling(self):
        store = self._store()
        try:
            filters = HardFilterContext(scope_ids=("album1",))
            hits = self._retriever(store).expand(["asset_1"], filters, limit=10)
            ids = {hit.asset_id for hit in hits}
            self.assertIn("asset_3", ids)  # same event via event_observations
        finally:
            store.close()

    def test_time_window_expansion(self):
        store = self._store()
        try:
            filters = HardFilterContext(scope_ids=("album1",))
            # asset_1 and asset_2 are 5 minutes apart -> within 120min window.
            hits = self._retriever(store).expand(["asset_1"], filters, limit=10)
            ids = {hit.asset_id for hit in hits}
            self.assertIn("asset_2", ids)
        finally:
            store.close()

    def test_scope_isolated(self):
        store = self._store()
        try:
            filters = HardFilterContext(scope_ids=("album1",))
            hits = self._retriever(store).expand(["asset_1"], filters, limit=10)
            self.assertNotIn("asset_4", {hit.asset_id for hit in hits})
        finally:
            store.close()

    def test_no_seeds_no_hits(self):
        store = self._store()
        try:
            filters = HardFilterContext(scope_ids=("album1",))
            self.assertEqual(self._retriever(store).expand([], filters, limit=10), [])
        finally:
            store.close()

    def test_seed_excluded_from_own_expansion(self):
        store = self._store()
        try:
            filters = HardFilterContext(scope_ids=("album1",))
            hits = self._retriever(store).expand(["asset_1"], filters, limit=10)
            self.assertNotIn("asset_1", {hit.asset_id for hit in hits})
        finally:
            store.close()


if __name__ == "__main__":
    unittest.main()
