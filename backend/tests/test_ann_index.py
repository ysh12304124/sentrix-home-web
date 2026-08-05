"""ANN adapter protocol tests (Phase 3.5.2).

The default backend is :class:`SqliteBaselineIndex`; real FAISS/HNSW backends
plug in once the 153 benchmark picks a library.  These tests lock down the
protocol so later backends can be added without breaking callers.
"""

import os
import tempfile
import unittest

from backend.retrieval_ann import SqliteBaselineIndex, create_index


class AnnIndexProtocolTests(unittest.TestCase):
    def setUp(self):
        self.index = SqliteBaselineIndex()
        self.vectors = [
            ("v1", [1.0, 0.0, 0.0], {"scope_id": "home", "asset_id": "a1"}),
            ("v2", [0.9, 0.1, 0.0], {"scope_id": "home", "asset_id": "a2"}),
            ("v3", [0.0, 1.0, 0.0], {"scope_id": "home", "asset_id": "a3"}),
            ("v4", [0.0, 0.0, 1.0], {"scope_id": "other", "asset_id": "a4"}),
        ]

    def test_build_and_search_returns_top_k_by_cosine(self):
        self.index.build(self.vectors)
        results = self.index.search([1.0, 0.0, 0.0], k=2)
        self.assertEqual([item[0] for item in results], ["v1", "v2"])

    def test_scope_filter_restricts_candidates(self):
        self.index.build(self.vectors)
        results = self.index.search([0.0, 0.0, 1.0], k=3, scope_id="home")
        self.assertNotIn("v4", [item[0] for item in results])

    def test_add_then_remove_updates_recall(self):
        self.index.build(self.vectors)
        self.index.add([("v5", [1.0, 0.0, 0.1], {"scope_id": "home", "asset_id": "a5"})])
        results = self.index.search([1.0, 0.0, 0.0], k=1)
        self.assertEqual(results[0][0], "v1")
        self.index.remove(["v1", "v2"])
        results = self.index.search([1.0, 0.0, 0.0], k=1)
        self.assertEqual(results[0][0], "v5")

    def test_save_and_load_roundtrip(self):
        self.index.build(self.vectors)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".json") as handle:
            path = handle.name
        try:
            self.index.save(path)
            fresh = SqliteBaselineIndex()
            fresh.load(path)
            results = fresh.search([1.0, 0.0, 0.0], k=1)
            self.assertEqual(results[0][0], "v1")
        finally:
            os.unlink(path)

    def test_factory_returns_baseline_by_default(self):
        index = create_index()
        self.assertIsInstance(index, SqliteBaselineIndex)


if __name__ == "__main__":
    unittest.main()
