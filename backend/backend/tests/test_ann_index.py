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


try:  # pragma: no cover - environment probe
    import hnswlib  # noqa: F401
    HNSWLIB_AVAILABLE = True
except Exception:
    HNSWLIB_AVAILABLE = False


@unittest.skipUnless(HNSWLIB_AVAILABLE, "hnswlib not installed in this environment")
class HnswlibIndexProtocolTests(unittest.TestCase):
    """Phase 3.5.2 production backend integration."""

    def setUp(self):
        from backend.retrieval_ann import HnswlibIndex

        self.index = HnswlibIndex(max_elements=64)
        self.vectors = [
            ("v1", [1.0, 0.0, 0.0], {"scope_id": "home", "asset_id": "a1"}),
            ("v2", [0.9, 0.1, 0.0], {"scope_id": "home", "asset_id": "a2"}),
            ("v3", [0.0, 1.0, 0.0], {"scope_id": "home", "asset_id": "a3"}),
            ("v4", [0.0, 0.0, 1.0], {"scope_id": "other", "asset_id": "a4"}),
        ]

    def test_search_returns_nearest_neighbors_in_cosine_order(self):
        self.index.build(self.vectors)
        results = self.index.search([1.0, 0.0, 0.0], k=2)
        ids = [item[0] for item in results]
        self.assertEqual(ids[:2], ["v1", "v2"])

    def test_incremental_add_extends_the_index_without_rebuild(self):
        self.index.build(self.vectors)
        self.index.add([("v5", [0.99, 0.01, 0.0], {"scope_id": "home", "asset_id": "a5"})])
        results = self.index.search([1.0, 0.0, 0.0], k=1)
        self.assertEqual(results[0][0], "v1")
        top3 = [item[0] for item in self.index.search([1.0, 0.0, 0.0], k=3)]
        self.assertIn("v5", top3)

    def test_mark_deleted_removes_id_from_results(self):
        self.index.build(self.vectors)
        self.index.remove(["v1"])
        results = self.index.search([1.0, 0.0, 0.0], k=3)
        self.assertNotIn("v1", [item[0] for item in results])

    def test_scope_filter_falls_back_to_kernel(self):
        self.index.build(self.vectors)
        results = self.index.search([0.0, 0.0, 1.0], k=3, scope_id="home")
        # v4 lives in a different scope and must not leak through.
        self.assertNotIn("v4", [item[0] for item in results])

    def test_save_load_roundtrip_preserves_recall(self):
        import os
        import tempfile

        self.index.build(self.vectors)
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "index")
            self.index.save(path)
            from backend.retrieval_ann import HnswlibIndex

            fresh = HnswlibIndex(max_elements=64)
            fresh.load(path)
            results = fresh.search([1.0, 0.0, 0.0], k=2)
            self.assertEqual([item[0] for item in results[:2]], ["v1", "v2"])

    def test_factory_returns_hnswlib_backend(self):
        index = create_index("hnswlib", max_elements=32)
        from backend.retrieval_ann import HnswlibIndex

        self.assertIsInstance(index, HnswlibIndex)


if __name__ == "__main__":
    unittest.main()
