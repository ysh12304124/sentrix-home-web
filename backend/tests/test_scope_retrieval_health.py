import tempfile
import unittest
from pathlib import Path

from backend.db import MemoryStore
from backend.scope_finalize import verify_scope_retrieval


class ScopeRetrievalHealthTests(unittest.TestCase):
    """核对必须能真的抓到"相册建好但检索失效"。"""

    def _store(self, directory):
        return MemoryStore(str(Path(directory) / "test.db"))

    @staticmethod
    def _add_image(store, asset_id, scope_id):
        store.create_asset(asset_id, f"{asset_id}.jpg", "image",
                           f"/tmp/{asset_id}.jpg", "image/jpeg", 1024, {}, scope_id=scope_id)

    def test_flags_scope_with_images_but_no_visual_vectors(self):
        """这正是 46 上踩过的状态：资产全 processed，视觉向量一条都没有。"""
        with tempfile.TemporaryDirectory() as directory:
            store = self._store(directory)
            try:
                for index in range(3):
                    self._add_image(store, f"asset_{index}", "scope-broken")
                health = verify_scope_retrieval(store, "scope-broken")
            finally:
                store.close()

        self.assertFalse(health["ok"])
        self.assertEqual(health["images"], 3)
        self.assertEqual(health["visual_vectors"], 0)
        self.assertTrue(any("视觉检索会永远返回空" in item for item in health["problems"]))

    def test_passes_when_every_image_has_a_visual_vector(self):
        with tempfile.TemporaryDirectory() as directory:
            store = self._store(directory)
            try:
                self._add_image(store, "asset_ok", "scope-ok")
                store.upsert_vector("visual", "asset", "asset_ok", [0.1] * 768,
                                    "chinese-clip-ViT-L-14", {"scope_id": "scope-ok"})
                health = verify_scope_retrieval(store, "scope-ok")
            finally:
                store.close()

        self.assertTrue(health["ok"], health["problems"])
        self.assertEqual(health["visual_vectors"], 1)

    def test_reports_partial_coverage(self):
        with tempfile.TemporaryDirectory() as directory:
            store = self._store(directory)
            try:
                self._add_image(store, "asset_a", "scope-partial")
                self._add_image(store, "asset_b", "scope-partial")
                store.upsert_vector("visual", "asset", "asset_a", [0.1] * 768,
                                    "chinese-clip-ViT-L-14", {"scope_id": "scope-partial"})
                health = verify_scope_retrieval(store, "scope-partial")
            finally:
                store.close()

        self.assertFalse(health["ok"])
        self.assertTrue(any("检索不到" in item for item in health["problems"]))


if __name__ == "__main__":
    unittest.main()
