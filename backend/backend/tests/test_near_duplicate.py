"""Phase R R3B — near-duplicate grouping (P0-13)."""

import unittest

from backend.db import MemoryStore
from backend.retrieval import NearDuplicateGrouper


class NearDuplicateGrouperTests(unittest.TestCase):
    def _store(self):
        store = MemoryStore(":memory:")
        rows = [
            ("asset_1", "IMG_1.JPG", "sha_a"),
            ("asset_2", "IMG_2.JPG", "sha_a"),
            ("asset_3", "IMG_3.JPG", "sha_b"),
            ("asset_4", "IMG_4.JPG", "sha_c"),
        ]
        for asset_id, file_name, sha in rows:
            store.create_asset(asset_id, file_name, "image", f"/tmp/{file_name}", "image/jpeg", 1,
                               {"scope_id": "album1", "content_sha256": sha})
        return store

    def test_groups_by_sha(self):
        store = self._store()
        try:
            assets = [{"asset_id": "asset_1"}, {"asset_id": "asset_2"},
                      {"asset_id": "asset_3"}, {"asset_id": "asset_4"}]
            groups = NearDuplicateGrouper(store).groups(assets)
            sha_a_group = [members for members in groups.values() if "asset_1" in members]
            self.assertEqual(len(sha_a_group), 1)
            self.assertIn("asset_2", sha_a_group[0])
            singles = [members for members in groups.values() if "asset_3" in members]
            self.assertEqual(len(singles[0]), 1)
        finally:
            store.close()

    def test_annotate_adds_group_size(self):
        store = self._store()
        try:
            assets = [{"asset_id": "asset_1"}, {"asset_id": "asset_2"}, {"asset_id": "asset_3"}]
            NearDuplicateGrouper(store).annotate(assets)
            by_id = {asset["asset_id"]: asset for asset in assets}
            self.assertEqual(by_id["asset_1"]["near_duplicate_size"], 2)
            self.assertEqual(by_id["asset_2"]["near_duplicate_size"], 2)
            self.assertEqual(by_id["asset_1"]["near_duplicate_group"], by_id["asset_2"]["near_duplicate_group"])
            self.assertEqual(by_id["asset_3"]["near_duplicate_size"], 1)
        finally:
            store.close()

    def test_all_members_retained(self):
        store = self._store()
        try:
            assets = [{"asset_id": "asset_1"}, {"asset_id": "asset_2"}]
            NearDuplicateGrouper(store).annotate(assets)
            # Grouping is annotation only — every asset stays in the result set.
            self.assertEqual({asset["asset_id"] for asset in assets}, {"asset_1", "asset_2"})
        finally:
            store.close()


if __name__ == "__main__":
    unittest.main()
