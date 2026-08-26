import sqlite3
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from backend.agent_runtime.tools import _confirmed_photo_identities, _preview_entry


class PhotoIdentityReadonlyTests(unittest.TestCase):
    def setUp(self):
        conn = sqlite3.connect(":memory:")
        conn.row_factory = sqlite3.Row
        conn.executescript("""
            CREATE TABLE face_instances (
                id TEXT, asset_id TEXT, observation_id TEXT, cluster_id TEXT,
                bbox_json TEXT, detection_confidence REAL, quality REAL
            );
            CREATE TABLE face_clusters (id TEXT, entity_id TEXT, status TEXT);
            CREATE TABLE entities (
                id TEXT, canonical_name TEXT, family_role TEXT,
                entity_type TEXT, status TEXT
            );
            CREATE TABLE entity_mentions (
                face_instance_id TEXT, entity_id TEXT, confidence REAL
            );
        """)
        self.conn = conn
        self.store = SimpleNamespace(connection=conn)

    def test_only_confirmed_cluster_entity_and_mention_returns_identity(self):
        self.conn.execute("INSERT INTO face_instances VALUES ('fi','a','o','fc','',.9,.8)")
        self.conn.execute("INSERT INTO face_clusters VALUES ('fc','e','confirmed')")
        self.conn.execute("INSERT INTO entities VALUES ('e','小明','朋友','person','confirmed')")
        self.conn.execute("INSERT INTO entity_mentions VALUES ('fi','e',.9)")
        self.conn.commit()
        before = self.conn.total_changes
        rows = _confirmed_photo_identities(self.store, "a")
        self.assertEqual(rows[0]["person_name"], "小明")
        self.assertEqual(rows[0]["identity_status"], "confirmed")
        self.assertEqual(self.conn.total_changes, before)

    def test_pending_missing_mention_unconfirmed_or_non_person_is_unknown(self):
        cases = [
            ("pending", "confirmed", "person", True),
            ("confirmed", "confirmed", "person", False),
            ("confirmed", "pending", "person", False),
            ("confirmed", "confirmed", "place", False),
        ]
        for index, (cluster_status, entity_status, entity_type, add_mention) in enumerate(cases):
            suffix = str(index)
            self.conn.execute("INSERT INTO face_instances VALUES (?,?,?,?,?,?,?)",
                              ("fi" + suffix, "a" + suffix, "o" + suffix, "fc" + suffix, "", .9, .8))
            self.conn.execute("INSERT INTO face_clusters VALUES (?,?,?)",
                              ("fc" + suffix, "e" + suffix, cluster_status))
            self.conn.execute("INSERT INTO entities VALUES (?,?,?,?,?)",
                              ("e" + suffix, "名字" + suffix, "", entity_type, entity_status))
            if add_mention:
                self.conn.execute("INSERT INTO entity_mentions VALUES (?,?,?)",
                                  ("fi" + suffix, "e" + suffix, .9))
        self.conn.commit()
        for index in range(len(cases)):
            self.assertEqual(_confirmed_photo_identities(self.store, "a" + str(index)), [])

    def test_search_preview_exposes_confirmed_people_without_guessing(self):
        store = SimpleNamespace(
            get_asset=lambda asset_id: {"id": asset_id, "captured_at": "2026-01-01"},
        )
        with patch("backend.agent_runtime.tools._observation_summary", return_value="一张照片"), \
             patch("backend.agent_runtime.tools._short_place_label", return_value="北京"), \
             patch("backend.agent_runtime.tools._confirmed_photo_identities", return_value=[{
                 "person_name": "明明", "family_role": "孩子", "identity_status": "confirmed",
             }]):
            preview = _preview_entry(store, "asset-1", "photo_1")

        self.assertEqual(preview["people"], [{
            "name": "明明", "family_role": "孩子", "identity_status": "confirmed",
        }])
        self.assertNotIn("face_instance_id", preview["people"][0])


if __name__ == "__main__":
    unittest.main()
