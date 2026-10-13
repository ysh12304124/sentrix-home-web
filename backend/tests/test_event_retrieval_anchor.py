import sqlite3
import unittest

from backend.agent_runtime.tools import _event_keyword_anchor, _expand_ranked_event_neighbors


class EventRetrievalAnchorTests(unittest.TestCase):
    @staticmethod
    def _store(events):
        connection = sqlite3.connect(":memory:")
        connection.row_factory = sqlite3.Row
        connection.executescript("""
            CREATE TABLE events (id TEXT, title TEXT, summary TEXT, scope_id TEXT);
            CREATE TABLE observations (
                id TEXT, asset_id TEXT, caption TEXT, activity TEXT,
                place TEXT, ocr_text TEXT
            );
            CREATE TABLE event_observations (event_id TEXT, observation_id TEXT);
            CREATE TABLE assets (id TEXT, metadata_json TEXT, scope_id TEXT);
        """)
        for event_id, title, summary in events:
            connection.execute(
                "INSERT INTO events VALUES (?, ?, ?, ?)",
                (event_id, title, summary, "scope-1"),
            )
            observation_id = f"obs-{event_id}"
            asset_id = f"asset-{event_id}"
            connection.execute(
                "INSERT INTO observations VALUES (?, ?, ?, ?, ?, ?)",
                (observation_id, asset_id, "多人在现场留影", "合影", "", ""),
            )
            connection.execute(
                "INSERT INTO event_observations VALUES (?, ?)",
                (event_id, observation_id),
            )
            connection.execute(
                "INSERT INTO assets VALUES (?, ?, ?)",
                (asset_id, "{}", "scope-1"),
            )
        connection.commit()
        return type("Store", (), {"connection": connection})()

    def test_unique_event_type_paraphrase_adds_event_members_as_soft_candidates(self):
        store = self._store([
            ("wedding-1", "婚礼现场合影活动", "夜间在宣传横幅旁多人合影。"),
            ("trip-1", "海边出游", "家人在海边散步。"),
        ])

        result = _event_keyword_anchor(
            "我记得参加亲友婚礼，在天台迎宾展架旁给同行亲友拍了留影，那次婚礼在哪里办？",
            store,
            "scope-1",
        )

        self.assertIsNotNone(result)
        self.assertEqual(result["event_id"], "wedding-1")
        self.assertEqual(result["event_title"], "婚礼现场合影活动")
        self.assertEqual(result["asset_ids"], ["asset-wedding-1"])

    def test_ambiguous_event_type_does_not_anchor_to_one_event(self):
        store = self._store([
            ("wedding-1", "婚礼现场合影活动", "夜间拍摄纪念照。"),
            ("wedding-2", "亲友婚礼聚会", "室内拍摄纪念照。"),
        ])

        result = _event_keyword_anchor("那次亲友婚礼在哪里办？", store, "scope-1")

        self.assertIsNone(result)

    def test_generic_photo_question_does_not_anchor_an_event(self):
        store = self._store([
            ("wedding-1", "婚礼现场合影活动", "夜间拍摄纪念照。"),
        ])

        result = _event_keyword_anchor("这张照片在哪里拍的？", store, "scope-1")

        self.assertIsNone(result)

    def test_ranked_seed_expands_only_its_small_in_scope_event(self):
        connection = sqlite3.connect(":memory:")
        connection.row_factory = sqlite3.Row
        connection.executescript("""
            CREATE TABLE observations (id TEXT, asset_id TEXT);
            CREATE TABLE event_observations (event_id TEXT, observation_id TEXT);
            CREATE TABLE assets (id TEXT, scope_id TEXT, media_type TEXT,
                                 captured_at TEXT, file_name TEXT, metadata_json TEXT);
        """)
        for asset_id, event_id, scope_id in (
            ("seed", "wedding", "scope-1"),
            ("sibling", "wedding", "scope-1"),
            ("other-scope", "wedding", "scope-2"),
            ("unrelated", "outing", "scope-1"),
        ):
            connection.execute("INSERT INTO assets VALUES (?,?,?,?,?,?)", (
                asset_id, scope_id, "image", "2017-12-16", f"{asset_id}.jpg", "{}"))
            connection.execute("INSERT INTO observations VALUES (?,?)", (
                f"obs-{asset_id}", asset_id))
            connection.execute("INSERT INTO event_observations VALUES (?,?)", (
                event_id, f"obs-{asset_id}"))
        store = type("Store", (), {
            "connection": connection,
            "get_asset": lambda self, aid: {
                "file_name": f"{aid}.jpg", "metadata_json": {}}
        })()
        scores = {"seed": 0.2}
        added = _expand_ranked_event_neighbors(scores, store, "scope-1", "image")
        self.assertEqual(added, {"sibling"})
        self.assertAlmostEqual(scores["sibling"], 0.14)
        self.assertNotIn("other-scope", scores)
        self.assertNotIn("unrelated", scores)


if __name__ == "__main__":
    unittest.main()
