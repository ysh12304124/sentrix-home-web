import tempfile
import unittest
from unittest.mock import patch

from backend.db import MemoryStore
from backend.graph_memory import GraphMemoryService


class GraphMemoryTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.store = MemoryStore(f"{self.temp_dir.name}/sentrix.db")
        self.scope = "video-test"
        self.store.create_memory_space(self.scope, "视频测试")
        self._seed_video_memory()

    def tearDown(self):
        self.store.close()
        self.temp_dir.cleanup()

    def _seed_video_memory(self):
        video = self.store.create_asset(
            "video-1", "family.mov", "video", "/tmp/family.mov",
            metadata={"scope_id": self.scope, "captured_at": "2026-08-27T10:00:00+08:00"},
        )
        observations = []
        for index, timestamp in enumerate((12.0, 18.0, 24.0), start=1):
            asset = self.store.create_asset(
                f"keyframe-{index}", f"family-{index}.webp", "image", f"/tmp/family-{index}.webp",
                metadata={
                    "scope_id": self.scope,
                    "parent_asset_id": video["id"],
                    "derived_kind": "video_keyframe_webp",
                    "source_timestamp_sec": timestamp,
                    "source_frame_index": index * 10,
                    "source_scene_index": 0 if index < 3 else 1,
                    "captured_at": f"2026-08-27T10:00:{index:02d}+08:00",
                },
            )
            observations.append(self.store.add_observation(asset["id"], {
                "id": f"observation-{index}", "scope_id": self.scope,
                "source_type": "video_keyframe",
                "captured_at": f"2026-08-27T10:00:{index:02d}+08:00",
                "caption": f"积木收纳盒关键帧 {index}", "activity": "整理积木",
                "place": "客厅", "objects": ["积木", "收纳盒"],
            }))

        scene_one = self.store.create_video_scene_event({
            "id": "scene-1", "scope_id": self.scope, "source_asset_id": video["id"],
            "source_scene_index": 0, "source_start_sec": 0, "source_end_sec": 20,
            "title": "整理积木第一段", "summary": "视频场景一",
        })
        scene_two = self.store.create_video_scene_event({
            "id": "scene-2", "scope_id": self.scope, "source_asset_id": video["id"],
            "source_scene_index": 1, "source_start_sec": 20, "source_end_sec": 30,
            "title": "整理积木第二段", "summary": "视频场景二",
        })
        self.store.attach_observation_to_event(scene_one["id"], observations[0]["id"])
        self.store.attach_observation_to_event(scene_one["id"], observations[1]["id"])
        self.store.attach_observation_to_event(scene_two["id"], observations[2]["id"])

        box = self.store.create_entity(
            "积木收纳盒", entity_type="object", status="active", confidence=0.9,
            summary="视频中反复出现的收纳盒", scope_id=self.scope,
        )
        self.store.connection.execute(
            "INSERT INTO entity_observations(entity_id, observation_id, confidence, source, created_at) VALUES (?, ?, ?, ?, datetime('now'))",
            (box["id"], observations[0]["id"], 0.9, "test"),
        )
        self.store.connection.commit()
        self.store.upsert_event_entity(scene_one["id"], box["id"], "contains", [observations[0]["id"]], 0.9)

    def _service(self):
        return GraphMemoryService(self.store.path)

    def test_video_mapping_order_and_non_causal_edges(self):
        service = self._service()
        try:
            result = service.rebuild(self.scope)
            self.assertEqual(result["causal_edges"], 0)
            self.assertEqual(result["causal_candidates"], 1)
            self.assertEqual(result["node_counts"]["EPISODE"], 2)
            self.assertEqual(result["node_counts"]["SESSION"], 2)
            self.assertEqual(result["node_counts"]["EVENT"], 3)
            edge_subtypes = set(result["edge_counts"])
            self.assertIn("SEMANTIC:PART_OF", edge_subtypes)
            self.assertIn("SEMANTIC:BELONGS_TO_SESSION", edge_subtypes)
            self.assertIn("TEMPORAL:PRECEDES", edge_subtypes)
            self.assertIn("SEMANTIC:REFERS_TO", edge_subtypes)
            edges = service._rows(
                "SELECT * FROM graph_memory_edges WHERE scope_id = ? AND edge_subtype = 'PRECEDES'",
                (self.scope,),
            )
            first = service._node_id(self.scope, "observation", "observation-1")
            second = service._node_id(self.scope, "observation", "observation-2")
            self.assertTrue(any(edge["source_node_id"] == first and edge["target_node_id"] == second for edge in edges))
            self.assertFalse(any(edge["source_node_id"] == second and edge["target_node_id"] == first for edge in edges))
            node = service.get_node(first, self.scope)
            self.assertEqual(node["attributes"]["observation_id"], "observation-1")
            self.assertEqual(node["attributes"]["asset_id"], "keyframe-1")
            self.assertEqual(node["attributes"]["event_id"], "scene-1")
        finally:
            service.close()

    def test_query_is_lexical_anchor_plus_bounded_graph_expansion(self):
        service = self._service()
        try:
            service.rebuild(self.scope)
            result = service.search("积木", self.scope, limit=8, expand_depth=2,
                                    node_types=["EVENT", "SESSION", "ENTITY"])
            self.assertEqual(result["scope_id"], self.scope)
            self.assertIn("积木", result["matched_terms"])
            self.assertTrue(result["anchors"])
            self.assertTrue(all(item["node_type"] in {"EVENT", "SESSION", "ENTITY"}
                                for item in result["anchors"] + result["expanded"]))
            self.assertTrue(any(item["node_type"] == "ENTITY" for item in result["expanded"] + result["anchors"]))
            self.assertLessEqual(max((item["depth"] for item in result["expanded"]), default=0), 2)
        finally:
            service.close()

    def test_rebuild_is_idempotent_and_does_not_touch_canonical_tables(self):
        service = self._service()
        try:
            tables = ("assets", "observations", "events", "entities", "relationships")
            canonical_before = {
                table: self.store.connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
                for table in tables
            }
            first = service.rebuild(self.scope)
            node_ids_first = [row["node_id"] for row in service._rows(
                "SELECT node_id FROM graph_memory_nodes WHERE scope_id = ? ORDER BY node_id", (self.scope,)
            )]
            second = service.rebuild(self.scope)
            node_ids_second = [row["node_id"] for row in service._rows(
                "SELECT node_id FROM graph_memory_nodes WHERE scope_id = ? ORDER BY node_id", (self.scope,)
            )]
            self.assertEqual(first["nodes"], second["nodes"])
            self.assertEqual(first["edges"], second["edges"])
            self.assertEqual(node_ids_first, node_ids_second)
            for table, count in canonical_before.items():
                self.assertEqual(self.store.connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0], count)
            self.assertEqual(self.store.connection.execute("PRAGMA foreign_key_check").fetchall(), [])
            self.assertEqual(service.stats(self.scope)["latest_build"]["status"], "completed")
        finally:
            service.close()

    def test_scope_isolation_and_failed_rebuild_rollback(self):
        service = self._service()
        try:
            service.rebuild(self.scope)
            other_scope = "other-scope"
            self.store.create_memory_space(other_scope, "另一个空间")
            service.rebuild(other_scope)
            self.assertEqual(service.stats(other_scope)["node_counts"], {"EPISODE": 1})
            self.assertEqual(service.search("积木", other_scope)["anchors"], [])
            node_id = service._node_id(self.scope, "observation", "observation-1")
            self.assertIsNone(service.get_node(node_id, other_scope))
            before = service.stats(self.scope)["nodes"]
            with patch.object(service, "_build_rows", side_effect=RuntimeError("synthetic failure")):
                with self.assertRaisesRegex(RuntimeError, "synthetic failure"):
                    service.rebuild(self.scope)
            self.assertEqual(service.stats(self.scope)["nodes"], before)
            self.assertEqual(service.stats(self.scope)["latest_build"]["status"], "failed")
        finally:
            service.close()


if __name__ == "__main__":
    unittest.main()
