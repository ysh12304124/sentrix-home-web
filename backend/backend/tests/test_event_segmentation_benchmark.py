import importlib.util
import json
import sqlite3
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "benchmarks" / "evaluate_event_segmentation.py"
SPEC = importlib.util.spec_from_file_location("evaluate_event_segmentation", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class EventSegmentationBenchmarkTests(unittest.TestCase):
    def test_reports_split_and_merge_metrics_without_writing_labels_to_database(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            db_path = root / "memory.db"
            manifest_path = root / "manifest.json"
            connection = sqlite3.connect(db_path)
            connection.executescript("""
                CREATE TABLE assets (id TEXT PRIMARY KEY, file_name TEXT NOT NULL, path TEXT NOT NULL);
                CREATE TABLE observations (id TEXT PRIMARY KEY, asset_id TEXT NOT NULL);
                CREATE TABLE event_observations (event_id TEXT NOT NULL, observation_id TEXT NOT NULL);
            """)
            connection.executemany("INSERT INTO assets VALUES (?, ?, ?)", [
                ("a1", "one.jpg", str(root / "one.jpg")), ("a2", "two.jpg", str(root / "two.jpg")),
                ("a3", "three.jpg", str(root / "three.jpg")), ("a4", "four.jpg", str(root / "four.jpg")),
            ])
            connection.executemany("INSERT INTO observations VALUES (?, ?)", [("o1", "a1"), ("o2", "a2"), ("o3", "a3"), ("o4", "a4")])
            connection.executemany("INSERT INTO event_observations VALUES (?, ?)", [("p1", "o1"), ("p2", "o2"), ("p3", "o3"), ("p3", "o4")])
            connection.commit()
            manifest_path.write_text(json.dumps({"assets": [
                {"file": "one.jpg", "event_id": "truth_a"}, {"file": "two.jpg", "event_id": "truth_a"},
                {"file": "three.jpg", "event_id": "truth_b"}, {"file": "four.jpg", "event_id": "truth_c"},
            ]}), encoding="utf-8")

            result = MODULE.evaluate(db_path, manifest_path)

            self.assertEqual(result["assets_evaluated"], 4)
            self.assertEqual(result["split_truth_events"], 1)
            self.assertEqual(result["merged_predicted_events"], 1)
            self.assertEqual(result["splits"]["truth_a"], ["p1", "p2"])
            self.assertEqual(result["merges"]["p3"], ["truth_b", "truth_c"])

    def test_reads_filename_keyed_metadata_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest_path = Path(directory) / "sentrix_metadata.json"
            manifest_path.write_text(json.dumps({
                "one.jpg": {"event_id": "truth_a", "activity_hint": "ignored"},
                "two.jpg": {"event_id": "truth_b"},
                "notes": "not an asset record",
            }), encoding="utf-8")

            truth = MODULE._truth_from_manifest(manifest_path)

            self.assertEqual(truth, {"one.jpg": "truth_a", "two.jpg": "truth_b"})

    def test_uses_asset_paths_to_keep_duplicate_filenames_distinct(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            db_path = root / "memory.db"
            manifest_path = root / "virtual_album_manifest.json"
            connection = sqlite3.connect(db_path)
            connection.executescript("""
                CREATE TABLE assets (id TEXT PRIMARY KEY, file_name TEXT NOT NULL, path TEXT NOT NULL);
                CREATE TABLE observations (id TEXT PRIMARY KEY, asset_id TEXT NOT NULL);
                CREATE TABLE event_observations (event_id TEXT NOT NULL, observation_id TEXT NOT NULL);
            """)
            connection.executemany("INSERT INTO assets VALUES (?, ?, ?)", [
                ("a1", "IMG_0001.jpg", str(root / "album-a" / "IMG_0001.jpg")),
                ("a2", "IMG_0001.jpg", str(root / "album-b" / "IMG_0001.jpg")),
            ])
            connection.executemany("INSERT INTO observations VALUES (?, ?)", [("o1", "a1"), ("o2", "a2")])
            connection.executemany("INSERT INTO event_observations VALUES (?, ?)", [("p1", "o1"), ("p2", "o2")])
            connection.commit()
            manifest_path.write_text(json.dumps({"assets": [
                {"file": "album-a/IMG_0001.jpg", "event_id": "truth_a"},
                {"file": "album-b/IMG_0001.jpg", "event_id": "truth_b"},
            ]}), encoding="utf-8")

            result = MODULE.evaluate(db_path, manifest_path)

            self.assertEqual(result["assets_evaluated"], 2)
            self.assertEqual(result["unmatched_manifest_assets"], [])


if __name__ == "__main__":
    unittest.main()
