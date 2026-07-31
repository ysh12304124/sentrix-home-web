import json
import tempfile
import unittest
from pathlib import Path

from backend.db import MemoryStore
from scripts.benchmarks.evaluate_household_benchmark import evaluate


class HouseholdBenchmarkEvaluatorTests(unittest.TestCase):
    def test_evaluator_reports_metrics_without_writing_evaluation_labels(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            database = root / "sentrix.db"
            image = root / "a1.jpg"
            image.write_bytes(b"image")
            store = MemoryStore(str(database))
            asset = store.create_asset("asset-a1", "a1.jpg", "image", str(image), metadata={"scope_id": "album1"}, scope_id="album1")
            observation = store.add_observation(asset["id"], {"caption": "一张测试图片", "scope_id": "album1"}, scope_id="album1")
            store.merge_observation_into_event(observation)
            store.close()
            manifest = root / "manifest.json"
            manifest.write_text(json.dumps({
                "version": 1,
                "spaces": [{
                    "scope_id": "album1",
                    "import": {"files": [{"file_name": "a1.jpg"}]},
                    "diagnostics": {"metadata_missing_images": [], "face_missing_images": []},
                    "evaluation": {"image_to_face_ids": {}, "queries": [{"query_cn": "a1.jpg", "ground_truth": ["a1.jpg"]}]},
                }],
            }), encoding="utf-8")

            result = evaluate(manifest, database)

            self.assertEqual(result["input_diagnostics"]["album1"]["images"], 1)
            self.assertIn("face_clustering", result["spaces"]["album1"])
            self.assertTrue(result["scope_isolation"]["passed"])
            self.assertEqual(result["spaces"]["album1"]["queries"]["hit_rate"], 1.0)


if __name__ == "__main__":
    unittest.main()
