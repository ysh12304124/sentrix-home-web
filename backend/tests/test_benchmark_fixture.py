import json
import tempfile
import unittest
from pathlib import Path

from backend.db import MemoryStore
from scripts.benchmarks.evaluate_lfw_clusters import evaluate as evaluate_lfw
from scripts.benchmarks.evaluate_lfw_clusters import meets_gate
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

    def test_lfw_evaluator_counts_missing_faces_and_enforces_pairwise_gate(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            database = root / "sentrix.db"
            store = MemoryStore(str(database))
            for index, (file_name, embedding) in enumerate((
                ("a1.jpg", [1.0, 0.0]),
                ("a2.jpg", [1.0, 0.0]),
                ("b1.jpg", [0.0, 1.0]),
                ("b2.jpg", None),
            )):
                asset = store.create_asset(f"asset-{index}", file_name, "image", str(root / file_name))
                observation = store.add_observation(asset["id"], {"caption": file_name})
                if embedding:
                    store.add_face_instance(
                        asset["id"],
                        observation["id"],
                        {"embedding": embedding, "quality": 0.9, "confidence": 0.9},
                        model_name="adaface",
                    )
            store.close()
            manifest = root / "lfw.json"
            manifest.write_text(json.dumps({"assets": [
                {"file": "a1.jpg", "source_identity": "person-a"},
                {"file": "a2.jpg", "source_identity": "person-a"},
                {"file": "b1.jpg", "source_identity": "person-b"},
                {"file": "b2.jpg", "source_identity": "person-b"},
            ]}), encoding="utf-8")

            result = evaluate_lfw(database, manifest)

            self.assertEqual(result["expected_samples"], 4)
            self.assertEqual(result["detected_samples"], 3)
            self.assertEqual(result["coverage"], 0.75)
            self.assertEqual(result["pairwise_f1"], 0.6667)
            self.assertFalse(meets_gate(result, minimum_f1=0.95, minimum_coverage=0.95))


if __name__ == "__main__":
    unittest.main()
