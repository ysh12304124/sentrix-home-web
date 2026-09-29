import unittest

from scripts.benchmarks.evaluate_vision_model import evaluate_images


class FakeVisionClient:
    model = "candidate"

    def analyze_image(self, path, metadata=None):
        return {
            "caption": "客厅里的生日蛋糕",
            "activity": "庆祝生日",
            "event_type": "家庭聚会",
            "objects": ["蛋糕"],
        }


class InvalidVisionClient(FakeVisionClient):
    def analyze_image(self, path, metadata=None):
        return {"caption": ""}


class VisionModelBenchmarkTests(unittest.TestCase):
    def test_passing_candidate_requires_chinese_fields_evidence_and_speed(self):
        report = evaluate_images(["first.jpg", "second.jpg"], FakeVisionClient(), baseline_seconds=18.14, min_speedup=0.01)

        self.assertTrue(report["summary"]["passed"])
        self.assertEqual(report["summary"]["passed_samples"], 2)
        self.assertEqual(report["samples"][0]["evidence_fields"], ["objects"])

    def test_missing_contract_fields_fails_even_when_fast(self):
        report = evaluate_images(["first.jpg"], InvalidVisionClient(), baseline_seconds=18.14, min_speedup=0.01)

        self.assertFalse(report["summary"]["passed"])
        self.assertIn("activity", report["samples"][0]["missing_required_fields"])


if __name__ == "__main__":
    unittest.main()
