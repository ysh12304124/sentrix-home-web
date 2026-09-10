import unittest

from scripts.maintenance.migrate_sentrix_lite import lite_analysis, lite_import_metadata


class SentrixLiteMigrationHelpersTest(unittest.TestCase):
    def test_caption_projects_into_native_semantic_contract(self):
        analysis = lite_analysis({
            "caption_model": "Qwen3-VL-4B-Instruct",
            "caption": {
                "summary": "一家人在婚礼现场合影。",
                "people": ["我", "芳芳"],
                "objects": ["婚纱", "花束"],
                "activities": ["合影"],
                "places": ["婚礼现场"],
                "visible_text": ["囍"],
            },
        })
        self.assertEqual(analysis["caption"], "一家人在婚礼现场合影。")
        self.assertEqual(analysis["activity"], "合影")
        self.assertEqual(analysis["ocr_text"], "囍")
        self.assertTrue(analysis["model"].startswith("migrated:"))
        self.assertEqual(analysis["detail"]["migration_source"], "sentrix-lite")
        self.assertEqual(analysis["objects"][0]["label"], "婚纱")

    def test_import_metadata_keeps_capture_provenance(self):
        metadata = lite_import_metadata({
            "source_metadata": {
                "capture_datetime": "2017-10-04T22:31:47",
                "readable_location": "易县, 保定市",
                "gps_coordinates": {"latitude": 39.4, "longitude": 115.1},
            }
        }, "scope-a", "batch-a")
        self.assertEqual(metadata["scope_id"], "scope-a")
        self.assertEqual(metadata["batch_id"], "batch-a")
        self.assertEqual(metadata["captured_at"], "2017-10-04T22:31:47")
        self.assertEqual(metadata["captured_location"], "易县, 保定市")
        self.assertEqual(metadata["gps"]["latitude"], 39.4)

    def test_format_fallback_recovers_embedded_summary(self):
        analysis = lite_analysis({
            "caption": {
                "summary": '{"summary":"机场安检口，多名旅客排队。","visible_text":["THAI PASSPORT"',
                "people": [], "objects": [], "activities": [], "places": [],
                "visible_text": [], "format_fallback": True,
            }
        })
        self.assertEqual(analysis["caption"], "机场安检口，多名旅客排队。")


if __name__ == "__main__":
    unittest.main()
