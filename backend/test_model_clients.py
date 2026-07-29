import unittest

from backend.model_clients import as_text, normalize_confidence, parse_json_response


class ModelClientTests(unittest.TestCase):
    def test_parses_json_inside_markdown_fence(self):
        result = parse_json_response('```json\n{"caption":"公园"}\n```')
        self.assertEqual(result["caption"], "公园")

    def test_invalid_model_output_is_empty_object(self):
        self.assertEqual(parse_json_response("无法确定"), {})

    def test_model_scalar_fields_can_be_written_to_sqlite(self):
        self.assertEqual(as_text(["厨房", "客厅"]), "厨房、客厅")
        self.assertEqual(as_text({"name": "厨房"}), '{"name": "厨房"}')

    def test_normalizes_chinese_confidence_labels_from_model_output(self):
        self.assertEqual(normalize_confidence("高", 0.5), 0.85)
        self.assertEqual(normalize_confidence("中等", 0.5), 0.6)
        self.assertEqual(normalize_confidence("80%", 0.5), 0.8)
        self.assertEqual(normalize_confidence("无效值", 0.65), 0.65)


if __name__ == "__main__":
    unittest.main()
