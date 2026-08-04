import unittest

from backend.semantic_taxonomy import (
    ATMOSPHERE_PRIMARY_TYPES,
    OBJECT_PRIMARY_TYPES,
    PLACE_PRIMARY_TYPES,
    normalize_semantic_analysis,
)


class SemanticTaxonomyTests(unittest.TestCase):
    def test_primary_vocabularies_cover_the_approved_daily_categories(self):
        self.assertIn("餐饮空间", PLACE_PRIMARY_TYPES)
        self.assertIn("医疗与公共服务", PLACE_PRIMARY_TYPES)
        self.assertIn("食品与饮品", OBJECT_PRIMARY_TYPES)
        self.assertIn("电子设备", OBJECT_PRIMARY_TYPES)
        self.assertIn("温馨", ATMOSPHERE_PRIMARY_TYPES)
        self.assertIn("其他或不确定", PLACE_PRIMARY_TYPES)
        self.assertIn("其他或不确定", OBJECT_PRIMARY_TYPES)
        self.assertIn("其他或不确定", ATMOSPHERE_PRIMARY_TYPES)

    def test_normalization_keeps_primary_details_and_raw_labels(self):
        result = normalize_semantic_analysis({
            "place": "湖边餐厅",
            "scene_type": "餐饮空间",
            "semantic": {
                "place": {"primary": "餐饮空间", "details": ["室内", "室内"]},
                "objects": [{"primary": "食品与饮品", "label": "蛋糕", "details": ["桌面"]}],
                "atmosphere": {"labels": ["温馨"], "details": ["暖色光线"]},
            },
        })

        self.assertEqual(result["semantic"]["place"]["primary"], "餐饮空间")
        self.assertEqual(result["semantic"]["place"]["details"], ["室内"])
        self.assertEqual(result["semantic"]["objects"][0]["label"], "蛋糕")
        self.assertEqual(result["semantic"]["atmosphere"]["labels"], ["温馨"])
        self.assertEqual(result["raw_labels"]["place"], "湖边餐厅")

    def test_invalid_primary_falls_back_without_losing_the_raw_value(self):
        result = normalize_semantic_analysis({
            "semantic": {
                "place": {"primary": "某个神秘地点", "details": ["无法判断"]},
                "objects": [{"primary": "未知物品", "label": "一件东西", "details": []}],
                "atmosphere": {"labels": ["不可描述"], "details": []},
            },
        })

        self.assertEqual(result["semantic"]["place"]["primary"], "其他或不确定")
        self.assertEqual(result["semantic"]["objects"][0]["primary"], "其他或不确定")
        self.assertEqual(result["semantic"]["atmosphere"]["labels"], ["其他或不确定"])
        self.assertEqual(result["raw_labels"]["place"], "")
        self.assertIn("某个神秘地点", result["raw_labels"]["semantic_place"])


if __name__ == "__main__":
    unittest.main()
