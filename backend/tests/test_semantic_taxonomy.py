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

    def test_legacy_object_labels_are_projected_to_primary_and_details(self):
        result = normalize_semantic_analysis({
            "place": "室内厨房",
            "scene_type": "餐饮空间",
            "objects": ["芒果蛋糕", "手机"],
            "emotions": ["平静"],
        })

        self.assertEqual(result["semantic"]["place"]["details"], ["室内", "厨房"])
        self.assertEqual(
            result["semantic"]["objects"],
            [
                {"primary": "食品与饮品", "label": "芒果蛋糕", "details": ["蛋糕", "水果"]},
                {"primary": "电子设备", "label": "手机", "details": ["手机"]},
            ],
        )
        self.assertEqual(result["semantic"]["atmosphere"]["labels"], ["平静"])

    def test_raw_place_and_object_labels_recover_when_model_selected_other(self):
        result = normalize_semantic_analysis({
            "place": "博物馆展厅",
            "semantic": {
                "place": {"primary": "其他或不确定", "details": []},
                "objects": [
                    {"primary": "其他或不确定", "label": "叉子", "details": []},
                    {"primary": "其他或不确定", "label": "手机", "details": []},
                ],
            },
        })

        self.assertEqual(result["semantic"]["place"]["primary"], "文化与展览")
        self.assertEqual(result["semantic"]["objects"][0]["primary"], "餐具与容器")
        self.assertEqual(result["semantic"]["objects"][1]["primary"], "电子设备")

    def test_common_scene_and_natural_object_labels_have_controlled_categories(self):
        result = normalize_semantic_analysis({
            "place": "户外公共广场",
            "objects": ["起重机", "大海", "艺术品", "麦克风"],
        })

        self.assertEqual(result["semantic"]["place"]["primary"], "街道与广场")
        self.assertEqual(
            [item["primary"] for item in result["semantic"]["objects"]],
            ["工业设备与设施", "自然景观与地貌", "艺术与展品", "演出与活动用品"],
        )


if __name__ == "__main__":
    unittest.main()
