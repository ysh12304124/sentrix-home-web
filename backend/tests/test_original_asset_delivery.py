"""RX-5 original-asset delivery contract: text and images must agree."""

import json
import unittest

from backend.answer_brief import build_answer_brief, VisibleAsset
from backend.evidence_retrieval import EvidencePacket
from backend.query_contracts import QueryAction, QuerySpec
from backend.response_plan import plan_response
from backend.response_validator import finalize_answer
from backend.response_writer import safe_fallback, write_response


class _Gamma:
    def __init__(self, result):
        self._result = result

    def chat(self, prompt, json_mode=True, role=None):
        if callable(self._result):
            return self._result(prompt)
        return self._result


def _spec():
    return QuerySpec("q", "single", ["home"], "owner", "c", "answer", "general",
                     constraints=[], actions=[QueryAction(type="return_assets")],
                     result_requirement={"return_original_assets": True})


def _packet():
    asset = {"asset_id": "asset-1", "file_name": "a.jpg", "media_type": "image",
             "observation_ids": ["obs-1"], "evidence_ids": ["asset-1"],
             "condition_results": {"image": {"status": "matched"}},
             "level": "exact", "captured_at": "2025-10-02"}
    return EvidencePacket("q", "home", "general", assets=[asset], exact_results=[asset])


class OriginalAssetDeliveryTests(unittest.TestCase):
    def test_delivery_text_and_images_agree(self):
        packet = _packet()
        visible = [VisibleAsset("asset-1", "照片1", captured_at="2025-10-02",
                                result_level="exact", supported_aspects=["image"])]
        brief = build_answer_brief("把原图给我", _spec(), packet, visible_assets=visible)
        self.assertEqual(brief.response_mode, "asset_delivery")
        plan = plan_response(brief)
        gamma = _Gamma(json.dumps({"text": "好的，已经找到并展示这1张照片。", "statements": []}))
        answer, statements = write_response(brief, plan, "把原图给我", gamma=gamma)
        final, final_statements, result = finalize_answer(
            answer, statements, brief, plan, plan.image_count,
            lambda: safe_fallback(brief, plan))
        self.assertTrue(result["valid"], result["failures"])
        self.assertEqual(plan.image_count, len(brief.visible_assets))
        self.assertNotIn("无法", final)

    def test_model_cannot_refuse_delivered_images(self):
        packet = _packet()
        visible = [VisibleAsset("asset-1", "照片1")]
        brief = build_answer_brief("把原图给我", _spec(), packet, visible_assets=visible)
        plan = plan_response(brief)
        # The model wrongly claims it cannot provide images that ARE delivered.
        gamma = _Gamma(json.dumps({"text": "我无法提供这些图片。", "statements": []}))
        answer, statements = write_response(brief, plan, "把原图给我", gamma=gamma)
        final, _, result = finalize_answer(
            answer, statements, brief, plan, plan.image_count,
            lambda: safe_fallback(brief, plan))
        self.assertTrue(result["valid"], result["failures"])
        self.assertNotIn("无法", final)

    def test_no_visible_assets_becomes_no_result(self):
        brief = build_answer_brief("把原图给我", _spec(), _packet(), visible_assets=[])
        self.assertEqual(brief.response_mode, "no_result")


if __name__ == "__main__":
    unittest.main()
