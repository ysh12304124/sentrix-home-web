"""RX-5 Response Validator tests."""

import unittest

from backend.answer_brief import AnswerBrief, Fact, Presentation, VisibleAsset
from backend.response_plan import plan_response
from backend.response_validator import finalize_answer, repair_response_once, validate_response


def _brief(mode="exact_result", facts=1, visible=1):
    return AnswerBrief(
        "b1", "find_and_explain_images", mode,
        facts=[Fact("fact_1", "记录中有「海边」", "confirmed", ["asset-1"])] * facts,
        uncertainties=[],
        visible_assets=[VisibleAsset("asset-1", "照片1", supported_aspects=["海边"])] * visible,
        presentation=Presentation(show_images=True),
    )


def _plan(mode="exact_result"):
    return plan_response(_brief(mode))


class ValidatorTests(unittest.TestCase):
    def test_clean_answer_passes(self):
        brief = _brief("exact_result")
        result = validate_response("我找到了去年十月在海边的照片，下面是其中最相关的几张。",
                                   brief, _plan(), 1,
                                   [{"text": "记录中有海边", "fact_id": "fact_1", "certainty": "confirmed"}])
        self.assertTrue(result["valid"])

    def test_internal_id_leak(self):
        result = validate_response("找到 asset_ff34f10b39c3 这张照片。", _brief(), _plan(), 1)
        self.assertFalse(result["valid"])
        self.assertTrue(any(f["rule"] == "internal_leak" for f in result["failures"]))

    def test_english_label_leak(self):
        result = validate_response("确定（matched）：2025年10月", _brief(), _plan(), 1)
        self.assertFalse(result["valid"])

    def test_cannot_provide_but_images(self):
        result = validate_response("我无法提供这些图片。", _brief("asset_delivery"), _plan("asset_delivery"), 3)
        self.assertFalse(result["valid"])
        self.assertTrue(any(f["rule"] == "cannot_provide_but_images" for f in result["failures"]))

    def test_claims_delivered_but_no_images(self):
        result = validate_response("已找到并展示这些照片。", _brief("asset_delivery"), _plan("asset_delivery"), 0)
        self.assertFalse(result["valid"])
        self.assertTrue(any(f["rule"] == "claims_delivered_but_no_images" for f in result["failures"]))

    def test_image_count_mismatch(self):
        result = validate_response("这里有 5 张照片。", _brief(), _plan(), 3)
        self.assertFalse(result["valid"])
        self.assertTrue(any(f["rule"] == "image_count_mismatch" for f in result["failures"]))

    def test_no_result_shows_images(self):
        result = validate_response("没有找到。", _brief("no_result", visible=0), _plan("no_result"), 2)
        self.assertFalse(result["valid"])

    def test_approximate_requires_disclosure(self):
        result = validate_response("这是相关照片。", _brief("approximate_result"), _plan("approximate_result"), 2)
        self.assertFalse(result["valid"])
        self.assertTrue(any(f["rule"] == "approximate_disclosure_missing" for f in result["failures"]))

    def test_approximate_with_disclosure_passes(self):
        result = validate_response("没有完全匹配；时间比较接近，但活动还不能确认。",
                                   _brief("approximate_result"), _plan("approximate_result"), 2)
        self.assertTrue(result["valid"])

    def test_person_no_evidence_claim(self):
        brief = _brief("person_summary", facts=0, visible=0)
        result = validate_response("明哥在这些记录中多次出现，性格很开朗。", brief, _plan("person_summary"), 0)
        self.assertFalse(result["valid"])
        self.assertTrue(any(f["rule"] == "person_no_evidence_claim" for f in result["failures"]))

    def test_person_gap_clean_passes(self):
        brief = _brief("person_summary", facts=0, visible=0)
        result = validate_response("目前还没有足够的照片或记录来介绍这个人，只确认了这个人。",
                                   brief, _plan("person_summary"), 0)
        self.assertTrue(result["valid"])

    def test_overstated_statement(self):
        brief = _brief("exact_result")
        brief.facts = [Fact("fact_1", "记录中可能有「爬山」", "possible", ["asset-1"])]
        result = validate_response("有爬山照片。", brief, _plan(), 1,
                                   [{"text": "有爬山照片", "fact_id": "fact_1", "certainty": "confirmed"}])
        self.assertFalse(result["valid"])
        self.assertTrue(any(f["rule"] == "fact_consistency" for f in result["failures"]))


class RepairTests(unittest.TestCase):
    def test_repair_removes_internal_ids(self):
        result = {"failures": [{"rule": "internal_leak", "detail": "asset_ff34f10b39c3"}]}
        repaired = repair_response_once("找到 asset_ff34f10b39c3 的照片。", _brief(), _plan(), 1, result)
        self.assertIsNotNone(repaired)
        self.assertNotIn("asset_", repaired)

    def test_repair_fixes_unable_when_images_present(self):
        result = {"failures": [{"rule": "cannot_provide_but_images", "detail": ""}]}
        repaired = repair_response_once("我无法提供这些图片。", _brief("asset_delivery"), _plan("asset_delivery"), 3, result)
        self.assertIn("已经找到", repaired)

    def test_finalize_falls_back_when_unrecoverable(self):
        brief = _brief("no_result", facts=0, visible=0)
        answer, statements, result = finalize_answer(
            "asset_ff34f10b39c3 matched possible unknown", [], brief, _plan("no_result"), 0,
            lambda: ("目前没有找到足够可靠的依据。可以补充人物、地点或日期再试试。", []))
        self.assertNotIn("asset_", answer)
        self.assertTrue(result["valid"])


if __name__ == "__main__":
    unittest.main()
