"""Phase G — G3/G4/G5 正式回归集（Phase H H7 改造后）。

覆盖：
- G3 Completion State / Gate：最小动态 requirements（retrieve/resolve_visual/resolve_ocr/deliver_media）
- G4 Guard 分层：Safety hard_block / Truth recoverable / Style advisory
- H7 拍板：L1 FinalGuard 只做结构性兜底（安全/占位符/交付完整性/流程结构），
  回答"是否合格"（数字等价/编造/矛盾/漏报/过度声称/缺口披露）由 L2 模型评审（judge.py）判断。
- G6 natural partial：不暴露工程错误、不猜答案
- G7 答案自然化：删除空壳收尾套话
"""

import unittest

from backend.agent_runtime.completion import (CompletionState, DELIVER_MEDIA,
                                              RESOLVE_OCR, RESOLVE_VISUAL,
                                              RETRIEVE_EVIDENCE)
from backend.agent_runtime.final_guard import FinalGuard
from backend.agent_runtime.final_writer import naturalize_answer
from backend.agent_runtime.guard_types import SEVERITY_HARD_BLOCK, SEVERITY_TRUTH
from backend.agent_runtime.runtime import _natural_partial


def _search_state(satisfaction="candidate_only", total=8, preview=None, inspect=None):
    tool_results = [{"tool_call_id": "tool_call_1", "tool": "search_memories",
                     "total": total, "preview": preview or []}]
    if inspect:
        tool_results.append({"tool_call_id": "tool_call_2", "tool": "inspect_photo",
                             "inspect_handle": "photo_1", "inspect_text": inspect})
    return {
        "search_satisfaction": satisfaction,
        "search_condition_summary": {"semantic": "unknown"},
        "tool_results": tool_results,
        "evidence_refs": ["tool_call_1"] + (["tool_call_2"] if inspect else []),
    }


class CompletionGateTests(unittest.TestCase):
    def test_retrieve_evidence_added_then_satisfied(self):
        cs = CompletionState("2019年7月明明和乐乐在哪拍的照片？")
        cs.update({"tool_results": []})
        self.assertTrue(cs.is_blocked())
        codes = [r.code for r in cs.blocking()]
        self.assertIn(RETRIEVE_EVIDENCE, codes)
        cs.update({"tool_results": [{"tool": "search_memories", "total": 2,
                                     "preview": [{"handle": "photo_1"}]}]})
        self.assertNotIn(RETRIEVE_EVIDENCE, [r.code for r in cs.blocking()])

    def test_resolve_visual_dynamic_requirement(self):
        cs = CompletionState("照片里明明穿什么颜色的衣服？")
        cs.update({"tool_results": [{"tool": "search_memories", "total": 2,
                                     "preview": [{"handle": "photo_1"}]}]})
        codes = [r.code for r in cs.blocking()]
        self.assertIn(RESOLVE_VISUAL, codes)
        cs.update({"tool_results": [
            {"tool": "search_memories", "total": 2, "preview": [{"handle": "photo_1"}]},
            {"tool": "inspect_photo", "inspect_handle": "photo_1", "inspect_text": "黄色上衣"}]})
        self.assertNotIn(RESOLVE_VISUAL, [r.code for r in cs.blocking()])

    def test_resolve_ocr_dynamic_requirement(self):
        cs = CompletionState("顶呱呱菜单上汉堡套餐多少钱？")
        cs.update({"tool_results": [{"tool": "search_memories", "total": 1,
                                     "preview": [{"handle": "photo_1"}]}]})
        self.assertIn(RESOLVE_OCR, [r.code for r in cs.blocking()])
        cs.update({"tool_results": [
            {"tool": "search_memories", "total": 1, "preview": [{"handle": "photo_1"}]},
            {"tool": "read_photo_text", "ocr_text": "汉堡套餐 34 元"}]})
        self.assertNotIn(RESOLVE_OCR, [r.code for r in cs.blocking()])

    def test_deliver_media_requirement(self):
        cs = CompletionState("把那张照片发给我看看")
        cs.update({"tool_results": [{"tool": "search_memories", "total": 1,
                                     "preview": [{"handle": "photo_1"}]}]})
        self.assertIn(DELIVER_MEDIA, [r.code for r in cs.blocking()])
        cs.update({"tool_results": [
            {"tool": "search_memories", "total": 1, "preview": [{"handle": "photo_1"}]},
            {"tool": "get_original_photos", "total": 1}]})
        self.assertNotIn(DELIVER_MEDIA, [r.code for r in cs.blocking()])

    def test_chat_only_has_no_requirements(self):
        cs = CompletionState("你好，谢谢")
        cs.update({"tool_results": []})
        self.assertFalse(cs.is_blocked())


class GuardTierTests(unittest.TestCase):
    """L1 结构性检查：安全/占位符/交付结构。事实合格性交给 L2 模型评审。"""

    def test_internal_id_leak_is_hard_block(self):
        problems = FinalGuard().check(
            "看到了 asset_abc123 的记录。", task_state={"tool_results": []})
        self.assertEqual(problems.severity, SEVERITY_HARD_BLOCK)
        self.assertIn("internal_id_leak", list(problems))

    def test_write_claim_is_hard_block(self):
        problems = FinalGuard().check(
            "我已经帮你删除了那条记录。", task_state={"tool_results": []})
        self.assertEqual(problems.severity, SEVERITY_HARD_BLOCK)
        self.assertIn("write_not_allowed", list(problems))

    def test_placeholder_leak_is_truth(self):
        problems = FinalGuard().check(
            "地点是[地点名称1]，时间是[时间]。", task_state={"tool_results": []})
        self.assertIn("placeholder_leak", list(problems))
        self.assertEqual(problems.severity, SEVERITY_TRUTH)

    def test_denial_without_search_is_truth(self):
        problems = FinalGuard().check("没有找到相关照片。", task_state={"tool_results": []})
        self.assertIn("denial_without_search", list(problems))
        self.assertEqual(problems.severity, SEVERITY_TRUTH)

    def test_denial_after_search_passes(self):
        problems = FinalGuard().check("没有找到相关照片。", task_state={
            "tool_results": [{"tool": "search_memories", "total": 0}]})
        self.assertEqual(list(problems), [])

    def test_all_but_has_more_blocked(self):
        problems = FinalGuard().check("都给你了。", task_state={
            "result_mode": "all", "has_more": True, "tool_results": []})
        self.assertIn("all_requested_but_has_more", list(problems))

    def test_delivery_contradiction_blocked(self):
        problems = FinalGuard().check("全部照片都交付了。", task_state={
            "delivery_state": "complete", "tool_results": []},
            delivered_count=0)
        self.assertIn("delivery_contradiction", list(problems))

    def test_candidate_claim_passes_l1(self):
        # candidate_only 却声称确认：事实合格性问题，L1 不再拦截，交给 L2 模型
        problems = FinalGuard().check(
            "确认就是这家店拍的。", task_state=_search_state("candidate_only"))
        self.assertEqual(list(problems), [])

    def test_boilerplate_tail_passes_l1(self):
        # 风格/披露问题由 L2 missing_disclosure（style advisory）处理，L1 不拦
        problems = FinalGuard().check(
            "是在秦皇岛如是海度假村。以上是我目前能确认的部分信息。",
            task_state=_search_state("full_support", total=2))
        self.assertEqual(list(problems), [])


class NaturalPartialTests(unittest.TestCase):
    def test_ocr_partial_is_natural_and_has_no_internal_codes(self):
        text = _natural_partial({
            "tool_results": [
                {"tool": "search_memories", "total": 1,
                 "preview": [{"handle": "photo_1", "place": "上海青杉路"}]},
            ]}, ["ocr_value_conflict"])
        self.assertIn("没能可靠读出", text)
        self.assertIn("原图", text)
        for bad in ("ocr_value_conflict", "blocked_by_guard", "tool_execution_error"):
            self.assertNotIn(bad, text)

    def test_generic_partial_shows_confirmed_facts(self):
        text = _natural_partial({
            "tool_results": [
                {"tool": "search_memories", "total": 2,
                 "preview": [{"handle": "photo_1", "place": "秦皇岛如是海度假村"}]},
                {"tool": "inspect_photo", "inspect_handle": "photo_1",
                 "inspect_text": "黄色上衣"},
            ]}, ["candidate_claimed_as_match"])
        self.assertIn("秦皇岛如是海度假村", text)
        self.assertIn("黄色上衣", text)
        self.assertNotIn("candidate_claimed_as_match", text)


class NaturalizeAnswerTests(unittest.TestCase):
    def test_strips_boilerplate_tail(self):
        out = naturalize_answer(
            "是在秦皇岛如是海度假村。；找到 2 张接近的照片。；部分信息能对上，还有细节不能完全确认。"
            "以上是我目前能确认的部分信息。")
        self.assertEqual(out, "是在秦皇岛如是海度假村。")

    def test_keeps_hard_values(self):
        out = naturalize_answer("汉堡套餐是 34 元。以上是我目前能确认的部分信息。")
        self.assertIn("34 元", out)
        self.assertNotIn("以上是我", out)


if __name__ == "__main__":
    unittest.main()
