"""Phase G — G3/G4/G5/G6/G7 正式回归集。

覆盖：
- G3 Completion State / Gate：最小动态 requirements（retrieve/resolve_visual/resolve_ocr/deliver_media）
- G4 Guard 三层：Safety hard_block / Truth recoverable / Style advisory
- G5 L2 Judge advisory：可确定性验证 → truth；否则 style，不得 hard block
- G6 natural partial：不暴露工程错误、不猜答案；OCR 失败显式 partial
- G7 答案自然化：删除空壳收尾套话
"""

import json
import os
import tempfile
import unittest

from backend.agent_runtime.completion import (CompletionState, DELIVER_MEDIA,
                                              RESOLVE_OCR, RESOLVE_VISUAL,
                                              RETRIEVE_EVIDENCE)
from backend.agent_runtime.final_guard import FinalGuard
from backend.agent_runtime.final_writer import naturalize_answer
from backend.agent_runtime.guard_types import (SEVERITY_HARD_BLOCK, SEVERITY_STYLE,
                                               SEVERITY_TRUTH)
from backend.agent_runtime.judge import deterministic_verify
from backend.agent_runtime.runtime import _natural_partial
from backend.db import MemoryStore


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
    def test_candidate_claimed_as_match_is_truth(self):
        problems = FinalGuard().check(
            "确认就是这家店拍的。", task_state=_search_state("candidate_only"))
        codes = {i.code: i for i in problems.issues}
        self.assertIn("candidate_claimed_as_match", codes)
        self.assertEqual(codes["candidate_claimed_as_match"].severity, SEVERITY_TRUTH)

    def test_certainty_upgrade_is_truth(self):
        state = _search_state("candidate_only")
        state["search_condition_summary"] = {"爬山": "unknown"}
        problems = FinalGuard().check(
            "确认就是爬山的活动。", task_state=state)
        codes = {i.code: i for i in problems.issues}
        self.assertIn("certainty_upgrade", codes)
        self.assertEqual(codes["certainty_upgrade"].severity, SEVERITY_TRUTH)

    def test_missing_disclosure_is_style(self):
        problems = FinalGuard().check(
            "找到了几张照片。", task_state=_search_state("candidate_only", total=8, preview=[]))
        for issue in problems.issues:
            if issue.code == "missing_disclosure":
                self.assertEqual(issue.severity, SEVERITY_STYLE)
                return
        self.fail("missing_disclosure 未触发")

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

    def test_boilerplate_tail_is_style_not_truth(self):
        problems = FinalGuard().check(
            "是在秦皇岛如是海度假村。以上是我目前能确认的部分信息。",
            task_state=_search_state("full_support", total=2))
        self.assertEqual(problems.severity, SEVERITY_STYLE)
        self.assertNotIn("candidate_claimed_as_match", list(problems))

    def test_truth_dominates_style(self):
        problems = FinalGuard().check(
            "确认就是这家店拍的。以上是我目前能确认的部分信息。",
            task_state=_search_state("candidate_only"))
        self.assertEqual(problems.severity, SEVERITY_TRUTH)


class JudgeAdvisoryTests(unittest.TestCase):
    def test_omission_verifiable_is_truth(self):
        self.assertTrue(deterministic_verify(
            "omission", [{"tool": "search_memories", "total": 5}], "没有找到相关照片"))

    def test_contradiction_not_verifiable_is_style(self):
        self.assertFalse(deterministic_verify(
            "contradiction", [{"tool": "search_memories", "total": 5}], "天气很好"))

    def test_missing_disclosure_not_verifiable(self):
        self.assertFalse(deterministic_verify(
            "missing_disclosure", [{"tool": "search_memories", "total": 5}], "不太确定"))

    def test_fabrication_verifiable_when_empty(self):
        self.assertTrue(deterministic_verify(
            "fabrication", [{"tool": "search_memories", "total": 0}], "我找到了那张照片"))


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


class FabricationFromEmptyRefTests(unittest.TestCase):
    """G6：引用 total=0 的工具却断言具体事实（人名/身份/价格）→ 编造拦截面。"""

    def _check(self, answer):
        return FinalGuard().check(answer, task_state={
            "tool_results": [
                {"tool_call_id": "tool_call_1", "tool": "search_memories", "total": 4},
                {"tool_call_id": "tool_call_2", "tool": "search_conversation_history", "total": 0}],
            "evidence_refs": ["tool_call_1", "tool_call_2"]})

    def test_fabricated_name_flagged(self):
        probs = self._check("和您一起去的朋友是小明。")
        self.assertIn("fabrication_from_empty_ref", list(probs))
        self.assertEqual(probs.severity, SEVERITY_TRUTH)

    def test_honest_denial_allowed(self):
        probs = self._check("现有记录中没有明确提到一起去的朋友名字。")
        self.assertNotIn("fabrication_from_empty_ref", list(probs))

    def test_positive_found_claim_flagged(self):
        probs = self._check("找到了那张照片。")
        self.assertIn("fabrication_from_empty_ref", list(probs))


class PersonFabricationTests(unittest.TestCase):
    """Phase G：同行/身份语境断言的人名必须有出处（问题/工具观察/最近对话），否则 truth 编造拦截。"""

    def _check(self, answer, q="", tools=None, history=""):
        return FinalGuard().check(answer, task_state={
            "user_query": q,
            "history_text": history,
            "tool_results": tools or [
                {"tool_call_id": "t1", "tool": "search_memories", "total": 4,
                 "preview": [{"handle": "photo_1", "place": "Hang Dong"}]}],
            "evidence_refs": ["t1"]})

    def test_fabricated_name_flagged_truth(self):
        probs = self._check("和你一起去的朋友是小明和李华。",
                            q="去清迈看表演时一起去的朋友是谁？")
        self.assertIn("person_fabrication", list(probs))
        self.assertEqual(probs.severity, SEVERITY_TRUTH)

    def test_name_in_question_allowed(self):
        probs = self._check("明明和乐乐在主题沙雕前合影。",
                            q="2019年7月明明和乐乐在哪合影？")
        self.assertNotIn("person_fabrication", list(probs))

    def test_name_in_tool_observation_allowed(self):
        tools = [
            {"tool_call_id": "t1", "tool": "search_memories", "total": 2,
             "preview": [{"handle": "photo_1"}]},
            {"tool_call_id": "t2", "tool": "query_memory_facts", "rows": [{"name": "小宇"}]},
        ]
        probs = self._check("另外那个男孩是小宇。", q="那个男孩是谁？", tools=tools)
        self.assertNotIn("person_fabrication", list(probs))

    def test_name_in_recent_history_allowed(self):
        probs = self._check("朋友是小明。", q="再说一下刚才那个人",
                            history="用户：和谁一起去的？\n助手：小明。")
        self.assertNotIn("person_fabrication", list(probs))

    def test_honest_denial_allowed(self):
        probs = self._check("现有记录中无法确认一起去的朋友名字。",
                            q="和你一起去的朋友是谁？")
        self.assertNotIn("person_fabrication", list(probs))

    def test_place_and_common_noun_not_flagged(self):
        probs = self._check("活动是在河北省秦皇岛市昌黎县的秦皇岛如是海度假村进行的。",
                            q="活动在哪进行？")
        self.assertNotIn("person_fabrication", list(probs))
        probs = self._check("你们主要和沙雕互动合影。", q="2023年8月6日你们主要做什么？")
        self.assertNotIn("person_fabrication", list(probs))
