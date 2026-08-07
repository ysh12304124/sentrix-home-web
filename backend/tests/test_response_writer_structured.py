"""TFPE v2: structured-mode writer prompts, safe fallback, and validator exactness."""

import unittest

from backend.answer_brief import build_structured_brief
from backend.query_contracts import QueryParseDraft, QuerySpec
from backend.response_plan import plan_response
from backend.response_validator import finalize_answer
from backend.response_writer import build_prompt, safe_fallback
from backend.structured_memory import StructuredResult


def _count_brief(total=2):
    draft = QueryParseDraft(answer_type="count", structured={})
    spec = QuerySpec("q", "single", ["home"], "owner", "c", "answer", "general", constraints=[])
    return build_structured_brief("去年拍了多少张照片", spec, draft,
                                  StructuredResult("count", total, total=total), "structured_fact")


class StructuredWriterTests(unittest.TestCase):
    def test_safe_fallback_restates_exact_count(self):
        brief = _count_brief(2)
        text, _ = safe_fallback(brief, plan_response(brief))
        self.assertEqual(text, "符合条件的记录共 2 条。")

    def test_safe_fallback_aggregate_joins_facts(self):
        draft = QueryParseDraft(answer_type="grouped_list", structured={})
        spec = QuerySpec("q", "single", ["home"], "owner", "c", "answer", "general", constraints=[])
        result = StructuredResult("grouped_list", [], rows=[{"group": "上海", "count": 2}], total=2)
        brief = build_structured_brief("去年主要在哪些地方拍照", spec, draft, result, "aggregation")
        text, _ = safe_fallback(brief, plan_response(brief))
        self.assertEqual(text, "上海有 2 条。")

    def test_build_prompt_includes_structured_instruction(self):
        prompt = build_prompt(_count_brief(), plan_response(_count_brief()), "去年拍了多少张照片")
        self.assertIn("数字和日期必须与事实完全一致", prompt)
        self.assertNotIn("structured_result", prompt)

    def test_validator_rejects_fabricated_count(self):
        brief = _count_brief(2)
        plan = plan_response(brief)
        answer, statements, validation = finalize_answer(
            "一共 99 条记录。", [], brief, plan, 0, lambda: safe_fallback(brief, plan))
        self.assertTrue(validation["fallback_used"])
        self.assertEqual(answer, "符合条件的记录共 2 条。")

    def test_validator_accepts_exact_count(self):
        brief = _count_brief(2)
        plan = plan_response(brief)
        answer, statements, validation = finalize_answer(
            "去年拍了 2 张照片。", [], brief, plan, 0, lambda: safe_fallback(brief, plan))
        self.assertFalse(validation["fallback_used"])
        self.assertNotIn("structured_number_mismatch", validation["reasons"])


if __name__ == "__main__":
    unittest.main()
