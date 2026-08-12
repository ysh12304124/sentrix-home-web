"""Phase H — H-A Deterministic Fact Delivery 单测（Phase H H7 改造后）。

覆盖：
- Answer Nucleus 提取（count/date/boolean/result_total/place/OCR 硬值/person）
- 简单确定性问题直接渲染（数量/日期/布尔）
- 硬值约束文本生成
- Nucleus 工具函数单测（check_nucleus_preservation 保留为纯工具校验函数）
- 集成：L1 不再用数字正则判定 final 合格性（"3 张" vs "三张" 等价），由 L2 模型评审
"""

import unittest

from backend.agent_runtime.answer_nucleus import (AnswerNucleus, build_nucleus,
                                                  check_nucleus_preservation,
                                                  classify_deterministic,
                                                  render_simple)
from backend.agent_runtime.final_guard import FinalGuard


def _search_state(total=5, satisfaction="full_support", place="上海", ocr=""):
    tool_results = [{"tool_call_id": "t1", "tool": "search_memories", "total": total,
                     "preview": [{"handle": "photo_1", "place": place,
                                  "condition_summary": {"place": "matched"}}]}]
    if ocr:
        tool_results.append({"tool_call_id": "t2", "tool": "read_photo_text",
                             "ocr_text": ocr})
    return {"user_query": "", "search_satisfaction": satisfaction,
            "tool_results": tool_results, "evidence_refs": ["t1"]}


class NucleusExtractionTests(unittest.TestCase):
    def test_result_total_from_search(self):
        n = build_nucleus(_search_state(total=5))
        v = n.get("result_total")
        self.assertIsNotNone(v)
        self.assertEqual(v.value, 5)
        self.assertEqual(v.certainty, "confirmed")
        self.assertIn("5", n.hard_values())

    def test_fact_count_and_boolean(self):
        n = build_nucleus({"fact_operation": "count", "fact_value": 3, "tool_results": []})
        self.assertEqual(n.get("count").value, 3)
        n = build_nucleus({"fact_operation": "exists", "fact_value": False, "tool_results": []})
        self.assertFalse(n.get("boolean").value)

    def test_ocr_hard_values(self):
        n = build_nucleus(_search_state(ocr="汉堡套餐 34元 电话 22048084 SINCE 1974"))
        self.assertIn("34元", n.hard_values())
        self.assertIn("22048084", n.hard_values())
        self.assertIn("1974", n.hard_values())

    def test_place_and_person(self):
        n = build_nucleus({**_search_state(), "active_person": "明明"})
        self.assertEqual(n.get("place").value, "上海")
        self.assertEqual(n.get("person").value, "明明")


class SimpleRenderTests(unittest.TestCase):
    def test_classify(self):
        self.assertEqual(classify_deterministic("一共有多少张照片？"), "count")
        self.assertEqual(classify_deterministic("具体是哪天拍的？"), "date")
        self.assertEqual(classify_deterministic("有没有去过北京？"), "boolean")
        self.assertIsNone(classify_deterministic("照片里有哪些人？"))

    def test_render_count(self):
        n = build_nucleus(_search_state(total=5))
        self.assertEqual(render_simple(n, "count"), "一共 5 张。")

    def test_render_date(self):
        n = build_nucleus({"fact_operation": "date", "fact_value": "2022-06-23",
                           "tool_results": []})
        self.assertEqual(render_simple(n, "date"), "相关时间是 2022年6月23日。")

    def test_render_boolean_false(self):
        n = build_nucleus({"fact_operation": "exists", "fact_value": False,
                           "tool_results": []})
        self.assertEqual(render_simple(n, "boolean"), "没有找到相关记录。")

    def test_render_none_when_missing(self):
        self.assertIsNone(render_simple(AnswerNucleus(), "count"))


class PreservationCheckTests(unittest.TestCase):
    """Nucleus 工具函数单测：check_nucleus_preservation 是纯工具，运行时不再由 L1 guard 调用。"""

    def test_count_conflict_detected(self):
        n = build_nucleus(_search_state(total=5))
        issues = check_nucleus_preservation("为您找到 3 张照片。", n, "一共有几张？")
        self.assertTrue(any("count_conflict:expected=5" in i for i in issues))

    def test_count_preserved(self):
        n = build_nucleus(_search_state(total=5))
        self.assertEqual(check_nucleus_preservation("为您找到 5 张照片。", n, "一共有几张？"), [])

    def test_date_conflict(self):
        n = build_nucleus({"fact_operation": "date", "fact_value": "2022-06-23",
                           "tool_results": []})
        issues = check_nucleus_preservation("是2022年6月25日拍的。", n, "哪天拍的？")
        self.assertTrue(any("date" in i for i in issues))

    def test_date_ok(self):
        n = build_nucleus({"fact_operation": "date", "fact_value": "2022-06-23",
                           "tool_results": []})
        self.assertEqual(check_nucleus_preservation("是2022年6月23日拍的。", n, "哪天拍的？"), [])


class GuardIntegrationTests(unittest.TestCase):
    """L1 不再拦截数字改写——硬值合格性由 L2 模型评审（judge.py）判断。"""

    def test_count_conflict_passes_l1(self):
        # 工具确认 5 张、回答写 3 张：这是实质冲突，L1 不做数字正则判断，交给 L2
        probs = FinalGuard().check("为您找到 3 张照片。", task_state=_search_state(total=5))
        self.assertEqual(list(probs), [])

    def test_chinese_number_equivalent_passes_l1(self):
        # "3 张" vs "三张"：表达等价，L1 不拦
        probs = FinalGuard().check("为您找到三张照片。", task_state=_search_state(total=3))
        self.assertEqual(list(probs), [])


if __name__ == "__main__":
    unittest.main()
