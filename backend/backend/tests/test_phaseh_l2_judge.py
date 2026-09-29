"""Phase H — H7 L2 模型评审 guard 单测。

覆盖：
- parse_verdict 解析（markdown 围栏 / 普通 JSON / 损坏输出）
- 严重级映射：fabrication/contradiction/omission/certainty_upgrade → truth recoverable；
  missing_disclosure → style advisory
- faithful=true 放行；输出不可解析/异常 → 降级放行（L1 已兜底结构性问题）
- 数字等价原则：L1 不再用正则判"三个人 vs 3"，交由模型判断（L1 pass）
"""

import unittest

from backend.agent_runtime.final_guard import FinalGuard
from backend.agent_runtime.guard_types import SEVERITY_STYLE, SEVERITY_TRUTH
from backend.agent_runtime.judge import judge_faithfulness, parse_verdict


def _chat(verdict: str):
    def chat_fn(messages):
        return verdict
    return chat_fn


class ParseVerdictTests(unittest.TestCase):
    def test_plain_json(self):
        self.assertEqual(parse_verdict('{"faithful": false, "problems": []}'),
                         {"faithful": False, "problems": []})

    def test_markdown_fence(self):
        out = parse_verdict('```json\n{"faithful": true}\n```')
        self.assertEqual(out, {"faithful": True})

    def test_text_around_json(self):
        out = parse_verdict('好的，我判断如下：\n{"faithful": false, "reason": "x"}')
        self.assertEqual(out["faithful"], False)

    def test_empty_and_garbage(self):
        self.assertIsNone(parse_verdict(""))
        self.assertIsNone(parse_verdict("不是json"))
        self.assertIsNone(parse_verdict("{broken"))


class SeverityMappingTests(unittest.TestCase):
    def _judge(self, ptype: str):
        verdict = json_dumps = __import__("json").dumps(
            {"faithful": False,
             "problems": [{"type": ptype, "detail": "test"}],
             "reason": "r"}, ensure_ascii=False)
        faithful, problems = judge_faithfulness(
            _chat(verdict), query="q", tool_results=[{"tool": "search_memories", "total": 5}],
            answer="答案", trusted_facts=[])
        self.assertFalse(faithful)
        return problems

    def test_truth_types_are_recoverable(self):
        for ptype in ("fabrication", "contradiction", "omission", "certainty_upgrade"):
            problems = self._judge(ptype)
            issue = problems.issues[0]
            self.assertEqual(issue.code, f"judge_{ptype}")
            self.assertEqual(issue.severity, SEVERITY_TRUTH, ptype)

    def test_missing_disclosure_is_style(self):
        problems = self._judge("missing_disclosure")
        self.assertEqual(problems.issues[0].severity, SEVERITY_STYLE)

    def test_unknown_type_defaults_style(self):
        problems = self._judge("unfamiliar_thing")
        self.assertEqual(problems.issues[0].severity, SEVERITY_STYLE)

    def test_faithful_passes(self):
        faithful, problems = judge_faithfulness(
            _chat('{"faithful": true, "problems": []}'),
            query="q", tool_results=[{"tool": "search_memories", "total": 1}],
            answer="好", trusted_facts=[])
        self.assertTrue(faithful)
        self.assertEqual(list(problems), [])

    def test_unparseable_degrades_to_pass(self):
        faithful, _ = judge_faithfulness(
            _chat("抱歉我无法判断"), query="q", tool_results=[{"tool": "search_memories", "total": 1}],
            answer="好", trusted_facts=[])
        self.assertTrue(faithful)

    def test_exception_degrades_to_pass(self):
        def boom(messages):
            raise RuntimeError("model down")
        faithful, _ = judge_faithfulness(boom, query="q",
                                         tool_results=[{"tool": "search_memories", "total": 1}],
                                         answer="好", trusted_facts=[])
        self.assertTrue(faithful)

    def test_unfaithful_without_problems_gets_default_issue(self):
        faithful, problems = judge_faithfulness(
            _chat('{"faithful": false, "problems": []}'),
            query="q", tool_results=[{"tool": "search_memories", "total": 1}],
            answer="好", trusted_facts=[])
        self.assertFalse(faithful)
        self.assertEqual(problems.issues[0].code, "judge_unfaithful")


class L1NumericEquivalenceTests(unittest.TestCase):
    """数字等价原则：L1 不做词语/数字正则合格性判断（"三个人" vs "3" 由 L2 模型判断）。"""

    def _check(self, answer, total=5):
        return FinalGuard().check(answer, task_state={
            "user_query": "照片里有几个人？",
            "search_satisfaction": "full_support",
            "tool_results": [{"tool_call_id": "t1", "tool": "search_memories", "total": total}],
            "evidence_refs": ["t1"],
        })

    def test_chinese_number_answer_passes_l1(self):
        # 模型答"三个人"，工具确认 3 人：L1 不得用"3"正则判错，交给 L2
        probs = self._check("照片里有三个人。", total=3)
        self.assertEqual(list(probs), [])

    def test_conflicting_number_answer_passes_l1(self):
        # 即使是实质冲突（3 vs 5），L1 也不拦，由 L2 模型判定
        probs = self._check("照片里有三个人。", total=5)
        self.assertEqual(list(probs), [])


if __name__ == "__main__":
    unittest.main()
