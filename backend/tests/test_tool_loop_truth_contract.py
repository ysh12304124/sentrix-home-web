"""Phase B B2/B2.1 — search_memories truth contract + Observation Faithfulness Guard."""
import unittest

from backend.agent_runtime.final_guard import FinalGuard
from backend.agent_runtime.tools import _truth_contract


class _Packet:
    def __init__(self, assets, gaps=None):
        self.assets = assets
        self.gaps = gaps or []


class TruthContractTests(unittest.TestCase):
    def test_full_support(self):
        p = _Packet([{"condition_results": {"semantic:爬山": {"status": "matched"}}}])
        summary, satisfaction, answerability = _truth_contract(p, 3)
        self.assertEqual(satisfaction, "full_support")
        self.assertEqual(summary["爬山"], "confirmed")

    def test_partial_support(self):
        p = _Packet([
            {"condition_results": {"semantic:爬山": {"status": "matched"},
                                   "time:2025年10月": {"status": "unknown"}}},
        ])
        _, satisfaction, answerability = _truth_contract(p, 3)
        self.assertEqual(satisfaction, "partial_support")
        self.assertEqual(answerability, "partial")

    def test_candidate_only(self):
        p = _Packet([{"condition_results": {"semantic:爬山": {"status": "unknown"}}}])
        _, satisfaction, _ = _truth_contract(p, 8)
        self.assertEqual(satisfaction, "candidate_only")

    def test_no_match(self):
        _, satisfaction, _ = _truth_contract(_Packet([]), 0)
        self.assertEqual(satisfaction, "no_match")

    def test_contradicted_not_confirmed(self):
        p = _Packet([{"condition_results": {"place:杭州": {"status": "contradicted"}}}])
        summary, satisfaction, _ = _truth_contract(p, 2)
        self.assertEqual(summary["杭州"], "contradicted")
        self.assertEqual(satisfaction, "candidate_only")


class FaithfulnessGuardTests(unittest.TestCase):
    """事实合格性由 L2 模型评审；L1 只做结构性检查（Phase H H7 拍板）。"""

    def _state(self, satisfaction="candidate_only", total=8, refs=("tool_call_1",), conditions=None):
        return {
            "search_satisfaction": satisfaction,
            "condition_summary": conditions or {"semantic": "unknown"},
            "tool_results": [{"tool_call_id": "tool_call_1", "tool": "search_memories", "total": total}],
            "evidence_refs": list(refs),
        }

    def test_candidate_claimed_as_match_passes_l1(self):
        g = FinalGuard()
        problems = g.check("我确认找到了爬山合影。", task_state=self._state())
        self.assertEqual(list(problems), [])

    def test_honest_candidate_disclosure_passes(self):
        g = FinalGuard()
        problems = g.check("找到几张接近的候选，还不能完全确认是爬山合影。", task_state=self._state())
        self.assertEqual(problems, [])

    def test_omission_conflict_blocked_by_l1(self):
        # 最小确定性存在性检查：工具确认存在（total>0）、回答整体否认 → L1 拦截
        g = FinalGuard()
        problems = g.check("我没有找到任何相关照片。", task_state=self._state(
            satisfaction="full_support", conditions={"semantic": "confirmed"}))
        self.assertIn("omission_conflict", list(problems))

    def test_condition_level_denial_passes_l1(self):
        # 条件级否认（"没找到能确认爬山的记录"）不是整体否认 → 放行交 L2
        g = FinalGuard()
        problems = g.check("我没找到能明确确认爬山的记录。", task_state=self._state(
            conditions={"爬山": "unknown"}))
        self.assertEqual(list(problems), [])

    def test_certainty_upgrade_passes_l1(self):
        g = FinalGuard()
        problems = g.check("确认就是爬山，不用再查了。", task_state=self._state())
        self.assertEqual(list(problems), [])

    def test_no_match_can_say_not_found(self):
        g = FinalGuard()
        problems = g.check("没有找到相关照片。", task_state=self._state(satisfaction="no_match", total=0))
        self.assertEqual(problems, [])


if __name__ == "__main__":
    unittest.main()
