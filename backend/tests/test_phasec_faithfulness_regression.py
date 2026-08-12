"""Phase C/C10 — Faithfulness 回归集（Phase H H7 改造后）。

原则：回答"是否合格"（编造/矛盾/漏报/过度声称/缺口披露/数字等价）由 L2 模型评审
（judge.py）判定；L1 FinalGuard 只做结构性兜底（安全/占位符/交付完整性/流程结构）。
本回归集验证：
- L1 不再用词语/数字正则误拦事实正确的回答（诚实否认、中文数字、等价表述均放行）；
- L2 judge 对编造/矛盾给出 truth recoverable；
- runtime 恢复路径在 L2 拦截后仍能恢复成功。
"""

import unittest

from backend.agent_runtime import tools as runtime_tools
from backend.agent_runtime.final_guard import FinalGuard
from backend.agent_runtime.judge import judge_faithfulness
from backend.db import MemoryStore


def _exists_state(value: bool):
    return {
        "last_tool": "query_memory_facts", "fact_operation": "exists",
        "fact_value": value,
        "tool_results": [{"tool": "query_memory_facts", "total": 3 if value else 0}],
    }


def _search_state(satisfaction: str, conditions: dict | None = None, total: int = 8,
                  inspect: str | None = None, selected: str | None = None):
    tool_results = [{"tool_call_id": "tool_call_1", "tool": "search_memories", "total": total}]
    if inspect:
        tool_results.append({"tool_call_id": "tool_call_2", "tool": "inspect_photo",
                             "inspect_handle": "photo_1", "inspect_text": inspect,
                             "confirms_visual_only": True})
    return {
        "search_satisfaction": satisfaction,
        "search_condition_summary": conditions or {},
        "tool_results": tool_results,
        "evidence_refs": ["tool_call_1"] + (["tool_call_2"] if inspect else []),
        "selected_asset_handle": selected,
    }


class ExistsRegressionTests(unittest.TestCase):
    """P0-3 / C10：q03 类 exists 场景。L1 放行，由 L2 模型判定存在性矛盾。"""

    def test_exists_true_honest_passes(self):
        self.assertEqual(list(FinalGuard().check(
            "查过了，2023年5月拍过照片。", task_state=_exists_state(True))), [])

    def test_exists_true_hedge_passes(self):
        self.assertEqual(list(FinalGuard().check(
            "我确认过记录，但没有完全确认时间范围。", task_state=_exists_state(True))), [])

    def test_exists_true_denial_passes_l1(self):
        # 存在性矛盾交给 L2；L1 不得用"没有找到"正则误伤
        problems = FinalGuard().check("没有找到任何相关照片。", task_state=_exists_state(True))
        self.assertEqual(list(problems), [])

    def test_exists_false_assertion_passes_l1(self):
        problems = FinalGuard().check("有照片，我找到了。", task_state=_exists_state(False))
        self.assertEqual(list(problems), [])

    def test_exists_false_denial_passes(self):
        self.assertEqual(list(FinalGuard().check(
            "没有找到相关记录。", task_state=_exists_state(False))), [])

    def test_l2_judge_blocks_omission(self):
        # L2 模型判定"工具确认存在、回答否认"为 omission → truth recoverable
        faithful, problems = judge_faithfulness(
            lambda msgs: '{"faithful": false, "problems": [{"type": "omission", "detail": "工具确认存在相关记录，回答却说没有找到"}]}',
            query="2023年5月拍过照片吗？",
            tool_results=_exists_state(True)["tool_results"],
            answer="没有找到任何相关照片。", trusted_facts=["已确认存在相关记录。"])
        self.assertFalse(faithful)
        self.assertEqual(problems.issues[0].code, "judge_omission")
        self.assertEqual(problems.severity, "truth")


class PlaceAggregationRegressionTests(unittest.TestCase):
    """P0-1 / C10："去年去过哪里" 类地点聚合。列举/编造判定交给 L2。"""

    def test_place_group_full_answer_passes(self):
        problems = FinalGuard().check(
            "去年去过杭州、绍兴和济南，另外还有 12 张照片没有可靠地点信息。",
            task_state={"last_tool": "query_memory_facts", "fact_operation": "group",
                        "fact_group_by": "place",
                        "fact_rows": [{"group": "杭州", "count": 150},
                                      {"group": "绍兴", "count": 34},
                                      {"group": "济南", "count": 28}]})
        self.assertEqual(list(problems), [])

    def test_place_group_fabricated_city_passes_l1(self):
        # 编造地点由 L2 判定；L1 不做列举项比对
        problems = FinalGuard().check(
            "去年去过杭州、绍兴、济南和拉萨。",
            task_state={"last_tool": "query_memory_facts", "fact_operation": "group",
                        "fact_group_by": "place",
                        "fact_rows": [{"group": "杭州", "count": 150},
                                      {"group": "绍兴", "count": 34},
                                      {"group": "济南", "count": 28}]})
        self.assertEqual(list(problems), [])

    def test_place_omission_passes_l1(self):
        problems = FinalGuard().check("去年没有去过任何地方。", task_state=_search_state(
            "full_support", {"杭州": "confirmed"}))
        self.assertEqual(list(problems), [])


class MealSummaryRegressionTests(unittest.TestCase):
    """P0-2 / C10："这两年吃过什么" 回归。食物编造交给 L2。"""

    def test_meal_answer_listing_explicit_foods_passes(self):
        problems = FinalGuard().check(
            "这两年吃过：火锅（2次）、烧烤（1次），还有 1 次只能确认是在吃饭。",
            task_state={"last_tool": "query_memory_facts", "fact_operation": "meal",
                        "tool_results": [{"tool": "query_memory_facts", "total": 3}]})
        self.assertEqual(list(problems), [])

    def test_meal_fabrication_passes_l1(self):
        problems = FinalGuard().check(
            "这两年吃过：火锅、烧烤和龙虾。",
            task_state={"last_tool": "query_memory_facts", "fact_operation": "meal",
                        "fact_rows": [{"food": "火锅", "events": 2}, {"food": "烧烤", "events": 1}],
                        "tool_results": [{"tool": "query_memory_facts", "total": 3}]})
        self.assertEqual(list(problems), [])


class CandidateUpgradeRegressionTests(unittest.TestCase):
    """C10：candidate_only -> full match 升级由 L2 判 certainty_upgrade（truth recoverable）。"""

    def test_candidate_claimed_as_full_match_passes_l1(self):
        problems = FinalGuard().check("找到了爬山的照片，确认是。", task_state=_search_state(
            "candidate_only", {"爬山": "unknown"}))
        self.assertEqual(list(problems), [])

    def test_candidate_with_disclosure_passes(self):
        self.assertEqual(list(FinalGuard().check(
            "找到几张接近的候选，还不能完全确认是爬山。", task_state=_search_state(
                "candidate_only", {"爬山": "unknown"}))), [])

    def test_l2_judge_blocks_certainty_upgrade(self):
        faithful, problems = judge_faithfulness(
            lambda msgs: '{"faithful": false, "problems": [{"type": "certainty_upgrade", "detail": "检索只是候选，回答却断言确认"}]}',
            query="是爬山吗？",
            tool_results=_search_state("candidate_only", {"爬山": "unknown"})["tool_results"],
            answer="确认是爬山。", trusted_facts=[])
        self.assertFalse(faithful)
        self.assertEqual(problems.issues[0].code, "judge_certainty_upgrade")
        self.assertEqual(problems.severity, "truth")


class L1StructuralBatteryTests(unittest.TestCase):
    """L1 结构性电池：只有结构性违规才拦；所有词语/数字合格性场景一律放行。"""

    BATTERY = [
        # (answer, state, expect_blocked)
        ("查过了，2023年5月拍过照片。", _exists_state(True), False),
        ("没有找到任何相关照片。", _exists_state(True), False),          # 交给 L2
        ("没有找到相关记录。", _exists_state(False), False),
        ("有照片，我找到了。", _exists_state(False), False),              # 交给 L2
        ("去年去过杭州和绍兴。", {"tool_results": [{"tool": "query_memory_facts", "total": 2}]}, False),
        ("去年去过杭州和拉萨。", {"tool_results": [{"tool": "query_memory_facts", "total": 2}]}, False),
        ("找到几张接近的候选，还不能完全确认。", _search_state("candidate_only", {"爬山": "unknown"}), False),
        ("找到了爬山的照片，确认是。", _search_state("candidate_only", {"爬山": "unknown"}), False),
        ("确认是爬山。", _search_state("candidate_only", {"爬山": "unknown"}, inspect="有积雪"), False),
        ("没有找到相关照片。", {"tool_results": []}, True),               # denial_without_search
        ("地点是[地点名称1]。", {"tool_results": [{"tool": "search_memories", "total": 2}]}, True),  # placeholder
    ]

    def test_l1_structural_battery(self):
        g = FinalGuard()
        for answer, state, expect_blocked in self.BATTERY:
            blocked = bool(list(g.check(answer, task_state=state)))
            self.assertEqual(blocked, expect_blocked, f"answer={answer!r} state={state}")


class RecoverySuccessMetricsTests(unittest.TestCase):
    """C10：L2 拦截后 runtime 恢复成功（Recovery Success）。"""

    def _run(self, store_seed, script, question):
        from backend.agent_runtime.runtime import AgentRuntime
        store = MemoryStore(":memory:")
        store.create_asset("a1", "a1.jpg", "image", "/x/a1.jpg",
                           metadata={"captured_at": store_seed, "captured_location": "上海"},
                           scope_id="home")
        store.add_observation("a1", {"id": "obs1", "captured_at": store_seed,
                                     "caption": "聚会", "objects": []}, scope_id="home")
        store.connection.commit()
        runtime_tools.bind_runtime(store)
        runtime_tools.register_tools()
        pool = list(script)

        def chat_fn(messages):
            return pool.pop(0)

        return AgentRuntime(chat_fn=chat_fn, scope_id="home", viewer_id="owner").run(question)

    def test_l2_blocked_recoverable_failure_recovers(self):
        cases = [
            # exists 否认 -> L2 omission 拦截 -> recovery 成功
            ("2023-05-12T10:00:00", [
                '{"action":"tool_call","tool":"query_memory_facts","arguments":{"operation":"exists","filters":{"time":"2023年5月"}},"public_status":"正在查询…"}',
                '{"action":"final","answer":"没有找到任何相关照片。","evidence_refs":["tool_call_1"]}',
                '{"faithful": false, "problems": [{"type": "omission", "detail": "工具确认存在相关记录，回答却说没有找到"}], "reason": "存在性矛盾"}',
                '{"action":"final","answer":"2023年5月拍过照片。","evidence_refs":["tool_call_1"]}',
                '{"faithful": true, "problems": []}',
            ], "2023年5月拍过照片吗？"),
        ]
        recovered = 0
        for seed, script, question in cases:
            turn = self._run(seed, script, question)
            self.assertEqual(turn.status, "complete", f"recovery failed for: {question}")
            judge_steps = [s for s in turn.steps if s.get("type") == "judge"]
            self.assertGreaterEqual(len(judge_steps), 2)
            if judge_steps[-1]["faithful"] is True and turn.status == "complete":
                recovered += 1
        self.assertGreaterEqual(recovered / len(cases), 0.9)


if __name__ == "__main__":
    unittest.main()
