"""Phase C/C10 — Faithfulness v2 正式回归集。

把 Phase C 的三个 P0 场景固化为确定性回归（guard 层 + runtime 恢复层），
并在电池上计算 Guard 指标：
  - Guard False Positive Rate（本应通过却被拦）<= 2%
  - Guard False Negative Rate（本应拦截却放行）= 0
  - Guard Recovery Success（可恢复失败中恢复成功的比例）>= 90%
  - candidate_only -> full match 升级 = 0
"""

import unittest

from backend.agent_runtime import tools as runtime_tools
from backend.agent_runtime.final_guard import FinalGuard
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


def _group_state(rows: list, group_by: str = "month"):
    return {
        "last_tool": "query_memory_facts", "fact_operation": "group",
        "fact_group_by": group_by, "fact_rows": rows,
        "tool_results": [{"tool": "query_memory_facts", "total": sum(r.get("count", 1) for r in rows)}],
    }


class ExistsRegressionTests(unittest.TestCase):
    """P0-3 / C10：q03 类 exists 误拦修复正式回归。"""

    def test_exists_true_honest_passes(self):
        self.assertEqual(list(FinalGuard().check(
            "查过了，2023年5月拍过照片。", task_state=_exists_state(True))), [])

    def test_exists_true_hedge_passes(self):
        self.assertEqual(list(FinalGuard().check(
            "我确认过记录，但没有完全确认时间范围。", task_state=_exists_state(True))), [])

    def test_exists_true_denial_blocked(self):
        problems = FinalGuard().check("没有找到任何相关照片。", task_state=_exists_state(True))
        self.assertTrue(any("fact_exists_contradiction" in p for p in problems))

    def test_exists_false_assertion_blocked(self):
        problems = FinalGuard().check("有照片，我找到了。", task_state=_exists_state(False))
        self.assertTrue(any("fact_exists_contradiction_false" in p for p in problems))

    def test_exists_false_denial_passes(self):
        self.assertEqual(list(FinalGuard().check(
            "没有找到相关记录。", task_state=_exists_state(False))), [])


class PlaceAggregationRegressionTests(unittest.TestCase):
    """P0-1 / C10："去年去过哪里" 类地点聚合回归。"""

    def test_place_group_full_answer_passes(self):
        problems = FinalGuard().check(
            "去年去过杭州、绍兴和济南，另外还有 12 张照片没有可靠地点信息。",
            task_state=_group_state([{"group": "杭州", "count": 150},
                                     {"group": "绍兴", "count": 34},
                                     {"group": "济南", "count": 28}], group_by="place"))
        self.assertEqual(list(problems), [])

    def test_place_group_fabricated_city_blocked(self):
        problems = FinalGuard().check(
            "去年去过杭州、绍兴、济南和拉萨。",
            task_state=_group_state([{"group": "杭州", "count": 150},
                                     {"group": "绍兴", "count": 34},
                                     {"group": "济南", "count": 28}], group_by="place"))
        self.assertTrue(any("group_fabrication" in p for p in problems))

    def test_partial_place_search_requires_disclosure(self):
        problems = FinalGuard().check("去过杭州。", task_state=_search_state(
            "partial_support", {"杭州": "confirmed", "爬山": "unknown"}))
        self.assertTrue(any("missing_disclosure" in p for p in problems))

    def test_place_bullet_list_fabricated_city_blocked(self):
        problems = FinalGuard().check(
            "根据记录，您去年去过以下地方：\n- 杭州市：150条记录\n- 拉萨市：5条记录",
            task_state=_group_state([{"group": "杭州市", "count": 150},
                                     {"group": "绍兴市", "count": 34}], group_by="place"))
        self.assertTrue(any("group_fabrication" in p for p in problems))

    def test_place_inline_parenthetical_passes(self):
        problems = FinalGuard().check(
            "去年春天，您的出行记录显示去过杭州市（共有100条记录）。",
            task_state=_group_state([{"group": "杭州市", "count": 100}], group_by="place"))
        self.assertEqual(list(problems), [])

    def test_place_structural_phrase_not_fabrication(self):
        problems = FinalGuard().check(
            "根据记录，您去年去过以下地方：\n- 杭州市：150条记录\n- 绍兴市：34条记录",
            task_state=_group_state([{"group": "杭州市", "count": 150},
                                     {"group": "绍兴市", "count": 34}], group_by="place"))
        self.assertEqual(list(problems), [])

    def test_place_bullet_fabrication_with_coverage_note_blocked(self):
        # "没有可靠地点信息" 的覆盖披露不应让编造地点漏网（没/未 不再跳过编造检查）
        problems = FinalGuard().check(
            "去年您去过的地方包括：\n- 北京：有 12 张照片\n- 广州：有 5 张照片\n\n此外，还有 15 张照片没有可靠的地点信息。",
            task_state=_group_state([{"group": "杭州市", "count": 150},
                                     {"group": "绍兴市", "count": 34}], group_by="place"))
        self.assertTrue(any("group_fabrication" in p for p in problems))

    def test_place_bullet_real_cities_with_structural_phrase_passes(self):
        problems = FinalGuard().check(
            "根据记录，您去年去过的地方包括：\n- 杭州市：150条记录\n- 绍兴市：34条记录",
            task_state=_group_state([{"group": "杭州市", "count": 150},
                                     {"group": "绍兴市", "count": 34}], group_by="place"))
        self.assertEqual(list(problems), [])

    def test_place_observation_scene_rows_pass(self):
        # 去年春天：rows 含观察地点（体育场等），"以下地点"结构短语不应误伤
        problems = FinalGuard().check(
            "去年春天，您的出行记录显示去过以下地点：\n- 杭州市（共100条记录）\n- 体育场或演艺场馆（2条记录）\n- 室内环境（2条记录）",
            task_state=_group_state([{"group": "杭州市", "count": 100},
                                     {"group": "体育场或演艺场馆", "count": 2},
                                     {"group": "室内环境", "count": 2}], group_by="place"))
        self.assertEqual(list(problems), [])

    def test_place_observation_scene_fabrication_blocked(self):
        problems = FinalGuard().check(
            "去年春天，您的出行记录显示去过以下地点：\n- 杭州市（共100条记录）\n- 月球基地（1条记录）",
            task_state=_group_state([{"group": "杭州市", "count": 100},
                                     {"group": "体育场或演艺场馆", "count": 2}], group_by="place"))
        self.assertTrue(any("group_fabrication" in p for p in problems))

    def test_place_omission_blocked(self):
        problems = FinalGuard().check("去年没有去过任何地方。", task_state=_search_state(
            "full_support", {"杭州": "confirmed"}))
        self.assertTrue(any("omission_conflict" in p for p in problems))


class MealSummaryRegressionTests(unittest.TestCase):
    """P0-2 / C10："这两年吃过什么" 回归（explicit_foods 答案形态放行 + 编造拦截）。"""

    def test_meal_answer_listing_explicit_foods_passes(self):
        problems = FinalGuard().check(
            "这两年吃过：火锅（2次）、烧烤（1次），还有 1 次只能确认是在吃饭。",
            task_state={"last_tool": "query_memory_facts", "fact_operation": "meal",
                        "tool_results": [{"tool": "query_memory_facts", "total": 3}]})
        self.assertEqual(list(problems), [])

    def test_meal_fabrication_blocked(self):
        # 工具只有火锅/烧烤，回答编造龙虾
        problems = FinalGuard().check(
            "这两年吃过：火锅、烧烤和龙虾。",
            task_state={"last_tool": "query_memory_facts", "fact_operation": "meal",
                        "fact_rows": [{"food": "火锅", "events": 2}, {"food": "烧烤", "events": 1}],
                        "tool_results": [{"tool": "query_memory_facts", "total": 3}]})
        self.assertTrue(any("group_fabrication" in p or "fabrication" in p for p in problems))


class CandidateUpgradeRegressionTests(unittest.TestCase):
    """C10：candidate_only -> full match 升级必须为 0。"""

    def test_candidate_claimed_as_full_match_blocked(self):
        problems = FinalGuard().check("找到了爬山的照片，确认是。", task_state=_search_state(
            "candidate_only", {"爬山": "unknown"}))
        self.assertTrue(any("candidate_claimed_as_match" in p for p in problems))

    def test_candidate_with_disclosure_passes(self):
        self.assertEqual(list(FinalGuard().check(
            "找到几张接近的候选，还不能完全确认是爬山。", task_state=_search_state(
                "candidate_only", {"爬山": "unknown"}))), [])


class GuardMetricsTests(unittest.TestCase):
    """C10：在确定性电池上计算 Guard 指标（FP/FN/Recovery Success）。"""

    # (answer, state, expect_blocked) —— 覆盖 P0 场景的通过/拦截期望
    BATTERY = [
        ("查过了，2023年5月拍过照片。", _exists_state(True), False),
        ("没有找到任何相关照片。", _exists_state(True), True),
        ("没有找到相关记录。", _exists_state(False), False),
        ("有照片，我找到了。", _exists_state(False), True),
        ("去年去过杭州和绍兴。", _group_state([{"group": "杭州", "count": 1},
                                              {"group": "绍兴", "count": 1}], "place"), False),
        ("去年去过杭州和拉萨。", _group_state([{"group": "杭州", "count": 1},
                                              {"group": "绍兴", "count": 1}], "place"), True),
        ("找到几张接近的候选，还不能完全确认。", _search_state("candidate_only", {"爬山": "unknown"}), False),
        ("找到了爬山的照片，确认是。", _search_state("candidate_only", {"爬山": "unknown"}), True),
        ("去过杭州。", _search_state("partial_support", {"杭州": "confirmed", "爬山": "unknown"}), True),
        ("照片里有 2 个人。", _search_state("candidate_only", {}, inspect="照片里有2个人", selected="photo_1"), False),
        ("我没找到能明确确认爬山的记录。不过最接近的一张里没有看到积雪。",
         _search_state("candidate_only", {"爬山": "unknown"}, inspect="没有积雪"), False),
        ("确认是爬山。", _search_state("candidate_only", {"爬山": "unknown"}, inspect="有积雪"), True),
    ]

    def test_battery_false_positive_rate_le_2pct(self):
        g = FinalGuard()
        fp, total = 0, 0
        for answer, state, expect_blocked in self.BATTERY:
            blocked = bool(list(g.check(answer, task_state=state)))
            total += 1
            if expect_blocked is False and blocked:
                fp += 1
        self.assertEqual(fp, 0, f"false positives: {fp}/{total}")
        self.assertLessEqual(fp / max(total, 1), 0.02)

    def test_battery_false_negative_rate_zero(self):
        g = FinalGuard()
        fn, total = 0, 0
        for answer, state, expect_blocked in self.BATTERY:
            blocked = bool(list(g.check(answer, task_state=state)))
            total += 1
            if expect_blocked is True and not blocked:
                fn += 1
        self.assertEqual(fn, 0, f"false negatives: {fn}/{total}")


class RecoverySuccessMetricsTests(unittest.TestCase):
    """C10：可恢复的 Guard 失败在 runtime 层恢复成功（Recovery Success >= 90%）。"""

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

    def test_all_recoverable_failures_recover(self):
        cases = [
            # exists 否认 -> 恢复
            ("2023-05-12T10:00:00", [
                '{"action":"tool_call","tool":"query_memory_facts","arguments":{"operation":"exists","filters":{"time":"2023年5月"}},"public_status":"正在查询…"}',
                '{"action":"final","answer":"没有找到任何相关照片。","evidence_refs":["tool_call_1"]}',
                '{"action":"final","answer":"2023年5月拍过照片。","evidence_refs":["tool_call_1"]}',
                '{"faithful": true, "problems": []}',
            ], "2023年5月拍过照片吗？"),
        ]
        recovered = 0
        for seed, script, question in cases:
            turn = self._run(seed, script, question)
            self.assertEqual(turn.status, "complete", f"recovery failed for: {question}")
            guard_steps = [s for s in turn.steps if s.get("type") == "guard"]
            self.assertGreaterEqual(len(guard_steps), 2)
            if guard_steps[-1]["status"] == "pass":
                recovered += 1
        self.assertGreaterEqual(recovered / len(cases), 0.9)


if __name__ == "__main__":
    unittest.main()
