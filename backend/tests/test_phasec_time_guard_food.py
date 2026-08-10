"""Phase C — C1 相对时间解析 / C2 Guard 结构化与 exists 误拦回归 / C4 地点覆盖 / C5 饮食聚合。"""

import unittest

from backend.agent_runtime import tools as runtime_tools
from backend.agent_runtime.final_guard import FinalGuard
from backend.db import MemoryStore
from backend.query_contracts import parse_time_expression


def _meal_store():
    store = MemoryStore(":memory:")
    seed = [
        # asset_id, captured_at, media, location, obs_id, activity, caption, objects
        ("a1", "2024-01-15T10:00:00", "image", "上海", "obs1", "吃火锅", "火锅聚会", ["火锅", "啤酒"]),
        ("a2", "2024-03-20T09:00:00", "image", "杭州", "obs2", "吃烧烤", "烧烤晚餐", ["烧烤", "可乐"]),
        ("a3", "2024-10-05T08:00:00", "image", "", "obs3", "逛街", "街边随手拍", []),
        ("a4", "2024-10-20T11:00:00", "image", "", "obs4", "吃火锅", "又吃火锅", ["火锅"]),
    ]
    for asset_id, captured, media, location, obs_id, activity, caption, objects in seed:
        store.create_asset(asset_id, f"{asset_id}.jpg", media, f"/x/{asset_id}.jpg",
                           metadata={"captured_at": captured, "captured_location": location},
                           scope_id="home")
        store.add_observation(asset_id, {"id": obs_id, "captured_at": captured,
                                         "activity": activity, "caption": caption,
                                         "objects": objects}, scope_id="home")
    # 事件级去重：a1+a4 同属一顿火锅 event
    store.connection.execute(
        "INSERT INTO events(id, scope_id, title, time_start, created_at, updated_at) "
        "VALUES ('ev-1', 'home', '火锅聚餐', '2024-01-15T10:00:00', '2024-01-01T00:00:00', '2024-01-01T00:00:00')")
    store.connection.execute(
        "INSERT INTO event_observations(event_id, observation_id) VALUES ('ev-1', 'obs1'), ('ev-1', 'obs4')")
    store.connection.commit()
    return store


class RelativeTimeResolverTests(unittest.TestCase):
    def test_last_year_is_absolute(self):
        self.assertEqual(runtime_tools._resolve_time_expression("去年"), "2025年")

    def test_this_and_previous_year(self):
        self.assertEqual(runtime_tools._resolve_time_expression("今年"), "2026年")
        self.assertEqual(runtime_tools._resolve_time_expression("前年"), "2024年")

    def test_last_two_years_range(self):
        for expr in ("这两年", "近两年", "最近两年"):
            self.assertEqual(runtime_tools._resolve_time_expression(expr), "2025年-2026年", expr)

    def test_last_month(self):
        self.assertEqual(runtime_tools._resolve_time_expression("上个月"), "2026年7月")
        self.assertEqual(runtime_tools._resolve_time_expression("上上个月"), "2026年6月")

    def test_last_year_month_chinese_numeral(self):
        self.assertEqual(runtime_tools._resolve_time_expression("去年十月"), "2025年10月")
        self.assertEqual(runtime_tools._resolve_time_expression("前年十一月"), "2024年11月")

    def test_last_year_season(self):
        self.assertEqual(runtime_tools._resolve_time_expression("去年春天"), "2025年3月-2025年5月")
        self.assertEqual(runtime_tools._resolve_time_expression("去年冬天"), "2025年12月-2026年2月")

    def test_recent_year_rolling(self):
        self.assertEqual(runtime_tools._resolve_time_expression("最近一年"), "2025年8月-2026年8月")

    def test_explicit_years_pass_through(self):
        self.assertEqual(runtime_tools._resolve_time_expression("2023"), "2023年")
        self.assertEqual(runtime_tools._resolve_time_expression("2023年"), "2023年")

    def test_parse_year_range(self):
        start, end = parse_time_expression("2025年-2026年")
        self.assertEqual(start.date().isoformat(), "2025-01-01")
        self.assertEqual(end.date().isoformat(), "2027-01-01")

    def test_parse_month_range(self):
        start, end = parse_time_expression("2025年3月-2025年5月")
        self.assertEqual(start.date().isoformat(), "2025-03-01")
        self.assertEqual(end.date().isoformat(), "2025-06-01")

    def test_parse_month_range_short_end(self):
        start, end = parse_time_expression("2025年3月-5月")
        self.assertEqual(start.date().isoformat(), "2025-03-01")
        self.assertEqual(end.date().isoformat(), "2025-06-01")


class GuardExistsRegressionTests(unittest.TestCase):
    def _state(self, value=True):
        return {
            "last_tool": "query_memory_facts", "fact_operation": "exists",
            "fact_value": value,
            "tool_results": [{"tool": "query_memory_facts",
                              "total": 3 if value else 0}],
        }

    def test_exists_true_honest_answer_passes(self):
        problems = FinalGuard().check("查过了，2023年5月拍过照片。", task_state=self._state(True))
        self.assertEqual(list(problems), [])

    def test_exists_true_hedge_passes(self):
        # P0-3 回归：模型用"没有完全确认"等措辞不应被当成否认
        problems = FinalGuard().check("我确认过记录，但没有完全确认时间范围。", task_state=self._state(True))
        self.assertEqual(list(problems), [])

    def test_exists_true_explicit_denial_blocked(self):
        problems = FinalGuard().check("没有找到任何相关照片。", task_state=self._state(True))
        self.assertTrue(any("fact_exists_contradiction" in p for p in problems))

    def test_exists_false_assertion_blocked(self):
        problems = FinalGuard().check("有照片，我找到了。", task_state=self._state(False))
        self.assertTrue(any("fact_exists_contradiction_false" in p for p in problems))

    def test_exists_false_denial_passes(self):
        problems = FinalGuard().check("没有找到相关记录。", task_state=self._state(False))
        self.assertEqual(list(problems), [])

    def test_placeholder_leak_blocked(self):
        problems = FinalGuard().check(
            "去年春天去过：\n- [地点名称1]\n- [地点名称2]",
            task_state={"search_satisfaction": "full_support", "condition_summary": {},
                        "tool_results": [{"tool_call_id": "tool_call_1", "tool": "search_memories", "total": 112}],
                        "evidence_refs": ["tool_call_1"]})
        self.assertTrue(any("placeholder_leak" in p for p in problems))

    def test_guard_messages_are_natural(self):
        problems = FinalGuard().check("我确认找到了爬山合影。", task_state={
            "search_satisfaction": "candidate_only", "condition_summary": {"semantic": "unknown"},
            "tool_results": [{"tool_call_id": "tool_call_1", "tool": "search_memories", "total": 8}],
            "evidence_refs": ["tool_call_1"],
        })
        self.assertTrue(problems)
        for issue in problems.issues:
            self.assertNotIn(issue.code, issue.message)
            self.assertNotRegex(issue.message, r"expected=|blocked=|conditions=")


class PlaceCoverageTests(unittest.TestCase):
    def setUp(self):
        store = MemoryStore(":memory:")
        seed = [
            ("a1", "2024-01-15T10:00:00", "image", "上海", "obs1", "上海"),
            ("a2", "2024-03-20T09:00:00", "image", "杭州", "obs2", "杭州"),
            ("a3", "2024-10-05T08:00:00", "image", "", "obs3", "上海"),
            ("a4", "2024-10-20T11:00:00", "image", "", "obs4", ""),
            ("a5", "2023-12-01T10:00:00", "image", "贵阳", "obs5", "贵阳"),
        ]
        for asset_id, captured, media, location, obs_id, place in seed:
            store.create_asset(asset_id, f"{asset_id}.jpg", media, f"/x/{asset_id}.jpg",
                               metadata={"captured_at": captured, "captured_location": location},
                               scope_id="home")
            store.add_observation(asset_id, {"id": obs_id, "captured_at": captured, "place": place},
                                  scope_id="home")
        store.connection.commit()
        runtime_tools.bind_runtime(store)
        self.store = store

    def test_place_group_returns_coverage(self):
        out = runtime_tools._query_memory_facts(
            {"operation": "group", "group_by": "place", "filters": {"time": "2024"}},
            context={"scope_id": "home"})
        self.assertEqual(out["operation"], "group")
        coverage = out["coverage"]
        self.assertEqual(coverage["known_location_assets"], 3)
        self.assertEqual(coverage["unknown_location_assets"], 1)
        self.assertEqual(coverage["total_assets"], 4)
        self.assertIn("地点", coverage["disclosure"])


class MealEvidenceTests(unittest.TestCase):
    def setUp(self):
        store = _meal_store()
        runtime_tools.bind_runtime(store)
        self.store = store

    def test_meal_aggregation_with_event_dedup(self):
        out = runtime_tools._query_memory_facts(
            {"operation": "meal", "filters": {"time": "2024"}},
            context={"scope_id": "home"})
        self.assertEqual(out["operation"], "meal")
        foods = {f["food"] for f in out["explicit_foods"]}
        self.assertIn("火锅", foods)
        self.assertIn("烧烤", foods)
        # a1+a4 同属一个 event，火锅只计 1 个事件
        hotpot = next(f for f in out["explicit_foods"] if f["food"] == "火锅")
        self.assertEqual(hotpot["events"], 1)
        # 无事件的"街边随手拍"不算用餐
        self.assertEqual(out["total_meal_observations"], 3)
        self.assertEqual(out["event_count"], 2)  # ev-1(火锅×2图) + obs2(烧烤)

    def test_meal_food_hint(self):
        out = runtime_tools._query_memory_facts(
            {"operation": "meal", "filters": {"time": "2024", "food": "火锅"}},
            context={"scope_id": "home"})
        foods = {f["food"] for f in out["explicit_foods"]}
        self.assertIn("火锅", foods)
        self.assertNotIn("烧烤", foods)

    def test_meal_coverage_disclosure(self):
        out = runtime_tools._query_memory_facts(
            {"operation": "meal", "filters": {"time": "2024"}},
            context={"scope_id": "home"})
        self.assertTrue(out["coverage"]["complete"])
        self.assertTrue(out["coverage"]["disclosure"])


if __name__ == "__main__":
    unittest.main()
