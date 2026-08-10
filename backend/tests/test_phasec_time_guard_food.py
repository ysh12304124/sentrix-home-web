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




class SearchInspectCertaintyTests(unittest.TestCase):
    """C8: search certainty 与 inspect certainty 分层；inspect 不能反向确认检索条件。"""

    def _state(self, satisfaction="candidate_only", condition=None, inspect=None,
               selected=None, total=8):
        tool_results = [{"tool_call_id": "tool_call_1", "tool": "search_memories", "total": total}]
        if inspect:
            tool_results.append({
                "tool_call_id": "tool_call_2", "tool": "inspect_photo",
                "inspect_handle": "photo_1", "inspect_text": inspect,
                "confirms_visual_only": True,
            })
        return {
            "search_satisfaction": satisfaction,
            "search_condition_summary": condition or {},
            "tool_results": tool_results,
            "evidence_refs": ["tool_call_1", "tool_call_2"],
            "selected_asset_handle": selected,
        }

    def test_candidate_plus_inspect_cannot_back_confirm_search(self):
        # 检索只是候选（爬山未确认），即使 inspect 看到积雪，也不能说"确认是爬山"
        problems = FinalGuard().check("找到爬山的照片了，确认是那次爬山。", task_state=self._state(
            condition={"爬山": "unknown"}, inspect="照片里有明显积雪"))
        self.assertTrue(any("candidate_claimed_as_match" in p or "certainty_upgrade" in p
                            for p in problems))

    def test_candidate_plus_inspect_natural_layered_answer_passes(self):
        # C8 目标形态：检索层披露 + 复核层观察，两层分开
        problems = FinalGuard().check(
            "我没找到能明确确认'爬山'的记录。不过最接近的一张里没有看到明显积雪。",
            task_state=self._state(condition={"爬山": "unknown"}, inspect="照片里没有明显积雪"))
        self.assertEqual(list(problems), [])

    def test_candidate_plus_inspect_visual_answer_requires_disclosure(self):
        # 只给视觉观察、不披露检索层缺口 → missing_disclosure（可被恢复循环修正）
        problems = FinalGuard().check("照片里没有看到积雪。", task_state=self._state(
            condition={"爬山": "unknown"}, inspect="照片里没有明显积雪"))
        self.assertTrue(any("missing_disclosure" in p for p in problems))

    def test_selected_photo_follow_up_exempt_from_disclosure(self):
        # 用户点选 photo_1 追问视觉细节：该照片的视觉回答以 inspect 为准，豁免检索层披露
        problems = FinalGuard().check("照片里有2个人。", task_state=self._state(
            inspect="照片里有2个人", selected="photo_1"))
        self.assertEqual(list(problems), [])

    def test_visual_confirm_claim_not_treated_as_condition_upgrade(self):
        # 规则4 标签感知：条件标签（爬山）没出现在回答里时，"我确认照片里没有雪"不应被误拦
        problems = FinalGuard().check("我确认照片里没有看到积雪。", task_state=self._state(
            condition={"爬山": "unknown"}, inspect="照片里没有明显积雪"))
        self.assertFalse(any("certainty_upgrade" in p for p in problems))

    def test_condition_label_claimed_confirmed_blocks(self):
        problems = FinalGuard().check("确认是爬山，照片里就是那座山。", task_state=self._state(
            condition={"爬山": "unknown"}, inspect="照片里有积雪"))
        self.assertTrue(any("certainty_upgrade" in p for p in problems))


class TaskStateInspectHandleTests(unittest.TestCase):
    def test_record_tool_result_keeps_inspect_handle(self):
        from backend.agent_runtime.result_set import TaskState
        ts = TaskState(user_goal="x")
        ts.record_tool_result("tool_call_2", "inspect_photo", {
            "asset_handle": "photo_3", "observation": "没有雪",
            "confirms_visual_only": True, "certainty": "supported"})
        row = ts.tool_results[-1]
        self.assertEqual(row["inspect_handle"], "photo_3")
        self.assertTrue(row["confirms_visual_only"])
        self.assertEqual(row["inspect_text"], "没有雪")

    def test_from_dict_restores_selected_asset_handle(self):
        from backend.agent_runtime.result_set import TaskState
        ts = TaskState.from_dict({"selected_asset_handle": "photo_1",
                                  "current_result_set": "rs_abc",
                                  "search_satisfaction": "candidate_only"},
                                 user_goal="这张有几个人？")
        self.assertEqual(ts.selected_asset_handle, "photo_1")
        self.assertEqual(ts.as_dict()["selected_asset_handle"], "photo_1")


class RepresentativePreviewTests(unittest.TestCase):
    def test_even_indices_spread(self):
        idx = runtime_tools._even_indices(52, 6)
        self.assertEqual(len(idx), 6)
        self.assertEqual(idx[0], 0)
        self.assertEqual(idx[-1], 51)
        self.assertEqual(idx, sorted(idx))
        self.assertGreaterEqual(min(idx[i + 1] - idx[i] for i in range(len(idx) - 1)), 5)
        # 均匀覆盖：相邻间隔差异不超过 2（不集中在某一端）

    def test_even_indices_small(self):
        self.assertEqual(runtime_tools._even_indices(3, 6), [0, 1, 2])

    def test_metadata_search_representative_preview(self):
        store = MemoryStore(":memory:")
        for i in range(13):
            store.create_asset(f"a{i:02d}", f"a{i:02d}.jpg", "image", f"/x/a{i:02d}.jpg",
                               metadata={"captured_at": f"2024-01-{i + 1:02d}T10:00:00",
                                         "captured_location": ""},
                               scope_id="home")
        store.connection.commit()
        runtime_tools.bind_runtime(store)
        draft = runtime_tools._draft_from_filters({"time": "2024"}, answer_type="asset_set")
        spec = runtime_tools._spec_for(draft, "home", "owner")
        out = runtime_tools._search_metadata_only(draft, spec, "home", "", "representative")
        self.assertEqual(out["mode"], "representative")
        self.assertEqual(out["total"], 13)
        handles = [p["handle"] for p in out["preview"]]
        self.assertEqual(len(handles), 6)
        # 均匀采样：预览横跨整个时间范围，而不是只取最新 6 张
        self.assertNotEqual(handles, ["photo_1", "photo_2", "photo_3",
                                      "photo_4", "photo_5", "photo_6"])
        self.assertEqual(handles[0], "photo_1")
        self.assertEqual(handles[-1], "photo_13")
        self.assertTrue(out["has_more"])
        self.assertEqual(out["remaining"], 7)
        self.assertTrue(out["can_inspect"])


if __name__ == "__main__":
    unittest.main()
