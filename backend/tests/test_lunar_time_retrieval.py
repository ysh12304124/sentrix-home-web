"""Calendar expansion is grounded in the question and the album, not QA labels."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from backend.db import MemoryStore
from backend.agent_runtime import tools
from backend.agent_runtime.lunar_time import lunar_holiday_dates, lunar_solar_dates
from backend.agent_runtime.canonical_intent import extract_time


class LunarTimeRetrievalTests(unittest.TestCase):
    def test_year_end_phrases_keep_the_user_day_or_month(self):
        self.assertEqual(extract_time("2018年最后一天全家去公园"), "2018年12月31日")
        self.assertEqual(extract_time("2018年底在古城留影"), "2018年12月")
        self.assertEqual(extract_time("2018年末的合影"), "2018年12月")
        self.assertEqual(extract_time("17年12月份参加婚礼"), "2017年12月")
        self.assertEqual(extract_time("18年12月31号去公园"), "2018年12月31日")

    def test_exact_day_rescues_media_and_overrides_year_only_slot(self):
        with tempfile.TemporaryDirectory() as directory:
            store = MemoryStore(str(Path(directory) / "memory.db"))
            try:
                target = store.create_asset(
                    "year-end-target", "garden.jpg", "image", str(Path(directory) / "garden.jpg"),
                    metadata={"captured_at": "2018-12-31 14:00:00"}, scope_id="album")
                store.add_observation(target["id"], {"caption": "室内植物和步道"}, scope_id="album")
                wrong = store.create_asset(
                    "same-year-noise", "other.jpg", "image", str(Path(directory) / "other.jpg"),
                    metadata={"captured_at": "2018-02-18 14:00:00"}, scope_id="album")
                store.add_observation(wrong["id"], {"caption": "室内植物和步道"}, scope_id="album")
                outside = store.create_asset(
                    "other-scope", "outside.jpg", "image", str(Path(directory) / "outside.jpg"),
                    metadata={"captured_at": "2018-12-31 15:00:00"}, scope_id="other")
                tools.bind_runtime(store)
                with patch.object(tools, "_relaxed_retrieve", return_value=(
                    tools._SlotRetrievalPacket([{"id": wrong["id"]}], retrieval_timing={}), 0)), \
                     patch.object(tools, "_trusted_query_constraints", return_value={
                         "time": "2018年12月31日", "place": None, "person": None}), \
                     patch("backend.agent_runtime.semantic_slots.parse_semantic_slots", return_value={
                         "time": {"year": 2018, "months": [], "days": [], "expr": "2018年"},
                         "place": {"name": "", "hint": ""}, "event": {"name": ""},
                         "objects": [], "query_core": "室内植物"}):
                    result = tools._search_memories(
                        {"query": "室内植物", "filters": {"time": "2018年"}},
                        context={"scope_id": "album", "task_state": {
                            "user_goal": "2018年最后一天在室内植物园拍的照片"}},
                    )
                ids = [item["asset_id"] for item in result["preview"]]
                self.assertIn(target["id"], ids)
                self.assertNotIn(wrong["id"], ids)
                self.assertNotIn(outside["id"], ids)
                self.assertGreaterEqual(result["retrieval_timing"]["semantic_retrieval"]
                                        ["calendar_day_added_count"], 1)
            finally:
                store.close()

    def test_explicit_and_unqualified_lunar_day(self):
        years = {2018, 2019}
        self.assertEqual(
            {str(day) for day in lunar_solar_dates("18年大年初三", years)},
            {"2018-02-18"})
        self.assertEqual(
            {str(day) for day in lunar_solar_dates("正月初三", years)},
            {"2018-02-18", "2019-02-07"})
        self.assertEqual(lunar_solar_dates("普通春游", years), set())

    def test_spring_festival_range_keeps_album_years_without_guessing_one(self):
        dates = {str(day) for day in lunar_holiday_dates("春节出游", {2018, 2019})}
        self.assertIn("2018-02-18", dates)  # third lunar day
        self.assertIn("2019-02-07", dates)
        self.assertNotIn("2019-01-01", dates)
        self.assertEqual(
            {str(day) for day in lunar_holiday_dates("2018年春节", {2018, 2019})},
            {str(day) for day in dates if day.startswith("2018-")},
        )
        self.assertEqual(
            {str(day) for day in lunar_holiday_dates("大年初三", {2018, 2019})},
            {"2018-02-18", "2019-02-07"},
        )

    def test_lunar_date_rescues_album_media_without_caption_keyword(self):
        with tempfile.TemporaryDirectory() as directory:
            store = MemoryStore(str(Path(directory) / "memory.db"))
            try:
                target = store.create_asset(
                    "lunar-target", "day3.jpg", "image", str(Path(directory) / "day3.jpg"),
                    metadata={"captured_at": "2018-02-18 11:54:19"}, scope_id="album")
                store.add_observation(target["id"], {
                    "caption": "一家人在公园里", "activity": "玩耍",
                }, scope_id="album")
                other = store.create_asset(
                    "other", "other.jpg", "image", str(Path(directory) / "other.jpg"),
                    metadata={"captured_at": "2023-08-05 11:54:19"}, scope_id="album")
                store.add_observation(other["id"], {
                    "caption": "室内桌上的文件", "activity": "整理文件",
                }, scope_id="album")
                outside = store.create_asset(
                    "outside", "outside.jpg", "image", str(Path(directory) / "outside.jpg"),
                    metadata={"captured_at": "2018-02-18 12:00:00"}, scope_id="other")
                tools.bind_runtime(store)
                with patch.object(tools, "_relaxed_retrieve", return_value=(
                    tools._SlotRetrievalPacket([], retrieval_timing={}), 0)):
                    result = tools._search_memories(
                        {"query": "大年初三出游", "filters": {"time": "大年初三"}},
                        context={"scope_id": "album", "task_state": {
                            "user_goal": "大年初三出去玩，找当天的照片"}},
                    )
                self.assertGreaterEqual(
                    result["retrieval_timing"]["semantic_retrieval"]
                    ["lunar_calendar_added_count"], 1)
                self.assertEqual(result["preview"][0]["asset_id"], target["id"])
                self.assertNotIn(outside["id"],
                                 [item.get("asset_id") for item in result["preview"]])
            finally:
                store.close()

    def test_spring_festival_rescues_day_three_without_explicit_day(self):
        with tempfile.TemporaryDirectory() as directory:
            store = MemoryStore(str(Path(directory) / "memory.db"))
            try:
                target = store.create_asset(
                    "festival-day3", "train.jpg", "image", str(Path(directory) / "train.jpg"),
                    metadata={"captured_at": "2018-02-18 11:54:19"}, scope_id="album")
                store.add_observation(target["id"], {
                    "caption": "儿童坐在小火车上", "activity": "乘坐小火车",
                }, scope_id="album")
                tools.bind_runtime(store)
                with patch.object(tools, "_relaxed_retrieve", return_value=(
                    tools._SlotRetrievalPacket([], retrieval_timing={}), 0)):
                    result = tools._search_memories(
                        {"query": "春节出游"},
                        context={"scope_id": "album", "task_state": {
                            "user_goal": "春节出游当天安排了哪些活动？"}},
                    )
                self.assertGreaterEqual(
                    result["retrieval_timing"]["semantic_retrieval"]
                    ["lunar_calendar_added_count"], 1)
                self.assertEqual(result["preview"][0]["asset_id"], target["id"])
            finally:
                store.close()
