"""Calendar expansion is grounded in the question and the album, not QA labels."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from backend.db import MemoryStore
from backend.agent_runtime import tools
from backend.agent_runtime.lunar_time import lunar_solar_dates


class LunarTimeRetrievalTests(unittest.TestCase):
    def test_explicit_and_unqualified_lunar_day(self):
        years = {2018, 2019}
        self.assertEqual(
            {str(day) for day in lunar_solar_dates("18年大年初三", years)},
            {"2018-02-18"})
        self.assertEqual(
            {str(day) for day in lunar_solar_dates("正月初三", years)},
            {"2018-02-18", "2019-02-07"})
        self.assertEqual(lunar_solar_dates("普通春游", years), set())

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
