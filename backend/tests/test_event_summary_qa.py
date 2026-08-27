import sqlite3
import unittest

from backend.agent_runtime import tools
from backend.agent_runtime.final_writer import build_final_context
from backend.agent_runtime.goal_planner import GoalPlanner, _creative_video_intent


class _Store:
    def __init__(self):
        self.connection = sqlite3.connect(":memory:")
        self.connection.row_factory = sqlite3.Row
        self.connection.executescript(
            """
            CREATE TABLE events (
                id TEXT, scope_id TEXT, title TEXT, event_type TEXT,
                time_start TEXT, time_end TEXT, place TEXT, activity TEXT,
                summary TEXT, cover_asset_id TEXT, status TEXT
            );
            CREATE TABLE event_observations (event_id TEXT, observation_id TEXT);
            CREATE TABLE observations (id TEXT, asset_id TEXT, captured_at TEXT);
            """
        )
        self.connection.execute(
            """INSERT INTO events VALUES
            ('event-1', 'album-test', '女性室内展示物品', '展示活动',
             '2026-08-27T10:43:00', '2026-08-27T10:45:00', '室内', '展示物品',
             '女性依次展示手机壳、透明收纳盒及四格照片。', 'asset-1', 'active')"""
        )
        self.connection.commit()


class EventSummaryQATests(unittest.TestCase):
    def setUp(self):
        self.previous = dict(tools._RUNTIME)
        self.store = _Store()
        tools._RUNTIME.clear()
        tools._RUNTIME["store"] = self.store

    def tearDown(self):
        tools._RUNTIME.clear()
        tools._RUNTIME.update(self.previous)

    def test_natural_chinese_question_matches_event_summary(self):
        result = tools._query_memory_metadata(
            {"operation": "event", "query": "视频里女性展示了什么"},
            context={"scope_id": "album-test"},
        )
        self.assertEqual(result["total"], 1)
        self.assertIn("手机壳", result["items"][0]["summary"])

    def test_event_summary_terms_do_not_require_whitespace(self):
        self.assertIn("女性展示", tools._event_summary_terms("视频里女性展示了什么"))

    def test_event_summary_is_answer_writer_fact(self):
        context = build_final_context(
            "视频里女性展示了什么",
            {"tool_results": [{
                "tool": "query_memory_metadata",
                "metadata_operation": "event",
                "items": [{"title": "女性室内展示物品", "summary": "展示手机壳和收纳盒。"}],
            }]},
        )
        self.assertTrue(context["facts_confirmed"])
        self.assertIn("收纳盒", context["facts"][0]["value"])

    def test_video_event_question_is_not_replanned_as_ocr(self):
        payload = GoalPlanner._normalize_payload(
            {"action": "declare", "declaration": {
                "goal": "展示购物清单时有什么物品",
                "scope_id": "album-test",
                "requirements": [{"id": "req_1", "evidence_type": "visible_text", "description": "物品"}],
            }},
            scope_id="album-test", default_goal="展示购物清单时有什么物品",
        )
        self.assertEqual(payload["declaration"]["requirements"][0]["evidence_type"], "structured_fact")

    def test_video_creation_request_is_recognized_as_a_timeline_task(self):
        self.assertTrue(_creative_video_intent("我想把这段内容剪成出发去韩国的一天，帮我理一下故事线"))
        self.assertTrue(_creative_video_intent("帮我写这段旅行视频的旁白和章节"))
        self.assertFalse(_creative_video_intent("这张照片上写了多少钱"))

    def test_timeline_operation_reads_all_events_without_keyword_filter(self):
        result = tools._query_memory_metadata(
            {"operation": "timeline"},
            context={"scope_id": "album-test"},
        )
        self.assertEqual(result["metadata_operation"], "timeline")
        self.assertEqual(result["total"], 1)


if __name__ == "__main__":
    unittest.main()
