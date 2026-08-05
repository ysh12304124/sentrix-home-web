"""Thin Agent runtime end-to-end tests.

Phase 2R rewired the parser boundary: tests now inject a scripted gamma so the
Thin Agent exercises the real ``QueryParser`` path without a live Ollama.
"""

import json
import os
import tempfile
import unittest

from backend.agent import MemoryAgent
from backend.db import MemoryStore


class ScriptedGamma:
    model = "scripted-runtime"

    def __init__(self, parser_response=None):
        self.calls = []
        self.parser_response = parser_response or {}

    def chat(self, prompt, *args, **kwargs):
        self.calls.append(prompt)
        if "查询解析器" in prompt and self.parser_response:
            return json.dumps(self.parser_response, ensure_ascii=False)
        if "QuerySpec 修复器" in prompt and self.parser_response:
            return json.dumps(self.parser_response, ensure_ascii=False)
        return "可以的。"


class ThinAgentRuntimeTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.store = MemoryStore(os.path.join(self.directory.name, "memory.db"))
        self.may = self.store.create_asset("asset-may", "may.jpg", "image", "/tmp/may.jpg",
                                            metadata={"captured_at": "2024-05-12T10:00:00"}, scope_id="home")
        self.store.add_observation(
            self.may["id"],
            {"id": "obs-may", "scope_id": "home", "captured_at": "2024-05-12T10:00:00",
             "caption": "明哥在厨房拿碗", "place": "厨房", "activity": "拿碗",
             "people": ["明哥"], "clothing": ["红色外套"], "confidence": 0.9},
            scope_id="home",
        )
        self.july = self.store.create_asset("asset-july", "july.jpg", "image", "/tmp/july.jpg",
                                             metadata={"captured_at": "2024-07-12T10:00:00"}, scope_id="home")
        self.store.add_observation(
            self.july["id"],
            {"id": "obs-july", "scope_id": "home", "captured_at": "2024-07-12T10:00:00",
             "caption": "明哥在厨房做晚饭", "place": "厨房", "activity": "做晚饭",
             "people": ["明哥"], "confidence": 0.9},
            scope_id="home",
        )
        self.store.create_entity("明哥", "person", "confirmed", scope_id="home")

    def tearDown(self):
        self.store.close()
        self.directory.cleanup()

    def _agent(self, parser_response):
        return MemoryAgent(self.store, gamma=ScriptedGamma(parser_response=parser_response))

    def _run(self, message, parser_response):
        old = os.environ.get("SENTRIX_THIN_AGENT_V1")
        os.environ["SENTRIX_THIN_AGENT_V1"] = "1"
        try:
            return self._agent(parser_response).answer_turn(message, scope_id="home", viewer_id="owner")
        finally:
            if old is None:
                os.environ.pop("SENTRIX_THIN_AGENT_V1", None)
            else:
                os.environ["SENTRIX_THIN_AGENT_V1"] = old

    def test_thin_runtime_person_introduction_is_summary_with_evidence(self):
        result = self._run(
            "介绍一下明哥",
            {"mode": "evidence",
             "actions": [{"type": "summarize_person", "target": "person"}],
             "answer_target": "person", "entity_names": ["明哥"]},
        )
        self.assertIn("明哥", result["answer"])
        self.assertNotIn("当前有多个可能的实体", result["answer"])
        self.assertTrue(result["evidence"])
        self.assertTrue(result["evidence_required"])
        self.assertTrue(result["claim_evidence_index"])

    def test_thin_runtime_enforces_month_and_does_not_state_dinner_for_bowl_observation(self):
        result = self._run(
            "2024 年 5 月厨房里做晚饭的照片",
            {"mode": "evidence",
             "actions": [{"type": "return_assets", "coverage": "best"}],
             "answer_target": "activity",
             "time_expression": "2024 年 5 月",
             "media_expressions": ["照片"],
             "semantic_conditions": [{"dimension": "activity", "value": "做晚饭"}]},
        )
        self.assertEqual({item["asset_id"] for item in result["evidence"]}, {"asset-may"})
        self.assertIn("无法确认", result["answer"])
        self.assertNotIn("确定在准备晚饭", result["answer"])

    def test_thin_runtime_normal_chat_does_not_read_memory(self):
        result = self._run("帮我写一段生日祝福", {})
        self.assertFalse(result["memory_used"])
        self.assertFalse(result["evidence"])
        self.assertEqual(result["retrieval_trace"][0]["counts"]["memory_tools"], 0)

    def test_unconfirmed_person_does_not_produce_cluster_selection(self):
        result = self._run(
            "介绍一下待确认人物",
            {"mode": "evidence",
             "actions": [{"type": "summarize_person", "target": "person"}],
             "answer_target": "person",
             "entity_names": ["待确认人物"]},
        )
        self.assertIn("已确认", result["answer"])
        self.assertFalse(result["evidence"])
        self.assertNotIn("cluster_", result["answer"])

    def test_original_request_shows_assets_without_visual_reinspection(self):
        result = self._run(
            "请直接给我相关原图",
            {"mode": "evidence",
             "actions": [{"type": "return_assets", "coverage": "best"}],
             "media_expressions": ["原图"],
             "result_requirement": {"return_original_assets": True}},
        )
        self.assertTrue(result["original_evidence_requested"])
        self.assertTrue(result["image_results"])
        self.assertFalse(any(item.get("tool") == "inspect_original_images" for item in result["tool_trace"]))

    def test_unbound_clothing_is_not_assigned_to_named_person(self):
        result = self._run(
            "明哥穿什么颜色的衣服",
            {"mode": "evidence",
             "actions": [{"type": "answer_question", "target": "clothing"}],
             "answer_target": "clothing",
             "entity_names": ["明哥"]},
        )
        self.assertIn("绑定", result["answer"])
        self.assertNotIn("红色外套", result["answer"])


if __name__ == "__main__":
    unittest.main()
