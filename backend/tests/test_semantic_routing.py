"""Phase 2R-1 semantic routing red tests.

Every case here targets the *post-2R* behaviour: paraphrases, contrast pairs and
composite tasks that a keyword table cannot handle.  Failing states are
expected until Phase 2R-2..2R-5 land.

The tests use a programmable ``FakeSemanticGamma`` so they do not require a
live Ollama endpoint.  When the QueryParser exists (Phase 2R-2) it will be
routed through this fake by way of ``MemoryAgent(gamma=...)``.
"""

import json
import os
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path

from backend.agent import MemoryAgent
from backend.db import MemoryStore


class FakeSemanticGamma:
    """Deterministic gamma stub with prompt-marker keyed responses."""

    model = "fake-semantic"

    def __init__(self, responses=None):
        self.calls = []
        self.responses = dict(responses or {})

    def chat(self, prompt, *args, **kwargs):
        self.calls.append(prompt)
        for marker, response in self.responses.items():
            if marker in prompt:
                payload = response(prompt) if callable(response) else response
                if isinstance(payload, (dict, list)):
                    return json.dumps(payload, ensure_ascii=False)
                return str(payload)
        return '{"answer": "我在听。"}' if kwargs.get("json_mode", True) else "我在听。"

    def answer(self, query, context):
        return {"answer": "证据不足", "confidence": 0.0, "evidence": [], "insufficient_evidence": True}

    def embed_text(self, text):
        return []


@contextmanager
def _flags(**overrides):
    original = {key: os.environ.get(key) for key in overrides}
    for key, value in overrides.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value
    try:
        yield
    finally:
        for key, value in original.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


class SemanticRoutingBase(unittest.TestCase):

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix="semantic-routing-")
        self.store = MemoryStore(str(Path(self.directory.name) / "memory.db"))
        self.store.create_entity("小黑", "person", "confirmed", scope_id="home")
        self.store.create_entity("明哥", "person", "confirmed", scope_id="home")
        self.may = self.store.create_asset(
            "asset-may", "may.jpg", "image", "/tmp/may.jpg",
            metadata={"captured_at": "2024-05-12T10:00:00"}, scope_id="home",
        )
        self.store.add_observation(
            self.may["id"],
            {"id": "obs-may", "scope_id": "home", "captured_at": "2024-05-12T10:00:00",
             "caption": "厨房拿碗", "place": "厨房", "activity": "拿碗",
             "people": ["明哥"], "clothing": ["红色外套"], "confidence": 0.9},
            scope_id="home",
        )

    def tearDown(self):
        self.store.close()
        self.directory.cleanup()

    def _make_agent(self, gamma):
        return MemoryAgent(self.store, gamma=gamma)


class WritingPromptRoutingTests(SemanticRoutingBase):
    """普通写作反例：即使包含 '照片'/'厨房'/'家庭' 等词也不能触发家庭检索。"""

    WRITING_PROMPTS = (
        "帮我写一段关于家庭照片的散文",
        "以相册为主题写一篇短文",
        "假设一家人在厨房做饭，写个故事",
        "为什么人们喜欢拍做饭照片",
        "不用查我的照片，帮我写一句文案",
    )

    def test_writing_prompts_never_trigger_evidence(self):
        gamma = FakeSemanticGamma(responses={
            "查询解析器": {"mode": "none", "actions": [{"type": "answer_question", "target": "general"}], "confidence": 0.9},
        })
        agent = self._make_agent(gamma)
        with _flags(SENTRIX_THIN_AGENT_V1="1", SENTRIX_SEMANTIC_QUERY_PARSER_V1="1"):
            for prompt in self.WRITING_PROMPTS:
                with self.subTest(prompt=prompt):
                    result = agent.answer_turn(prompt, scope_id="home", viewer_id="owner")
                    self.assertFalse(result.get("memory_used"), f"memory_used unexpectedly true for: {prompt}")
                    self.assertFalse(result.get("evidence"), f"evidence non-empty for: {prompt}")


class EvidenceParaphraseRoutingTests(SemanticRoutingBase):
    """家庭证据 paraphrase：不同措辞应稳定进入 evidence 模式。"""

    EVIDENCE_PROMPTS = (
        "聊聊那次做菜时她穿的什么",
        "我记得有人在灶台旁端着碗，是哪一次",
        "找找之前下厨时拍的图",
        "哪些照片看起来是在准备晚餐",
        "不要只是拿碗的，我想找真正烹饪的",
    )

    def test_evidence_paraphrases_route_to_evidence(self):
        gamma = FakeSemanticGamma(responses={
            "查询解析器": {"mode": "evidence",
                          "actions": [{"type": "answer_question", "target": "activity"},
                                      {"type": "return_assets", "coverage": "best"}],
                          "facets": [{"dimension": "activity", "surface_text": "做饭"}],
                          "confidence": 0.85},
        })
        agent = self._make_agent(gamma)
        with _flags(SENTRIX_THIN_AGENT_V1="1", SENTRIX_SEMANTIC_QUERY_PARSER_V1="1"):
            for prompt in self.EVIDENCE_PROMPTS:
                with self.subTest(prompt=prompt):
                    result = agent.answer_turn(prompt, scope_id="home", viewer_id="owner")
                    self.assertTrue(result.get("memory_used"), f"expected memory_used=True for: {prompt}")
                    self.assertTrue(result.get("evidence_required"), f"expected evidence_required=True for: {prompt}")


class ContextualParaphraseRoutingTests(SemanticRoutingBase):
    """自然人物承接 paraphrase：不同措辞应稳定进入 contextual 模式。"""

    CONTEXTUAL_PROMPTS = (
        "今晚回家时想起小黑了",
        "突然有点怀念小黑",
        "今天看到一只猫让我想到小黑",
        "小黑啊，最近总会想到它",
        "刚才路过宠物店，想起家里的小黑",
    )

    def test_contextual_paraphrases_route_to_contextual(self):
        gamma = FakeSemanticGamma(responses={
            "查询解析器": {"mode": "contextual",
                          "actions": [{"type": "answer_question", "target": "general"}],
                          "facets": [{"dimension": "person", "surface_text": "小黑"}],
                          "confidence": 0.75},
        })
        agent = self._make_agent(gamma)
        with _flags(SENTRIX_THIN_AGENT_V1="1", SENTRIX_SEMANTIC_QUERY_PARSER_V1="1"):
            for prompt in self.CONTEXTUAL_PROMPTS:
                with self.subTest(prompt=prompt):
                    result = agent.answer_turn(prompt, scope_id="home", viewer_id="owner")
                    trace_status = None
                    for stage in result.get("retrieval_trace") or []:
                        if stage.get("stage") == "gate":
                            trace_status = stage.get("status")
                    self.assertEqual(trace_status, "contextual", f"expected contextual for: {prompt}")
                    self.assertFalse(any((item.get("kind") == "observation") for item in result.get("evidence") or []),
                                      f"contextual should not surface Observation evidence for: {prompt}")


class ContrastPairTests(SemanticRoutingBase):
    """相同关键词、不同意图 vs 不同词汇、相同意图。"""

    PAIRS = (
        {"chat": "帮我写关于厨房做饭的故事", "evidence": "找厨房里真正做饭的照片"},
        {"chat": "为什么人喜欢拍照片", "evidence": "把去年拍的照片给我"},
        {"chat": "我想写一篇明哥的虚构故事", "evidence": "我想问问明哥去年做了什么"},
        {"chat": '介绍一下"家庭相册"这个产品概念', "evidence": "介绍一下明哥"},
    )

    def _run(self, prompt, mode_expected):
        gamma = FakeSemanticGamma(responses={
            "查询解析器": {"mode": mode_expected,
                          "actions": [{"type": "answer_question", "target": "general"}],
                          "confidence": 0.9},
        })
        agent = self._make_agent(gamma)
        with _flags(SENTRIX_THIN_AGENT_V1="1", SENTRIX_SEMANTIC_QUERY_PARSER_V1="1"):
            return agent.answer_turn(prompt, scope_id="home", viewer_id="owner")

    def test_contrast_pairs_route_correctly(self):
        for pair in self.PAIRS:
            with self.subTest(pair=pair):
                chat_result = self._run(pair["chat"], "none")
                self.assertFalse(chat_result.get("memory_used"), f"chat side leaked memory: {pair['chat']}")
                self.assertFalse(chat_result.get("evidence"), f"chat side surfaced evidence: {pair['chat']}")
                evidence_result = self._run(pair["evidence"], "evidence")
                self.assertTrue(evidence_result.get("memory_used"), f"evidence side missed memory: {pair['evidence']}")


class CompositeGoalTests(SemanticRoutingBase):
    """复合任务：answer + return_assets 同时保留，多个 facets 不丢失。"""

    def test_composite_answer_and_return_assets_are_preserved(self):
        gamma = FakeSemanticGamma(responses={
            "查询解析器": {
                "mode": "evidence",
                "actions": [
                    {"type": "answer_question", "target": "clothing"},
                    {"type": "return_assets", "coverage": "best"},
                ],
                "facets": [
                    {"dimension": "person", "surface_text": "妈妈"},
                    {"dimension": "time", "surface_text": "去年春节"},
                    {"dimension": "activity", "surface_text": "家庭聚餐"},
                    {"dimension": "clothing", "surface_text": "穿了什么"},
                ],
                "confidence": 0.9,
            },
        })
        agent = self._make_agent(gamma)
        with _flags(SENTRIX_THIN_AGENT_V1="1", SENTRIX_SEMANTIC_QUERY_PARSER_V1="1"):
            result = agent.answer_turn(
                "说说去年春节妈妈穿了什么，再把最相关的照片给我",
                scope_id="home", viewer_id="owner",
            )
        actions = [action.get("type") for action in result.get("actions") or []]
        facets = [facet.get("dimension") for facet in result.get("facets") or []]
        self.assertIn("answer_question", actions)
        self.assertIn("return_assets", actions)
        for dimension in ("person", "time", "activity", "clothing"):
            self.assertIn(dimension, facets, f"missing facet: {dimension}")


class ModelCallBudgetTests(SemanticRoutingBase):
    """明确普通聊天不能触发 QuerySpec/Gate 模型调用。"""

    def test_normal_chat_does_not_call_query_parser(self):
        gamma = FakeSemanticGamma()
        agent = self._make_agent(gamma)
        with _flags(SENTRIX_THIN_AGENT_V1="1", SENTRIX_SEMANTIC_QUERY_PARSER_V1="1"):
            agent.answer_turn("帮我写一段生日祝福", scope_id="home", viewer_id="owner")
        parser_calls = [call for call in gamma.calls if "查询解析器" in call]
        self.assertEqual(parser_calls, [], "普通聊天不得调用 QuerySpec Parser")


class SanitizerAndFallbackTests(SemanticRoutingBase):
    """Parser 输出必须经 sanitizer；模型失败不得放宽硬条件。"""

    def test_model_ids_are_discarded(self):
        gamma = FakeSemanticGamma(responses={
            "查询解析器": {
                "mode": "evidence",
                "actions": [{"type": "answer_question", "target": "person"}],
                "scope_id": "attacker-scope",
                "viewer_id": "attacker-viewer",
                "entity_ids": ["attacker-entity"],
                "facets": [{"dimension": "person", "surface_text": "明哥"}],
            },
        })
        agent = self._make_agent(gamma)
        with _flags(SENTRIX_THIN_AGENT_V1="1", SENTRIX_SEMANTIC_QUERY_PARSER_V1="1"):
            result = agent.answer_turn("介绍一下明哥", scope_id="home", viewer_id="owner")
        self.assertEqual(result.get("scope_id"), "home")
        self.assertEqual(result.get("viewer_id"), "owner")
        # Serialize only the JSON-safe payload; QuerySpec/Constraint objects
        # remain out of the wire response so their reprs do not leak identity.
        wire = {key: value for key, value in result.items() if isinstance(value, (str, int, float, bool, list, dict, type(None)))}
        self.assertNotIn("attacker", json.dumps(wire, ensure_ascii=False, default=str))

    def test_parser_failure_does_not_return_keyword_classification(self):
        gamma = FakeSemanticGamma(responses={
            "查询解析器": "not-a-json",
        })
        agent = self._make_agent(gamma)
        with _flags(SENTRIX_THIN_AGENT_V1="1", SENTRIX_SEMANTIC_QUERY_PARSER_V1="1"):
            result = agent.answer_turn(
                "帮我写一段关于家庭照片的散文",
                scope_id="home", viewer_id="owner",
            )
        self.assertFalse(result.get("evidence"), "parser 失败不得因关键词'照片'把普通写作误路由到 evidence")


if __name__ == "__main__":
    unittest.main()
