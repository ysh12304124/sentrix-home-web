"""Phase 2R-1 red tests for the model Query Parser.

The parser does not exist yet — this file will import it once Phase 2R-2 adds
``backend/query_parser.py``.  Every test below encodes an invariant from the
supplementary plan §5-§8.
"""

import json
import unittest


try:
    from backend.query_parser import QueryParser  # type: ignore
    QUERY_PARSER_AVAILABLE = True
except Exception:
    QueryParser = None  # type: ignore
    QUERY_PARSER_AVAILABLE = False


class ScriptedGamma:
    """Programmable gamma with per-prompt-marker responses."""

    model = "scripted-parser"

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
        return "{}"


@unittest.skipUnless(QUERY_PARSER_AVAILABLE, "backend.query_parser not implemented yet (Phase 2R-2)")
class QueryParserOpenVocabularyTests(unittest.TestCase):
    """开放词汇不改代码就能进 constraints。"""

    OPEN_VOCAB_CASES = (
        ("公园里散步的照片", "place", "公园"),
        ("找找我们炒菜的时候", "activity", "炒菜"),
        ("穿米白色毛领棉衣的那次", "clothing", "米白色毛领棉衣"),
        ("有没有毛领棉衣", "clothing", "毛领棉衣"),
    )

    def test_open_vocabulary_conditions_are_preserved(self):
        for message, dimension, surface in self.OPEN_VOCAB_CASES:
            with self.subTest(message=message):
                gamma = ScriptedGamma(responses={
                    "查询解析器": {
                        "mode": "evidence",
                        "actions": [{"type": "answer_question", "target": "general"}],
                        "facets": [{"dimension": dimension, "surface_text": surface}],
                        "semantic_conditions": [{"dimension": dimension, "value": surface, "strictness": "semantic_required"}],
                    },
                })
                parser = QueryParser(gamma=gamma)
                draft = parser.parse(message, recent_turns="")
                terms = [item.get("value") for item in draft.semantic_conditions]
                self.assertIn(surface, terms, f"open vocabulary {surface} lost from parser output")


@unittest.skipUnless(QUERY_PARSER_AVAILABLE, "backend.query_parser not implemented yet (Phase 2R-2)")
class QueryParserSanitizerTests(unittest.TestCase):
    """模型返回运行时身份字段必须被丢弃。"""

    def test_model_scope_viewer_entity_ids_are_ignored(self):
        gamma = ScriptedGamma(responses={
            "查询解析器": {
                "mode": "evidence",
                "actions": [{"type": "answer_question", "target": "person"}],
                "scope_id": "attacker-scope",
                "scope_mode": "single",
                "viewer_id": "attacker-viewer",
                "entity_ids": ["attacker-entity"],
                "conversation_id": "attacker-conv",
            },
        })
        parser = QueryParser(gamma=gamma)
        draft = parser.parse("介绍一下明哥", recent_turns="")
        payload = json.dumps(draft.__dict__, ensure_ascii=False, default=str)
        self.assertNotIn("attacker", payload, "sanitizer must strip identity fields returned by the model")


@unittest.skipUnless(QUERY_PARSER_AVAILABLE, "backend.query_parser not implemented yet (Phase 2R-2)")
class QueryParserFailureFallbackTests(unittest.TestCase):
    """模型失败必须走安全降级，不能退化成关键词分类器。"""

    def test_illegal_json_triggers_repair_then_safe_fallback(self):
        gamma = ScriptedGamma(responses={"查询解析器": "not-json"})
        parser = QueryParser(gamma=gamma)
        draft = parser.parse("帮我写关于家庭照片的散文", recent_turns="")
        self.assertIn(draft.intent, {"answer", "none"}, "failure must produce safe intent")
        self.assertFalse([item for item in draft.semantic_conditions if item.get("value") in ("照片", "家庭")],
                          "safe fallback must not slip keywords back in")

    def test_repair_is_called_at_most_once(self):
        calls = []

        def responder(prompt):
            calls.append(prompt)
            return "still-not-json"

        gamma = ScriptedGamma(responses={
            "查询解析器": responder,
            "修复": responder,
        })
        parser = QueryParser(gamma=gamma)
        parser.parse("有没有妈妈的照片", recent_turns="")
        repair_calls = [call for call in calls if "修复" in call]
        self.assertLessEqual(len(repair_calls), 1, "repair should run at most once per turn")


@unittest.skipUnless(QUERY_PARSER_AVAILABLE, "backend.query_parser not implemented yet (Phase 2R-2)")
class QueryParserDeterministicHardTests(unittest.TestCase):
    """代码从原文恢复日期/media/否定；模型不能把这些降为软条件。"""

    def test_missing_date_is_recovered_from_user_message(self):
        gamma = ScriptedGamma(responses={
            "查询解析器": {"mode": "evidence", "actions": [{"type": "return_assets", "coverage": "best"}]},
        })
        parser = QueryParser(gamma=gamma)
        draft = parser.parse("2024 年 5 月厨房的照片", recent_turns="")
        self.assertTrue(getattr(draft, "time_expression", None), "explicit date must be recovered even when model omits it")

    def test_explicit_negation_stays_hard(self):
        gamma = ScriptedGamma(responses={
            "查询解析器": {
                "mode": "evidence",
                "actions": [{"type": "return_assets", "coverage": "best"}],
                "negative_conditions": [{"dimension": "media", "value": "video"}],
            },
        })
        parser = QueryParser(gamma=gamma)
        draft = parser.parse("不要视频，只要照片", recent_turns="")
        negatives = [item.get("value") for item in getattr(draft, "negative_conditions", [])]
        self.assertIn("video", negatives, "explicit negation must survive the parser")


if __name__ == "__main__":
    unittest.main()
