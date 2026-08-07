"""RX-4 Response Writer tests: prompt hygiene + parsing + safe fallback."""

import json
import unittest

from backend.answer_brief import AnswerBrief, Fact, Presentation, VisibleAsset
from backend.response_plan import plan_response
from backend.response_writer import build_prompt, safe_fallback, write_response


class _Gamma:
    def __init__(self, result):
        self._result = result
        self.prompt = None
        self.role = None

    def chat(self, prompt, json_mode=True, role=None):
        self.prompt = prompt
        self.role = role
        if callable(self._result):
            return self._result(prompt)
        return self._result


def _brief(mode="exact_result"):
    return AnswerBrief(
        "b1", "find_and_explain_images", mode,
        facts=[Fact("fact_1", "记录中有「海边」", "confirmed", ["asset-1"])],
        uncertainties=[],
        visible_assets=[VisibleAsset("asset-1", "照片1", supported_aspects=["海边"])],
        presentation=Presentation(show_images=True),
    )


class PromptHygieneTests(unittest.TestCase):
    def test_prompt_has_no_internal_fields(self):
        prompt = build_prompt(_brief("exact_result"), plan_response(_brief("exact_result")), "去年十月爬山拍的合影")
        # Real internal identifiers never reach the writer — the literal token
        # "asset_" appears only as a ban instruction, so assert on actual ids.
        self.assertNotIn("asset-1", prompt)
        self.assertNotIn("obs-", prompt)
        self.assertNotIn("condition_results", prompt)
        self.assertNotIn("recall_strength", prompt)
        self.assertIn("照片1", prompt)
        self.assertIn("fact_1", prompt)

    def test_prompt_carries_must_not_say(self):
        brief = _brief("approximate_result")
        brief.must_not_say = ["确定是爬山合影"]
        prompt = build_prompt(brief, plan_response(brief), "水族馆海豚跃出水面")
        self.assertIn("确定是爬山合影", prompt)


class WriteResponseTests(unittest.TestCase):
    def test_parses_dict_output(self):
        gamma = _Gamma(json.dumps({
            "text": "我找到了去年十月在照片里记录的海边。",
            "statements": [{"text": "记录中有海边", "fact_id": "fact_1", "certainty": "confirmed"}],
        }))
        answer, statements = write_response(_brief("exact_result"), plan_response(_brief()), "去年十月", gamma=gamma)
        self.assertEqual(gamma.role, "answer")
        self.assertIn("海边", answer)
        self.assertEqual(statements[0]["fact_id"], "fact_1")

    def test_unknown_fact_id_downgraded_to_null(self):
        gamma = _Gamma(json.dumps({
            "text": "找到了。",
            "statements": [{"text": "不存在的", "fact_id": "fact_99", "certainty": "confirmed"}],
        }))
        _, statements = write_response(_brief("exact_result"), plan_response(_brief()), "去年十月", gamma=gamma)
        self.assertIsNone(statements[0]["fact_id"])

    def test_gamma_exception_returns_none(self):
        class Boom(_Gamma):
            def chat(self, prompt, json_mode=True, role=None):
                raise RuntimeError("down")
        answer, statements = write_response(_brief(), plan_response(_brief()), "x", gamma=Boom(None))
        self.assertIsNone(answer)
        self.assertEqual(statements, [])

    def test_no_model_returns_none(self):
        answer, _ = write_response(_brief(), plan_response(_brief()), "x")
        self.assertIsNone(answer)


class SafeFallbackTests(unittest.TestCase):
    def test_no_result_offers_direction(self):
        text, statements = safe_fallback(_brief("no_result"), plan_response(_brief("no_result")))
        self.assertIn("没有找到", text)
        self.assertEqual(statements, [])

    def test_person_gap_no_claim(self):
        brief = _brief("person_summary")
        brief.facts = []
        text, _ = safe_fallback(brief, plan_response(brief))
        self.assertNotIn("多次出现", text)
        self.assertIn("还没有足够", text)


if __name__ == "__main__":
    unittest.main()
