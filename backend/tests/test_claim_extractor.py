"""ClaimExtractor path invariants (Phase 2R-1 + 2R-7)."""

import json
import unittest

from backend.claim_extractor import ClaimExtractor


try:
    from backend.claim_extractor import LLMClaimExtractor  # type: ignore
    LLM_CLAIM_EXTRACTOR_AVAILABLE = True
except Exception:
    LLMClaimExtractor = None  # type: ignore
    LLM_CLAIM_EXTRACTOR_AVAILABLE = False


class RegexClaimExtractorTests(unittest.TestCase):
    """The regex extractor stays available as a fallback but must not carry semantics."""

    def test_scans_full_writer_text_even_when_writer_omits_candidate(self):
        text = "明哥经常参加展览。他应该很喜欢艺术。"
        claims = ClaimExtractor().scan(text, [{"text": "明哥经常参加展览", "candidate_evidence_ids": ["event-1"]}])
        self.assertEqual(len(claims), 2)
        self.assertEqual(claims[1]["text"], "他应该很喜欢艺术")
        self.assertEqual(claims[1]["candidate_evidence_ids"], [])


class ScriptedGamma:
    model = "scripted-claim"

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


@unittest.skipUnless(LLM_CLAIM_EXTRACTOR_AVAILABLE, "LLMClaimExtractor not implemented yet (Phase 2R-7)")
class LLMClaimExtractorTests(unittest.TestCase):
    """Complex answers must go through the model-driven extractor."""

    def test_single_sentence_can_yield_multiple_claims(self):
        gamma = ScriptedGamma(responses={
            "Claim Extractor": {
                "claims": [
                    {"claim_id": "claim_1", "text": "明哥常和家人一起做饭",
                     "start": 0, "end": 12, "intended_type": "derived_pattern"},
                    {"claim_id": "claim_2", "text": "他对烹饪似乎充满兴趣",
                     "start": 13, "end": 25, "intended_type": "inference"},
                ]
            },
        })
        extractor = LLMClaimExtractor(gamma=gamma)
        claims = extractor.scan("明哥常和家人一起做饭，他对烹饪似乎充满兴趣。", writer_candidates=[])
        types = [claim.get("intended_type") for claim in claims]
        self.assertIn("derived_pattern", types)
        self.assertIn("inference", types)

    def test_covers_negations_and_unknowns(self):
        gamma = ScriptedGamma(responses={
            "Claim Extractor": {
                "claims": [
                    {"claim_id": "c1", "text": "记录里没有明哥的生日",
                     "start": 0, "end": 10, "intended_type": "negative"},
                    {"claim_id": "c2", "text": "目前无法确认他和小黑的关系",
                     "start": 11, "end": 25, "intended_type": "uncertainty"},
                ]
            },
        })
        extractor = LLMClaimExtractor(gamma=gamma)
        claims = extractor.scan("记录里没有明哥的生日，目前无法确认他和小黑的关系。", writer_candidates=[])
        types = [claim.get("intended_type") for claim in claims]
        self.assertIn("negative", types)
        self.assertIn("uncertainty", types)

    def test_extracts_facts_without_hint_keywords(self):
        gamma = ScriptedGamma(responses={
            "Claim Extractor": {
                "claims": [
                    {"claim_id": "c1", "text": "明哥去年在厨房拍了一张照片",
                     "start": 0, "end": 15, "intended_type": "fact"},
                ]
            },
        })
        extractor = LLMClaimExtractor(gamma=gamma)
        claims = extractor.scan("明哥去年在厨房拍了一张照片。", writer_candidates=[])
        # 没有 "经常/通常/喜欢" 等模式关键词，仍应被识别为事实主张
        self.assertEqual(claims[0]["intended_type"], "fact")


@unittest.skipUnless(LLM_CLAIM_EXTRACTOR_AVAILABLE, "LLMClaimExtractor not implemented yet (Phase 2R-7)")
class LLMClaimExtractorFallbackTests(unittest.TestCase):
    """When the model is unavailable, extractor must not fabricate claims."""

    def test_model_failure_returns_no_claims(self):
        gamma = ScriptedGamma(responses={"Claim Extractor": "not-json"})
        extractor = LLMClaimExtractor(gamma=gamma)
        claims = extractor.scan("这是一段自由文本。", writer_candidates=[])
        self.assertEqual(claims, [])


if __name__ == "__main__":
    unittest.main()
