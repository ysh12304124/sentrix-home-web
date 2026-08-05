"""Phase 4 complex-path tests for the Thin Agent."""

import json
import os
import tempfile
import unittest

from backend.agent import MemoryAgent
from backend.db import MemoryStore
from backend.complex_answer import ComplexAnswerBuilder
from backend.evidence_retrieval import EvidencePacket
from backend.query_contracts import (
    Constraint, HARD, QueryAction, QueryFacet, QueryParseDraft, QuerySpec,
)


class ScriptedGamma:
    model = "scripted-complex"

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
        return "我在听。"


def _packet(assets=None):
    return EvidencePacket(
        "q1", "home", "person",
        assets=assets or [{
            "asset_id": "asset-1", "file_name": "a.jpg", "media_type": "image",
            "observation_ids": ["obs-1"], "evidence_ids": ["asset-1", "obs-1"],
            "condition_results": {}, "level": "exact", "score": 1.0,
            "observation_fields": {"place": "厨房", "activity": "做饭"},
            "captured_at": "2024-05-12T10:00:00",
        }],
        exact_results=[],
    )


def _spec():
    return QuerySpec(
        query_id="q1", scope_mode="single", scope_ids=["home"], viewer_id="owner",
        conversation_id="c1", intent="answer", answer_target="person",
        constraints=[Constraint("person", "明哥", HARD, "confirmed_bridge", source_text="明哥")],
        entity_ids=["entity-ming"], actions=[QueryAction("summarize_person", "person")],
        facets=[QueryFacet("person", "明哥")],
    )


class ComplexAnswerBuilderTests(unittest.TestCase):
    def test_writer_and_verifier_produce_statements_with_evidence(self):
        # NB: dict order matters — Claim Extractor prompt contains the word
        # "Writer" ("不信任 Writer 提供的 claim 列表") so its marker must be
        # inspected first.
        gamma = ScriptedGamma(responses={
            "独立的家庭事实 Claim Extractor": {
                "claims": [
                    {"claim_id": "claim_1", "text": "明哥经常出现在厨房相关记录里",
                     "start": 0, "end": 15, "intended_type": "observed_pattern"},
                ],
            },
            "人物叙事 Writer": {
                "text": "明哥经常出现在厨房相关记录里。",
                "candidate_claims": [
                    {"text": "明哥经常出现在厨房相关记录里",
                     "intended_type": "observed_pattern",
                     "candidate_evidence_ids": ["asset-1", "obs-1"]},
                ],
                "unknowns": [],
            },
        })
        builder = ComplexAnswerBuilder(gamma=gamma)
        result = builder.build("介绍一下明哥", _spec(), _packet())
        self.assertFalse(result["fallback"])
        self.assertTrue(result["answer"])
        self.assertTrue(result["statements"])
        self.assertTrue(result["statements"][0]["evidence_ids"])

    def test_writer_unavailable_returns_safe_fallback(self):
        gamma = ScriptedGamma(responses={})  # No Writer response
        builder = ComplexAnswerBuilder(gamma=gamma)
        result = builder.build("介绍一下明哥", _spec(), _packet())
        self.assertTrue(result["fallback"])
        # Fallback never surfaces unverified free text.
        self.assertIn("明哥", result["answer"])

    def test_claim_extractor_unavailable_falls_back(self):
        gamma = ScriptedGamma(responses={
            "人物叙事 Writer": {"text": "明哥经常出现。", "candidate_claims": [], "unknowns": []},
            # No Claim Extractor response — extractor returns []
        })
        builder = ComplexAnswerBuilder(gamma=gamma)
        result = builder.build("介绍一下明哥", _spec(), _packet())
        self.assertTrue(result["fallback"])


class ThinAgentComplexRoutingTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix="thin-complex-")
        self.store = MemoryStore(os.path.join(self.directory.name, "memory.db"))
        self.asset = self.store.create_asset("asset-may", "may.jpg", "image", "/tmp/may.jpg",
                                              metadata={"captured_at": "2024-05-12T10:00:00"}, scope_id="home")
        self.store.add_observation(
            self.asset["id"],
            {"id": "obs-may", "scope_id": "home", "captured_at": "2024-05-12T10:00:00",
             "caption": "明哥在厨房", "place": "厨房", "activity": "做饭",
             "people": ["明哥"], "confidence": 0.9},
            scope_id="home",
        )
        self.store.create_entity("明哥", "person", "confirmed", scope_id="home")

    def tearDown(self):
        self.store.close()
        self.directory.cleanup()

    def test_complex_path_used_when_flag_on(self):
        gamma = ScriptedGamma(responses={
            "查询解析器": {"mode": "evidence",
                          "actions": [{"type": "summarize_person", "target": "person"}],
                          "answer_target": "person", "entity_names": ["明哥"]},
            "独立的家庭事实 Claim Extractor": {
                "claims": [
                    {"claim_id": "claim_1", "text": "明哥经常出现在厨房相关场景中",
                     "start": 8, "end": 22, "intended_type": "observed_pattern"},
                ],
            },
            "人物叙事 Writer": {
                "text": "从现有记录看，明哥经常出现在厨房相关场景中。",
                "candidate_claims": [
                    {"text": "明哥经常出现在厨房相关场景中",
                     "intended_type": "observed_pattern",
                     "candidate_evidence_ids": ["asset-may", "obs-may"]},
                ],
                "unknowns": [],
            },
        })
        os.environ["SENTRIX_THIN_AGENT_V1"] = "1"
        os.environ["SENTRIX_LLM_CLAIM_EXTRACTOR_V1"] = "1"
        try:
            agent = MemoryAgent(self.store, gamma=gamma)
            result = agent.answer_turn("介绍一下明哥", scope_id="home", viewer_id="owner")
        finally:
            os.environ.pop("SENTRIX_LLM_CLAIM_EXTRACTOR_V1", None)
        writer_calls = [call for call in gamma.calls if "人物叙事 Writer" in call]
        self.assertTrue(writer_calls, "complex path should invoke the Writer prompt")
        self.assertIn("明哥", result["answer"])


if __name__ == "__main__":
    unittest.main()
