"""RX-2 (U4): the Writer -> Claim -> Verify chain must actually run.

Before this fix the complex chain silently skipped the Claim Extractor when the
Writer returned legal-but-unusable JSON (a bare list or an empty object), and
even a successful extraction dropped the Writer's candidate evidence anchors so
every claim verified as unsupported.  These tests pin the corrected behaviour
with a stub gamma (no network).
"""

import json
import unittest

from backend.complex_answer import ComplexAnswerBuilder, _propagate_candidate_evidence
from backend.evidence_retrieval import EvidencePacket
from backend.query_contracts import Constraint, HARD, QuerySpec


class StubGamma:
    def __init__(self, responses):
        self.responses = dict(responses)
        self.calls = []

    def chat(self, prompt, json_mode=True, role=None):
        self.calls.append(role)
        fn = self.responses.get(role)
        return fn(prompt) if callable(fn) else fn


def _spec():
    return QuerySpec("q", "single", ["home"], "owner", "c", "answer", "person",
                     constraints=[Constraint("person", "明哥", HARD, "confirmed_bridge")])


def _packet():
    asset = {
        "asset_id": "asset-1", "file_name": "a.jpg", "media_type": "image",
        "observation_ids": ["obs-1"], "evidence_ids": ["asset-1", "obs-1"],
        "condition_results": {"person:明哥": {"status": "matched"}},
        "level": "exact", "captured_at": "2025-10-02",
        "observation_fields": {"place": "海边", "activity": "爬山"},
    }
    return EvidencePacket("q", "home", "person", assets=[asset],
                          exact_results=[asset])


_WRITER_DICT = {
    "text": "明哥出现在这些记录中，记录里有海边爬山。",
    "candidate_claims": [
        {"text": "明哥出现在这些记录中", "intended_type": "observed_pattern",
         "candidate_evidence_ids": ["asset-1"]},
        {"text": "记录里有海边爬山", "intended_type": "confirmed_fact",
         "candidate_evidence_ids": ["asset-1"]},
    ],
    "unknowns": ["他的性格还不能确定"],
}

_CLAIM_JSON = json.dumps({
    "claims": [
        {"claim_id": "claim_1", "text": "明哥出现在这些记录中", "start": 0, "end": 11,
         "intended_type": "observed_pattern"},
        {"claim_id": "claim_2", "text": "记录里有海边爬山", "start": 12, "end": 21,
         "intended_type": "confirmed_fact"},
    ],
})


class PersonChainTests(unittest.TestCase):
    def test_writer_then_claim_both_called_and_verified(self):
        gamma = StubGamma({"writer": json.dumps(_WRITER_DICT), "claim": _CLAIM_JSON})
        builder = ComplexAnswerBuilder(gamma=gamma)
        result = builder.build("介绍一下明哥", _spec(), _packet())
        self.assertEqual(gamma.calls, ["writer", "claim"])
        self.assertFalse(result.get("fallback"))
        self.assertTrue(result["claims"])
        self.assertTrue(all(v.get("status") == "reasonable_summary"
                            for v in result["verifications"]))
        self.assertTrue(all("asset-1" in (s.get("evidence_ids") or [])
                            for s in result["statements"]))

    def test_writer_returns_list_json_still_reaches_claim(self):
        # A valid JSON list (not an object) previously short-circuited to the
        # safe fallback before the Claim Extractor ever ran.
        gamma = StubGamma({"writer": json.dumps([_WRITER_DICT]), "claim": _CLAIM_JSON})
        builder = ComplexAnswerBuilder(gamma=gamma)
        result = builder.build("介绍一下明哥", _spec(), _packet())
        self.assertEqual(gamma.calls, ["writer", "claim"])
        self.assertFalse(result.get("fallback"))

    def test_empty_writer_object_falls_back_with_reason(self):
        gamma = StubGamma({"writer": "{}"})
        builder = ComplexAnswerBuilder(gamma=gamma)
        result = builder.build("介绍一下明哥", _spec(), _packet())
        self.assertEqual(gamma.calls, ["writer"])
        self.assertTrue(result.get("fallback"))
        self.assertEqual(builder.last_fallback_reason, "writer_unavailable")

    def test_empty_claim_list_falls_back_with_reason(self):
        gamma = StubGamma({"writer": json.dumps(_WRITER_DICT),
                           "claim": json.dumps({"claims": []})})
        builder = ComplexAnswerBuilder(gamma=gamma)
        result = builder.build("介绍一下明哥", _spec(), _packet())
        self.assertEqual(gamma.calls, ["writer", "claim"])
        self.assertTrue(result.get("fallback"))
        self.assertEqual(builder.last_fallback_reason, "claim_extractor_unavailable")


class PropagateEvidenceTests(unittest.TestCase):
    def test_propagates_matching_candidate_ids(self):
        claims = [{"text": "明哥出现在这些记录中", "candidate_evidence_ids": []}]
        candidates = [{"text": "明哥出现在这些记录中", "candidate_evidence_ids": ["asset-1", "obs-1"]}]
        _propagate_candidate_evidence(claims, candidates)
        self.assertEqual(claims[0]["candidate_evidence_ids"], ["asset-1", "obs-1"])


if __name__ == "__main__":
    unittest.main()
