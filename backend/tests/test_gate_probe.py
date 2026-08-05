"""Phase R R4 — Gate decision + Neutral Probe (P0-6/P0-7/P0-8)."""

import unittest

from backend.memory_gate import MemoryGate
from backend.query_contracts import QueryFacet, QueryParseDraft, QueryAction
from backend.retrieval import NeutralProbe, RetrievalConfig
from backend.retrieval.base import CandidateHit
from backend.retrieval.probes import ProbeOutcome


def _draft(mode, **kwargs):
    draft = QueryParseDraft(intent="answer", answer_target="general", mode=mode)
    for key, value in kwargs.items():
        setattr(draft, key, value)
    return draft


class GateDecisionTests(unittest.TestCase):
    def test_writing_prefix_none_no_probe(self):
        decision = MemoryGate().classify("帮我写一段生日祝福", draft=_draft("none"))
        self.assertEqual(decision.mode, "none")
        self.assertFalse(decision.allow_probe)

    def test_no_length_heuristic(self):
        # Long message that is a general task -> none (no length>6 repair rule).
        decision = MemoryGate().classify("请解释一下量子纠缠为什么不等于超光速通信", draft=_draft("none"))
        self.assertEqual(decision.mode, "none")

    def test_parser_none_bare_noun_routes_to_probe(self):
        decision = MemoryGate().classify("银色心形手镯", draft=_draft("none"))
        self.assertEqual(decision.mode, "ambiguous")
        self.assertTrue(decision.allow_probe)

    def test_parser_none_with_household_facet_routes_to_probe(self):
        draft = _draft("none", facets=[QueryFacet("person", "明哥")])
        decision = MemoryGate().classify("介绍一下明哥", draft=draft)
        self.assertEqual(decision.mode, "ambiguous")
        self.assertTrue(decision.allow_probe)

    def test_parser_evidence_stays_evidence(self):
        draft = _draft("evidence", actions=[QueryAction("answer_question", "person")])
        decision = MemoryGate().classify("介绍一下明哥", draft=draft)
        self.assertEqual(decision.mode, "evidence")
        self.assertFalse(decision.allow_probe)

    def test_confidence_not_a_single_point_routing_input(self):
        low = MemoryGate().classify("银色心形手镯", draft=_draft("none", confidence=0.91))
        high = MemoryGate().classify("银色心形手镯", draft=_draft("none", confidence=0.1))
        # Both route to probe regardless of self-reported confidence.
        self.assertEqual(low.mode, high.mode)


class NeutralProbeTests(unittest.TestCase):
    def _hits(self, asset_ids, retriever="lexical"):
        return [CandidateHit(asset_id=asset_id, retriever=retriever, raw_score=float(i + 1),
                             score_kind="discrete", higher_is_better=True, rank=i + 1)
                for i, asset_id in enumerate(asset_ids)]

    def _probe(self):
        return NeutralProbe(RetrievalConfig())

    def test_multi_channel_agreement_upgrades(self):
        channel_hits = {
            "lexical": self._hits(["asset_1", "asset_2"]),
            "visual_ann": self._hits(["asset_1"]),
        }
        outcome = self._probe().run("银色心形手镯", channel_hits, scope_id="album1")
        self.assertEqual(outcome.decision, "upgrade")
        self.assertIn("asset_1", outcome.signals["shared_assets"])

    def test_single_weak_channel_clarifies(self):
        channel_hits = {"lexical": self._hits(["asset_1"])}
        outcome = self._probe().run("今天的晚饭", channel_hits, scope_id="album1")
        self.assertEqual(outcome.decision, "clarify")

    def test_no_hits_clarifies(self):
        outcome = self._probe().run("随机词", {}, scope_id="album1")
        self.assertEqual(outcome.decision, "clarify")

    def test_exact_lexical_phrase_upgrades(self):
        hit = CandidateHit(asset_id="asset_1", retriever="lexical", raw_score=3.0,
                           score_kind="token_hits", higher_is_better=True, rank=1,
                           matched_text="银色心形手镯")
        outcome = self._probe().run("银色心形手镯", {"lexical": [hit]}, scope_id="album1")
        self.assertEqual(outcome.decision, "upgrade")

    def test_probe_never_emits_household_facts(self):
        # A ProbeOutcome is a routing signal only — no answer text, no evidence.
        outcome = ProbeOutcome(decision="upgrade", reason="test")
        self.assertEqual(outcome.decision, "upgrade")
        self.assertFalse(hasattr(outcome, "answer"))
        self.assertFalse(hasattr(outcome, "evidence"))


if __name__ == "__main__":
    unittest.main()
