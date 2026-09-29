"""Phase R9-2 — NeutralProbe v2 tests.

The probe is a routing signal only: it reports agreement / candidates /
conflicts / index health, accepts session focus and media hints, and returns
``no_household_match`` when nothing matched.  It never produces user facts.
"""

import unittest

from backend.retrieval import NeutralProbe, RetrievalConfig
from backend.retrieval.base import CandidateHit
from backend.retrieval.probes import ProbeOutcome


def _hit(asset_id, retriever="lexical", score=1.0, matched_text=""):
    return CandidateHit(asset_id=asset_id, retriever=retriever, raw_score=score,
                        score_kind="token_hits", higher_is_better=True, rank=1,
                        matched_text=matched_text)


class NeutralProbeV2Tests(unittest.TestCase):
    def _probe(self):
        return NeutralProbe(RetrievalConfig())

    def test_empty_channels_is_no_household_match(self):
        outcome = self._probe().run("随机词", {})
        self.assertEqual(outcome.decision, "no_household_match")
        self.assertEqual(outcome.channel_agreement, 0)
        self.assertEqual(outcome.top_candidates, [])

    def test_agreement_upgrade_carries_health_and_candidates(self):
        health = {"visual_ann": {"status": "ok", "hits": 1}}
        outcome = self._probe().run(
            "银色心形手镯",
            {"lexical": [_hit("a1")], "visual_ann": [_hit("a1", "visual_ann")]},
            index_health=health,
        )
        self.assertEqual(outcome.decision, "upgrade")
        self.assertGreaterEqual(outcome.channel_agreement, 2)
        self.assertIn("a1", outcome.top_candidates)
        self.assertEqual(outcome.index_health, health)

    def test_conflicting_channels_reported(self):
        outcome = self._probe().run(
            "银色心形手镯",
            {"lexical": [_hit("a1")], "visual_ann": [_hit("a2", "visual_ann")]},
        )
        self.assertEqual(outcome.decision, "upgrade")  # 2 channels -> upgrade
        self.assertIn("a1", outcome.top_candidates)
        self.assertIn("a2", outcome.top_candidates)
        self.assertTrue(outcome.conflicts)  # channels disagree on the asset

    def test_focus_and_media_hint_recorded(self):
        outcome = self._probe().run("那件黄色的", {}, focus={"active_entity_ids": ["e1"]},
                                    media_hint="image")
        self.assertEqual(outcome.decision, "no_household_match")
        self.assertTrue(outcome.signals["focus_active"])
        self.assertEqual(outcome.signals["media_hint"], "image")

    def test_probe_never_emits_user_facts(self):
        outcome = ProbeOutcome(decision="upgrade", reason="x",
                               top_candidates=["a1"], index_health={})
        self.assertFalse(hasattr(outcome, "answer"))
        self.assertFalse(hasattr(outcome, "evidence"))
        self.assertFalse(hasattr(outcome, "confirmed_fact"))


if __name__ == "__main__":
    unittest.main()
