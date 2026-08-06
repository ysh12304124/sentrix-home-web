"""Phase R9-1 — Final Router decision tests.

The Router owns the final route; the Parser's proposed_mode is advisory only.
These cases pin the R9 contract: family queries must survive a parser "none",
general-intro verbs must not decide "none" by themselves, ambiguous phrases must
not fall into normal chat, and explicit writing must never probe memory.
"""

import unittest

from backend.query_contracts import QueryAction, QueryFacet, QueryParseDraft
from backend.retrieval.probes import ProbeOutcome
from backend.router import Router


def _draft(proposed_mode="none", **kwargs):
    return QueryParseDraft(intent="answer", answer_target="general",
                           proposed_mode=proposed_mode, **kwargs)


class RouterRouteTests(unittest.TestCase):
    def setUp(self):
        # "明哥" is a confirmed person in the scope.
        self.router = Router(
            entity_resolver=lambda name: "ent-mg" if name == "明哥" else None,
            message_entity_resolver=lambda msg: ["ent-mg"] if "明哥" in str(msg) else [],
        )

    def test_writing_prefix_none_no_probe(self):
        decision = self.router.route("帮我写一段生日祝福", _draft("none"))
        self.assertEqual(decision.mode, "none")
        self.assertFalse(decision.probe_required)

    def test_mid_sentence_compose_is_writing_not_household(self):
        decision = self.router.route("我想写一篇明哥的虚构故事", _draft("none"))
        self.assertEqual(decision.mode, "none")

    def test_album_themed_writing_stays_none(self):
        decision = self.router.route("以相册为主题写一篇短文", _draft("none"))
        self.assertEqual(decision.mode, "none")

    def test_photo_reads_is_not_writing(self):
        # "照片里写着什么？" contains 写着 but asks what is written — household.
        draft = _draft("evidence", media_expressions=["照片"])
        decision = self.router.route("照片里写着什么？", draft)
        self.assertEqual(decision.mode, "evidence")

    def test_confirmed_person_introduction_is_evidence(self):
        draft = _draft("none", facets=[QueryFacet("person", "明哥")])
        decision = self.router.route("介绍一下明哥", draft)
        self.assertEqual(decision.mode, "evidence")
        self.assertEqual(decision.answer_target, "person")

    def test_intro_verb_with_confirmed_person_in_message_is_evidence(self):
        # "介绍一下明哥" with a broken parser (no facets) must NOT go to chat
        # when 明哥 is a confirmed entity — the raw-message mention routes it
        # to evidence.
        decision = self.router.route("介绍一下明哥", _draft("none"))
        self.assertEqual(decision.mode, "evidence")
        self.assertEqual(decision.answer_target, "person")

    def test_intro_verb_unconfirmed_parser_none_is_general(self):
        # A parser-none "介绍一下X" with no confirmed entity and no household
        # signal is a general-intro question.  The confirmed-entity mechanism
        # (test above) is the protection for real family members; this edge is
        # the documented tradeoff, never a keyword classification.
        router = Router()  # nothing confirmed
        decision = router.route("介绍一下明哥", _draft("none"))
        self.assertEqual(decision.mode, "none")

    def test_parser_failure_intro_verb_routes_to_clarify(self):
        # R9-6: when the parser TIMES OUT (parser_failed), an intro verb must
        # not decide "none" (family query lost) and must not probe into visual
        # noise — it clarifies immediately.
        router = Router()
        failed = _draft("none", parser_failed=True)
        decision = router.route("介绍一下明哥", failed)
        self.assertEqual(decision.mode, "clarify")

    def test_parser_failure_bare_noun_still_probes(self):
        # A parser-down bare noun (no general verb) is unaffected and probes.
        router = Router()
        failed = _draft("none", parser_failed=True)
        decision = router.route("银色心形手镯", failed)
        self.assertEqual(decision.mode, "ambiguous")
        self.assertTrue(decision.probe_required)

    def test_concept_question_is_none(self):
        decision = self.router.route("请解释一下量子纠缠为什么不等于超光速", _draft("none"))
        self.assertEqual(decision.mode, "none")

    def test_general_verb_with_date_anchor_stays_household(self):
        draft = _draft("none")
        decision = self.router.route("为什么去年春节没有小黑的照片", draft)
        # date anchor -> weak -> probe (not none).
        self.assertIn(decision.mode, {"evidence", "ambiguous"})
        self.assertTrue(decision.probe_required)

    def test_product_concept_intro_is_none(self):
        decision = self.router.route('介绍一下"家庭相册"这个产品概念', _draft("none"))
        self.assertEqual(decision.mode, "none")

    def test_bare_noun_phrase_routes_to_probe(self):
        decision = self.router.route("银色心形手镯", _draft("none"))
        self.assertEqual(decision.mode, "ambiguous")
        self.assertTrue(decision.probe_required)

    def test_parser_none_with_person_facet_routes_to_probe_when_unresolved(self):
        router = Router()  # no resolver
        draft = _draft("none", facets=[QueryFacet("person", "某人")])
        decision = router.route("介绍一下某人", draft)
        self.assertEqual(decision.mode, "ambiguous")
        self.assertTrue(decision.probe_required)

    def test_contextual_person_mention_stays_contextual(self):
        draft = _draft("contextual", facets=[QueryFacet("person", "小黑")],
                       actions=[QueryAction("answer_question", "general")])
        decision = self.router.route("今晚回家时想起小黑了", draft)
        self.assertEqual(decision.mode, "contextual")

    def test_strong_evidence_action_is_evidence(self):
        draft = _draft("evidence", actions=[QueryAction("return_assets", "general")])
        decision = self.router.route("把去年拍的照片给我", draft)
        self.assertEqual(decision.mode, "evidence")

    def test_session_follow_up_reuses_focus(self):
        focus = {"active_entity_ids": ["ent-mg"], "active_event_ids": []}
        decision = self.router.route("上次说的那件黄色的", _draft("none"), focus=focus)
        self.assertEqual(decision.mode, "evidence")
        self.assertEqual(decision.focus_ids, ["ent-mg"])


class RouterProbeFinalizeTests(unittest.TestCase):
    def test_probe_upgrade_routes_to_evidence(self):
        router = Router()
        decision = router.route("银色心形手镯", _draft("none"))
        final = router.resolve_after_probe(
            ProbeOutcome("upgrade", {"visual_ann": 1}, {}, "2 channels"),
            "银色心形手镯", decision, _draft("none"),
        )
        self.assertEqual(final.mode, "evidence")

    def test_probe_clarify_stays_clarify(self):
        router = Router()
        decision = router.route("银色心形手镯", _draft("none"))
        final = router.resolve_after_probe(
            ProbeOutcome("clarify", {}, {}, "weak candidates"),
            "银色心形手镯", decision, _draft("none"),
        )
        self.assertEqual(final.mode, "clarify")

    def test_probe_no_hit_ambiguous_phrase_clarifies_not_chat(self):
        router = Router()
        decision = router.route("银色心形手镯", _draft("none"))
        final = router.resolve_after_probe(
            ProbeOutcome("no_household_match", {}, {}, "no hit"),
            "银色心形手镯", decision, _draft("none"),
        )
        self.assertEqual(final.mode, "clarify")

    def test_probe_no_hit_clear_general_goes_chat(self):
        router = Router()
        decision = router.route("解释一下量子纠缠", _draft("none"))
        final = router.resolve_after_probe(
            ProbeOutcome("no_household_match", {}, {}, "no hit"),
            "解释一下量子纠缠", decision, _draft("none"),
        )
        self.assertEqual(final.mode, "none")


if __name__ == "__main__":
    unittest.main()
