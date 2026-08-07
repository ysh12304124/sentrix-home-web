"""RX-1 AnswerBrief contract tests."""

import unittest

from backend.answer_brief import (
    AnswerBrief,
    Fact,
    VisibleAsset,
    build_answer_brief,
    build_facts_and_uncertainties,
    condition_aspects,
    derive_response_mode,
    user_goal,
)
from backend.evidence_retrieval import EvidencePacket
from backend.query_contracts import Constraint, HARD, SEMANTIC, QueryAction, QuerySpec


def _spec(answer_target="general", actions=None, return_original=False):
    return QuerySpec(
        query_id="q1", scope_mode="single", scope_ids=["home"], viewer_id="owner",
        conversation_id="c", intent="answer", answer_target=answer_target,
        constraints=[Constraint("time", "2025年10月", HARD, "asset_metadata")],
        actions=actions or [],
        result_requirement={"return_original_assets": return_original},
    )


def _asset(asset_id="asset-1", conds=None, level="exact", evidence_ids=None):
    return {
        "asset_id": asset_id, "file_name": "a.jpg", "media_type": "image",
        "observation_ids": ["obs-1"], "evidence_ids": evidence_ids or [asset_id],
        "condition_results": conds or {}, "level": level, "score": 0.9,
        "captured_at": "2025-10-02T10:00:00",
    }


def _packet(assets=None, exact=None, strong=None, approx=None, gaps=None):
    return EvidencePacket("q1", "home", "general",
                          assets=assets or [],
                          exact_results=exact or [],
                          strong_results=strong or [],
                          approximate_results=approx or [],
                          gaps=gaps or [])


class UserGoalTests(unittest.TestCase):
    def test_return_assets_is_deliver_images(self):
        spec = _spec(actions=[QueryAction(type="return_assets")])
        self.assertEqual(user_goal(spec), "deliver_images")

    def test_return_original_assets_flag(self):
        spec = _spec(return_original=True)
        self.assertEqual(user_goal(spec), "deliver_images")

    def test_person_target(self):
        self.assertEqual(user_goal(_spec(answer_target="person")), "person_summary")

    def test_default_find_and_explain(self):
        self.assertEqual(user_goal(_spec()), "find_and_explain_images")


class ResponseModeTests(unittest.TestCase):
    def test_exact(self):
        a = _asset(conds={"time:2025年10月": {"status": "matched"}})
        packet = _packet(assets=[a], exact=[a])
        self.assertEqual(derive_response_mode("find_and_explain_images", packet), "exact_result")

    def test_approximate(self):
        a = _asset(conds={"semantic:海豚": {"status": "unknown"}}, level="approximate")
        packet = _packet(assets=[a], approx=[a])
        self.assertEqual(derive_response_mode("find_and_explain_images", packet), "approximate_result")

    def test_no_result(self):
        self.assertEqual(derive_response_mode("find_and_explain_images", _packet()), "no_result")

    def test_asset_delivery_with_assets(self):
        a = _asset()
        self.assertEqual(derive_response_mode("deliver_images", _packet(assets=[a])), "asset_delivery")

    def test_asset_delivery_no_assets(self):
        self.assertEqual(derive_response_mode("deliver_images", _packet()), "no_result")

    def test_person_summary_always(self):
        self.assertEqual(derive_response_mode("person_summary", _packet()), "person_summary")


class FactUncertaintyTests(unittest.TestCase):
    def test_matched_becomes_confirmed_fact(self):
        a = _asset(conds={"time:2025年10月": {"status": "matched"},
                          "activity:爬山": {"status": "unknown"}},
                   evidence_ids=["asset-1", "obs-1"])
        facts, uncertainties = build_facts_and_uncertainties(_packet(assets=[a]))
        self.assertEqual(len(facts), 1)
        self.assertEqual(facts[0].certainty, "confirmed")
        self.assertIn("2025年10月", facts[0].text)
        self.assertEqual(facts[0].evidence_ids, ["asset-1", "obs-1"])
        self.assertTrue(any(u.topic == "爬山" for u in uncertainties))

    def test_possible_becomes_possible_fact(self):
        a = _asset(conds={"activity:爬山": {"status": "possible"}})
        facts, _ = build_facts_and_uncertainties(_packet(assets=[a]))
        self.assertEqual(facts[0].certainty, "possible")

    def test_no_evidence_no_facts(self):
        facts, uncertainties = build_facts_and_uncertainties(_packet(gaps=[
            {"condition": "place:贵阳", "reason": "no_direct_support"}]))
        self.assertEqual(facts, [])
        self.assertTrue(any(u.topic == "贵阳" for u in uncertainties))

    def test_gap_becomes_uncertainty(self):
        packet = _packet(gaps=[{"condition": "activity:爬山", "reason": "no_direct_support"}])
        facts, uncertainties = build_facts_and_uncertainties(packet)
        self.assertEqual(facts, [])
        self.assertTrue(any(u.topic == "爬山" and u.status == "unknown" for u in uncertainties))


class ConditionAspectTests(unittest.TestCase):
    def test_aspects_split(self):
        a = _asset(conds={"time:2025年10月": {"status": "matched"},
                          "activity:爬山": {"status": "possible"},
                          "semantic:合影": {"status": "unknown"}})
        supported, uncertain = condition_aspects(a)
        self.assertIn("2025年10月", supported)
        self.assertTrue(any("爬山" in item for item in uncertain))
        self.assertTrue(any("合影" in item for item in uncertain))


class BuildBriefTests(unittest.TestCase):
    def test_no_result_hides_images(self):
        packet = _packet(gaps=[{"condition": "place:贵阳夜晚步行街", "reason": "no_direct_support"}])
        brief = build_answer_brief("贵阳夜晚步行街", _spec(), packet)
        self.assertEqual(brief.response_mode, "no_result")
        self.assertEqual(brief.facts, [])
        self.assertFalse(brief.presentation.show_images)
        self.assertTrue(brief.presentation.show_evidence_entry)

    def test_asset_delivery_requires_visible(self):
        a = _asset(conds={"image": {"status": "matched"}})
        packet = _packet(assets=[a], exact=[a])
        visible = [VisibleAsset(asset_id="asset-1", display_handle="照片1",
                                supported_aspects=["image"])]
        brief = build_answer_brief("把原图给我", _spec(actions=[QueryAction(type="return_assets")]),
                                   packet, visible_assets=visible)
        self.assertEqual(brief.response_mode, "asset_delivery")
        self.assertTrue(brief.presentation.show_images)
        self.assertTrue(brief.presentation.auto_expand_images)
        self.assertEqual(brief.hidden_assets_count, 0)

    def test_asset_delivery_empty_visible_becomes_no_result(self):
        a = _asset()
        packet = _packet(assets=[a], exact=[a])
        brief = build_answer_brief("把原图给我", _spec(actions=[QueryAction(type="return_assets")]),
                                   packet, visible_assets=[])
        self.assertEqual(brief.response_mode, "no_result")
        self.assertFalse(brief.presentation.show_images)

    def test_person_gap_forbids_family_claims(self):
        packet = _packet()
        brief = build_answer_brief("介绍一下明哥", _spec(answer_target="person"), packet)
        self.assertEqual(brief.facts, [])
        self.assertIn("多次出现", brief.must_not_say)

    def test_writer_payload_has_no_internal_keys(self):
        a = _asset(conds={"time:2025年10月": {"status": "matched"},
                          "activity:爬山": {"status": "unknown"}})
        packet = _packet(assets=[a], exact=[a])
        visible = [VisibleAsset(asset_id="asset-1", display_handle="照片1",
                                supported_aspects=["2025年10月"], uncertain_aspects=["爬山"])]
        brief = build_answer_brief("去年十月爬山拍的合影", _spec(), packet, visible_assets=visible)
        payload = json_dumps(brief.writer_payload())
        self.assertNotIn("asset_id", payload)
        self.assertNotIn("condition", payload)
        self.assertNotIn("fusion_score", payload)
        self.assertIn("照片1", payload)


def json_dumps(value):
    import json
    return json.dumps(value, ensure_ascii=False)


if __name__ == "__main__":
    unittest.main()
