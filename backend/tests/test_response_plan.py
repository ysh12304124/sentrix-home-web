"""RX-2 ResponsePlan contract tests."""

import unittest

from backend.answer_brief import AnswerBrief, Fact, Presentation, VisibleAsset
from backend.response_plan import plan_response


def _brief(mode, visible=0, facts=0):
    return AnswerBrief(
        "b1", "find_and_explain_images", mode,
        visible_assets=[VisibleAsset(f"asset-{i + 1}", f"照片{i + 1}")
                        for i in range(visible)],
        facts=[Fact(f"f{i + 1}", "t", "confirmed") for i in range(facts)],
        presentation=Presentation(show_images=visible > 0),
    )


class ResponsePlanTests(unittest.TestCase):
    def test_chat_hides_everything(self):
        plan = plan_response(_brief("chat"))
        self.assertEqual(plan.image_count, 0)
        self.assertEqual(plan.evidence_entry, "hidden")
        self.assertFalse(plan.include_uncertainty)

    def test_asset_delivery_short_image_first(self):
        plan = plan_response(_brief("asset_delivery", visible=4))
        self.assertEqual(plan.max_paragraphs, 1)
        self.assertEqual(plan.image_count, 4)
        self.assertFalse(plan.include_uncertainty)

    def test_exact_caps_images_at_three(self):
        plan = plan_response(_brief("exact_result", visible=5))
        self.assertEqual(plan.image_count, 3)

    def test_approximate_caps_at_three(self):
        plan = plan_response(_brief("approximate_result", visible=5))
        self.assertEqual(plan.image_count, 3)
        self.assertTrue(plan.include_uncertainty)

    def test_no_result_zero_images(self):
        plan = plan_response(_brief("no_result"))
        self.assertEqual(plan.image_count, 0)

    def test_person_summary_with_facts(self):
        plan = plan_response(_brief("person_summary", facts=2))
        self.assertTrue(plan.include_uncertainty)
        self.assertEqual(plan.evidence_entry, "collapsed")

    def test_person_summary_no_facts_expands_gap(self):
        plan = plan_response(_brief("person_summary", facts=0))
        self.assertEqual(plan.evidence_entry, "expanded")

    def test_clarify_hides_evidence(self):
        plan = plan_response(_brief("clarify"))
        self.assertEqual(plan.evidence_entry, "hidden")
        self.assertEqual(plan.image_count, 0)


if __name__ == "__main__":
    unittest.main()
