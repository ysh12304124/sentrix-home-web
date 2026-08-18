import unittest

from backend.agent_runtime.disagreement import audit_counterfactual_turn


class DisagreementAuditTests(unittest.TestCase):
    def test_no_disagreement_when_actions_match(self):
        item = audit_counterfactual_turn(
            case_id="case-1",
            turn_id=1,
            legacy_action="search_memories",
            planner_action="search_memories",
        )
        self.assertIsNone(item)

    def test_classifies_tool_divergence(self):
        item = audit_counterfactual_turn(
            case_id="case-1",
            turn_id=1,
            legacy_action="inspect_photo",
            planner_action="read_photo_text",
        )
        self.assertIsNotNone(item)
        self.assertEqual(item.disagreement_kind, "tool_divergence")

    def test_classifies_premature_final(self):
        item = audit_counterfactual_turn(
            case_id="case-2",
            turn_id=2,
            legacy_action="search_memories",
            planner_action="final",
        )
        self.assertIsNotNone(item)
        self.assertEqual(item.disagreement_kind, "premature_final")

    def test_classifies_over_planning(self):
        item = audit_counterfactual_turn(
            case_id="case-3",
            turn_id=3,
            legacy_action="final",
            planner_action="inspect_photo",
        )
        self.assertIsNotNone(item)
        self.assertEqual(item.disagreement_kind, "over_planning")


if __name__ == "__main__":
    unittest.main()
