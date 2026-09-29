import unittest

from backend.agent_runtime.planner_contracts import parse_planner_action


class PlannerContractTests(unittest.TestCase):
    def test_parses_declaration_with_open_evidence_requirements(self):
        action = parse_planner_action({
            "action": "declare",
            "declaration": {
                "goal": "identify the text on a selected sign",
                "scope_id": "album1",
                "requirements": [
                    {"id": "sign_text", "evidence_type": "visible_text"},
                ],
            },
        })

        self.assertEqual(action.kind, "declare")
        self.assertEqual(action.declaration.requirements[0].evidence_type, "visible_text")

    def test_rejects_unknown_action_and_untyped_requirement(self):
        with self.assertRaisesRegex(ValueError, "unsupported planner action"):
            parse_planner_action({"action": "browse_the_web"})

        with self.assertRaisesRegex(ValueError, "unsupported evidence type"):
            parse_planner_action({
                "action": "declare",
                "declaration": {
                    "goal": "guess intent",
                    "scope_id": "album1",
                    "requirements": [{"id": "intent", "evidence_type": "question_kind"}],
                },
            })

    def test_clarify_only_accepts_known_missing_requirement_ids(self):
        action = parse_planner_action({
            "action": "clarify",
            "missing_requirement_ids": ["event"],
            "candidate_refs": ["photo_1", "photo_2"],
        }, known_requirement_ids={"event"})

        self.assertEqual(action.missing_requirement_ids, ("event",))
        self.assertEqual(action.candidate_refs, ("photo_1", "photo_2"))

        with self.assertRaisesRegex(ValueError, "unknown requirement"):
            parse_planner_action({
                "action": "clarify",
                "missing_requirement_ids": ["invented"],
            }, known_requirement_ids={"event"})


if __name__ == "__main__":
    unittest.main()
