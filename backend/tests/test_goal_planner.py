import unittest

from backend.agent_runtime.goal_planner import GoalPlanner


class GoalPlannerTests(unittest.TestCase):
    def test_returns_typed_declaration_from_planner_response(self):
        planner = GoalPlanner(chat_fn=lambda messages: '''{
          "action":"declare",
          "declaration":{
            "goal":"read the sign next to the red shirt",
            "scope_id":"album1",
            "requirements":[
              {"id":"scene","evidence_type":"visual_observation"},
              {"id":"text","evidence_type":"visible_text"}
            ]
          }
        }''')

        result = planner.declare("What does the sign say?", scope_id="album1")

        self.assertTrue(result.ok)
        self.assertEqual(result.declaration.goal, "read the sign next to the red shirt")
        self.assertEqual(
            [item.evidence_type for item in result.declaration.requirements],
            ["visual_observation", "visible_text"],
        )

    def test_records_fallback_when_planner_response_is_invalid_or_cross_scope(self):
        invalid = GoalPlanner(chat_fn=lambda messages: "not json")
        invalid_result = invalid.declare("find it", scope_id="album1")
        self.assertFalse(invalid_result.ok)
        self.assertEqual(invalid_result.fallback_reason, "invalid_planner_action")

        cross_scope = GoalPlanner(chat_fn=lambda messages: '''{
          "action":"declare",
          "declaration":{
            "goal":"read text", "scope_id":"album2",
            "requirements":[{"id":"text","evidence_type":"visible_text"}]
          }
        }''')
        scope_result = cross_scope.declare("read this", scope_id="album1")
        self.assertFalse(scope_result.ok)
        self.assertEqual(scope_result.fallback_reason, "scope_mismatch")

    def test_recovers_only_a_missing_outer_object_brace(self):
        planner = GoalPlanner(chat_fn=lambda messages: (
            '{"action":"declare","declaration":{"goal":"读招牌","scope_id":"album1",'
            '"requirements":[{"id":"text","evidence_type":"visible_text"}]}'
        ))
        result = planner.declare("读招牌", scope_id="album1")
        self.assertTrue(result.ok)
        self.assertEqual(result.declaration.requirements[0].id, "text")

    def test_invalid_json_gets_one_format_rewrite_before_blocking(self):
        calls = []

        def chat(messages, call_type=None, **kwargs):
            calls.append(call_type)
            if len(calls) == 1:
                return "not-json"
            return ('{"action":"declare","declaration":{"goal":"读招牌","scope_id":"album1",'
                    '"requirements":[{"id":"text","evidence_type":"visible_text"}]}}')

        result = GoalPlanner(chat_fn=chat).declare("读招牌", scope_id="album1")
        self.assertTrue(result.ok)
        self.assertEqual(calls, ["planner", "planner_format_rewrite"])


if __name__ == "__main__":
    unittest.main()
