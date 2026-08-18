import unittest

from backend.agent_runtime.runtime import AgentRuntime


class Agent2ShadowRuntimeTests(unittest.TestCase):
    def test_shadow_profile_records_declaration_without_changing_legacy_answer_path(self):
        responses = iter((
            '''{"action":"declare","declaration":{"goal":"greet the user",
            "scope_id":"album1","requirements":[{"id":"statement",
            "evidence_type":"user_statement"}]}}''',
            '{"action":"final","answer":"你好"}',
        ))
        runtime = AgentRuntime(
            chat_fn=lambda messages: next(responses),
            profile_name="goal_driven_shadow",
            scope_id="album1",
        )

        turn = runtime.run("你好")

        self.assertEqual(turn.final_answer, "你好。")
        self.assertEqual(turn.profile, "goal_driven_shadow")
        self.assertEqual(turn.agent2_trace["task_declaration"]["goal"], "greet the user")
        self.assertEqual(turn.agent2_trace["terminal_reason"], "shadow_only")
        self.assertEqual(turn.agent2_trace["planner_decisions"][0]["status"], "accepted")

    def test_shadow_profile_records_fallback_without_blocking_legacy_loop(self):
        responses = iter(("not-json", '{"action":"final","answer":"你好"}'))
        runtime = AgentRuntime(
            chat_fn=lambda messages: next(responses),
            profile_name="goal_driven_shadow",
            scope_id="album1",
        )

        turn = runtime.run("你好")

        self.assertEqual(turn.final_answer, "你好。")
        self.assertEqual(turn.agent2_trace["planner_decisions"][0]["status"], "fallback")
        self.assertEqual(turn.agent2_trace["planner_decisions"][0]["reason"], "invalid_planner_action")


if __name__ == "__main__":
    unittest.main()
