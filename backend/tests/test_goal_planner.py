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

    def test_declaration_without_scope_id_is_accepted(self):
        """模型不再被要求输出 scope_id；缺省时由系统注入。"""
        planner = GoalPlanner(chat_fn=lambda messages: '''{
          "action":"declare",
          "declaration":{
            "goal":"读招牌",
            "requirements":[{"id":"text","evidence_type":"visible_text"}]
          }
        }''')

        result = planner.declare("读招牌", scope_id="album1")

        self.assertTrue(result.ok)
        self.assertEqual(result.declaration.scope_id, "album1")

    def test_prompt_never_contains_a_real_scope_id(self):
        """E1 回归守卫：示例里泄漏真实 scope_id 会让小模型照抄并抄错。"""
        prompts = []

        def chat(messages, **kwargs):
            prompts.append(messages)
            return ('{"action":"declare","declaration":{"goal":"找照片",'
                    '"requirements":[{"id":"m","evidence_type":"memory_asset"}]}}')

        GoalPlanner(chat_fn=chat).declare("找我拍的照片", scope_id="album_9f8e7d6c5b4a")

        system = prompts[0][0]["content"]
        self.assertNotIn("album_9f8e7d6c5b4a", system)
        self.assertIn("不要输出 scope_id", system)

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

    def test_strips_trailing_fence_without_leading_fence(self):
        """E3 回归守卫：只有结尾 ``` 时也要剥离，否则补 } 的修复路径会被跳过。"""
        planner = GoalPlanner(chat_fn=lambda messages: (
            '{"action":"declare","declaration":{"goal":"读招牌","scope_id":"album1",'
            '"requirements":[{"id":"text","evidence_type":"visible_text"}]}}\n```'
        ))
        self.assertTrue(planner.declare("读招牌", scope_id="album1").ok)

    def test_invalid_json_is_retried_with_the_failure_reason(self):
        """失败后必须把「上一轮原文 + 失败原因」回灌，而不是重发同样的提示词。"""
        calls = []

        def chat(messages, call_type=None, **kwargs):
            calls.append(messages)
            if len(calls) == 1:
                return "not-json"
            return ('{"action":"declare","declaration":{"goal":"读招牌","scope_id":"album1",'
                    '"requirements":[{"id":"text","evidence_type":"visible_text"}]}}')

        result = GoalPlanner(chat_fn=chat).declare("读招牌", scope_id="album1")

        self.assertTrue(result.ok)
        self.assertEqual(len(calls), 2)
        retry = calls[1]
        self.assertEqual(retry[-2]["role"], "assistant")
        self.assertEqual(retry[-2]["content"], "not-json")
        self.assertEqual(retry[-1]["role"], "user")
        self.assertIn("invalid_planner_action", retry[-1]["content"])

    def test_gives_up_after_max_plan_attempts(self):
        calls = []

        def chat(messages, **kwargs):
            calls.append(messages)
            return "still-not-json"

        result = GoalPlanner(chat_fn=chat).declare("读招牌", scope_id="album1")

        self.assertFalse(result.ok)
        self.assertEqual(result.fallback_reason, "invalid_planner_action")
        self.assertEqual(len(calls), 3)  # SENTRIX_PLANNER_MAX_ATTEMPTS 默认 3

    def test_history_is_merged_into_the_leading_system_message(self):
        """严格 chat template 不允许中途出现 system 消息。"""
        prompts = []

        def chat(messages, **kwargs):
            prompts.append(messages)
            return ('{"action":"declare","declaration":{"goal":"读招牌","scope_id":"album1",'
                    '"requirements":[{"id":"text","evidence_type":"visible_text"}]}}')

        GoalPlanner(chat_fn=chat).declare("读招牌", scope_id="album1", history="上一轮问了价格")

        roles = [item["role"] for item in prompts[0]]
        self.assertEqual(roles, ["system", "user"])
        self.assertIn("上一轮问了价格", prompts[0][0]["content"])

    def test_prompt_marks_single_album_counts_as_structured_facts(self):
        prompts = []

        def chat(messages, **kwargs):
            prompts.append(messages)
            return ('{"action":"declare","declaration":{"goal":"统计相册照片",'
                    '"scope_id":"album1","requirements":[{"id":"fact",'
                    '"evidence_type":"structured_fact"}]}}')

        self.assertTrue(GoalPlanner(chat_fn=chat).declare("相册有多少张照片", scope_id="album1").ok)
        self.assertIn("相册数量、拍摄时间、地点、媒体、已命名人物、处理状态：声明 structured_fact",
                      prompts[0][0]["content"])


if __name__ == "__main__":
    unittest.main()
