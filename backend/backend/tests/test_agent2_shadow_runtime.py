import unittest
from types import SimpleNamespace
from unittest.mock import patch

from backend.agent_runtime.evidence_ledger import EvidenceLedger
from backend.agent_runtime.runtime import AgentRuntime, _prompt_annotations, record_agent2_tool_evidence
from backend.agent_runtime.task_state import EvidenceRequirement, TaskDeclaration, TaskState
from backend.agent_runtime.tool_registry import ToolSpec


class Agent2ShadowRuntimeTests(unittest.TestCase):
    def test_recovery_prompt_is_marked_as_internal_trace_message(self):
        annotations = _prompt_annotations([
            {"role": "system", "content": "system"},
            {"role": "user", "content": "用户原话"},
            {"role": "user", "content": "请先调用 search_memories 检索相关照片。"},
        ])
        self.assertEqual(len(annotations), 1)
        self.assertEqual(annotations[0]["message_origin"], "system_recovery")

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

    def test_empty_tool_observation_records_failure_without_satisfying_requirement(self):
        task_state = TaskState.from_declaration(TaskDeclaration(
            goal="read text", scope_id="album1",
            requirements=(EvidenceRequirement(id="text", evidence_type="visible_text"),),
        ))
        ledger = EvidenceLedger(scope_id="album1")
        spec = SimpleNamespace(
            name="read_photo_text",
            produces_evidence=("visible_text",),
            can_satisfy=lambda evidence_type: evidence_type == "visible_text",
        )

        recorded = record_agent2_tool_evidence(
            task_state, ledger, spec, tool_call_id="ocr_1", observation={})

        self.assertTrue(recorded)
        self.assertEqual(task_state.requirement("text").status, "running")
        self.assertEqual(task_state.requirement("text").coverage_status, "failed")
        self.assertEqual(task_state.requirement("text").attempts[0]["outcome"], "failed")
        self.assertEqual(ledger.entries[0].failure_reason, "empty_tool_observation")

    def test_valid_evidence_with_no_matching_requirement_is_explicitly_unmatched(self):
        task_state = TaskState.from_declaration(TaskDeclaration(
            goal="identify person", scope_id="album1",
            requirements=(EvidenceRequirement(id="identity", evidence_type="confirmed_identity"),),
        ))
        ledger = EvidenceLedger(scope_id="album1")
        spec = SimpleNamespace(
            name="search_memories",
            produces_evidence=("memory_asset", "location_metadata"),
            can_satisfy=lambda evidence_type: evidence_type in {"memory_asset", "location_metadata"},
        )
        recorded = record_agent2_tool_evidence(
            task_state, ledger, spec, tool_call_id="search_1",
            observation={"total": 1, "preview": [{"handle": "photo_1", "place": "上海"}]},
        )
        self.assertTrue(recorded)
        self.assertTrue(ledger.entries)
        self.assertTrue(all(entry.requirement_refs == () for entry in ledger.entries))
        self.assertTrue(all(entry.unmatched_reason == "evidence_incompatible" for entry in ledger.entries))
        self.assertEqual(task_state.requirement("identity").status, "open")

    def test_failed_visual_resolution_retries_next_preview_handle(self):
        """A failed inspect must not close the task or keep retrying photo_1."""
        inspect_calls = []

        def search_executor(arguments, *, context=None):
            return {
                "result_set_id": "rs_test",
                "total": 2,
                "preview": [{"handle": "photo_1"}, {"handle": "photo_2"}],
                "can_inspect": True,
                "recommended_resolution": {"needed": True, "tool": "inspect_photo"},
            }

        def inspect_executor(arguments, *, context=None):
            handle = str(arguments.get("asset_handle"))
            inspect_calls.append(handle)
            if len(inspect_calls) == 1:
                return {"asset_handle": handle, "status": "partial",
                        "reason": "model_unavailable", "observation": ""}
            return {"asset_handle": handle, "status": "ok", "certainty": "supported",
                    "observation": "蓝色灯光", "confirms_visual_only": True}

        specs = {
            "search_memories": ToolSpec(
                name="search_memories", description="search", input_schema={},
                executor=search_executor, produces_evidence=("memory_asset",)),
            "inspect_photo": ToolSpec(
                name="inspect_photo", description="inspect", input_schema={},
                executor=inspect_executor, cost_class="expensive",
                produces_evidence=("visual_observation",)),
        }

        class ScriptedChat:
            def __init__(self):
                self.agent_responses = iter((
                    '{"action":"tool_call","tool":"search_memories","arguments":{"query":"灯光"}}',
                    '{"action":"tool_call","tool":"inspect_photo","arguments":{"asset_handle":"photo_1"}}',
                    '{"action":"final","answer":"我看到了相关照片","evidence_refs":[]}',
                    '{"action":"final","answer":"我看到了相关照片","evidence_refs":[]}',
                    '{"action":"final","answer":"照片中有蓝色灯光","evidence_refs":[]}',
                    '{"action":"final","answer":"照片中有蓝色灯光","evidence_refs":[]}',
                    '{"action":"final","answer":"照片中有蓝色灯光","evidence_refs":[]}',
                ))

            def __call__(self, messages, **kwargs):
                if kwargs.get("call_type") == "planner":
                    return ('{"action":"declare","declaration":{"goal":"看照片细节",'
                            '"scope_id":"album1","requirements":[{"id":"visual",'
                            '"evidence_type":"visual_observation","description":"颜色"}]}}')
                if kwargs.get("call_type") == "agent":
                    return next(self.agent_responses)
                return '{"faithful":true,"problems":[],"reason":"ok"}'

        runtime = AgentRuntime(
            chat_fn=ScriptedChat(), profile_name="goal_driven_shadow", scope_id="album1")
        with patch("backend.agent_runtime.runtime.get_tool", side_effect=specs.get):
            turn = runtime.run("哪张照片里有蓝色灯光？")

        self.assertEqual(turn.status, "complete")
        self.assertEqual(inspect_calls, ["photo_1", "photo_2"])
        auto_steps = [step for step in turn.steps
                      if step.get("type") == "tool" and step.get("auto_resolution")]
        self.assertEqual(len(auto_steps), 1)
        self.assertEqual(auto_steps[0]["arguments"]["asset_handle"], "photo_2")


if __name__ == "__main__":
    unittest.main()
