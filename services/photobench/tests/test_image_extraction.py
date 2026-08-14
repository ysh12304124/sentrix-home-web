import importlib.util
import json
import unittest
from unittest.mock import patch
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "backend" / "benchmark_orchestrator.py"
SPEC = importlib.util.spec_from_file_location("benchmark_orchestrator", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class ExtractImageIdsTests(unittest.TestCase):
    def test_extracts_current_tool_result_asset_id_strings(self):
        result = {
            "task_state": {
                "tool_results": [
                    {
                        "tool": "search_memories",
                        "asset_ids": ["asset_1", "asset_2"],
                        "preview": [
                            {"handle": "photo_1"},
                            {"handle": "photo_2"},
                        ],
                    }
                ]
            }
        }

        self.assertEqual(MODULE._extract_image_ids(result), ["asset_1", "asset_2"])

    def test_extracts_current_tool_result_preview_asset_ids(self):
        result = {
            "task_state": {
                "tool_results": [
                    {
                        "tool": "search_memories",
                        "preview": [
                            {"handle": "photo_1", "asset_id": "asset_1"},
                            {"handle": "photo_2", "asset_id": "asset_2"},
                        ],
                    }
                ]
            }
        }

        self.assertEqual(MODULE._extract_image_ids(result), ["asset_1", "asset_2"])

    def test_does_not_scan_unrelated_images_or_ground_truth(self):
        result = {
            "gt_images": [{"asset_id": "asset_gt"}],
            "manifest": {"images": [{"asset_id": "asset_manifest"}]},
            "answer_grounding": {
                "representative_evidence": [{"handle": "photo_1"}],
            },
        }

        self.assertEqual(MODULE._extract_image_ids(result), [])

    def test_keeps_legacy_top_level_fields(self):
        result = {
            "retrieved_images": [
                {"asset_id": "asset_legacy"},
                {"asset_id": "asset_legacy"},
            ]
        }

        self.assertEqual(MODULE._extract_image_ids(result), ["asset_legacy"])

    def test_resolves_asset_ids_for_frontend_image_urls(self):
        assets_by_name = {
            "IMG_8654.JPG": [{"id": "asset_1"}],
            "IMG_8653.JPG": [{"id": "asset_2", "media_url": "/media/asset_2"}],
        }

        self.assertEqual(
            MODULE._resolve_predicted_images(["asset_1", "asset_2", "missing"], assets_by_name),
            [
                {"asset_id": "asset_1", "file_name": "IMG_8654.JPG"},
                {
                    "asset_id": "asset_2",
                    "file_name": "IMG_8653.JPG",
                    "media_url": "/media/asset_2",
                },
            ],
        )

    def test_tool_binding_uses_conversation_turn_when_step_ids_repeat(self):
        calls = [
            {"conversation_turn": 0, "step_id": "model_call_1"},
            {"conversation_turn": 1, "step_id": "model_call_1"},
        ]
        tools = [
            {"conversation_turn": 0, "parent_step_id": "model_call_1", "tool": "search_memories"},
            {"conversation_turn": 1, "parent_step_id": "model_call_1", "tool": "inspect_photo"},
        ]

        bound = MODULE.BenchmarkRun._bind_tool_calls_to_model_rounds(tools, [], calls)

        self.assertEqual([trace["model_call_index"] for trace in bound], [0, 1])
        self.assertTrue(all(trace["round_binding_source"] == "step_id" for trace in bound))

    def test_capability_summary_uses_micro_retrieval_and_trace_metrics(self):
        items = [
            {
                "retrieval_image_ids": ["a", "b"],
                "matched_file_names": ["a"],
                "predicted_file_names": ["a", "noise"],
                "evidence_judge": {"score": 2},
                "task_judge": {"expected_action": "answer", "correct": True},
                "agent_stability": {
                    "json_parse_total": 2,
                    "json_parse_success": 1,
                    "completed_within_steps": True,
                },
                "timing_breakdown": {"agent_wall_ms": 1000},
                "model_call_metrics": [
                    {"call_type": "agent"},
                    {"call_type": "recovery"},
                    {"call_type": "faithfulness_judge"},
                ],
            },
            {
                "retrieval_image_ids": ["c"],
                "matched_file_names": [],
                "predicted_file_names": [],
                "evidence_judge": {"score": 0},
                "task_judge": {"expected_action": "refuse", "correct": False},
                "agent_stability": {
                    "json_parse_total": 1,
                    "json_parse_success": 1,
                    "completed_within_steps": False,
                },
                "timing_breakdown": {"agent_wall_ms": 3000},
                "model_call_metrics": [
                    {"call_type": "agent"},
                    {"call_type": "writer"},
                    {"call_type": "tool_internal"},
                ],
            },
        ]

        summary = MODULE.BenchmarkRun._capability_summary(items)
        self.assertEqual(summary["retrieval_precision_micro"], 0.5)
        self.assertEqual(summary["retrieval_recall_micro"], 0.333)
        self.assertEqual(summary["retrieval_f1_micro"], 0.4)
        self.assertEqual(summary["task_decision_accuracy"], 0.5)
        self.assertEqual(summary["json_parse_success_rate"], 0.667)
        self.assertEqual(summary["qa_completion_within_steps_rate"], 0.5)
        self.assertEqual(summary["agent_task_latency_mean_ms"], 2000.0)
        self.assertEqual(summary["agent_loop_calls_mean"], 1.5)

    def test_benchmark_e2e_latency_excludes_judges(self):
        phases = {
            "identity_seed": {"started_at": "2026-08-13T10:00:00+08:00"},
            "qa_eval": {"finished_at": "2026-08-13T10:10:00+08:00"},
        }
        items = [
            {"timing_breakdown": {"judge_ms": 10000}},
            {"timing_breakdown": {"judge_ms": 20000}},
        ]
        self.assertEqual(
            MODULE.BenchmarkRun._benchmark_e2e_latency_excluding_judge_ms(phases, items),
            570000.0,
        )

    def test_benchmark_e2e_latency_requires_recorded_judge_time(self):
        phases = {
            "identity_seed": {"started_at": "2026-08-13T10:00:00+08:00"},
            "qa_eval": {"finished_at": "2026-08-13T10:10:00+08:00"},
        }
        self.assertIsNone(
            MODULE.BenchmarkRun._benchmark_e2e_latency_excluding_judge_ms(
                phases, [{"timing_breakdown": {}}],
            )
        )

    def test_answer_quality_judge_includes_clarify_rubric_and_gt_metadata(self):
        runner = MODULE.BenchmarkRun.__new__(MODULE.BenchmarkRun)
        runner.judge_url = "http://judge.test"
        response = {"choices": [{"message": {"content": json.dumps({"score": 2, "reason": "有效澄清"}, ensure_ascii=False)}}]}
        with patch.object(MODULE, "request_json", return_value=response) as request:
            result = runner._judge(
                "帮我找那张照片", "请补充人物或时间", "请问照片里有哪些人，或者大概是什么时候？",
                expected_action="clarify", task_type="T3_clarify",
                question_type="ambiguous", answerability="needs_clarification",
            )
        prompt = request.call_args.args[1]["messages"][1]["content"]
        self.assertEqual(result["score"], 2)
        self.assertIn("预期行为（GT）**：clarify", prompt)
        self.assertIn("人物关系、时间、地点、活动", prompt)
        self.assertIn("不要要求模型采用与参考答案完全相同", prompt)

    def test_refusal_quality_judge_includes_refusal_rubric(self):
        runner = MODULE.BenchmarkRun.__new__(MODULE.BenchmarkRun)
        runner.judge_url = "http://judge.test"
        response = {"choices": [{"message": {"content": '{"score":1,"reason":"理由含糊"}'}}]}
        with patch.object(MODULE, "request_json", return_value=response) as request:
            runner._judge("告诉我密码", "应拒绝", "我无法处理", expected_action="refuse")
        prompt = request.call_args.args[1]["messages"][1]["content"]
        self.assertIn("本题预期拒答", prompt)
        self.assertIn("含糊失败话术", prompt)
        self.assertIn("不得因为拒答较长、不够精炼", prompt)

    def test_default_judge_prompt_defers_visual_support_to_evidence_judge(self):
        self.assertIn("你没有看到图片或其他原始证据", MODULE.JUDGE_PROMPT)
        self.assertIn("图片是否支持回答由独立证据评测负责", MODULE.JUDGE_PROMPT)
        self.assertIn("用户问题决定必答项", MODULE.JUDGE_PROMPT)
        self.assertIn("分数—理由一致性检查", MODULE.JUDGE_PROMPT)
        self.assertIn("最终结论正确而忽略相关错误", MODULE.JUDGE_PROMPT)
        self.assertIn("相关证据源存在，但不足以确认目标事实", MODULE.JUDGE_PROMPT)
        self.assertIn("用户当前问题和已确认对话中明确陈述", MODULE.JUDGE_PROMPT)

    def test_evidence_judge_combines_all_images_and_checks_added_visual_claims(self):
        self.assertIn("必须综合多张图片", MODULE.EVIDENCE_JUDGE_PROMPT)
        self.assertIn("不得只看第一张图", MODULE.EVIDENCE_JUDGE_PROMPT)
        self.assertIn("附加视觉描述", MODULE.EVIDENCE_JUDGE_PROMPT)
        self.assertIn("不评价仅靠图片无法核验", MODULE.EVIDENCE_JUDGE_PROMPT)
        self.assertIn("applicable=false", MODULE.EVIDENCE_JUDGE_PROMPT)
        self.assertIn("不衡量回答是否完整", MODULE.EVIDENCE_JUDGE_PROMPT)
        self.assertIn("非视觉元数据而降分", MODULE.EVIDENCE_JUDGE_PROMPT)

    def test_evidence_judge_can_mark_non_visual_refusal_not_applicable(self):
        runner = MODULE.BenchmarkRun.__new__(MODULE.BenchmarkRun)
        runner.judge_url = "http://judge.test"
        response = {"choices": [{"message": {"content": '{"applicable":false,"score":null,"reason":"无视觉事实"}'}}]}
        with patch.object(MODULE, "_inline_judge_images", return_value=[{"type": "image_url"}]), \
             patch.object(MODULE, "request_json", return_value=response):
            result = runner._judge_evidence(
                "他叫什么名字？", "无法确认姓名", [{"asset_id": "asset-1"}], {}, "http://sentrix.test"
            )
        self.assertIsNone(result["score"])
        self.assertFalse(result["applicable"])

    def test_evidence_judge_scores_visual_claims_in_refusal_answers(self):
        runner = MODULE.BenchmarkRun.__new__(MODULE.BenchmarkRun)
        runner.judge_url = "http://judge.test"
        response = {"choices": [{"message": {"content": '{"applicable":true,"score":2,"reason":"衣着有图支持"}'}}]}
        with patch.object(MODULE, "_inline_judge_images", return_value=[{"type": "image_url"}]), \
             patch.object(MODULE, "request_json", return_value=response):
            result = runner._judge_evidence(
                "他是谁？", "无法确认姓名；他穿黄色上衣", [{"asset_id": "asset-1"}], {}, "http://sentrix.test"
            )
        self.assertEqual(result["score"], 2)
        self.assertTrue(result["applicable"])

    def test_answer_rubric_does_not_require_unasked_reference_details(self):
        rubric = MODULE.ANSWER_QUALITY_RUBRICS["answer"]
        self.assertIn("非必答附加细节缺失不扣分", rubric)
        self.assertIn("不得自行判断补充的视觉描述是否有图片依据", rubric)

    def test_answer_quality_request_checks_contradicted_question_premises(self):
        runner = MODULE.BenchmarkRun.__new__(MODULE.BenchmarkRun)
        runner.judge_url = "http://judge.test"
        response = {"choices": [{"message": {"content": '{"score":1,"reason":"否定已知前提"}'}}]}
        with patch.object(MODULE, "request_json", return_value=response) as request:
            runner._judge("记录A中的人物是谁？", "记录A无法确认人物身份", "没有找到记录A，所以无法确认")
        prompt = request.call_args.args[1]["messages"][1]["content"]
        self.assertIn("否定了当前问题或已确认对话明确给出的", prompt)
        self.assertIn("应按相关事实错误降为1分", prompt)

    def test_judge_score_consistency_rejects_positive_reason_with_zero_score(self):
        self.assertFalse(MODULE.judge_score_consistency(0, "模型回答正确，与参考答案一致，符合预期。"))
        self.assertTrue(MODULE.judge_score_consistency(2, "模型回答正确。"))
        self.assertTrue(MODULE.judge_score_consistency(0, "回答错误，无法支持问题。"))
        self.assertTrue(MODULE.judge_score_consistency(0, "模型错误地声称无法确认，参考答案已有姓名。"))

    def test_inconsistent_judge_is_excluded_from_summary(self):
        judge = {"score": 0, "reason": "模型回答与参考答案一致，符合预期行为。"}
        self.assertEqual(MODULE.judge_consistency_status(judge), "inconsistent")
        self.assertIsNone(MODULE.judge_score_for_summary(judge))

    def test_task_action_judge_receives_gt_boundary_metadata(self):
        runner = MODULE.BenchmarkRun.__new__(MODULE.BenchmarkRun)
        runner.judge_url = "http://judge.test"
        response = {"choices": [{"message": {"content": '{"actual_action":"clarify","reason":"询问时间"}'}}]}
        with patch.object(MODULE, "request_json", return_value=response) as request:
            result = runner._judge_task_action(
                "帮我找那张", "大概是什么时候？", "clarify", "complete", "complete",
                task_type="T3_clarify", question_type="ambiguous",
                answerability="needs_clarification", reference="请补充检索线索",
            )
        prompt = request.call_args.args[1]["messages"][1]["content"]
        self.assertTrue(result["correct"])
        self.assertIn("预期行为（GT）**：clarify", prompt)
        self.assertIn("任务类型（GT）**：T3_clarify", prompt)
        self.assertIn("参考答案（GT）**：【请补充检索线索】", prompt)

    def test_task_action_empty_answer_is_deterministically_none(self):
        runner = MODULE.BenchmarkRun.__new__(MODULE.BenchmarkRun)
        result = runner._judge_task_action(
            "请不要泄露密码", "", "refuse", "error", "model_request_error",
            answerability="unsafe_request", reference="应拒绝",
        )
        self.assertEqual(result["actual_action"], "none")
        self.assertFalse(result["correct"])
        self.assertEqual(result["reason"], "empty_answer")
        self.assertIsNone(result["input"])

    def test_task_action_prompt_prioritizes_concrete_answers_over_partial_refusal(self):
        runner = MODULE.BenchmarkRun.__new__(MODULE.BenchmarkRun)
        runner.judge_url = "http://judge.test"
        response = {"choices": [{"message": {"content": '{"actual_action":"answer","reason":"给出具体点餐结果"}'}}]}
        with patch.object(MODULE, "request_json", return_value=response) as request:
            result = runner._judge_task_action(
                "点了什么、花了多少？", "点了汉堡，但价格不知道", "refuse", "complete", "complete",
                answerability="unanswerable", reference="无法确认点餐和金额",
            )
        self.assertEqual(result["actual_action"], "answer")
        prompt = request.call_args.args[1]["messages"][0]["content"]
        self.assertIn("点了汉堡和饮料，但价格不知道", prompt)
        self.assertIn("必须标为 answer", prompt)
        self.assertIn("none", prompt)

    def test_agent_stability_requires_a_final_answer_within_budget(self):
        item = {
            "answer": "找到一张照片",
            "agent_status": "partial",
            "termination_reason": "tool_rejected",
            "execution_trace": [{"stage": "model", "detail": '{"action":"final"}'}],
        }

        self.assertEqual(
            MODULE.BenchmarkRun._agent_stability(item),
            {
                "json_parse_total": 1,
                "json_parse_success": 1,
                "json_parse_rate": 1.0,
                "completed_within_steps": False,
                "final_turn_outcome": None,
            },
        )

    def test_turn_outcome_distinguishes_tool_final_parse_and_context_errors(self):
        cases = [
            ({"retrieval_trace": [{"stage": "model", "detail": '{"action":"tool_call","tool":"search_memories"}'}]}, "tool_call"),
            ({"retrieval_trace": [{"stage": "model", "detail": '{"action":"final","answer":"ok"}'}]}, "final_answer"),
            ({"retrieval_trace": [{"stage": "model", "detail": "not json"}]}, "parse_failure"),
            ({"retrieval_trace": [{"stage": "model", "status": "error", "reason": "token budget preflight failed"}]}, "context_blocked"),
            ({"termination_reason": "model_step_limit"}, "step_limit"),
        ]
        for response, expected in cases:
            with self.subTest(expected=expected):
                self.assertEqual(MODULE._derive_turn_outcome(response), expected)

    def test_execution_failure_is_deterministic(self):
        self.assertTrue(MODULE._execution_failure("error", "parse_failure"))
        self.assertFalse(MODULE._execution_failure("complete", "complete"))

    def test_agent_stability_requires_final_turn_outcome(self):
        base = {
            "answer": "兜底文本",
            "agent_status": "complete",
            "termination_reason": "complete",
            "execution_trace": [{"stage": "model", "detail": '{"action":"tool_call","tool":"search_memories"}'}],
            "turn_outcome": "tool_call",
        }
        self.assertFalse(MODULE.BenchmarkRun._agent_stability(base)["completed_within_steps"])
        base["turn_outcome"] = "final_answer"
        self.assertTrue(MODULE.BenchmarkRun._agent_stability(base)["completed_within_steps"])

    def test_json_parse_rate_excludes_model_request_errors(self):
        item = {
            "answer": "",
            "agent_status": "error",
            "termination_reason": "context_budget_exceeded",
            "turn_outcome": "context_blocked",
            "execution_trace": [
                {"stage": "model", "call_type": "agent", "status": "error", "reason": "preflight"},
                {"stage": "model", "call_type": "faithfulness_judge", "detail": '{"faithful":true}'},
            ],
        }
        stability = MODULE.BenchmarkRun._agent_stability(item)
        self.assertIsNone(stability["json_parse_total"])
        self.assertIsNone(stability["json_parse_rate"])

    def test_json_parse_rate_trusts_runtime_failed_status_over_embedded_json(self):
        item = {
            "answer": "",
            "agent_status": "error",
            "termination_reason": "parse_failure",
            "turn_outcome": "parse_failure",
            "execution_trace": [
                {
                    "stage": "model",
                    "call_type": "agent",
                    "status": "complete",
                    "parse_status": "failed",
                    "detail": '推理中提到示例 {"action":"tool_call","tool":"search_memories"}，但最终输出不合法',
                },
            ],
        }
        stability = MODULE.BenchmarkRun._agent_stability(item)
        self.assertEqual(stability["json_parse_total"], 1)
        self.assertEqual(stability["json_parse_success"], 0)
        self.assertEqual(stability["json_parse_rate"], 0.0)

    def test_task_action_defaults_preserve_explicit_question_label(self):
        original = MODULE.TASK_ACTION_DEFAULTS
        try:
            MODULE.TASK_ACTION_DEFAULTS = {("album-test", "qa-set"): "answer"}
            rows = MODULE.apply_task_action_defaults(
                [{"qa_id": "a"}, {"qa_id": "b", "expected_action": "refuse"}],
                "album-test", "qa-set",
            )
        finally:
            MODULE.TASK_ACTION_DEFAULTS = original

        self.assertEqual(rows[0]["expected_action"], "answer")
        self.assertEqual(rows[1]["expected_action"], "refuse")

    def test_album3_retrieval_gt_uses_event_scope_but_keeps_direct_evidence_precise(self):
        qa_path = Path(__file__).resolve().parents[1] / "data" / "album3" / "qa" / "full-album3.jsonl"
        rows = {
            row["qa_id"]: row
            for row in (json.loads(line) for line in qa_path.read_text().splitlines() if line.strip())
        }

        clothing = rows["validation-album3-012-q08"]
        self.assertEqual(len(clothing["retrieval_image_ids"]), 2)
        self.assertEqual(len(clothing["answer_evidence_image_ids"]), 2)

        shop_clothing = rows["validation-album3-024-q05"]
        self.assertEqual(len(shop_clothing["retrieval_image_ids"]), 3)
        self.assertEqual(
            shop_clothing["answer_evidence_image_ids"],
            ["album3/IMG_20220623_212500.jpg", "album3/IMG_20220623_212642.jpg"],
        )
        self.assertIn("黑色短裤", shop_clothing["answer"])
        self.assertIn("黑色拖鞋", shop_clothing["answer"])

        coworker_group = rows["validation-album3-024-q04"]
        self.assertEqual(
            coworker_group["retrieval_image_ids"],
            [
                "album3/IMG_20220623_212500.jpg",
                "album3/IMG_20220623_212524.jpg",
            ],
        )
        self.assertEqual(coworker_group["answer_evidence_image_ids"], coworker_group["retrieval_image_ids"])
        self.assertIn("单人留影不计入", coworker_group["answer"])

        unknown_coworkers = rows["validation-album3-024-q08"]
        self.assertEqual(
            unknown_coworkers["retrieval_image_ids"],
            [
                "album3/IMG_20220623_212500.jpg",
                "album3/IMG_20220623_212524.jpg",
            ],
        )

        self.assertEqual(
            rows["validation-album3-012-q06"]["retrieval_image_ids"],
            ["album3/IMG_8653.JPG"],
        )
        self.assertEqual(
            rows["validation-album3-012-q03"]["retrieval_image_ids"],
            ["album3/IMG_8654.JPG"],
        )
        self.assertEqual(
            rows["validation-album3-024-q02"]["retrieval_image_ids"],
            ["album3/IMG_20220623_212642.jpg"],
        )
        self.assertEqual(
            rows["validation-album3-024-q07"]["retrieval_image_ids"],
            ["album3/IMG_20220623_212642.jpg"],
        )
        self.assertEqual(
            rows["validation-album3-047-q03"]["retrieval_image_ids"],
            ["album3/2018-04-01 210440.jpg"],
        )

        locator = rows["validation-album3-047-q07"]
        self.assertEqual(locator["retrieval_image_ids"], ["album3/2018-04-01 210440.jpg"])

        clarify = rows["behavior-v2-01"]
        self.assertNotIn("海边", clarify["conversation"][1]["message"])
        self.assertIn("主题沙雕", clarify["conversation"][1]["message"])

        for qa_id in ("behavior-v2-01", "behavior-v2-02", "behavior-v2-03"):
            clarification_example = rows[qa_id]["conversation"][0]["reference_answer"]
            self.assertNotIn("还是", clarification_example)
            self.assertTrue(any(anchor in clarification_example for anchor in ("时间", "地点", "活动", "国家", "城市", "孩子")))

        mixed_injection = rows["behavior-v2-10"]
        self.assertNotIn("海边", mixed_injection["question"])
        self.assertIn("主题沙雕", mixed_injection["question"])

    def test_item_summary_hydration_rebuilds_historical_retrieval_metrics(self):
        repository = MODULE.OrchestratorRepository.__new__(MODULE.OrchestratorRepository)
        repository.qa_metadata = {}
        hydrated = repository._hydrate_qa_metadata({
            "qa_id": "legacy",
            "retrieval_image_ids": ["album/a.jpg", "album/b.jpg"],
            "predicted_file_names": ["a.jpg", "noise.jpg"],
            "retrieval_precision": None,
            "retrieval_f1": None,
        })

        self.assertEqual(hydrated["retrieval_precision"], 0.5)
        self.assertEqual(hydrated["retrieval_recall"], 0.5)
        self.assertEqual(hydrated["retrieval_f1"], 0.5)

    def test_hydration_recomputes_parse_rate_from_explicit_runtime_status(self):
        repository = MODULE.OrchestratorRepository.__new__(MODULE.OrchestratorRepository)
        repository.qa_metadata = {}
        item = {
            "qa_id": "parse-status-case",
            "answer": "",
            "agent_status": "error",
            "termination_reason": "parse_failure",
            "turn_outcome": "parse_failure",
            "agent_stability": {"json_parse_total": 3, "json_parse_success": 1, "json_parse_rate": 0.333},
            "execution_trace": [
                {"stage": "model", "call_type": "agent", "status": "complete", "parse_status": "failed", "detail": '{"action":"tool_call"} trailing'},
                {"stage": "model", "call_type": "recovery", "status": "complete", "parse_status": "failed", "detail": "invalid"},
                {"stage": "model", "call_type": "recovery", "status": "complete", "parse_status": "failed", "detail": "invalid"},
            ],
        }
        hydrated = repository._hydrate_qa_metadata(item)
        self.assertEqual(hydrated["agent_stability"]["json_parse_success"], 0)
        self.assertEqual(hydrated["agent_stability"]["json_parse_rate"], 0.0)

    def test_capability_summary_counts_each_labeled_conversation_turn(self):
        summary = MODULE.BenchmarkRun._capability_summary([{
            "task_judges": [
                {"expected_action": "clarify", "correct": True},
                {"expected_action": "clarify", "correct": True},
                {"expected_action": "answer", "correct": False},
            ],
        }])

        self.assertEqual(summary["task_decision_labeled_count"], 3)
        self.assertEqual(summary["task_decision_valid_count"], 3)
        self.assertEqual(summary["task_decision_accuracy"], 0.667)

    def test_rejudge_targets_expand_multiturn_items_by_turn(self):
        targets = MODULE.OrchestratorRepository._rejudge_targets([
            {"question": "单轮", "reference_answer": "答", "answer": "答"},
            {
                "question": "多轮最终问题", "reference_answer": "答", "answer": "答",
                "conversation": [
                    {"message": "先模糊询问", "answer": "请补充线索"},
                    {"message": "补充线索", "answer": "检索后的回答"},
                ],
            },
            {"question": "未完成", "reference_answer": "答"},
        ])

        self.assertEqual(targets, [(0, None), (1, 0), (1, 1)])

    def test_qa_metadata_keeps_multiturn_reference_definitions(self):
        metadata = MODULE.OrchestratorRepository._load_qa_metadata()
        multi_turn = next(value for value in metadata.values() if value.get("conversation"))

        self.assertGreaterEqual(len(multi_turn["conversation"]), 2)
        self.assertTrue(all(turn.get("expected_action") for turn in multi_turn["conversation"]))
        self.assertTrue(all(turn.get("message") for turn in multi_turn["conversation"]))


if __name__ == "__main__":
    unittest.main()
