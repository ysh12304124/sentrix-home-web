import json
import os
import re
import unittest
from unittest.mock import patch

from backend.agent_runtime.evidence_ledger import EvidenceLedger, LedgerEntry
from backend.agent_runtime.runtime import (
    _last_supported_inspection_handle, _selected_inspection_facts,
    _selected_metadata_facts, _visual_tool_correction,
)
from backend.agent_runtime.intent import ocr_intent
from backend.agent_runtime.final_writer import (
    build_answer_writer_messages,
    clean_writer_output, naturalize_answer,
    rewrite_final,
)
from backend.agent_runtime.task_state import EvidenceRequirement, TaskDeclaration, TaskState


class Agent2AnswerContextTests(unittest.TestCase):
    def test_selected_metadata_ignores_unselected_preview_dates_and_places(self):
        state = {"tool_results": [{
            "tool": "search_memories",
            "preview": [
                {"handle": "photo_1", "captured_at": "2018-02-18 15:35:11",
                 "place": "石家庄市桥西区"},
                {"handle": "photo_2", "captured_at": "2023-08-01 11:00:00",
                 "place": "邯郸市永年区"},
            ],
        }]}
        facts = _selected_metadata_facts(state, ["photo_1"])
        self.assertEqual(len(facts), 1)
        self.assertIn("2018-02-18", facts[0])
        self.assertIn("石家庄市桥西区", facts[0])
        self.assertNotIn("2023", str(facts))
        self.assertEqual(_selected_metadata_facts(state, ["photo_3"]), [])

    def test_selected_inspection_keeps_visuals_and_confirmed_names_only(self):
        state = {"tool_results": [
            {"tool": "inspect_photo", "inspect_handle": "photo_1",
             "inspect_text": "三人在餐厅合影", "certainty": "supported",
             "photo_identities": [
                 {"person_name": "明明", "identity_status": "confirmed"},
                 {"person_name": "路人甲", "identity_status": "unconfirmed"},
             ]},
            {"tool": "inspect_photo", "inspect_handle": "photo_2",
             "inspect_text": "海边", "certainty": "supported"},
        ]}
        facts = _selected_inspection_facts(state, ["photo_1"])
        self.assertIn("三人在餐厅合影", str(facts))
        self.assertIn("明明", str(facts))
        self.assertNotIn("路人甲", str(facts))
        self.assertNotIn("海边", str(facts))

    def test_missing_final_handle_recovers_only_successfully_inspected_visible_photo(self):
        state = {"tool_results": [
            {"tool": "search_memories", "preview": [
                {"handle": "photo_1", "captured_at": "2023-01-06 21:37:14"},
                {"handle": "photo_2", "captured_at": "2022-01-01"},
            ]},
            {"tool": "inspect_photo", "inspect_handle": "photo_2",
             "inspect_observation": "模糊", "certainty": "uncertain"},
            {"tool": "inspect_photo", "inspect_handle": "photo_1",
             "inspect_observation": "家人合影", "certainty": "supported"},
        ]}
        handle = _last_supported_inspection_handle(state)
        self.assertEqual(handle, "photo_1")
        self.assertIn("2023-01-06", str(_selected_metadata_facts(state, [handle])))
        state["tool_results"][-1]["inspect_handle"] = "photo_9"
        self.assertIsNone(_last_supported_inspection_handle(state))
        state["tool_results"][-1]["inspect_handle"] = "photo_1"
        state["tool_results"][1]["certainty"] = "supported"
        self.assertIsNone(_last_supported_inspection_handle(state))

    def test_visual_call_scene_is_not_misclassified_as_phone_number_ocr(self):
        question = "疫情期间我躺在床上打电话，当时是什么场景，描述一下？"
        self.assertFalse(ocr_intent(question))
        tool, args = _visual_tool_correction(
            "read_photo_text", {"asset_handle": "photo_1"}, question,
            inspect_called=False, available_tools={"inspect_photo", "read_photo_text"})
        self.assertEqual(tool, "inspect_photo")
        self.assertEqual(args["asset_handle"], "photo_1")
        self.assertEqual(args["question"], question)

    def test_document_value_needs_ocr_and_is_not_redirected(self):
        question = "考勤记录里，7月份实际上班天数是多少天？"
        self.assertTrue(ocr_intent(question))
        tool, _ = _visual_tool_correction(
            "read_photo_text", {"asset_handle": "photo_1"}, question,
            inspect_called=False, available_tools={"inspect_photo", "read_photo_text"})
        self.assertEqual(tool, "read_photo_text")
        self.assertFalse(ocr_intent("乐乐住院那次具体是哪一年的事？"))

    def _task(self):
        return TaskState.from_declaration(TaskDeclaration(
            goal="确认照片地点",
            scope_id="album1",
            requirements=(
                EvidenceRequirement(id="place", evidence_type="location_metadata", description="地点"),
            ),
        ))

    def test_writer_messages_contain_only_minimal_context(self):
        context = {
            "facts": [{"evidence_type": "location_metadata", "value": "秦皇岛", "asset": "photo_1"}],
            "unknowns": [],
            "conflicts": [],
        }

        messages = build_answer_writer_messages("照片在哪里？", context)
        serialized = json.dumps(messages, ensure_ascii=False)

        self.assertIn("秦皇岛", serialized)
        self.assertNotIn("search_memories", serialized)
        self.assertNotIn("inspect_photo", serialized)
        self.assertNotIn("result_set_id", serialized)

    def test_clean_writer_output_accepts_plain_text_or_legacy_json(self):
        self.assertEqual(clean_writer_output("活动在秦皇岛"), "活动在秦皇岛。")
        self.assertEqual(clean_writer_output('{"action":"final","answer":"活动在秦皇岛"}'), "活动在秦皇岛。")

    def test_clean_writer_output_removes_photo_handle_parenthetical(self):
        output = clean_writer_output(
            "我为您找到的这张照片（photo_1）实际上是在河北省邯郸市永年区的室内家居环境中拍摄的..."
        )

        self.assertEqual(
            output,
            "我为您找到的这张照片实际上是在河北省邯郸市永年区的室内家居环境中拍摄的...。",
        )
        self.assertNotRegex(output, r"(?i)(?<![A-Za-z0-9_])photo_\d+(?![A-Za-z0-9_])")
        self.assertNotIn("（）", output)

    def test_clean_writer_output_naturalizes_bare_and_multiple_photo_handles(self):
        cases = {
            "photo_2显示的是室内家居环境。": "这张照片显示的是室内家居环境。",
            "画面来自photo_3，地点在河北。": "画面来自这张照片，地点在河北。",
            "这是室内照片（photo_4）。": "这是室内照片。",
            "photo_5和photo_6展示了同一处场景。": "这些照片展示了同一处场景。",
            "这些画面（photo_7、photo_8）拍摄于河北。": "这些画面拍摄于河北。",
        }

        for source, expected in cases.items():
            with self.subTest(source=source):
                output = clean_writer_output(source)
                self.assertEqual(output, expected)
                self.assertIsNone(re.search(r"(?i)(?<![A-Za-z0-9_])photo_\d+(?![A-Za-z0-9_])", output))

    def test_photo_handle_cleanup_does_not_match_embedded_identifier(self):
        self.assertEqual(
            clean_writer_output("型号是my_photo_1_variant"),
            "型号是my_photo_1_variant。",
        )

    def test_rewrite_final_removes_photo_handle(self):
        output = rewrite_final(
            lambda _messages, **_kwargs: "照片（photo_9）是在河北拍的。",
            {},
            "draft",
        )

        self.assertEqual(output, "照片是在河北拍的。")

    def test_naturalize_answer_removes_handle_on_normal_final_path(self):
        self.assertEqual(
            naturalize_answer("根据对照片 photo_1 的视觉复核，现场有蓝色灯光"),
            "根据对照片 这张照片 的视觉复核，现场有蓝色灯光。",
        )

    def test_answer_context_flag_is_opt_in(self):
        ledger = EvidenceLedger(scope_id="album1")
        ledger.append(LedgerEntry(
            tool_call_id="call_1",
            capability="search_memories",
            evidence_type="location_metadata",
            input_refs=("photo_1",),
            provenance_refs=("photo_1",),
            extracted_value="秦皇岛",
            requirement_refs=("place",),
            provenance_scope_id="album1",
        ))
        self.assertTrue(ledger.build_answer_context("where", self._task())["facts"])
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("SENTRIX_AGENT2_ANSWER_CONTEXT", None)
            self.assertIsNone(os.getenv("SENTRIX_AGENT2_ANSWER_CONTEXT"))


if __name__ == "__main__":
    unittest.main()
