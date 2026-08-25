import json
import os
import re
import unittest
from unittest.mock import patch

from backend.agent_runtime.evidence_ledger import EvidenceLedger, LedgerEntry
from backend.agent_runtime.final_writer import (
    build_answer_writer_messages,
    clean_writer_output, naturalize_answer,
    rewrite_final,
)
from backend.agent_runtime.task_state import EvidenceRequirement, TaskDeclaration, TaskState


class Agent2AnswerContextTests(unittest.TestCase):
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
