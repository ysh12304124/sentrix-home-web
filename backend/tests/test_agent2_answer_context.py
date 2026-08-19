import json
import os
import unittest
from unittest.mock import patch

from backend.agent_runtime.evidence_ledger import EvidenceLedger, LedgerEntry
from backend.agent_runtime.final_writer import build_answer_writer_messages, clean_writer_output
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
