import unittest

from backend.agent_runtime import runtime
from backend.agent_runtime.runtime import _merge_system_constraint


class RuntimeMessageOrderTests(unittest.TestCase):
    def test_guard_recovery_requires_field_evidence_before_numeric_direct_answer(self):
        guidance = runtime._ocr_recovery_instruction(
            "礼金总额是多少？",
            [{"tool": "read_photo_text", "ocr_text": "回看1", "exact_values": []}],
        )

        self.assertIn("现有记录不足以确认", guidance)
        self.assertNotIn("直接给数字", guidance)

    def test_guard_recovery_marks_internal_prompt_origin(self):
        messages = [
            {"role": "system", "content": "base"},
            {"role": "user", "content": "礼金总额是多少？"},
            {"role": "assistant", "content": "{\"action\":\"final\"}"},
            {"role": "user", "content": "上一条（你刚输出的内容）是你的最终回答。基于这些读到的文字直接回答具体内容。"},
        ]

        annotations = runtime._prompt_annotations(messages)

        self.assertEqual(annotations, [{
            "message_index": 3,
            "message_origin": "system_recovery",
            "content_preview": messages[3]["content"],
        }])

    def test_constraint_is_merged_into_leading_system_message(self):
        messages = [
            {"role": "system", "content": "base"},
            {"role": "user", "content": "question"},
            {"role": "assistant", "content": "tool action"},
            {"role": "tool", "content": "tool result"},
        ]

        _merge_system_constraint(messages, "confirmed facts")

        self.assertEqual([message["role"] for message in messages],
                         ["system", "user", "assistant", "tool"])
        self.assertEqual(messages[0]["content"], "base\n\nconfirmed facts")

    def test_empty_constraint_does_not_mutate_messages(self):
        messages = [{"role": "system", "content": "base"}]

        _merge_system_constraint(messages, "")

        self.assertEqual(messages, [{"role": "system", "content": "base"}])

    def test_missing_leading_system_message_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "must start with a system message"):
            _merge_system_constraint([{"role": "user", "content": "question"}],
                                     "confirmed facts")


if __name__ == "__main__":
    unittest.main()
