import unittest

from backend.agent_runtime.runtime import _merge_system_constraint


class RuntimeMessageOrderTests(unittest.TestCase):
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
