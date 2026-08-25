import unittest

from backend.agent_runtime import final_writer


class WriterAuthorityTests(unittest.TestCase):
    def test_deterministic_answer_override_is_removed(self):
        self.assertFalse(hasattr(final_writer, "deterministic_answer"))


if __name__ == "__main__":
    unittest.main()
