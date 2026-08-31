import unittest

from backend.agent_runtime.runtime import _normalize_preview_handle


class HandleLifecycleTests(unittest.TestCase):
    def test_stale_handle_is_not_silently_remapped(self):
        arguments, requested = _normalize_preview_handle(
            {"asset_handle": "photo_99", "question": "谁"}, ["photo_7", "photo_14"])
        self.assertEqual(arguments["asset_handle"], "photo_99")
        self.assertEqual(requested, "photo_99")


if __name__ == "__main__":
    unittest.main()
