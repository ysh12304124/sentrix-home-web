import unittest

from backend.agent_runtime.runtime import _normalize_preview_handle


class HandleLifecycleTests(unittest.TestCase):
    def test_handle_alias_is_not_silently_rebound_to_first_preview(self):
        arguments, requested = _normalize_preview_handle(
            {"handle": "photo_2", "question": "看这张照片"},
            ["photo_1", "photo_2"],
        )
        self.assertEqual(arguments["asset_handle"], "photo_2")
        self.assertEqual(requested, None)

    def test_canonical_asset_handle_wins_over_alias(self):
        arguments, requested = _normalize_preview_handle(
            {"asset_handle": "photo_2", "handle": "photo_1"},
            ["photo_1", "photo_2"],
        )
        self.assertEqual(arguments["asset_handle"], "photo_2")
        self.assertEqual(requested, None)

    def test_stale_handle_is_not_silently_remapped(self):
        arguments, requested = _normalize_preview_handle(
            {"asset_handle": "photo_99", "question": "谁"}, ["photo_7", "photo_14"])
        self.assertEqual(arguments["asset_handle"], "photo_99")
        self.assertEqual(requested, "photo_99")

    def test_stale_handle_alias_is_not_silently_remapped(self):
        arguments, requested = _normalize_preview_handle(
            {"handle": "photo_99"}, ["photo_7", "photo_14"])
        self.assertEqual(arguments["asset_handle"], "photo_99")
        self.assertEqual(requested, "photo_99")


if __name__ == "__main__":
    unittest.main()
