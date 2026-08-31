import unittest

from backend.agent_runtime.ocr_tool import resolve_small_ocr_enabled


class SmallOCRSettingsTests(unittest.TestCase):
    def test_legacy_false_without_explicit_marker_keeps_available_ocr_enabled(self):
        self.assertTrue(resolve_small_ocr_enabled("false", None, available=True))

    def test_explicit_user_opt_out_disables_available_ocr(self):
        self.assertFalse(resolve_small_ocr_enabled("false", "true", available=True))

    def test_explicit_user_opt_in_enables_available_ocr(self):
        self.assertTrue(resolve_small_ocr_enabled("true", "true", available=True))

    def test_unavailable_ocr_is_never_enabled(self):
        self.assertFalse(resolve_small_ocr_enabled("true", "true", available=False))
