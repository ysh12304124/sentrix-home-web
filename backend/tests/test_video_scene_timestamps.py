import unittest

from backend.video.processor import _captured_at


class VideoSceneTimestampTests(unittest.TestCase):
    def test_quicktime_compact_offset_receives_frame_seconds(self):
        self.assertEqual(
            _captured_at("2026-08-02T14:55:56+0800", 6.200880681818181),
            "2026-08-02T14:56:02.200881+08:00",
        )


if __name__ == "__main__":
    unittest.main()
