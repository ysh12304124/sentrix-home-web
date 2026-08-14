import json
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from backend.video.metadata import probe_video_metadata


class VideoMetadataTests(unittest.TestCase):
    def test_prefers_quicktime_creationdate_and_parses_iso6709(self):
        payload = {
            "format": {"duration": "11.735", "tags": {
                "creation_time": "2026-08-02T06:55:56Z",
                "com.apple.quicktime.creationdate": "2026-08-02T14:55:56+0800",
                "com.apple.quicktime.location.ISO6709": "+22.4846+114.5438+005.469/",
                "com.apple.quicktime.make": "Apple", "com.apple.quicktime.model": "iPhone 14 Plus",
            }},
            "streams": [{"codec_type": "video", "codec_name": "hevc", "width": 1920, "height": 1080,
                         "avg_frame_rate": "70400/2347", "tags": {}}],
        }
        completed = Mock(returncode=0, stdout=json.dumps(payload), stderr="")
        with patch("backend.video.metadata.subprocess.run", return_value=completed):
            result = probe_video_metadata(Path("movie.mov"))
        self.assertEqual(result.captured_at, "2026-08-02T14:55:56+08:00")
        self.assertAlmostEqual(result.latitude, 22.4846)
        self.assertAlmostEqual(result.longitude, 114.5438)
        self.assertEqual(result.codec, "hevc")
        self.assertEqual(result.device, "Apple iPhone 14 Plus")


if __name__ == "__main__": unittest.main()
