import json
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from backend.pipeline import IngestionPipeline
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

    def test_parses_ffprobe_location_tag_used_by_tagged_mp4(self):
        payload = {
            "format": {"duration": "6", "tags": {
                "creation_time": "2017-09-27T03:36:15.000000Z",
                "location": "+36.54452778+115.27778611+41.7909/",
            }},
            "streams": [{"codec_type": "video", "codec_name": "h264", "width": 720, "height": 1280,
                         "avg_frame_rate": "24/1", "tags": {}}],
        }
        completed = Mock(returncode=0, stdout=json.dumps(payload), stderr="")
        with patch("backend.video.metadata.subprocess.run", return_value=completed):
            result = probe_video_metadata(Path("kling.mp4"))
        self.assertEqual(result.captured_at, "2017-09-27T03:36:15.000000+00:00")
        self.assertAlmostEqual(result.latitude, 36.54452778)
        self.assertAlmostEqual(result.longitude, 115.27778611)

    def test_prepare_asset_copies_video_capture_time_and_gps(self):
        store = Mock()
        store.find_asset_by_hash.return_value = None
        geocoder = Mock()
        geocoder.lookup.return_value = {"city": "馆陶县"}
        pipeline = IngestionPipeline.__new__(IngestionPipeline)
        pipeline.store = store
        pipeline.geocoder = geocoder
        capture = {
            "captured_at": "2017-09-27T11:36:15+08:00",
            "gps": {"latitude": 36.5445, "longitude": 115.2778},
            "captured_location": "36.544500,115.277800",
            "device": "iPhone",
        }
        with patch.object(IngestionPipeline, "_sha256", return_value="abc123"), \
                patch.object(IngestionPipeline, "_extract_video_capture_metadata", return_value=capture):
            prepared = IngestionPipeline.prepare_asset(
                pipeline, Path("clip.mp4"), media_type="video",
            )
        metadata = prepared["metadata"]
        self.assertEqual(metadata["captured_at"], "2017-09-27T11:36:15+08:00")
        self.assertEqual(metadata["captured_location"], "36.544500,115.277800")
        self.assertEqual(metadata["source_device_id"], "iPhone")
        geocoder.lookup.assert_called_once()


if __name__ == "__main__": unittest.main()
