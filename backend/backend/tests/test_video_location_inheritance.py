import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from backend.db import MemoryStore
from backend.pipeline import IngestionPipeline
from backend.tests.video_fakes import FakeClip, FakeFace, FakeGamma, FakeGeocoder, FixtureWorldMM
from backend.video.contracts import VideoMetadata
from backend.video.processor import VideoMemoryAdapter


class VideoLocationInheritanceTests(unittest.TestCase):
    def test_source_gps_is_inherited_without_visual_place_overwrite(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {"SENTRIX_DATA_DIR": directory}):
            root = Path(directory)
            video = root / "movie.mov"
            video.write_bytes(b"video")
            store = MemoryStore(str(root / "memory.db"))
            pipeline = IngestionPipeline(store, gamma=FakeGamma(), face=FakeFace(), clip=FakeClip(), geocoder=FakeGeocoder())
            pipeline.video_memory_adapter = VideoMemoryAdapter(FixtureWorldMM(root, counts=(1,), ranges=((0, 10),)))
            asset = pipeline.create_asset(video, media_type="video", metadata={"scope_id": "location-test"})
            metadata = VideoMetadata(
                captured_at="2026-08-13T14:30:00+08:00", latitude=22.4846, longitude=114.5438,
                captured_location="22.484600,114.543800", duration_sec=10, fps=30,
                width=1920, height=1080, codec="hevc",
            )
            with patch("backend.video.processor.probe_video_metadata", return_value=metadata):
                pipeline.process(asset["id"])

            scene = store.list_video_scene_events(asset["id"])[0]
            frame = store.list_derived_assets(asset["id"])[0]
            self.assertEqual(scene["place"], "深圳市龙岗区")
            self.assertEqual(frame["captured_location"], "22.484600,114.543800")
            self.assertEqual(frame["metadata_json"]["location_source"], "video_metadata")


if __name__ == "__main__":
    unittest.main()
