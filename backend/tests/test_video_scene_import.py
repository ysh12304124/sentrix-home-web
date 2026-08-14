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


class VideoSceneImportTests(unittest.TestCase):
    def test_three_scenes_create_19_assets_observations_and_three_events(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {"SENTRIX_DATA_DIR": directory}):
            root = Path(directory)
            video = root / "movie.mov"; video.write_bytes(b"video")
            store = MemoryStore(str(root / "memory.db"))
            pipeline = IngestionPipeline(store, gamma=FakeGamma(), face=FakeFace(), clip=FakeClip(), geocoder=FakeGeocoder())
            pipeline.video_memory_adapter = VideoMemoryAdapter(FixtureWorldMM(root))
            asset = pipeline.create_asset(video, media_type="video", metadata={"scope_id": "family-video"})
            metadata = VideoMetadata(captured_at="2026-08-13T14:30:00+08:00", latitude=22.4846, longitude=114.5438,
                                     captured_location="22.484600,114.543800", duration_sec=600, fps=30, width=1920,
                                     height=1080, codec="hevc", device="Apple iPhone 14 Plus")
            with patch("backend.video.processor.probe_video_metadata", return_value=metadata):
                result = pipeline.process(asset["id"])

            self.assertEqual(result["status"], "processed")
            self.assertEqual(store.count("assets"), 20)
            self.assertEqual(store.count("observations"), 19)
            scenes = store.list_video_scene_events(asset["id"])
            self.assertEqual(len(scenes), 3)
            self.assertEqual([len(item["observation_ids"]) for item in scenes], [5, 8, 6])
            self.assertEqual(scenes[1]["time_start"], "2026-08-13T14:32:00+08:00")
            self.assertEqual(scenes[1]["time_end"], "2026-08-13T14:37:00+08:00")
            self.assertEqual(scenes[1]["place"], "深圳市龙岗区")
            derived = store.list_derived_assets(asset["id"])
            self.assertEqual(len(derived), 19)
            self.assertEqual(derived[0]["parent_asset_id"], asset["id"])
            self.assertNotEqual(derived[0]["captured_at"], derived[1]["captured_at"])
            self.assertEqual(derived[0]["captured_location"], "22.484600,114.543800")
            self.assertTrue(all(item["derived_kind"] == "video_keyframe" for item in derived))


if __name__ == "__main__": unittest.main()
