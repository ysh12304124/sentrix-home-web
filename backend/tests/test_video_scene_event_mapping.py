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


class VideoSceneEventMappingTests(unittest.TestCase):
    def test_each_worldmm_scene_maps_to_exactly_one_event(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {"SENTRIX_DATA_DIR": directory}):
            root = Path(directory)
            video = root / "movie.mov"
            video.write_bytes(b"video")
            store = MemoryStore(str(root / "memory.db"))
            pipeline = IngestionPipeline(store, gamma=FakeGamma(), face=FakeFace(), clip=FakeClip(), geocoder=FakeGeocoder())
            pipeline.video_memory_adapter = VideoMemoryAdapter(FixtureWorldMM(root, counts=(2, 3, 1), ranges=((0, 10), (10, 20), (20, 30))))
            asset = pipeline.create_asset(video, media_type="video", metadata={"scope_id": "mapping-test"})
            metadata = VideoMetadata(
                captured_at="2026-08-13T14:30:00+08:00", duration_sec=30, fps=30,
                width=1920, height=1080, codec="h264",
            )
            with patch("backend.video.processor.probe_video_metadata", return_value=metadata):
                pipeline.process(asset["id"])

            scenes = store.list_video_scene_events(asset["id"])
            self.assertEqual([item["source_scene_index"] for item in scenes], [0, 1, 2])
            self.assertEqual([len(item["observation_ids"]) for item in scenes], [2, 3, 1])
            self.assertTrue(all(item["source_asset_id"] == asset["id"] for item in scenes))
            self.assertEqual(store.count("events"), 3)


if __name__ == "__main__":
    unittest.main()
