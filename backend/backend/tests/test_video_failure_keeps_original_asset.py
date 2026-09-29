import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from backend.db import MemoryStore
from backend.pipeline import IngestionPipeline
from backend.tests.video_fakes import FakeClip, FakeFace, FakeGamma
from backend.video.contracts import VideoMetadata
from backend.video.processor import VideoMemoryAdapter


class BrokenWorldMM:
    def run(self, *args): raise RuntimeError("synthetic WorldMM failure")


class VideoFailureTests(unittest.TestCase):
    def test_failure_keeps_original_and_records_retryable_stage(self):
        with tempfile.TemporaryDirectory() as directory:
            video = Path(directory) / "movie.mov"; video.write_bytes(b"original-video")
            store = MemoryStore(str(Path(directory) / "memory.db"))
            pipeline = IngestionPipeline(store, gamma=FakeGamma(), face=FakeFace(), clip=FakeClip())
            pipeline.video_memory_adapter = VideoMemoryAdapter(BrokenWorldMM())
            asset = pipeline.create_asset(video, media_type="video")
            with patch("backend.video.processor.probe_video_metadata", return_value=VideoMetadata(duration_sec=10, fps=30)):
                result = pipeline.process(asset["id"])
            self.assertEqual(result["status"], "video-processing-failed")
            self.assertTrue(video.is_file())
            self.assertEqual(video.read_bytes(), b"original-video")
            self.assertTrue(result["metadata_json"]["retryable"])
            self.assertEqual(store.count("assets"), 1)


if __name__ == "__main__": unittest.main()
