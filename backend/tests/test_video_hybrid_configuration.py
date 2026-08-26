import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from backend.video.contracts import VideoMetadata
from backend.video.hybrid_keyframe import run as run_hybrid_keyframes
from backend.video.processor import VideoMemoryAdapter


class HybridVideoConfigurationTests(unittest.TestCase):
    def test_hybrid_path_keeps_browser_preview_generation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            video = root / "movie.mov"
            video.write_bytes(b"video")
            store = MagicMock()
            pipeline = SimpleNamespace(store=store, geocoder=MagicMock())
            adapter = VideoMemoryAdapter(keyframe_algorithm="hybrid_webp")
            expected = {"status": "processed"}

            with (
                patch("backend.video.processor.probe_video_metadata", return_value=VideoMetadata(codec="hevc")),
                patch("backend.video.processor._browser_preview", return_value=str(root / "preview.mp4")) as preview,
                patch.object(adapter, "_process_hybrid_webp", return_value=expected) as hybrid,
                patch.dict(os.environ, {"SENTRIX_DATA_DIR": directory}),
            ):
                result = adapter.process({"id": "video-1", "path": str(video)}, pipeline)

            self.assertEqual(result, expected)
            preview.assert_called_once()
            hybrid.assert_called_once()
            self.assertTrue(any(
                call.args[2].get("browser_preview_path") == str(root / "preview.mp4")
                for call in store.update_asset.call_args_list
            ))

    def test_hybrid_extractor_honors_webp_quality(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            video = root / "movie.mov"
            video.write_bytes(b"video")
            output = root / "output"
            output.mkdir()
            (output / "frame_map.json").write_text('{"frames": []}', encoding="utf-8")
            (output / "stats.json").write_text('{"preliminary_event_count": 0}', encoding="utf-8")
            completed = SimpleNamespace(returncode=0, stdout="", stderr="")

            with (
                patch("backend.video.hybrid_keyframe.subprocess.run", return_value=completed) as command_run,
                patch.dict(os.environ, {"SENTRIX_VIDEO_WEBP_QUALITY": "73"}),
            ):
                run_hybrid_keyframes(video, output, "video-1")

            command = command_run.call_args.args[0]
            quality_index = command.index("--webp-quality")
            self.assertEqual(command[quality_index + 1], "73")


if __name__ == "__main__":
    unittest.main()
