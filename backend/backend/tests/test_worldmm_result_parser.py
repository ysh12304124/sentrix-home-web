import json
import tempfile
import unittest
from pathlib import Path

from backend.video.contracts import WorldMMResult


class WorldMMParserTests(unittest.TestCase):
    def test_uses_worldmm_scene_boundaries_without_reclustering(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "keyframe_package/original").mkdir(parents=True)
            image = root / "keyframe_package/original/k1.jpg"
            image.write_bytes(b"frame")
            (root / "memory.json").write_text(json.dumps({
                "video": {"duration_sec": 10},
                "memory_keyframes": [{"keyframe_code": "k1", "source_image": "keyframe_package/original/k1.jpg", "timestamp": 3.5, "frame_index": 105}],
                "scenes": [{"scene_id": "scene_0042", "start_time": 2.0, "end_time": 9.0, "keyframe_codes": ["k1"]}],
            }), encoding="utf-8")
            result = WorldMMResult.from_output(root)
            self.assertEqual(result.scenes[0].scene_id, "scene_0042")
            self.assertEqual((result.scenes[0].start_sec, result.scenes[0].end_sec), (2.0, 9.0))
            self.assertEqual(result.scenes[0].keyframes[0].timestamp_sec, 3.5)


if __name__ == "__main__": unittest.main()
