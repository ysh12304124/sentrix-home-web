import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from backend.video.contracts import VideoMetadata
from backend.video.mlt_keyframe import _sliding_window_spans, merge_and_analyze_windows
from backend.video.processor import VideoMemoryAdapter, _frame_analysis_from_event


class WindowGamma:
    def __init__(self, decisions):
        self.decisions = list(decisions)
        self.calls = []

    def analyze_video_scene_window(self, paths, metadata=None):
        self.calls.append((list(paths), metadata))
        return self.decisions.pop(0)


def scene(root, index):
    image = root / f"scene-{index}.webp"
    image.write_bytes(f"image-{index}".encode())
    row = {
        "webp_path": str(image), "source_timestamp_sec": index * 2.0,
        "source_frame_index": index * 60,
    }
    return {
        "event_id": f"scene-{index}", "source_event_ids": [f"scene-{index}"],
        "start_sec": index * 2.0, "end_sec": (index + 1) * 2.0,
        "representative": row, "representatives": [row],
        "objects": [], "actions": [], "expressions": [], "yolo_timeline": [],
        "source_frame_count": 4, "duplicate_frame_count": 0,
        "visual_duplicate_count": 0, "memory_keyframe_count": 1,
        "mlt_scene_count": 1, "boundary_confidence": 0.8,
    }


class MLTVideoTests(unittest.TestCase):
    def test_sliding_windows_cover_every_adjacent_pair(self):
        for count in range(3, 41):
            spans = _sliding_window_spans(count, maximum=5, stride=4)
            self.assertTrue(all(3 <= end - start <= 5 for start, end in spans))
            covered_edges = {
                edge for start, end in spans for edge in range(start, end - 1)
            }
            self.assertEqual(covered_edges, set(range(count - 1)))

    def test_three_to_five_frame_calls_merge_and_generate_memory_once(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            items = [scene(root, index) for index in range(6)]
            gamma = WindowGamma([
                {"groups": [
                    {"indices": [0, 1, 2], "merge_reason": "同一场景连续镜头",
                     "caption": "连续活动记录", "representative_indices": [1]},
                    {"indices": [3, 4], "merge_reason": "另一连续场景",
                     "caption": "另一活动记录", "representative_indices": [3]},
                ]},
                {"groups": [
                    {"indices": [0, 1], "merge_reason": "同一家庭互动",
                     "caption": "家人互动", "representative_indices": [0]},
                    {"indices": [2], "merge_reason": "地点变化",
                     "caption": "独立场景", "representative_indices": [2]},
                ]},
            ])
            merged, stats = merge_and_analyze_windows(items, gamma, {"video_id": "v1"})

            self.assertEqual(len(merged), 3)
            self.assertEqual(merged[0]["source_event_ids"], ["scene-0", "scene-1", "scene-2"])
            self.assertEqual(merged[1]["source_event_ids"], ["scene-3", "scene-4"])
            self.assertEqual(merged[2]["source_event_ids"], ["scene-5"])
            self.assertEqual(stats["calls"], 2)
            self.assertEqual(stats["merged_away"], 3)
            self.assertEqual(len(merged[0]["representative"]["vlm_evidence"]), 3)
            self.assertEqual(merged[0]["event_analysis"]["caption"], "连续活动记录")
            self.assertEqual(merged[0]["event_analysis"]["representative_indices"], [1])
            self.assertTrue(all(3 <= len(call[0]) <= 5 for call in gamma.calls))
            self.assertEqual([len(call[0]) for call in gamma.calls], [5, 3])
            self.assertEqual(stats["window_stride"], 4)
            self.assertTrue(stats["sliding_windows"])

    def test_overlapping_anchor_connects_scene_across_window_boundary(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            items = [scene(root, index) for index in range(7)]
            gamma = WindowGamma([
                {"groups": [
                    {"indices": [0], "caption": "片段0"},
                    {"indices": [1], "caption": "片段1"},
                    {"indices": [2], "caption": "片段2"},
                    {"indices": [3, 4], "caption": "连续场景前半", "representative_indices": [4]},
                ]},
                {"groups": [{
                    "indices": [0, 1, 2], "caption": "连续场景后半",
                    "representative_indices": [1],
                }]},
            ])

            merged, stats = merge_and_analyze_windows(items, gamma)

            self.assertEqual([item["source_event_ids"] for item in merged], [
                ["scene-0"], ["scene-1"], ["scene-2"],
                ["scene-3", "scene-4", "scene-5", "scene-6"],
            ])
            self.assertEqual(stats["calls"], 2)
            self.assertEqual(stats["windows"][1]["overlap_with_previous"], 1)
            self.assertEqual(stats["edge_consensus"][4]["same_votes"], 1)

    def test_generic_mlt_strong_boundary_guard_rejects_over_merge(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            items = [scene(root, index) for index in range(3)]
            items[0]["_mlt_embedding"] = [1.0, 0.0]
            items[1]["_mlt_embedding"] = [0.99, 0.01]
            items[2]["_mlt_embedding"] = [-1.0, 0.0]
            gamma = WindowGamma([{"groups": [{
                "indices": [0, 1, 2], "merge_reason": "模型建议合并",
                "caption": "连续记录", "representative_indices": [0, 2],
                "frame_observations": [
                    {"index": 0, "caption": "场景甲", "place": "地点甲"},
                    {"index": 1, "caption": "场景甲后续", "place": "地点甲"},
                    {"index": 2, "caption": "场景乙", "place": "地点乙"},
                ],
            }]}])

            merged, stats = merge_and_analyze_windows(items, gamma)

            self.assertEqual([item["source_event_ids"] for item in merged], [
                ["scene-0", "scene-1"], ["scene-2"],
            ])
            self.assertEqual(stats["strong_boundary_splits"], 1)
            self.assertEqual(merged[1]["event_analysis"]["caption"], "场景乙")
            self.assertEqual(merged[0]["event_analysis"]["representative_indices"], [0, 1])
            self.assertEqual(merged[0]["event_analysis"]["coverage_required_indices"], [0, 1])

    def test_frame_analysis_uses_only_the_matching_frame_observation(self):
        event_analysis = {
            "caption": "人物甲观察目标物，人物乙在前方指向目标物",
            "people": ["人物甲", "人物乙"], "objects": ["目标物"],
            "clothing": ["人物甲浅色上衣", "人物乙深色上衣"],
            "frame_observations": [
                {"index": 0, "caption": "人物甲观察目标物", "activity": "观察",
                 "place": "场景区域", "people": ["人物甲"], "objects": ["目标物"]},
                {"index": 1, "caption": "人物乙指向目标物", "activity": "指向",
                 "place": "场景区域", "people": ["人物乙"], "objects": ["目标物"]},
            ],
            "representative_indices": [0, 1], "coverage_required_indices": [0, 1],
        }

        frame = _frame_analysis_from_event(event_analysis, 0)

        self.assertEqual(frame["caption"], "人物甲观察目标物")
        self.assertEqual(frame["people"], ["人物甲"])
        self.assertNotIn("人物乙", frame["caption"])
        self.assertEqual(frame["clothing"], [])
        self.assertNotIn("representative_indices", frame)

    def test_mlt_configuration_dispatches_to_mlt_processor(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            video = root / "movie.mp4"
            video.write_bytes(b"video")
            store = MagicMock()
            pipeline = SimpleNamespace(store=store, geocoder=MagicMock())
            adapter = VideoMemoryAdapter(keyframe_algorithm="mlt_semantic")
            expected = {"status": "processed"}

            with (
                patch("backend.video.processor.probe_video_metadata", return_value=VideoMetadata(codec="h264")),
                patch.object(adapter, "_process_mlt_semantic", return_value=expected) as mlt,
                patch.dict(os.environ, {"SENTRIX_DATA_DIR": directory}),
            ):
                result = adapter.process({"id": "video-1", "path": str(video)}, pipeline)

            self.assertEqual(result, expected)
            mlt.assert_called_once()


if __name__ == "__main__":
    unittest.main()
