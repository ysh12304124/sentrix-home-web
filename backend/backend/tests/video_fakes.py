from dataclasses import dataclass
from pathlib import Path

from backend.video.contracts import WorldMMKeyframe, WorldMMResult, WorldMMScene


class FakeGamma:
    model = "video-test-gamma"

    def analyze_image(self, path, metadata=None):
        return {
            "caption": "家人在客厅活动", "activity": "家庭活动", "place": "客厅",
            "people": [], "objects": ["桌子"], "ocr_text": "", "event_type": "家庭记录",
            "facts": [], "confidence": 0.8, "model": self.model,
        }

    def summarize_event(self, event, observations):
        return {
            "title": "客厅里的家庭活动", "event_type": "视频场景", "activity": "家庭活动",
            "summary": f"这个视频场景包含 {len(observations)} 张关键帧。",
            "confidence": 0.8, "model": self.model,
        }


class FakeClip:
    model_name = "video-test-clip"

    def embed_image(self, path): return [1.0, 0.0]
    def embed_text(self, text): return [0.0, 1.0]


class FakeFace:
    enabled = True
    error = None
    identity_model = "video-test-face"
    identity_ready = True

    def detect(self, path): return []


class FakeGeocoder:
    def lookup(self, gps):
        return {"source": "offline", "country": "CN", "city": "深圳市", "district": "龙岗区", "label": "深圳市龙岗区"}


class FixtureWorldMM:
    def __init__(self, root, counts=(5, 8, 6), ranges=((0, 120), (120, 420), (420, 600))):
        self.root = Path(root)
        self.counts = counts
        self.ranges = ranges

    def run(self, video_path, video_id, output_dir):
        scenes = []
        for scene_index, (count, bounds) in enumerate(zip(self.counts, self.ranges)):
            frames = []
            for index in range(count):
                timestamp = bounds[0] + (index + 1) * (bounds[1] - bounds[0]) / (count + 1)
                path = self.root / f"s{scene_index}_k{index}.jpg"
                path.write_bytes(f"{scene_index}-{index}".encode())
                frames.append(WorldMMKeyframe(
                    code=f"s{scene_index}k{index}", path=str(path), timestamp_sec=timestamp,
                    frame_index=int(timestamp * 30), score=0.5, selection_reason="fixture",
                ))
            scenes.append(WorldMMScene(
                scene_id=f"scene_{scene_index + 1:04d}", index=scene_index,
                start_sec=bounds[0], end_sec=bounds[1], keyframes=frames,
            ))
        return WorldMMResult(video={"video_id": video_id}, scenes=scenes, output_dir=str(output_dir))
