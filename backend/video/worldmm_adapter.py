from __future__ import annotations

import os
import subprocess
import sys
import json
from dataclasses import dataclass
from pathlib import Path

from .contracts import WorldMMKeyframe, WorldMMResult, WorldMMScene


HYBRID_METHOD_VERSION = "sentrix-keyframe-hybrid-v2.0.0"


@dataclass
class HybridResult:
    """Normalized result returned by the frozen YOLO/Katna/NVDEC pipeline."""

    video: dict
    scenes: list[WorldMMScene]
    output_dir: str
    manifest: dict
    full_keyframe_count: int
    summary_keyframe_count: int
    selected_keyframe_count: int

    @property
    def keyframe_count(self):
        return sum(len(scene.keyframes) for scene in self.scenes)

    @classmethod
    def from_output(cls, output_dir: Path, video_path: str, video_id: str):
        semantic_path = output_dir / "semantic.json"
        map_path = output_dir / "frame_map.json"
        stats_path = output_dir / "stats.json"
        if not semantic_path.is_file() or not map_path.is_file():
            raise ValueError("hybrid keyframe output is missing semantic.json or frame_map.json")
        semantic = json.loads(semantic_path.read_text(encoding="utf-8"))
        frame_map = json.loads(map_path.read_text(encoding="utf-8"))
        stats = json.loads(stats_path.read_text(encoding="utf-8")) if stats_path.is_file() else {}
        frames = [item for item in frame_map.get("frames", []) if isinstance(item, dict)]
        semantic_by_index = {
            int(item.get("encoded_frame_index")): item
            for item in semantic.get("frames", [])
            if isinstance(item, dict) and str(item.get("encoded_frame_index", "")).lstrip("-").isdigit()
        }
        event_specs = {
            str(item.get("event_id")): item
            for item in semantic.get("events", [])
            if isinstance(item, dict) and item.get("event_id")
        }
        grouped = {}
        for raw in frames:
            frame_index = int(raw.get("encoded_frame_index", 0) or 0)
            frame = {**raw, **semantic_by_index.get(frame_index, {})}
            event_id = str(frame.get("event_id") or f"event_{frame_index:05d}")
            grouped.setdefault(event_id, []).append(frame)
        scenes = []
        for index, (event_id, members) in enumerate(sorted(
            grouped.items(), key=lambda pair: float(pair[1][0].get("event_start_sec") or pair[1][0].get("source_timestamp_sec") or 0)
        )):
            spec = dict(event_specs.get(event_id) or {})
            representative = min(members, key=lambda item: float(item.get("source_timestamp_sec") or 0))
            image_path = Path(str(representative.get("webp_path") or ""))
            if not image_path.is_absolute():
                image_path = output_dir / image_path
            if not image_path.is_file():
                raise ValueError(f"hybrid representative WebP is missing: {image_path}")
            objects = list(representative.get("objects") or [])
            actions = list(representative.get("actions") or [])
            expressions = list(representative.get("expressions") or [])
            labels = list(dict.fromkeys(
                [str(item.get("label") or "") for item in objects + actions + expressions if isinstance(item, dict)]
                + [str(item) for item in spec.get("objects") or []]
                + [str(item) for item in spec.get("actions") or []]
            ))
            keyframe = WorldMMKeyframe(
                code=event_id, path=str(image_path.resolve()),
                timestamp_sec=float(representative.get("source_timestamp_sec") or 0),
                frame_index=int(representative.get("source_frame_index") or 0),
                score=float(representative.get("sharpness") or 0),
                selection_reason="hybrid-event-representative",
                objects=objects, actions=actions, expressions=expressions,
                raw={**representative, "method_version": HYBRID_METHOD_VERSION},
            )
            start = float(spec.get("start_sec") or representative.get("event_start_sec") or keyframe.timestamp_sec)
            end = float(spec.get("end_sec") or representative.get("event_end_sec") or keyframe.timestamp_sec)
            scenes.append(WorldMMScene(
                scene_id=event_id, index=index, start_sec=start, end_sec=end,
                keyframes=[keyframe], semantic_labels=labels[:80], raw={
                    **spec, "method_version": HYBRID_METHOD_VERSION,
                    "event_kind": spec.get("kind") or representative.get("event_kind") or "scene",
                    "representative_webp": str(image_path.resolve()),
                },
            ))
        manifest = {"method_version": HYBRID_METHOD_VERSION, "stats": stats, "frame_map": str(map_path), "semantic": str(semantic_path)}
        return cls(
            video={"path": str(video_path), "video_id": video_id}, scenes=scenes,
            output_dir=str(output_dir), manifest=manifest,
            full_keyframe_count=len(frames), summary_keyframe_count=len(frames),
            selected_keyframe_count=len(frames),
        )


class WorldMMAdapter:
    """Execute the frozen Hybrid v2 pipeline, with explicit legacy fallback."""

    def __init__(self, root=None):
        self.repo_root = Path(__file__).resolve().parents[2]
        self.root = Path(root or self.repo_root / "tools" / "video_keyframe").resolve()
        self.script = self.root / "worldmm_keyframe_pipeline.py"
        self.hybrid_script = self.root / "katna" / "run_yolo_prefilter_event_webp.py"

    def run(self, video_path, video_id, output_dir):
        method = os.getenv("SENTRIX_VIDEO_METHOD", "hybrid_v2").strip().lower()
        if method in {"hybrid", "hybrid_v2", "hybrid_v2.1_eventagg", "eventagg", "eventagg_v21", "current"}:
            return self._run_hybrid(video_path, video_id, output_dir)
        return self._run_legacy(video_path, video_id, output_dir)

    def _run_legacy(self, video_path, video_id, output_dir):
        if not self.script.is_file():
            raise RuntimeError(f"vendored WorldMM pipeline is missing: {self.script}")
        output = Path(output_dir).resolve()
        output.mkdir(parents=True, exist_ok=True)
        yolo = Path(os.getenv("SENTRIX_VIDEO_YOLO_MODEL", self.root / "models/keyframe/yolo11n.pt"))
        pose = Path(os.getenv("SENTRIX_VIDEO_POSE_MODEL", self.root / "models/keyframe/yolo11n-pose.pt"))
        command = [
            os.getenv("SENTRIX_VIDEO_PYTHON", sys.executable), str(self.script),
            "--video", str(Path(video_path).resolve()), "--video-id", str(video_id),
            "--output", str(output), "--width", os.getenv("SENTRIX_VIDEO_WIDTH", "640"),
            "--sample-fps", os.getenv("SENTRIX_VIDEO_SAMPLE_FPS", "10"),
            "--analysis-fps", os.getenv("SENTRIX_VIDEO_ANALYSIS_FPS", "5"),
            "--device", os.getenv("SENTRIX_VIDEO_DEVICE", "cpu"),
            "--yolo-model", str(yolo), "--pose-model", str(pose),
        ]
        if os.getenv("SENTRIX_VIDEO_DISABLE_SEMANTICS", "0").lower() in {"1", "true", "yes"}:
            command.append("--disable-semantics")
        timeout = int(os.getenv("SENTRIX_VIDEO_TIMEOUT_SECONDS", "7200"))
        process = subprocess.run(command, check=False, capture_output=True, text=True, timeout=timeout)
        (output / "sentrix-worldmm.log").write_text(
            process.stdout + ("\nSTDERR\n" + process.stderr if process.stderr else ""), encoding="utf-8")
        if process.returncode:
            raise RuntimeError(f"WorldMM failed ({process.returncode}): {process.stderr.strip()[-2000:]}")
        return WorldMMResult.from_output(output)

    def _run_hybrid(self, video_path, video_id, output_dir):
        if not self.hybrid_script.is_file():
            raise RuntimeError(f"frozen hybrid keyframe pipeline is missing: {self.hybrid_script}")
        output = Path(output_dir).resolve() / "hybrid-v2"
        output.mkdir(parents=True, exist_ok=True)
        yolo = Path(os.getenv("SENTRIX_VIDEO_YOLO_MODEL", self.root / "models/keyframe/yolo11n.pt"))
        pose = Path(os.getenv("SENTRIX_VIDEO_POSE_MODEL", self.root / "models/keyframe/yolo11n-pose.pt"))
        command = [
            os.getenv("SENTRIX_VIDEO_PYTHON", sys.executable), str(self.hybrid_script),
            "--video", str(Path(video_path).resolve()), "--video-id", str(video_id),
            "--output", str(output), "--katna-root", str(self.root / "katna"),
            "--pipeline-root", str(self.root), "--yolo-model", str(yolo), "--pose-model", str(pose),
            "--scan-fps", os.getenv("SENTRIX_VIDEO_SCAN_FPS", "10"),
            "--yolo-batch-size", os.getenv("SENTRIX_VIDEO_YOLO_BATCH_SIZE", "16"),
            "--katna-scan-fps", os.getenv("SENTRIX_VIDEO_KATNA_SCAN_FPS", "10"),
            "--katna-unstable-percentile", os.getenv("SENTRIX_VIDEO_KATNA_UNSTABLE_PERCENTILE", "75"),
            "--merge-max-sec", os.getenv("SENTRIX_VIDEO_MERGE_MAX_SEC", "12"),
            "--target-decode-workers", os.getenv("SENTRIX_VIDEO_TARGET_DECODE_WORKERS", "4"),
            "--webp-quality", os.getenv("SENTRIX_VIDEO_WEBP_QUALITY", "80"),
            "--device", os.getenv("SENTRIX_VIDEO_DEVICE", "0"),
        ]
        timeout = int(os.getenv("SENTRIX_VIDEO_TIMEOUT_SECONDS", "7200"))
        process = subprocess.run(command, check=False, capture_output=True, text=True, timeout=timeout)
        (output / "sentrix-hybrid-v2.log").write_text(
            process.stdout + ("\nSTDERR\n" + process.stderr if process.stderr else ""), encoding="utf-8")
        if process.returncode:
            raise RuntimeError(f"Hybrid v2 failed ({process.returncode}): {process.stderr.strip()[-2000:]}")
        return HybridResult.from_output(output, str(video_path), str(video_id))
