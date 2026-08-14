from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class VideoMetadata:
    captured_at: str | None = None
    latitude: float | None = None
    longitude: float | None = None
    captured_location: str | None = None
    duration_sec: float = 0.0
    fps: float = 0.0
    width: int = 0
    height: int = 0
    codec: str = ""
    rotation: int = 0
    device: str = ""
    creation_source: str = ""
    raw: dict[str, Any] = field(default_factory=dict)

    def as_dict(self):
        return asdict(self)


@dataclass
class WorldMMKeyframe:
    code: str
    path: str
    timestamp_sec: float
    frame_index: int
    score: float = 0.0
    selection_reason: str = ""
    objects: list[dict[str, Any]] = field(default_factory=list)
    actions: list[dict[str, Any]] = field(default_factory=list)
    expressions: list[dict[str, Any]] = field(default_factory=list)
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass
class WorldMMScene:
    scene_id: str
    index: int
    start_sec: float
    end_sec: float
    keyframes: list[WorldMMKeyframe] = field(default_factory=list)
    semantic_labels: list[str] = field(default_factory=list)
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass
class WorldMMResult:
    video: dict[str, Any]
    scenes: list[WorldMMScene]
    output_dir: str
    manifest: dict[str, Any] = field(default_factory=dict)
    full_keyframe_count: int = 0
    summary_keyframe_count: int = 0
    selected_keyframe_count: int = 0

    @property
    def keyframe_count(self):
        return sum(len(scene.keyframes) for scene in self.scenes)

    @classmethod
    def from_output(cls, output_dir: str | Path) -> "WorldMMResult":
        root = Path(output_dir).resolve()
        memory_path = root / "memory.json"
        if not memory_path.is_file():
            raise ValueError(f"WorldMM memory.json is missing: {memory_path}")
        memory = json.loads(memory_path.read_text(encoding="utf-8"))
        frames = memory.get("memory_keyframes")
        scenes_payload = memory.get("scenes")
        if not isinstance(frames, list) or not isinstance(scenes_payload, list):
            raise ValueError("WorldMM memory.json has no memory_keyframes/scenes arrays")
        frame_by_code = {str(item.get("keyframe_code")): item for item in frames if isinstance(item, dict)}

        # WorldMM keeps a complete evidence set in memory.json.  That set is
        # intentionally dense (often about one frame per second) and is not a
        # sensible number of images to send through Sentrix's image pipeline.
        # The package's research/summary_keyframes.json is the recommended
        # downstream set, followed by a product-level budget for very long or
        # highly dynamic videos.
        summary_path = root / "research" / "summary_keyframes.json"
        summary = []
        if summary_path.is_file():
            try:
                payload = json.loads(summary_path.read_text(encoding="utf-8"))
                if isinstance(payload, list):
                    summary = [item for item in payload if isinstance(item, dict)]
            except (OSError, json.JSONDecodeError):
                summary = []
        summary_codes = [str(item.get("keyframe_code")) for item in summary if item.get("keyframe_code") in frame_by_code]
        candidate_codes = summary_codes or list(frame_by_code)
        try:
            max_keyframes = int(os.getenv("SENTRIX_VIDEO_MAX_KEYFRAMES", "160"))
        except ValueError:
            max_keyframes = 160

        selected_codes = set(candidate_codes)
        if max_keyframes > 0 and len(candidate_codes) > max_keyframes:
            # Keep the highest-scoring summary frame from each scene first so
            # scene coverage survives the global budget. Fill the remainder
            # in temporal order, which keeps the resulting timeline coherent.
            scene_picks = []
            score_by_code = {
                str(item.get("keyframe_code")): float(item.get("summary_score") or item.get("information_gain") or 0)
                for item in summary
            }
            for raw_scene in scenes_payload:
                scene_codes = [str(code) for code in (raw_scene.get("keyframe_codes") or []) if str(code) in candidate_codes]
                if scene_codes:
                    scene_picks.append(max(scene_codes, key=lambda code: score_by_code.get(code, 0.0)))
            selected = list(dict.fromkeys(scene_picks))
            if len(selected) > max_keyframes:
                stride = len(selected) / max_keyframes
                selected = [selected[min(int(i * stride), len(selected) - 1)] for i in range(max_keyframes)]
            selected_set = set(selected)
            remaining = [code for code in candidate_codes if code not in selected_set]
            slots = max_keyframes - len(selected)
            if slots > 0 and remaining:
                stride = len(remaining) / slots
                selected.extend(remaining[min(int(i * stride), len(remaining) - 1)] for i in range(slots))
            selected_codes = set(selected)
        scenes = []
        for index, raw_scene in enumerate(scenes_payload):
            if not isinstance(raw_scene, dict):
                raise ValueError(f"WorldMM scene {index} is not an object")
            keyframes = []
            for code in raw_scene.get("keyframe_codes") or []:
                if str(code) not in selected_codes:
                    continue
                raw = frame_by_code.get(str(code))
                if raw is None:
                    raise ValueError(f"WorldMM scene references unknown keyframe: {code}")
                relative = raw.get("source_image") or raw.get("image_path")
                if not relative:
                    raise ValueError(f"WorldMM keyframe has no source image: {code}")
                image_path = Path(relative)
                if not image_path.is_absolute():
                    image_path = root / image_path
                if not image_path.is_file():
                    raise ValueError(f"WorldMM keyframe image is missing: {image_path}")
                keyframes.append(WorldMMKeyframe(
                    code=str(code), path=str(image_path.resolve()),
                    timestamp_sec=float(raw.get("timestamp") or raw.get("time_sec") or 0),
                    frame_index=int(raw.get("frame_index") or 0),
                    score=float(raw.get("information_gain") or raw.get("summary_score") or 0),
                    selection_reason=str(raw.get("selection_reason") or ""),
                    objects=list(raw.get("objects") or []), actions=list(raw.get("actions") or []),
                    expressions=list(raw.get("expressions") or []), raw=raw,
                ))
            scenes.append(WorldMMScene(
                scene_id=str(raw_scene.get("scene_id") or f"scene_{index + 1:04d}"),
                index=index, start_sec=float(raw_scene.get("start_time") or 0),
                end_sec=float(raw_scene.get("end_time") or 0), keyframes=keyframes,
                semantic_labels=list(raw_scene.get("semantic_labels") or []), raw=raw_scene,
            ))
        manifest_path = root / "semantic" / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.is_file() else {}
        return cls(
            video=dict(memory.get("video") or {}), scenes=scenes, output_dir=str(root), manifest=manifest,
            full_keyframe_count=len(frames), summary_keyframe_count=len(summary),
            selected_keyframe_count=sum(len(scene.keyframes) for scene in scenes),
        )
