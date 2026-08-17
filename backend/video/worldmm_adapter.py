from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from .contracts import WorldMMResult


class WorldMMAdapter:
    """Execute the vendored WorldMM-a pipeline and parse its stable output."""

    def __init__(self, root=None):
        self.repo_root = Path(__file__).resolve().parents[2]
        self.root = Path(root or self.repo_root / "tools" / "video_keyframe").resolve()
        self.script = self.root / "worldmm_keyframe_pipeline.py"

    def run(self, video_path, video_id, output_dir):
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
