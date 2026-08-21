#!/usr/bin/env python3
"""NVDEC-backed targeted Katna primitives for the hybrid keyframe method."""

from __future__ import annotations

import subprocess
from pathlib import Path

import cv2
import numpy as np


def gpu_katna_candidates(
    video: Path,
    resize_to: int,
    chunk_size: int,
    fps: float,
    width: int,
    height: int,
    scan_fps: float = 10.0,
    windows: list[tuple[int, int]] | None = None,
):
    """Run LUV/Hanning local maxima only inside requested frame windows.

    Each window is independently seeked with FFmpeg/NVDEC. Stable portions of
    a long source video therefore do not pass through the Katna stage.
    """
    from extract_keyframes import Candidate, hanning_smooth

    out_w = int(resize_to)
    out_h = max(1, round(height * out_w / max(width, 1)))
    if out_h % 2:
        out_h += 1
    frame_bytes = out_w * out_h * 3
    stride = max(1, int(np.ceil(float(fps) / max(float(scan_fps), 0.1))))
    normalized = sorted((max(0, int(start)), max(0, int(end)))
                        for start, end in (windows or []))
    if not normalized:
        return []
    result = []

    def scan_window(start: int, end: int) -> None:
        duration = max(1.0 / max(float(fps), 0.1),
                       (end - start + 1) / max(float(fps), 0.1))
        command = [
            "ffmpeg", "-hide_banner", "-loglevel", "error",
            "-ss", f"{start / max(float(fps), 0.1):.6f}",
            "-t", f"{duration:.6f}",
            "-hwaccel", "cuda", "-hwaccel_output_format", "cuda",
            "-i", str(video), "-an",
            # Decimate inside the NVDEC/FFmpeg window.  The previous helper
            # read every raw frame but closed the pipe as soon as the
            # synthetic index crossed ``end``, which surfaced as a false
            # Broken pipe from FFmpeg.
            "-vf", f"hwdownload,format=nv12,scale={out_w}:{out_h},fps={float(scan_fps):g},format=bgr24",
            "-f", "rawvideo", "-pix_fmt", "bgr24", "pipe:1",
        ]
        process = subprocess.Popen(command, stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE)
        previous_luv = None
        images, indices, differences = [], [], []

        def flush() -> None:
            nonlocal images, indices, differences
            if len(differences) >= 3:
                smooth = hanning_smooth(np.asarray(differences, dtype=np.float64), 20)
                maxima = np.where((smooth[1:-1] > smooth[:-2]) &
                                  (smooth[1:-1] > smooth[2:]))[0] + 1
                for pos in maxima.tolist():
                    if pos < len(images):
                        result.append(Candidate(indices[pos], indices[pos] / fps,
                                                images[pos]))
            images, indices, differences = [], [], []

        local_index = 0
        try:
            while True:
                raw = process.stdout.read(frame_bytes)
                if len(raw) != frame_bytes:
                    break
                frame_index = start + local_index * stride
                local_index += 1
                frame = np.frombuffer(raw, dtype=np.uint8).reshape(
                    (out_h, out_w, 3)).copy()
                luv = cv2.cvtColor(frame, cv2.COLOR_BGR2LUV)
                if previous_luv is not None:
                    differences.append(float(np.sum(cv2.absdiff(luv, previous_luv))))
                    images.append(frame)
                    indices.append(frame_index)
                    if len(differences) >= max(3, int(chunk_size)):
                        flush()
                previous_luv = luv
        finally:
            if process.stdout:
                process.stdout.close()
            stderr = process.stderr.read().decode("utf-8", errors="replace")
            process.wait()
        flush()
        if process.returncode != 0:
            raise RuntimeError(
                f"NVDEC targeted Katna scan failed ({process.returncode}): {stderr[-1000:]}"
            )

    for start, end in normalized:
        scan_window(start, end)
    return sorted({item.frame_index: item for item in result}.values(),
                  key=lambda item: item.frame_index)
