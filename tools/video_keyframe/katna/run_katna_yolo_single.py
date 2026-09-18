#!/usr/bin/env python3
"""NVDEC-backed targeted Katna primitives for the hybrid keyframe method."""

from __future__ import annotations

import subprocess
import threading
from pathlib import Path

import sys
import cv2
import numpy as np

# 把本目录放进 sys.path：decode_strategy / extract_keyframes 是同目录模块，
# 但本文件既会被直接运行、也会被 run_yolo_prefilter_event_webp 导入、还会被
# 测试用 importlib 加载 —— 只有直接运行时脚本目录才自动在 sys.path 上。
_HERE = str(Path(__file__).resolve().parent)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from decode_strategy import detect_codec, select_strategy  # noqa: E402

_semaphores: dict[int, threading.Semaphore] = {}
_semaphore_lock = threading.Lock()


def _gate(limit: int) -> threading.Semaphore:
    """按并发上限取一个**进程内**信号量。

    为什么进程内就够：视频在导入管线里是**串行**处理的（`_process_ingest_asset_group`
    对非图片资产直接同步调用 process_asset），同一时刻只会有一个抽帧子进程，
    信号量用于约束它内部的解码线程池。跨进程上限由那个串行化保证——为此引入
    文件锁只会给并不存在的并发场景增加故障面。
    """
    with _semaphore_lock:
        semaphore = _semaphores.get(limit)
        if semaphore is None:
            semaphore = threading.Semaphore(limit)
            _semaphores[limit] = semaphore
        return semaphore


def _run_scan(command: list[str], frame_bytes: int, start: int, stride: int,
              fps: float, chunk_size: int, out_w: int, out_h: int,
              hanning_smooth, Candidate) -> list:
    """跑一次 ffmpeg 扫描窗口，返回候选帧。失败时抛 RuntimeError。"""
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    previous_luv = None
    images, indices, differences = [], [], []
    result: list = []

    def flush() -> None:
        nonlocal images, indices, differences
        if len(differences) >= 3:
            smooth = hanning_smooth(np.asarray(differences, dtype=np.float64), 20)
            maxima = np.where((smooth[1:-1] > smooth[:-2]) &
                              (smooth[1:-1] > smooth[2:]))[0] + 1
            for pos in maxima.tolist():
                if pos < len(images):
                    result.append(Candidate(indices[pos], indices[pos] / fps, images[pos]))
        images, indices, differences = [], [], []

    local_index = 0
    try:
        while True:
            raw = process.stdout.read(frame_bytes)
            if len(raw) != frame_bytes:
                break
            frame_index = start + local_index * stride
            local_index += 1
            frame = np.frombuffer(raw, dtype=np.uint8).reshape((out_h, out_w, 3)).copy()
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
        raise RuntimeError(f"hardware Katna scan failed ({process.returncode}): {stderr[-1000:]}")
    return result


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

    # 解码路径由这台机器**实测**的能力决定，不写死：
    # Jetson 走 nvv4l2dec，x86+NVIDIA 走 -hwaccel cuda，无 GPU 走软解。
    strategy = select_strategy(detect_codec(video))
    cpu_strategy = select_strategy("", available=frozenset())
    semaphore = _gate(strategy.concurrency)
    result: list = []

    def scan_window(start: int, end: int) -> None:
        duration = max(1.0 / max(float(fps), 0.1),
                       (end - start + 1) / max(float(fps), 0.1))
        base = [
            "ffmpeg", "-hide_banner", "-loglevel", "error",
            "-ss", f"{start / max(float(fps), 0.1):.6f}",
            "-t", f"{duration:.6f}",
            *strategy.input_args,
            "-i", str(video), "-an",
            # Decimate inside the decode window. Reading every raw frame and
            # closing the pipe early surfaced as a false Broken pipe.
            "-vf", f"{strategy.vf_prefix}scale={out_w}:{out_h},fps={float(scan_fps):g},format=bgr24",
            "-f", "rawvideo", "-pix_fmt", "bgr24", "pipe:1",
        ]
        semaphore.acquire()
        try:
            result.extend(_run_scan(base, frame_bytes, start, stride, fps,
                                    chunk_size, out_w, out_h, hanning_smooth, Candidate))
        except RuntimeError:
            if strategy.name == "cpu":
                raise
            # 显存被别的进程（VLM / 人脸）占满时硬件解码会失败；退回软解跑完这一窗，
            # 总比整段视频抽帧失败好。
            fallback = [
                "ffmpeg", "-hide_banner", "-loglevel", "error",
                "-ss", f"{start / max(float(fps), 0.1):.6f}",
                "-t", f"{duration:.6f}",
                *cpu_strategy.input_args,
                "-i", str(video), "-an",
                "-vf", f"{cpu_strategy.vf_prefix}scale={out_w}:{out_h},fps={float(scan_fps):g},format=bgr24",
                "-f", "rawvideo", "-pix_fmt", "bgr24", "pipe:1",
            ]
            result.extend(_run_scan(fallback, frame_bytes, start, stride, fps,
                                    chunk_size, out_w, out_h, hanning_smooth, Candidate))
        finally:
            semaphore.release()

    for start, end in normalized:
        scan_window(start, end)
    return sorted({item.frame_index: item for item in result}.values(),
                  key=lambda item: item.frame_index)
