"""按这台机器**实测**的硬件解码能力选解码路径，不写死。

为什么必须有这一层
------------------
三台机器是三条不同的路，写死任何一条都会让另外两台失败或静默退化：

- **Jetson（46 / 118）**：ffmpeg 编了 `nvv4l2dec`，但**没有** `cuda` hwaccel
  （实测 `ffmpeg -hwaccels` 只列出 drm / opencl）。用 `-hwaccel cuda` 会直接失败。
- **x86 + NVIDIA（153）**：没有 `nvv4l2dec`，但有 `cuda` hwaccel，
  帧留在显存里，必须 `hwdownload` 取回。
- **无 GPU**：纯 CPU 软解。

映射表本身也不可信：表里写了 `av1_nvv4l2dec`，但 46 与 118 的 ffmpeg（4.4.2）
实际只编了 h264/hevc/mpeg2/mpeg4/vp8/vp9 六个。**必须拿实测列表求交**，
否则每个 av1 视频都会先走一次必然失败的硬解、再回退 CPU。

并发上限同样按平台给：Jetson 的 NVDEC 会话数是硬限制（Orin NX 16 路、AGX 8 路），
超发直接解码失败；x86 侧受显存限制但余量大。
"""

from __future__ import annotations

import functools
import subprocess
from dataclasses import dataclass
from pathlib import Path

_HW_SUFFIX = "_nvv4l2dec"

# 编解码名 → nvv4l2dec 解码器名。**只是候选**，实际用哪个要和实测列表求交。
_NVV4L2 = {
    "h264": "h264_nvv4l2dec",
    "avc": "h264_nvv4l2dec",
    "hevc": "hevc_nvv4l2dec",
    "h265": "hevc_nvv4l2dec",
    "vp9": "vp9_nvv4l2dec",
    "vp8": "vp8_nvv4l2dec",
    "mpeg4": "mpeg4_nvv4l2dec",
    "msmpeg4v3": "mpeg4_nvv4l2dec",
    "mpeg2video": "mpeg2_nvv4l2dec",
    "av1": "av1_nvv4l2dec",
}


@dataclass(frozen=True)
class DecodeStrategy:
    """一条解码路径。`input_args` 放在 `-i` 之前，`vf_prefix` 拼在 `-vf` 最前面。"""

    name: str
    input_args: tuple[str, ...]
    vf_prefix: str
    concurrency: int


_CPU = DecodeStrategy(name="cpu", input_args=(), vf_prefix="", concurrency=4)


@functools.lru_cache(maxsize=1)
def nvv4l2_decoders() -> frozenset[str]:
    """实测这台机器的 ffmpeg 有哪些 nvv4l2 解码器。探测失败返回空集（=走 CPU）。"""
    try:
        result = subprocess.run(
            ["ffmpeg", "-hide_banner", "-decoders"],
            capture_output=True, text=True, timeout=20,
        )
    except (OSError, subprocess.SubprocessError):
        return frozenset()
    found = set()
    for line in result.stdout.splitlines():
        for token in line.split():
            if token.endswith(_HW_SUFFIX):
                found.add(token)
    return frozenset(found)


@functools.lru_cache(maxsize=1)
def _is_jetson() -> bool:
    try:
        model = Path("/proc/device-tree/model").read_bytes().replace(b"\0", b"").decode("utf-8", "replace")
    except OSError:
        return False
    return "jetson" in model.lower()


@functools.lru_cache(maxsize=1)
def nvdec_concurrency() -> int:
    """NVDEC 并发上限（留出余量）。

    这是**设备级**硬限制，不是配置项：Orin NX 单引擎 16 路 → 留 4 路余量取 12；
    AGX Orin 8 路 → 留 1 路取 7。取不准时按最保守的 7。
    """
    try:
        model = Path("/proc/device-tree/model").read_bytes().replace(b"\0", b"").decode("utf-8", "replace").lower()
    except OSError:
        return 7
    if "orin" in model and "nx" in model:
        return 12
    return 7


def select_strategy(codec: str, available: frozenset[str] | None = None) -> DecodeStrategy:
    """为 *codec* 选一条真的能用的解码路径。

    顺序：实测可用的 nvv4l2dec → 实测可用的 cuda hwaccel → CPU 软解。
    绝不返回"表里有、这台机器上没有"的解码器。
    """
    decoders = nvv4l2_decoders() if available is None else available
    candidate = _NVV4L2.get(str(codec or "").strip().lower())
    if candidate and candidate in decoders:
        # nvv4l2dec 直接输出 NV12 软件帧，不需要 hwdownload。
        return DecodeStrategy(
            name="nvv4l2dec",
            input_args=("-c:v", candidate),
            vf_prefix="",
            concurrency=nvdec_concurrency(),
        )
    if _cuda_hwaccel_available():
        return DecodeStrategy(
            name="cuda_hwaccel",
            input_args=("-hwaccel", "cuda", "-hwaccel_output_format", "cuda"),
            # 帧留在显存，必须取回系统内存才能交给 numpy / OpenCV。
            vf_prefix="hwdownload,format=nv12,",
            concurrency=8,
        )
    return _CPU


@functools.lru_cache(maxsize=1)
def _cuda_hwaccel_available() -> bool:
    """ffmpeg 是否支持 `-hwaccel cuda`。Jetson 的 ffmpeg 只有 drm/opencl，不支持。"""
    try:
        result = subprocess.run(
            ["ffmpeg", "-hide_banner", "-hwaccels"],
            capture_output=True, text=True, timeout=20,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return "cuda" in result.stdout.lower().split()


def detect_codec(video) -> str:
    """用 ffprobe 取视频编码名（小写）。取不到返回空串 → 走 CPU 软解。"""
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=codec_name",
             "-of", "default=nw=1:nk=1", str(video)],
            capture_output=True, text=True, timeout=15,
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    return result.stdout.strip().lower()
