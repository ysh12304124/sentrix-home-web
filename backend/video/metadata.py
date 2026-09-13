from __future__ import annotations

import json
import re
import os
import shutil
import subprocess
from pathlib import Path

from .contracts import VideoMetadata


ISO6709_RE = re.compile(r"^([+-]\d+(?:\.\d+)?)([+-]\d+(?:\.\d+)?)([+-]\d+(?:\.\d+)?)?/?$")
COMPACT_TZ_RE = re.compile(r"([+-]\d{2})(\d{2})$")


_FFMPEG_BINARIES = {}


def resolve_ffmpeg_binary(name: str) -> str:
    """Resolve ffmpeg/ffprobe on Windows and Unix without changing the pipeline.

    The video pipeline intentionally shells out to FFmpeg.  On Windows a
    WinGet installation is not visible to an already-running Python process,
    so ``shutil.which`` alone leaves uploads stuck at ``video-metadata``.
    Prefer an explicit project setting, then PATH, then common WinGet/Scoop
    locations; returning the bare command preserves the normal OS lookup and
    keeps tests/mocks unchanged.
    """
    cached = _FFMPEG_BINARIES.get(name)
    if cached:
        return cached
    candidates = []
    configured = os.getenv("SENTRIX_FFMPEG_BIN", "").strip()
    if configured:
        configured_path = Path(configured)
        candidates.append(configured_path / f"{name}.exe" if configured_path.is_dir() else configured_path)
    found = shutil.which(name)
    if found:
        candidates.append(Path(found))
    if os.name == "nt":
        local = Path(os.getenv("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
        candidates.extend([
            local / "Microsoft" / "WinGet" / "Links" / f"{name}.exe",
            local / "Programs" / "ffmpeg" / "bin" / f"{name}.exe",
            Path.home() / "scoop" / "apps" / "ffmpeg" / "current" / "bin" / f"{name}.exe",
        ])
        packages = local / "Microsoft" / "WinGet" / "Packages"
        if packages.is_dir():
            # WinGet package layout is <package>/<archive>/bin/<binary>.exe.
            candidates.extend(packages.glob(f"*/*/bin/{name}.exe"))
    for candidate in candidates:
        try:
            if candidate.is_file():
                resolved = str(candidate.resolve())
                _FFMPEG_BINARIES[name] = resolved
                return resolved
        except OSError:
            continue
    _FFMPEG_BINARIES[name] = name
    return name


def _rate(value):
    try:
        numerator, denominator = str(value or "0/1").split("/", 1)
        return float(numerator) / float(denominator) if float(denominator) else 0.0
    except (TypeError, ValueError, ZeroDivisionError):
        return 0.0


def _location(tags):
    raw = (tags.get("com.apple.quicktime.location.ISO6709") or tags.get("location") or "").strip()
    match = ISO6709_RE.match(raw)
    if not match:
        return None, None
    return float(match.group(1)), float(match.group(2))


def _normalized_datetime(value):
    value = str(value or "").strip()
    if not value:
        return None
    value = COMPACT_TZ_RE.sub(r"\1:\2", value)
    return value[:-1] + "+00:00" if value.endswith("Z") else value


def probe_video_metadata(path: str | Path) -> VideoMetadata:
    path = Path(path).resolve()
    process = subprocess.run(
        [resolve_ffmpeg_binary("ffprobe"), "-v", "error", "-print_format", "json", "-show_format", "-show_streams", str(path)],
        check=False, capture_output=True, text=True, timeout=60,
    )
    if process.returncode:
        raise RuntimeError(f"ffprobe failed: {process.stderr.strip()[-1000:]}")
    payload = json.loads(process.stdout)
    format_info = payload.get("format") or {}
    format_tags = format_info.get("tags") or {}
    video = next((item for item in payload.get("streams") or [] if item.get("codec_type") == "video"), None)
    if not video:
        raise ValueError("ffprobe found no video stream")
    stream_tags = video.get("tags") or {}
    captured_at = (
        format_tags.get("com.apple.quicktime.creationdate")
        or format_tags.get("creation_time") or stream_tags.get("creation_time")
    )
    creation_source = (
        "com.apple.quicktime.creationdate" if format_tags.get("com.apple.quicktime.creationdate")
        else "format.creation_time" if format_tags.get("creation_time") else "stream.creation_time"
    )
    latitude, longitude = _location({**stream_tags, **format_tags})
    rotation = 0
    for item in video.get("side_data_list") or []:
        if "rotation" in item:
            rotation = int(float(item["rotation"]))
            break
    if not rotation:
        try:
            rotation = int(float(stream_tags.get("rotate") or 0))
        except ValueError:
            rotation = 0
    fps = _rate(video.get("avg_frame_rate")) or _rate(video.get("r_frame_rate"))
    duration = float(format_info.get("duration") or video.get("duration") or 0)
    device = " ".join(filter(None, [format_tags.get("com.apple.quicktime.make"), format_tags.get("com.apple.quicktime.model")])).strip()
    return VideoMetadata(
        captured_at=_normalized_datetime(captured_at),
        latitude=latitude, longitude=longitude,
        captured_location=f"{latitude:.6f},{longitude:.6f}" if latitude is not None else None,
        duration_sec=duration, fps=fps, width=int(video.get("width") or 0),
        height=int(video.get("height") or 0), codec=str(video.get("codec_name") or ""),
        rotation=rotation, device=device, creation_source=creation_source, raw=payload,
    )
