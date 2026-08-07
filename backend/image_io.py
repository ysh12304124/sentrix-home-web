"""Shared image open helpers, including Apple HEIC/HEIF support."""

from __future__ import annotations

import mimetypes
from pathlib import Path

_HEIF_REGISTERED = False

HEIF_SUFFIXES = {".heic", ".heif"}
HEIF_MIME_TYPES = {
    ".heic": "image/heic",
    ".heif": "image/heif",
}


def ensure_heif_support() -> bool:
    """Register pillow-heif with Pillow once. Returns True when HEIC is available."""
    global _HEIF_REGISTERED
    if _HEIF_REGISTERED:
        return True
    try:
        from pillow_heif import register_heif_opener

        register_heif_opener()
        _HEIF_REGISTERED = True
        return True
    except Exception:
        return False


def guess_mime_type(path) -> str:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix in HEIF_MIME_TYPES:
        return HEIF_MIME_TYPES[suffix]
    return mimetypes.guess_type(path.name)[0] or "application/octet-stream"


def media_type_for_path(path) -> str:
    mime = guess_mime_type(path)
    if "/" in mime:
        return mime.split("/", 1)[0]
    suffix = Path(path).suffix.lower()
    if suffix in HEIF_SUFFIXES:
        return "image"
    return "text"


def media_type_from_upload(content_type: str | None, file_name: str | None) -> str:
    """Resolve upload media type; prefer MIME, fall back to filename for HEIC."""
    mime = (content_type or "").split(";", 1)[0].strip().lower()
    if mime.startswith("image/") or mime in {"image/heic", "image/heif"}:
        return "image"
    if mime.startswith("audio/"):
        return "audio"
    if mime.startswith("video/"):
        return "video"
    if mime.startswith("text/"):
        return "text"
    if file_name:
        return media_type_for_path(file_name)
    return "text"


def needs_browser_transcode(path, mime_type: str | None = None) -> bool:
    """Browsers generally cannot render Apple HEIC/HEIF natively."""
    suffix = Path(path).suffix.lower()
    mime = (mime_type or "").split(";", 1)[0].strip().lower()
    return suffix in HEIF_SUFFIXES or mime in {"image/heic", "image/heif"}


def encode_jpeg_preview(path, *, max_dimension: int = 1600, quality: int = 85) -> bytes:
    """Decode an image (including HEIC) and return browser-safe JPEG bytes."""
    from io import BytesIO

    from PIL import Image

    ensure_heif_support()
    with Image.open(path) as source:
        image = source.convert("RGB")
        if max(image.size) > max_dimension:
            image.thumbnail((max_dimension, max_dimension))
        output = BytesIO()
        image.save(output, format="JPEG", quality=quality, optimize=True)
        return output.getvalue()
