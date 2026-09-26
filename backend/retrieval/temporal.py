"""Helpers for treating asset timestamps as evidence only when trustworthy."""

from __future__ import annotations

import json


_UNTRUSTED_CAPTURE_SOURCES = {"file_mtime_fallback", "filesystem_mtime", "import_time"}


def _metadata(asset: dict | None) -> dict:
    value = (asset or {}).get("metadata_json") or {}
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except (TypeError, ValueError):
            return {}
    return value if isinstance(value, dict) else {}


def capture_time_source(asset: dict | None, store=None) -> str:
    """Return the capture-time provenance, consulting the parent for keyframes."""
    asset = asset or {}
    metadata = _metadata(asset)
    source = (metadata.get("capture_time_source")
              or metadata.get("captured_at_source")
              or metadata.get("creation_source"))
    video_metadata = metadata.get("video_metadata")
    if not source and isinstance(video_metadata, dict):
        source = video_metadata.get("creation_source")

    # Older derived frames did not persist their parent's time provenance.
    # Recover it from the source video so file modification times are not
    # mistaken for when the recording actually happened.
    if (not source and store is not None
            and asset.get("derived_kind") in {"video_keyframe", "video_keyframe_webp"}
            and asset.get("parent_asset_id")):
        try:
            parent = store.get_asset(asset["parent_asset_id"]) or {}
            parent_metadata = _metadata(parent)
            source = (parent_metadata.get("capture_time_source")
                      or parent_metadata.get("captured_at_source")
                      or parent_metadata.get("creation_source"))
            parent_video_metadata = parent_metadata.get("video_metadata")
            if not source and isinstance(parent_video_metadata, dict):
                source = parent_video_metadata.get("creation_source")
        except Exception:
            pass

    return str(source or "asset_metadata").strip().lower()


def trusted_captured_at(asset: dict | None, observation: dict | None = None, store=None):
    """Capture timestamp usable for hard filtering; unknown stays open-world."""
    asset = asset or {}
    if capture_time_source(asset, store) in _UNTRUSTED_CAPTURE_SOURCES:
        return None
    return asset.get("captured_at") or (observation or {}).get("captured_at")
