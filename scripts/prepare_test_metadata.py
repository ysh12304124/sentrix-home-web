#!/usr/bin/env python3
"""Create auditable source/time/location metadata for public image test batches."""

import argparse
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path


ALBUM_SOURCES = [
    ("test-album-owner-a", "测试相册归属成员 A", "test-device-a", "test-album-a"),
    ("test-album-owner-b", "测试相册归属成员 B", "test-device-b", "test-album-b"),
    ("test-album-owner-c", "测试相册归属成员 C", "test-device-c", "test-album-c"),
    ("test-album-owner-d", "测试相册归属成员 D", "test-device-d", "test-album-d"),
]


def exif_time(path):
    try:
        from PIL import Image
        value = Image.open(path).getexif().get(36867) or Image.open(path).getexif().get(306)
        if value:
            return datetime.strptime(str(value), "%Y:%m:%d %H:%M:%S").replace(tzinfo=timezone.utc).isoformat()
    except (OSError, ValueError):
        pass
    return None


def build_metadata(source):
    files = sorted(path for path in source.iterdir() if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"})
    base = datetime(2025, 5, 10, 18, 0, tzinfo=timezone.utc)
    groups = [
        (range(0, 18), "家中餐厅", base),
        (range(18, 30), "家中客厅", base + timedelta(hours=3)),
        (range(30, 43), "城市公园", base + timedelta(days=1)),
        (range(43, len(files)), "家附近", base + timedelta(days=3)),
    ]
    result = {}
    for index, path in enumerate(files):
        location, group_time = "家附近", base
        for indexes, candidate_location, candidate_time in groups:
            if index in indexes:
                location, group_time = candidate_location, candidate_time
                break
        owner_id, owner_label, device_id, album_id = ALBUM_SOURCES[index % len(ALBUM_SOURCES)]
        captured_at = exif_time(path)
        if not captured_at:
            captured_at = (group_time + timedelta(minutes=(index % 6) * 3)).isoformat()
        result[path.name] = {
            "source_owner_id": owner_id,
            "source_owner_label": owner_label,
            "source_device_id": device_id,
            "source_album_id": album_id,
            "source_confidence": 1.0,
            "captured_at": captured_at,
            "captured_location": location,
            "metadata_origin": "exif_time" if exif_time(path) else "synthetic_time_location_for_clustering",
        }
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    output = args.output or args.source / "sentrix_metadata.json"
    output.write_text(json.dumps(build_metadata(args.source), ensure_ascii=False, indent=2), encoding="utf-8")
    print({"source": str(args.source), "output": str(output), "assets": len(json.loads(output.read_text(encoding="utf-8")))})
