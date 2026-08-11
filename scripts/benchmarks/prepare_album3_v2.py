#!/usr/bin/env python3
"""准备 album3-v2 导入清单：从 EXIF 提取 captured_at / captured_location(GPS)。

输出 manifest（旧版 import 契约）：
  {"version": 1, "source_root": ..., "spaces": [{"scope_id": "album3-v2", "import": {"files": [...]}}]}

用法:
  python prepare_album3_v2.py /Users/rm001/Downloads/album3 /tmp/album3-v2-manifest.json
"""
import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

try:
    from PIL import Image
    from PIL.ExifTags import TAGS
except ImportError:
    print("need Pillow", file=sys.stderr)
    sys.exit(2)

ALBUM_SCOPE = "album3-v2"


def _conv(value):
    if not value or not isinstance(value, tuple) or len(value) != 3:
        return None
    try:
        return float(value[0]) + float(value[1]) / 60 + float(value[2]) / 3600
    except (TypeError, ValueError):
        return None


def photo_meta(path):
    with Image.open(path) as img:
        exif = img.getexif()
    captured_at = None
    for key, name in TAGS.items():
        if name == "DateTime" and exif.get(key):
            try:
                captured_at = datetime.strptime(str(exif[key]), "%Y:%m:%d %H:%M:%S").isoformat()
            except ValueError:
                pass
    lat = lon = None
    try:
        gps = exif.get_ifd(0x8825)
        lat = _conv(gps.get(2))
        lon = _conv(gps.get(4))
        if gps.get(1) in ("S", "s"):
            lat = -lat if lat else None
        if gps.get(3) in ("W", "w"):
            lon = -lon if lon else None
    except Exception:
        pass
    captured_location = f"{lat:.6f},{lon:.6f}" if lat is not None and lon is not None else None
    return captured_at, captured_location


def prepare(album_root):
    root = Path(album_root).expanduser()
    photos_dir = root / "photos"
    if not photos_dir.is_dir():
        raise ValueError(f"photos dir not found: {photos_dir}")
    files = []
    for path in sorted(photos_dir.iterdir()):
        if not path.is_file():
            continue
        captured_at, captured_location = photo_meta(path)
        files.append({
            "file_name": path.name,
            "relative_path": f"photos/{path.name}",
            "captured_at": captured_at,
            "captured_location": captured_location,
            "source_album_id": ALBUM_SCOPE,
            "scope_id": ALBUM_SCOPE,
        })
    no_time = [f["file_name"] for f in files if not f["captured_at"]]
    no_gps = [f["file_name"] for f in files if not f["captured_location"]]
    manifest = {
        "version": 1,
        "source_root": str(root),
        "import_contract": "source image + EXIF capture time/location + album scope only",
        "spaces": [{"scope_id": ALBUM_SCOPE, "name": ALBUM_SCOPE,
                    "import": {"files": files}}],
    }
    return manifest, {"files": len(files), "no_time": no_time, "no_gps": no_gps}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("album_root", type=Path)
    ap.add_argument("output", type=Path)
    args = ap.parse_args()
    manifest, diag = prepare(args.album_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(diag, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
