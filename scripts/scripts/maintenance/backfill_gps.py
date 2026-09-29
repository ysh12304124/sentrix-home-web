"""Backfill EXIF GPS + PyGeoCN reverse_geocode for all existing image assets."""
import json
import os
import sqlite3
import sys
from pathlib import Path

DB_PATH = os.getenv("SENTRIX_DB_PATH", os.path.expanduser("~/Github/Sentrix-Home-Web/data/sentrix.db"))


def gps_value(value):
    try:
        return float(value[0]) / float(value[1])
    except (TypeError, ValueError, ZeroDivisionError, IndexError):
        return float(value)


def extract_exif_gps(path):
    try:
        from PIL import Image, ExifTags
        with Image.open(path) as img:
            raw = img.getexif()
            try:
                gps = raw.get_ifd(ExifTags.IFD.GPSInfo) or {}
            except Exception:
                gps = {}
            tags = {ExifTags.TAGS.get(k, str(k)): v for k, v in raw.items()}

        result = {}
        if tags.get("DateTimeOriginal") or tags.get("DateTime"):
            raw_time = tags.get("DateTimeOriginal") or tags.get("DateTime")
            normalized = str(raw_time).replace(":", "-", 2)
            offset = str(tags.get("OffsetTimeOriginal") or tags.get("OffsetTime") or "").strip()
            result["captured_at"] = normalized + (offset if offset.startswith(("+", "-")) else "")
        if tags.get("Model") or tags.get("Make"):
            result["device"] = tags.get("Model") or tags.get("Make")

        if not gps or not gps.get(2) or not gps.get(4):
            return result  # No GPS data in EXIF

        lat = sum(gps_value(item) / (60 ** idx) for idx, item in enumerate(gps[2]))
        lon = sum(gps_value(item) / (60 ** idx) for idx, item in enumerate(gps[4]))
        if str(gps.get(1, "N")).upper() == "S":
            lat *= -1
        if str(gps.get(3, "E")).upper() == "W":
            lon *= -1
        result["gps"] = {"latitude": lat, "longitude": lon}
        return result
    except Exception:
        return {}


def geocode(latitude, longitude):
    try:
        from PyGeoCN.regeo import regeo
        result = regeo(latitude, longitude)
        if result and result.get("status") == 1:
            address = result["address"]
            province = str(address.get("province") or "").strip()
            city = str(address.get("city") or "").strip()
            district = str(address.get("district") or "").strip()
            parts = [p for p in (province, city, district) if p]
            return {
                "source": "tianditu",
                "precision": "district" if district else "city",
                "label": "".join(parts),
                "province": province,
                "city": city,
                "district": district,
                "latitude": latitude,
                "longitude": longitude,
                "confidence": 0.90,
            }
    except Exception:
        pass
    return None


def main():
    db = sqlite3.connect(DB_PATH)
    db.row_factory = sqlite3.Row
    db.execute("PRAGMA journal_mode = WAL")

    assets = db.execute(
        "SELECT id, path, file_name, metadata_json FROM assets WHERE media_type = 'image'"
    ).fetchall()

    updated_gps = 0
    updated_geo = 0
    skipped_missing = 0
    skipped_no_gps = 0

    for row in assets:
        path = row["path"]
        if not path or not Path(path).exists():
            skipped_missing += 1
            continue

        metadata = json.loads(row["metadata_json"] or "{}")

        # Only backfill if GPS is missing
        has_gps = bool(metadata.get("exif", {}).get("gps") or metadata.get("gps"))
        if has_gps and "reverse_geocode" in metadata:
            continue  # Already complete

        exif = extract_exif_gps(path)

        if not exif.get("gps"):
            skipped_no_gps += 1
            continue

        # Merge EXIF data into metadata
        metadata.setdefault("exif", {})
        metadata["exif"].update(exif)
        updated_gps += 1

        # Geocode
        gps_data = exif["gps"]
        geo = geocode(gps_data["latitude"], gps_data["longitude"])
        if geo:
            metadata["reverse_geocode"] = geo
            if not metadata.get("captured_location"):
                metadata["captured_location"] = geo["label"]
            updated_geo += 1

        # Save
        db.execute(
            "UPDATE assets SET metadata_json = ?, captured_location = ? WHERE id = ?",
            (json.dumps(metadata, ensure_ascii=False), metadata.get("captured_location"), row["id"]),
        )

    db.commit()
    db.close()

    print(f"Total images scanned: {len(assets)}")
    print(f"GPS backfilled:     {updated_gps}")
    print(f"Geocoded:           {updated_geo}")
    print(f"Missing source:     {skipped_missing}")
    print(f"No EXIF GPS:        {skipped_no_gps}")
    print("DONE")


if __name__ == "__main__":
    main()
