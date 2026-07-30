#!/usr/bin/env python3
"""Build an auditable 120-image virtual family album from public LFW identities.

The images retain their LFW identity provenance. Capture times and locations
are synthetic test provenance; identity and event labels remain external
evaluation data and are never imported into Sentrix.
"""

import argparse
import hashlib
import json
import shutil
from datetime import datetime, timedelta, timezone
from pathlib import Path


IDENTITIES = ["George_W_Bush", "Colin_Powell", "Tony_Blair", "Donald_Rumsfeld"]

EVENTS = [
    ("evt_birthday", "2025-05-10T18:00:00+00:00", "家中餐厅", "生日庆祝", "父亲"),
    ("evt_repair", "2025-05-10T18:00:00+00:00", "家中餐厅", "家庭维修", "母亲"),
    ("evt_park", "2025-05-11T10:00:00+00:00", "城市公园", "公园散步", "哥哥"),
    ("evt_kitchen", "2025-05-12T12:00:00+00:00", "家中厨房", "午餐准备", "妹妹"),
    ("evt_trip", "2025-05-13T09:00:00+00:00", "湖边步道", "家庭出游", "父亲"),
    ("evt_portrait", "2025-05-14T16:00:00+00:00", "家中客厅", "家庭合影", "母亲"),
]


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build(source, output, per_identity=30):
    output.mkdir(parents=True, exist_ok=True)
    manifest = {
        "dataset": "LFW funneled subset",
        "dataset_source": "https://ndownloader.figshare.com/files/5976015",
        "license_note": "LFW is a public research benchmark; verify downstream use terms before redistribution.",
        "identity_count": len(IDENTITIES),
        "synthetic_metadata": True,
        "metadata_note": "event labels and identity labels are external evaluation data only; imported metadata contains provenance, capture time, and capture location only.",
        "events": [
            {"event_id": event_id, "captured_at": captured_at, "location": location, "activity": activity, "photographer": photographer}
            for event_id, captured_at, location, activity, photographer in EVENTS
        ],
        "assets": [],
    }
    files = []
    for source_identity in IDENTITIES:
        candidates = sorted((source / source_identity).glob("*.jpg"))[:per_identity]
        if len(candidates) < per_identity:
            raise ValueError(f"{source_identity} has only {len(candidates)} images")
        files.extend((source_identity, path) for path in candidates)

    base = datetime(2025, 5, 10, 18, 0, tzinfo=timezone.utc)
    for index, (source_identity, source_path) in enumerate(files):
        # Keep every event populated by all four fixed family identities while
        # making the first two events share time and place but differ in activity.
        event_index = (index // len(IDENTITIES)) % len(EVENTS)
        event_id, event_time, location, activity, photographer = EVENTS[event_index]
        target_name = f"asset_{index + 1:03d}.jpg"
        target = output / target_name
        shutil.copy2(source_path, target)
        captured = datetime.fromisoformat(event_time) + timedelta(minutes=index % 6)
        manifest["assets"].append({
            "file": target_name,
            "source_identity": source_identity,
            "captured_at": captured.isoformat(),
            "captured_location": location,
            "photographer": photographer,
            "activity": activity,
            "event_id": event_id,
            "source_sha256": sha256(source_path),
            "metadata_origin": "synthetic_virtual_album",
        })
    (output / "sentrix_metadata.json").write_text(json.dumps({
        item["file"]: {
            "captured_at": item["captured_at"],
            "captured_location": item["captured_location"],
            "source_owner_id": f"test-album-owner-{index % len(IDENTITIES) + 1}",
            "source_owner_label": f"测试相册归属成员 {index % len(IDENTITIES) + 1}",
            "source_device_id": "virtual-device-1",
            "source_album_id": f"test-album-{index // len(IDENTITIES) + 1}",
            "source_confidence": 1.0,
            "metadata_origin": item["metadata_origin"],
        }
        for index, item in enumerate(manifest["assets"])
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "virtual_album_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"assets": len(manifest["assets"]), "identities": len(IDENTITIES), "events": len(EVENTS), "output": str(output)}, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--per-identity", type=int, default=30)
    args = parser.parse_args()
    build(args.source, args.output, args.per_identity)
