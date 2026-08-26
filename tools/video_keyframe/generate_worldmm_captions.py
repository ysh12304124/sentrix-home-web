#!/usr/bin/env python3
"""Batch-caption existing WebP representatives with the configured VLM."""

from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path

from backend.model_clients import GammaClient, parse_json_response


def image_payload(path):
    return {"base64": base64.b64encode(Path(path).read_bytes()).decode("ascii"), "mime_type": "image/webp"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--package", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()
    root = Path(args.package)
    frame_map = json.loads((root / "frame_map.json").read_text(encoding="utf-8"))
    semantic = json.loads((root / "semantic.json").read_text(encoding="utf-8"))
    semantic = semantic if isinstance(semantic, list) else list(semantic.get("frames") or [])
    by_index = {int(row.get("encoded_frame_index")): row for row in semantic if str(row.get("encoded_frame_index", "")).lstrip("-").isdigit()}
    frames = list(frame_map.get("frames") or [])
    items = []
    for index, frame in enumerate(frames):
        encoded = int(frame.get("encoded_frame_index", index))
        row = by_index.get(encoded, frame)
        event_id = str(row.get("event_id") or frame.get("event_id") or f"event_{encoded:05d}")
        path = root / "webp" / f"event_{event_id}.webp"
        if not path.is_file():
            candidates = sorted((root / "webp").glob(f"*{event_id}*.webp"))
            if not candidates:
                continue
            path = candidates[0]
        items.append({"event_id": event_id, "timestamp_sec": frame.get("source_timestamp_sec", 0), "image_path": str(path)})
    gamma = GammaClient()
    results = []
    batch_size = max(1, min(8, int(args.batch_size)))
    for start in range(0, len(items), batch_size):
        batch = items[start:start + batch_size]
        prompt = (
            "Analyze each attached video keyframe independently. Return JSON only as an array with exactly "
            f"{len(batch)} objects in the same order, fields: caption, objects (array), actions (array), setting. "
            "Use concise English descriptions grounded only in the image; do not infer names, speech, or time."
        )
        try:
            raw = gamma.chat(prompt, images=[image_payload(row["image_path"]) for row in batch], json_mode=True, role="parser")
            parsed = parse_json_response(raw)
            if isinstance(parsed, dict):
                parsed = parsed.get("items") or parsed.get("results") or [parsed]
            if not isinstance(parsed, list):
                parsed = []
        except Exception as error:
            parsed = [{"caption": "", "objects": [], "actions": [], "error": str(error)} for _ in batch]
        for row, description in zip(batch, parsed):
            description = description if isinstance(description, dict) else {}
            results.append({"event_id": row["event_id"], "timestamp_sec": row["timestamp_sec"], "caption": str(description.get("caption") or description.get("setting") or ""), "objects": [str(x) for x in description.get("objects") or []], "actions": [str(x) for x in description.get("actions") or []], "image_path": row["image_path"]})
        print(json.dumps({"done": min(start + batch_size, len(items)), "total": len(items)}, ensure_ascii=False), flush=True)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"count": len(results), "output": args.output}, ensure_ascii=False))


if __name__ == "__main__":
    main()
