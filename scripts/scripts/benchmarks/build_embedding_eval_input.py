#!/usr/bin/env python3
"""Build R1B Development-Set inputs from the real DB (run on 153).

Emits the JSON files ``evaluate_embedding_quality.py`` consumes:

  --images-json      [{"id": asset_id, "path": media_file_path}]
  --corpus-json      [{"id": observation_id, "text": caption|activity|...|event summary, "field": ...}]

The Development labels (which query text goes with which image / which
observation is the target) are authored separately as a label file; the
retrieval target for the visual set is resolved by file name so the label file
can stay human-authored and DB-independent.

This script never reads benchmark ground truth; it only walks the current DB.
"""

import argparse
import json
import os
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", default=os.getenv("SENTRIX_DB_PATH", "data/sentrix.db"))
    parser.add_argument("--media-root", default=os.getenv("SENTRIX_MEDIA_DIR", "data/media"))
    parser.add_argument("--images-json", default=None, help="output: candidate image paths")
    parser.add_argument("--corpus-json", default=None, help="output: candidate observation/event text")
    parser.add_argument("--event-text", action="store_true", help="also emit event summaries into corpus")
    args = parser.parse_args()

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from backend.db import MemoryStore

    store = MemoryStore(args.db)
    media_root = Path(args.media_root)

    if args.images_json:
        images = []
        for asset in store.list_assets(media_type="image", limit=100_000):
            # The canonical path lives on the assets.path column (benchmark
            # sources live under data/household-benchmark-source, not media/).
            path = asset.get("path") or asset.get("file_path") or str(media_root / (asset.get("file_name") or ""))
            if not Path(path).is_file():
                path = str(media_root / (asset.get("file_name") or ""))
            images.append({"id": asset["id"], "path": str(path)})
        Path(args.images_json).write_text(json.dumps(images, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"wrote {args.images_json} ({len(images)} images)")

    if args.corpus_json:
        corpus = []
        observations = store.list_observations(limit=100_000)
        for obs in observations:
            for field in ("caption", "activity", "place"):
                text = obs.get(field)
                if text:
                    corpus.append({"id": obs["id"], "text": str(text), "field": field, "asset_id": obs.get("asset_id")})
            for field, key in (("object", "objects_json"), ("clothing", "clothing_json")):
                try:
                    values = json.loads(obs.get(key) or "[]")
                except (TypeError, ValueError):
                    values = []
                for value in values:
                    if value:
                        corpus.append({"id": obs["id"], "text": str(value), "field": field, "asset_id": obs.get("asset_id")})
            if obs.get("ocr_text"):
                corpus.append({"id": obs["id"], "text": str(obs.get("ocr_text"))[:300], "field": "ocr", "asset_id": obs.get("asset_id")})
        if args.event_text:
            for event in store.connection.execute("SELECT * FROM events").fetchall():
                event = dict(event)
                if event.get("summary"):
                    corpus.append({"id": event["id"], "text": str(event["summary"]), "field": "event_summary"})
        Path(args.corpus_json).write_text(json.dumps(corpus, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"wrote {args.corpus_json} ({len(corpus)} records)")
    store.close()


if __name__ == "__main__":
    main()
