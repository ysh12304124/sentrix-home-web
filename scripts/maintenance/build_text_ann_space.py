#!/usr/bin/env python3
"""Build a standalone bge-m3 text ANN space (Phase R9-4 shadow).

Embeds every observation with the same text assembly the pipeline uses
(``backend/pipeline.py``), via the bge-m3 sidecar HTTP API, then writes a
standalone hnswlib index + manifest into ``data/ann/{space}`` with
``model_id=BAAI/bge-m3`` / ``dimension=1024``.  It never touches the runtime
``memory_vectors`` table — the index is a derived shadow artifact.

Run after the sidecar is up:
  PYTHONPATH=. .venv-mac/bin/python scripts/maintenance/build_text_ann_space.py
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def observation_text(obs) -> str:
    clothing = " ".join(str(item) for item in (obs.get("clothing") or []))
    facts = obs.get("facts") or []
    fact_text = " ".join(
        f"{f.get('subject', '')} {f.get('predicate', '')} {f.get('object', '')}"
        for f in facts if isinstance(f, dict)
    )
    detail = obs.get("detail") or {}
    detail_text = " ".join(
        str(item.get("text") or item.get("label") or item)
        for item in (detail.get("visible_details") or [])
    ) if isinstance(detail, dict) else ""
    return " ".join(filter(None, [
        obs.get("caption"), obs.get("activity"), obs.get("place"),
        obs.get("ocr_text"), obs.get("transcript"), clothing, fact_text, detail_text,
    ]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", default=os.getenv("SENTRIX_DB_PATH", "data/sentrix.db"))
    parser.add_argument("--ann-dir", default=str(REPO_ROOT / "data" / "ann"))
    parser.add_argument("--spaces", default="text_bge_semantic,text_bge_episodic")
    parser.add_argument("--embedder-url", default=os.getenv("SENTRIX_TEXT_EMBEDDER_URL", "http://127.0.0.1:8101"))
    parser.add_argument("--model-id", default=os.getenv("SENTRIX_TEXT_EMBED_MODEL", "BAAI/bge-m3"))
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    sys.path.insert(0, str(REPO_ROOT))
    import httpx
    from backend.db import MemoryStore
    from backend.retrieval_ann import create_index

    store = MemoryStore(args.db)
    observations = store.list_observations(limit=args.limit or 1_000_000)
    ann_dir = Path(args.ann_dir)
    ann_dir.mkdir(parents=True, exist_ok=True)
    spaces = [space.strip() for space in args.spaces.split(",") if space.strip()]

    buckets = {space: [] for space in spaces}
    skipped = 0
    for obs in observations:
        text = observation_text(obs)
        if not text.strip():
            skipped += 1
            continue
        try:
            resp = httpx.post(f"{args.embedder_url}/embed", json={"text": text}, timeout=30)
            resp.raise_for_status()
            vector = resp.json()["vector"]
        except Exception as exc:
            print(f"embed failed for observation {obs.get('id')}: {exc}")
            continue
        asset_id = obs.get("asset_id")
        event_id = obs.get("event_id")
        for space in spaces:
            if space.endswith("episodic") and event_id:
                buckets[space].append((event_id, vector, {"asset_id": asset_id, "event_id": event_id}))
            else:
                buckets[space].append((obs.get("id"), vector, {"asset_id": asset_id, "event_id": event_id}))

    summary = {}
    for space, rows in buckets.items():
        if not rows:
            summary[space] = {"count": 0}
            continue
        dimension = len(rows[0][1])
        index = create_index("hnswlib", dim=dimension, M=16, ef_construction=200, ef_search=50)
        index.add(rows)
        index.set_manifest_extra(model_id=args.model_id, source_type="observation")
        base = ann_dir / space
        index.save(str(base))
        summary[space] = {"count": len(rows), "dim": dimension, "model": args.model_id}
        print(f"built {space}: {len(rows)} vectors -> {base}.hnsw (dim {dimension})")

    print(json.dumps({"summary": summary, "observations_scanned": len(observations),
                      "skipped_empty": skipped}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
