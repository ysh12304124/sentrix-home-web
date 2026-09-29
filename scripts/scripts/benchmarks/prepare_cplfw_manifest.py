#!/usr/bin/env python3
"""Build a CPLFW verification/clustering manifest from the marcelohaps HuggingFace dataset.

Reads train/metadata.csv (per-image identity) and pairs.csv (LFW 10-fold
protocol) and emits the same manifest schema used by ingest_face_benchmark.py and
evaluate_lfw_clusters.py. A --fold filter keeps the benchmark small and
high-signal while still exposing cross-pose pairs.
"""

import argparse
import csv
import json
from pathlib import Path


def build(source, metadata_path, pairs_path, fold):
    metadata = {}
    with open(metadata_path, encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            metadata[row["source_filename"]] = row["identity"]

    pairs = []
    missing = []
    with open(pairs_path, encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if fold is not None and row["fold_id"] != str(fold):
                continue
            files = [row["image_a_path"], row["image_b_path"]]
            absent = [f for f in files if not (source / f).is_file()]
            if absent:
                missing.extend(absent)
                continue
            pairs.append({
                "label": 1 if row["is_same"] == "1" else 0,
                "files": files,
            })

    involved = {path for pair in pairs for path in pair["files"]}
    assets = []
    for path in sorted(involved):
        name = Path(path).name
        identity = metadata.get(name)
        if identity is None:
            continue
        assets.append({"file": path, "source_identity": identity})

    manifest = {
        "dataset": "cplfw",
        "pair_semantics": "same_identity",
        "source_root": str(source.resolve()),
        "assets": assets,
        "pairs": pairs,
        "diagnostics": {
            "fold": fold,
            "pairs_kept": len(pairs),
            "same_pairs": sum(1 for p in pairs if p["label"] == 1),
            "diff_pairs": sum(1 for p in pairs if p["label"] == 0),
            "unique_images": len(involved),
            "assets_with_identity": len(assets),
            "identities": len({a["source_identity"] for a in assets}),
            "pairs_skipped_missing_images": len(set(missing)),
        },
    }
    return manifest


def main():
    parser = argparse.ArgumentParser(description="Generate a CPLFW manifest.")
    parser.add_argument("--source", type=Path, required=True, help="Root containing the images/ directory.")
    parser.add_argument("--metadata", type=Path, required=True, help="train/metadata.csv")
    parser.add_argument("--pairs", type=Path, required=True, help="pairs.csv")
    parser.add_argument("--fold", type=int, help="Only keep one protocol fold (1-10).")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    manifest = build(args.source, args.metadata, args.pairs, args.fold)
    args.out.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest["diagnostics"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
