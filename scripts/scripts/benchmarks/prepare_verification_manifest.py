#!/usr/bin/env python3
"""Generate a small, high-signal face-verification manifest from an LFW-style protocol.

Supported datasets:
  - cplfw     cross-pose pairs (frontal/profile). label=1 means the SAME person.
  - kinfacew  kinship pairs (FS/FD/MS/MD). label=1 means blood relatives, NOT the
              same person -- deliberately hard negatives for a family-album product.

The manifest keeps only the requested subset of protocol pairs, so the benchmark
stays small while still exposing the pose-varied positives and kinship hard
negatives that decide whether the embedding / clustering rules hold. Identity and
pair annotations live only in the manifest; ingest_face_benchmark.py never writes
them into SQLite.
"""

import argparse
import json
from pathlib import Path


def _is_filename(value):
    return "." in value or "/" in value


def _label(value):
    return 1 if value.strip().lower() in {"1", "same", "kin", "kinship", "true", "related"} else 0


def _coerce_pair(tokens):
    def idx(value):
        return value.isdigit()

    if len(tokens) == 2:
        return tokens[0], tokens[1], 1
    if len(tokens) == 3 and _is_filename(tokens[0]) and _is_filename(tokens[1]):
        return tokens[0], tokens[1], _label(tokens[2])
    if len(tokens) == 3 and idx(tokens[1]) and idx(tokens[2]):
        return (tokens[0], int(tokens[1])), (tokens[0], int(tokens[2])), 1
    if len(tokens) == 4 and idx(tokens[1]) and idx(tokens[3]):
        return (tokens[0], int(tokens[1])), (tokens[2], int(tokens[3])), 0
    return None


def _resolve_ref(image_dir, ref):
    if isinstance(ref, tuple):
        name, index = ref
        candidates = [
            image_dir / f"{name}_{index:04d}.jpg",
            image_dir / name / f"{name}_{index:04d}.jpg",
            image_dir / f"{name}_{index}.jpg",
        ]
    else:
        candidates = [image_dir / ref, image_dir / ref.replace("_", "/")]
    return next((candidate for candidate in candidates if candidate.is_file()), None)


def build(source, pairs_path, dataset, max_pairs):
    image_dir = Path(source)
    if not image_dir.is_dir():
        raise SystemExit(f"source directory not found: {image_dir}")
    parsed = []
    for tokens in (line.split() for line in pairs_path.read_text(encoding="utf-8").splitlines()):
        if not tokens or tokens[0].startswith("#"):
            continue
        pair = _coerce_pair(tokens)
        if pair:
            parsed.append(pair)
    if not parsed:
        raise SystemExit(f"no pairs parsed from {pairs_path}")
    assets = {}
    pairs = []
    missing = []
    kept = 0
    for ref_a, ref_b, same in parsed:
        if max_pairs is not None and kept >= max_pairs:
            break
        image_a = _resolve_ref(image_dir, ref_a)
        image_b = _resolve_ref(image_dir, ref_b)
        if image_a is None or image_b is None:
            missing.append([str(ref_a), str(ref_b)])
            continue
        pairs.append({
            "label": int(same),
            "files": [image_a.relative_to(image_dir).as_posix(), image_b.relative_to(image_dir).as_posix()],
        })
        for image, ref in ((image_a, ref_a), (image_b, ref_b)):
            rel = image.relative_to(image_dir).as_posix()
            if rel not in assets:
                identity = (
                    Path(image.name).stem
                    if dataset == "kinfacew"
                    else (ref[0] if isinstance(ref, tuple) else Path(str(ref)).stem)
                )
                assets[rel] = identity
        kept += 1
    return {
        "dataset": dataset,
        "pair_semantics": "kinship" if dataset == "kinfacew" else "same_identity",
        "source_root": str(image_dir.resolve()),
        "assets": [{"file": rel, "source_identity": identity} for rel, identity in sorted(assets.items())],
        "pairs": pairs,
        "diagnostics": {
            "pairs_parsed": len(parsed),
            "pairs_kept": len(pairs),
            "same_pairs": sum(1 for item in pairs if item["label"] == 1),
            "diff_pairs": sum(1 for item in pairs if item["label"] == 0),
            "unique_identities": len(set(assets.values())),
            "missing_pair_images": len(missing),
        },
    }


def main():
    parser = argparse.ArgumentParser(description="Generate a small face-verification manifest.")
    parser.add_argument("--source", type=Path, required=True, help="Directory containing dataset images.")
    parser.add_argument("--pairs", type=Path, required=True, help="LFW-style pairs.txt protocol file.")
    parser.add_argument("--dataset", choices=["cplfw", "kinfacew"], required=True)
    parser.add_argument("--out", type=Path, required=True, help="Output manifest JSON path.")
    parser.add_argument("--max-pairs", type=int, help="Keep only the first N protocol pairs.")
    args = parser.parse_args()
    manifest = build(args.source, args.pairs, args.dataset, args.max_pairs)
    args.out.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest["diagnostics"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
