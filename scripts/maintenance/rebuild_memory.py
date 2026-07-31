#!/usr/bin/env python3
"""Discard derived Sentrix memory and rebuild it from a source directory."""

import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from backend.db import MemoryStore
from backend.model_clients import ClipAdapter, FaceAdapter, FunASRClient, GammaClient
from backend.pipeline import IngestionPipeline


SUPPORTED = {".jpg", ".jpeg", ".png", ".webp", ".gif", ".wav", ".mp3", ".m4a", ".flac", ".txt", ".md"}


def metadata_for_path(metadata, source, path):
    """Prefer a source-relative metadata key; retain flat-album compatibility."""
    if not isinstance(metadata, dict):
        return {}
    try:
        relative_path = path.relative_to(source).as_posix()
    except ValueError:
        relative_path = ""
    return metadata.get(relative_path) or metadata.get(path.name) or {}


def benchmark_imports(manifest, source_root=None, scope_id=None):
    """Yield only original benchmark files and allowlisted provenance fields."""
    root = Path(source_root or manifest.get("source_root", "")).expanduser().resolve()
    for space in manifest.get("spaces", []):
        if scope_id and space.get("scope_id") != scope_id:
            continue
        current_scope = space["scope_id"]
        for record in space.get("import", {}).get("files", []):
            relative_path = Path(record["relative_path"])
            path = root / relative_path
            if not path.is_file():
                raise FileNotFoundError(f"benchmark source file not found: {path}")
            yield current_scope, path, {
                "captured_at": record.get("captured_at"),
                "captured_location": record.get("captured_location"),
                "source_album_id": record.get("source_album_id") or current_scope,
                "scope_id": current_scope,
            }


def rebuild(root, source=None, benchmark_manifest=None, scope_id=None):
    face = FaceAdapter()
    if face.enabled and face.identity_model in {"adaface", "magface"} and not face.identity_configured:
        raise RuntimeError(
            f"{face.identity_model} identity embedding is not configured: "
            f"{face.identity_error or 'missing model configuration'}"
        )
    data_dir = root / "data"
    db_path = data_dir / "sentrix.db"
    media_dir = data_dir / "media"
    if db_path.exists():
        db_path.unlink()
    if media_dir.exists():
        shutil.rmtree(media_dir)
    media_dir.mkdir(parents=True, exist_ok=True)

    store = MemoryStore(str(db_path))
    manifest = None
    if benchmark_manifest:
        manifest = json.loads(Path(benchmark_manifest).read_text(encoding="utf-8"))
        source = Path(manifest.get("source_root", source or ""))
        files = list(benchmark_imports(manifest, scope_id=scope_id))
        for current_scope in sorted({item[0] for item in files}):
            store.create_memory_space(current_scope, current_scope, kind="benchmark", source_path=str(source / current_scope))
        run_scope = scope_id or "benchmark"
    else:
        source = Path(source)
        files = [("home-default", path, metadata_for_path(json.loads((source / "sentrix_metadata.json").read_text(encoding="utf-8")) if (source / "sentrix_metadata.json").is_file() else {}, source, path)) for path in sorted(source.rglob("*")) if path.is_file() and path.name != "sentrix_metadata.json" and path.suffix.lower() in SUPPORTED]
        run_scope = str(source)
    run = store.start_rebuild("sentrix-rebuild-v1", str(run_scope))
    pipeline = IngestionPipeline(store, gamma=GammaClient(), asr=FunASRClient(), face=face, clip=ClipAdapter())
    processed = 0
    failed = 0
    for current_scope, path, metadata in files:
        asset = pipeline.create_asset(path, metadata=metadata)
        result = pipeline.process(asset["id"], summarize_event=False)
        if result.get("status") == "failed":
            failed += 1
            print(f"FAILED {path}: {result.get('metadata_json', result)}")
        else:
            processed += 1
            print(f"OK {processed}/{len(files)} {path}")
    event_consolidation = {}
    event_summaries = {}
    for current_scope in sorted({item[0] for item in files}):
        event_consolidation[current_scope] = store.consolidate_events(current_scope)
        event_summaries[current_scope] = len(pipeline.summarize_events(current_scope))
    recluster = {current_scope: store.recluster_faces(scope_id=current_scope) for current_scope in sorted({item[0] for item in files})}
    stats = {"files": len(files), "processed": processed, "failed": failed, "assets": store.count("assets"), "observations": store.count("observations"), "events": store.count("events"), "event_consolidation": event_consolidation, "event_summaries": event_summaries, "entities": store.count("entities"), "clusters": store.count("face_clusters"), "facts": store.count("facts"), "recluster": recluster}
    store.finish_rebuild(run["id"], "completed" if failed == 0 else "completed_with_failures", stats)
    print(stats)
    store.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--source", type=Path, default=None)
    parser.add_argument("--benchmark-manifest", type=Path, default=None)
    parser.add_argument("--scope-id", default=None)
    args = parser.parse_args()
    source = args.source or (args.root / "data" / "test-albums")
    rebuild(args.root, source, args.benchmark_manifest, args.scope_id)
