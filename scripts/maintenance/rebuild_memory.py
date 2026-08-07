#!/usr/bin/env python3
"""Discard derived Sentrix memory and rebuild it from a source directory."""

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from backend.db import MemoryStore
from backend.image_io import HEIF_SUFFIXES, ensure_heif_support
from backend.model_clients import ClipAdapter, FaceAdapter, FunASRClient, GammaClient
from backend.pipeline import IngestionPipeline


SUPPORTED = {
    ".jpg", ".jpeg", ".png", ".webp", ".gif", ".heic", ".heif",
    ".wav", ".mp3", ".m4a", ".flac", ".txt", ".md",
} | HEIF_SUFFIXES


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


def rebuild(root, source=None, benchmark_manifest=None, scope_id=None, keep_db=False):
    ensure_heif_support()
    face = FaceAdapter()
    if face.enabled and face.identity_model in {"adaface", "magface"} and not face.identity_configured:
        raise RuntimeError(
            f"{face.identity_model} identity embedding is not configured: "
            f"{face.identity_error or 'missing model configuration'}"
        )
    data_dir = Path(os.getenv("SENTRIX_DATA_DIR", root / "data"))
    db_path = Path(os.getenv("SENTRIX_DB_PATH", data_dir / "sentrix.db"))
    media_dir = data_dir / "media"
    incremental = bool(benchmark_manifest and scope_id) or keep_db
    if not incremental:
        if db_path.exists():
            db_path.unlink()
        if media_dir.exists():
            shutil.rmtree(media_dir)
    media_dir.mkdir(parents=True, exist_ok=True)
    db_path.parent.mkdir(parents=True, exist_ok=True)

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
        source = Path(source).expanduser().resolve()
        if not source.is_dir():
            raise FileNotFoundError(f"source album directory not found: {source}")
        current_scope = scope_id or "home-default"
        store.create_memory_space(
            current_scope,
            current_scope,
            kind="household",
            source_path=str(source),
        )
        metadata_path = source / "sentrix_metadata.json"
        album_metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.is_file() else {}
        files = []
        for path in sorted(source.rglob("*")):
            if not path.is_file() or path.name == "sentrix_metadata.json":
                continue
            if path.suffix.lower() not in SUPPORTED:
                continue
            metadata = metadata_for_path(album_metadata, source, path)
            metadata.setdefault("scope_id", current_scope)
            metadata.setdefault("source_album_id", current_scope)
            files.append((current_scope, path, metadata))
        run_scope = current_scope
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
    purged_facts = {}
    for current_scope in sorted({item[0] for item in files}):
        event_consolidation[current_scope] = store.consolidate_events(current_scope)
        event_summaries[current_scope] = len(pipeline.summarize_events(current_scope))
        purged_facts[current_scope] = store.purge_unanchored_facts(current_scope)
    recluster = {current_scope: store.recluster_faces(scope_id=current_scope) for current_scope in sorted({item[0] for item in files})}
    stats = {"files": len(files), "processed": processed, "failed": failed, "assets": store.count("assets"), "observations": store.count("observations"), "events": store.count("events"), "event_consolidation": event_consolidation, "event_summaries": event_summaries, "purged_unanchored_facts": purged_facts, "entities": store.count("entities"), "clusters": store.count("face_clusters"), "facts": store.count("facts"), "recluster": recluster}
    store.finish_rebuild(run["id"], "completed" if failed == 0 else "completed_with_failures", stats)
    print(stats)
    store.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--source", type=Path, default=None, help="Album directory to scan (e.g. data/my-album)")
    parser.add_argument("--benchmark-manifest", type=Path, default=None)
    parser.add_argument("--scope-id", default=None, help="MemorySpace id for the scanned album")
    parser.add_argument(
        "--keep-db",
        action="store_true",
        help="Do not delete sentrix.db / data/media before scanning (safe for adding a new album)",
    )
    args = parser.parse_args()
    source = args.source or (args.root / "data" / "test-albums")
    rebuild(args.root, source, args.benchmark_manifest, args.scope_id, keep_db=args.keep_db)
