#!/usr/bin/env python3
"""Migrate one processed Sentrix Lite scope into the native Sentrix store.

The Lite SQLite/Qdrant layout is not binary-compatible with the native store.
This tool therefore preserves source media, generated captions, confirmed
identity labels, and family relationships, while rebuilding native events and
CLIP/BGE/Qdrant vectors.  The resulting scope is deliberately marked as a
legacy import so it is not confused with a clean from-scratch benchmark run.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from backend.db import MemoryStore, now_iso
from backend.image_io import guess_mime_type
from backend.model_clients import ClipAdapter, FaceAdapter, FunASRClient, GammaClient
from backend.pipeline import IngestionPipeline
from backend.semantic_taxonomy import normalize_semantic_analysis


def json_object(value) -> dict:
    if isinstance(value, dict):
        return value
    try:
        parsed = json.loads(value or "{}")
    except (TypeError, json.JSONDecodeError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def first_text(values) -> str:
    for value in values or []:
        text = str(value or "").strip()
        if text:
            return text
    return ""


def lite_analysis(metadata: dict) -> dict:
    """Project a Lite caption into the native semantic-analysis contract."""
    caption = json_object(metadata.get("caption"))
    summary = str(caption.get("summary") or "").strip()
    # Two historical Lite responses were saved through the format-fallback
    # path.  One contains a truncated JSON object (and hundreds of repeated OCR
    # tokens) inside ``summary``. Recover the actual first summary string so a
    # malformed legacy response cannot poison event text or tokenizer input.
    if caption.get("format_fallback") and summary.startswith("{"):
        try:
            embedded = json.loads(summary)
        except json.JSONDecodeError:
            embedded = {}
            match = re.search(r'"summary"\s*:\s*"((?:\\.|[^"\\])*)"', summary)
            if match:
                try:
                    embedded["summary"] = json.loads(f'"{match.group(1)}"')
                except json.JSONDecodeError:
                    embedded["summary"] = match.group(1)
        if isinstance(embedded, dict):
            summary = str(embedded.get("summary") or summary[:800]).strip()
            for key in ("people", "objects", "activities", "places", "visible_text", "time_clues"):
                if embedded.get(key) and not caption.get(key):
                    caption[key] = embedded[key]
    activities = [str(item).strip() for item in caption.get("activities") or [] if str(item).strip()]
    visible_text = [str(item).strip() for item in caption.get("visible_text") or [] if str(item).strip()]
    analysis = {
        "caption": summary,
        "people": caption.get("people") or [],
        "objects": caption.get("objects") or [],
        "activity": "、".join(activities),
        "place": first_text(caption.get("places")),
        "ocr_text": "\n".join(visible_text),
        "event_type": first_text(activities),
        "facts": [],
        "confidence": 0.75,
        "model": f"migrated:{metadata.get('caption_model') or 'sentrix-lite'}",
        "detail": {
            "schema_version": 1,
            "caption": summary,
            "people": caption.get("people") or [],
            "objects": caption.get("objects") or [],
            "activity": "、".join(activities),
            "place": first_text(caption.get("places")),
            "ocr_text": "\n".join(visible_text),
            "time_clues": caption.get("time_clues") or [],
            "migration_source": "sentrix-lite",
        },
    }
    return normalize_semantic_analysis(analysis)


def lite_import_metadata(metadata: dict, scope_id: str, batch_id: str) -> dict:
    source = json_object(metadata.get("source_metadata"))
    capture_times = source.get("source_capture_times") or []
    captured_at = source.get("capture_datetime") or first_text(capture_times)
    captured_location = source.get("readable_location") or source.get("title")
    gps = source.get("gps_coordinates")
    result = {
        "scope_id": scope_id,
        "batch_id": batch_id,
        "source_album_id": scope_id,
        "captured_at": captured_at,
        "captured_location": captured_location,
    }
    if isinstance(gps, dict):
        result["gps"] = gps
    return {key: value for key, value in result.items() if value not in (None, "", {})}


def _legacy_text_summary(
    store: MemoryStore,
    pipeline: IngestionPipeline,
    asset: dict,
    analysis: dict,
    *,
    source_type: str,
    migration_note: str,
) -> dict:
    """Persist a Lite text summary when native visual processing is unavailable."""
    asset_id = asset["id"]
    store.update_asset(asset_id, "processing", {})
    analysis = dict(analysis)
    analysis["captured_at"] = asset.get("captured_at")
    analysis["source_type"] = source_type
    analysis["canonical"] = {
        key: analysis.get(key)
        for key in (
            "caption", "activity", "place", "scene_type", "semantic", "raw_labels",
            "people", "objects", "clothing", "spatial_relations", "ocr_text", "event_type",
        )
    }
    analysis["raw"] = {
        "legacy_sentrix_lite": True,
        "gamma": {key: value for key, value in analysis.items() if key not in {"raw", "canonical"}},
    }
    observation = store.add_observation(asset_id, analysis)
    event = store.merge_observation_into_event(observation)
    entity_ids = [
        item["id"] for item in store.maintain_observation_entities(observation["id"], event["id"])
    ]
    objects = " ".join(
        str(item.get("label") or item.get("primary") or "") if isinstance(item, dict) else str(item)
        for item in observation.get("objects") or []
    )
    text = " ".join(filter(None, [
        observation.get("caption"), observation.get("activity"), observation.get("place"),
        observation.get("ocr_text"), objects,
    ]))
    vector, model = pipeline._text_embed(text)
    metadata = {"asset_id": asset_id, "event_id": event["id"], "migration_note": migration_note}
    store.upsert_vector("episodic", "observation", observation["id"], vector, model, metadata)
    store.upsert_vector("semantic", "observation", observation["id"], vector, model, metadata)
    store.upsert_vector(
        "episodic", "event", event["id"], vector, model,
        {"observation_id": observation["id"], "migration_note": migration_note},
    )
    return store.update_asset(asset_id, "processed", {
        "observation_id": observation["id"], "event_id": event["id"],
        "entity_ids": entity_ids, "semantic_status": "migrated",
        "migration_source": "sentrix-lite", "migration_note": migration_note,
    })


def _identity_annotations(path: Path) -> dict[str, list[str]]:
    result: dict[str, list[str]] = {}
    if not path.is_file():
        return result
    for line in path.read_text(encoding="utf-8").splitlines():
        item = json_object(line)
        filename = str(item.get("filename") or "").strip()
        if not filename:
            continue
        face_ids = [
            str(person.get("face_id") or "").strip()
            for person in item.get("people") or [] if isinstance(person, dict)
        ]
        result[filename] = [value for value in face_ids if value]
    return result


def _migrate_people(
    lite: sqlite3.Connection,
    store: MemoryStore,
    lite_scope_id: str,
    scope_id: str,
    media_root: Path,
    new_assets: dict[str, dict],
) -> tuple[int, int, int]:
    people_by_face: dict[str, dict] = {}
    people_by_name: dict[str, dict] = {}
    for row in lite.execute(
        "SELECT entity_id, name, family_role, payload_json FROM people WHERE scope_id = ?",
        (lite_scope_id,),
    ):
        payload = json_object(row["payload_json"])
        entity = store.create_entity(
            row["name"], "person", "confirmed", row["family_role"], 1.0,
            "从 Sentrix Lite 迁移的已确认身份", scope_id=scope_id,
        )
        store.connection.execute(
            "UPDATE entities SET identity_state = 'stable', role_state = ?, name_state = 'confirmed' WHERE id = ?",
            ("confirmed" if row["family_role"] else "unknown", entity["id"]),
        )
        store.connection.commit()
        aliases = payload.get("aliases") or []
        store.set_person_aliases(entity["id"], aliases)
        store.maintain_entity_property(
            entity["id"], "migration_provenance",
            {"source": "sentrix-lite", "legacy_entity_id": row["entity_id"]},
            1.0, source="sentrix-lite-migration",
        )
        for reference in payload.get("reference_paths") or []:
            name = Path(str(reference)).name
            if name.lower().startswith("faceid_"):
                face_id = name.rsplit("_", 1)[-1].split(".", 1)[0]
                people_by_face[face_id] = entity
        people_by_name[str(row["name"])] = entity
        for alias in aliases:
            people_by_name[str(alias)] = entity

    annotations = _identity_annotations(media_root / "metadata" / "image_metadata.jsonl")
    mention_count = 0
    touched_observations = []
    for filename, face_ids in annotations.items():
        asset = new_assets.get(filename)
        if not asset:
            continue
        metadata = asset.get("metadata_json") or {}
        observation_id = metadata.get("observation_id")
        if not observation_id:
            continue
        for face_id in face_ids:
            entity = people_by_face.get(face_id)
            if not entity:
                continue
            store._link_confirmed_entity_mention(entity, observation_id, None, 1.0)
            mention_count += 1
        touched_observations.append(observation_id)
    if touched_observations:
        store._refresh_event_participants(list(dict.fromkeys(touched_observations)))

    relationship_count = 0
    for row in lite.execute(
        "SELECT payload_json FROM relationships WHERE scope_id = ? ORDER BY ordinal",
        (lite_scope_id,),
    ):
        item = json_object(row["payload_json"])
        subject = people_by_name.get(str(item.get("subject") or ""))
        object_entity = people_by_name.get(str(item.get("object") or ""))
        predicate = str(item.get("predicate") or "").strip()
        if not subject or not object_entity or not predicate:
            continue
        store.create_relationship(
            subject["id"], predicate, object_entity["id"], [],
            float(item.get("confidence") or 1.0), "active",
        )
        relationship_count += 1
    for entity in people_by_face.values():
        store.rebuild_person_memory(entity["id"])
    return len({item["id"] for item in people_by_face.values()}), mention_count, relationship_count


def _finalize_events(store: MemoryStore, pipeline: IngestionPipeline, scope_id: str) -> int:
    # Do not call the global ``consolidate_events`` pass here. The observations
    # were already assigned incrementally, while the global O(n^2) comparison
    # can hold SQLite's writer lock for many minutes on a 300+ item Windows
    # migration. A later maintenance run may consolidate them independently.
    events = store.list_events(5000, scope_id)
    for event in events:
        detail = store.get_event_detail(event["id"]) or {}
        observations = detail.get("observations") or []
        captions = list(dict.fromkeys(
            str(item.get("caption") or "").strip() for item in observations
            if str(item.get("caption") or "").strip()
        ))
        activities = list(dict.fromkeys(
            str(item.get("activity") or "").strip() for item in observations
            if str(item.get("activity") or "").strip()
        ))
        event_types = list(dict.fromkeys(
            str(item.get("event_type") or "").strip() for item in observations
            if str(item.get("event_type") or "").strip()
        ))
        day = str(event.get("time_start") or "")[:10]
        title = f"{day} 相册事件" if day else "迁移相册事件"
        activity = "、".join(activities[:3]) or "媒体记录"
        summary = "；".join(captions[:4]) or activity
        event_type = event_types[0] if event_types else "家庭记录"
        store.connection.execute(
            """UPDATE events SET title = ?, event_type = ?, activity = ?, summary = ?,
            revision = revision + 1, updated_at = ? WHERE id = ?""",
            (title, event_type, activity, summary[:800], now_iso(), event["id"]),
        )
        store.connection.commit()
        vector, model = pipeline._text_embed(" ".join([title, event_type, activity, summary]))
        store.upsert_vector(
            "episodic", "event", event["id"], vector, model,
            {"event_summary": True, "summary_model": "sentrix-lite-migration"},
        )
    return len(events)


def migrate(args) -> dict:
    lite_db = args.lite_db.expanduser().resolve()
    media_root = args.media_root.expanduser().resolve()
    if not lite_db.is_file():
        raise FileNotFoundError(lite_db)
    if not media_root.is_dir():
        raise FileNotFoundError(media_root)
    lite = sqlite3.connect(str(lite_db))
    lite.row_factory = sqlite3.Row
    try:
        rows = list(lite.execute(
            """SELECT id, file_name, media_type, status, metadata_json
            FROM assets WHERE scope_id = ? AND status = 'processed'
            AND media_type IN ('image', 'video') ORDER BY created_at, file_name""",
            (args.lite_scope_id,),
        ))
        if not rows:
            raise ValueError(f"no processed media found in Lite scope {args.lite_scope_id}")
        resolved = []
        missing = []
        for row in rows:
            subdir = "photos" if row["media_type"] == "image" else "videos"
            path = media_root / subdir / row["file_name"]
            if path.is_file():
                resolved.append((row, path))
            else:
                missing.append(str(path))
        if missing:
            raise FileNotFoundError(f"{len(missing)} migrated media files are missing; first: {missing[0]}")
        if args.dry_run:
            return {
                "scope_id": args.scope_id, "lite_scope_id": args.lite_scope_id,
                "assets": len(resolved),
                "images": sum(row["media_type"] == "image" for row, _ in resolved),
                "videos": sum(row["media_type"] == "video" for row, _ in resolved),
            }

        store = MemoryStore(str(args.db_path.expanduser().resolve()))
        gamma = GammaClient()
        gamma.bind_store(store)
        pipeline = IngestionPipeline(
            store, gamma=gamma, asr=FunASRClient(), face=FaceAdapter(), clip=ClipAdapter(),
        )
        batch_id = args.batch_id or f"lite-migration-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}"
        store.create_memory_space(
            args.scope_id, args.scope_name or f"{args.scope_id}（Lite 迁移）",
            kind="legacy-import", source_path=str(media_root), include_in_people=True,
        )
        store.create_ingest_batch(batch_id, args.scope_id)
        new_assets: dict[str, dict] = {}
        failed = []
        for index, (row, path) in enumerate(resolved, start=1):
            metadata = json_object(row["metadata_json"])
            import_metadata = lite_import_metadata(metadata, args.scope_id, batch_id)
            existing_row = store._row(
                "SELECT id FROM assets WHERE scope_id = ? AND file_name = ? ORDER BY created_at LIMIT 1",
                (args.scope_id, row["file_name"]),
            )
            asset = store.get_asset(existing_row["id"]) if existing_row else None
            if not asset:
                prepared = pipeline.prepare_asset(
                    path, file_name=row["file_name"], media_type=row["media_type"],
                    mime_type=guess_mime_type(path), metadata=import_metadata,
                )
                asset = pipeline.create_asset(path, prepared=prepared)
            if asset.get("status") in {"processing", "semantic_enriching", "failed"}:
                store.cleanup_asset_derivatives(asset["id"])
                asset = store.update_asset(asset["id"], "queued", {
                    "migration_recovered_from_status": asset.get("status"),
                })
            store.update_asset(asset["id"], asset.get("status") or "queued", {
                "migration_source": "sentrix-lite",
                "legacy_scope_id": args.lite_scope_id,
                "legacy_asset_id": row["id"],
                "legacy_caption_model": metadata.get("caption_model"),
            })
            analysis = lite_analysis(metadata)
            try:
                caption = json_object(metadata.get("caption"))
                if asset.get("status") == "processed":
                    saved = asset
                elif row["media_type"] == "image" and not caption.get("format_fallback"):
                    saved = pipeline.process(
                        asset["id"], summarize_event=False, image_analysis=analysis,
                    )
                else:
                    is_video = row["media_type"] == "video"
                    saved = _legacy_text_summary(
                        store, pipeline, store.get_asset(asset["id"]), analysis,
                        source_type="legacy_video_summary" if is_video else "legacy_image_summary",
                        migration_note="whole_video_only" if is_video else "format_fallback_visual_skipped",
                    )
                if saved.get("status") != "processed":
                    failed.append({"file_name": row["file_name"], "error": (saved.get("metadata_json") or {}).get("error")})
                new_assets[row["file_name"]] = saved
            except Exception as error:
                store.update_asset(asset["id"], "failed", {"error": str(error), "migration_source": "sentrix-lite"})
                failed.append({"file_name": row["file_name"], "error": str(error)})
            except SystemExit as error:
                message = f"dependency exited with code {error.code}"
                store.cleanup_asset_derivatives(asset["id"])
                store.update_asset(asset["id"], "failed", {"error": message, "migration_source": "sentrix-lite"})
                failed.append({"file_name": row["file_name"], "error": message})
            if index == 1 or index % 10 == 0 or index == len(resolved):
                print(f"migrated {index}/{len(resolved)} (failed={len(failed)})", flush=True)

        event_count = _finalize_events(store, pipeline, args.scope_id)
        person_count = mention_count = relationship_count = 0
        if args.include_identities:
            person_count, mention_count, relationship_count = _migrate_people(
                lite, store, args.lite_scope_id, args.scope_id, media_root, new_assets,
            )
        store.complete_ingest_batch(batch_id)
        if store.claim_ingest_batch_summary(batch_id):
            store.finish_ingest_batch(batch_id)
        store.update_ingest_batch_metadata(batch_id, {
            "migration": {
                "source": "sentrix-lite", "lite_db": str(lite_db),
                "lite_scope_id": args.lite_scope_id, "reused_generated_captions": True,
                "clean_benchmark_eligible": False,
            },
            "pipeline_metrics": {
                "status": "completed" if not failed else "completed_with_failures",
                "asset_count": len(resolved), "failed": len(failed),
            },
        })
        stats = {
            "scope_id": args.scope_id, "batch_id": batch_id,
            "assets": len(resolved), "processed": len(resolved) - len(failed),
            "failed": failed, "events": event_count, "people": person_count,
            "identity_mentions": mention_count, "relationships": relationship_count,
            "vectors": store._row(
                "SELECT COUNT(*) AS count FROM memory_vectors WHERE scope_id = ?", (args.scope_id,),
            )["count"],
        }
        store.close()
        try:
            from backend.qdrant_memory import close_qdrant_clients
            close_qdrant_clients()
        except Exception:
            pass
        return stats
    finally:
        lite.close()


def parse_args():
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lite-db", type=Path, required=True)
    parser.add_argument("--lite-scope-id", required=True)
    parser.add_argument("--media-root", type=Path, required=True)
    parser.add_argument("--db-path", type=Path, default=root / "data" / "sentrix.db")
    parser.add_argument("--scope-id", default="album3-max-video10-migrated")
    parser.add_argument("--scope-name", default="PhotoBench album3 混合版（Lite 迁移）")
    parser.add_argument("--batch-id", default=None)
    parser.add_argument("--include-identities", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(migrate(parse_args()), ensure_ascii=False, indent=2))
