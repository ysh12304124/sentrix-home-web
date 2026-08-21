import json
import hashlib
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

from .db import MemoryStore, make_id
from .geocoding import OfflineReverseGeocoder
from .image_io import ensure_heif_support, guess_mime_type, media_type_for_path
from .model_clients import ClipAdapter, FaceAdapter, FunASRClient, GammaClient, ModelError, normalize_confidence
from .semantic_taxonomy import normalize_semantic_analysis

ensure_heif_support()


IMPORT_METADATA_KEYS = {
    "content_sha256", "sha256", "exif", "captured_at", "captured_location",
    "source_owner_id", "source_owner_label", "source_device_id", "source_album_id",
    "source_confidence", "scope_id", "batch_id", "gps", "import_timings",
}


def file_time(path):
    return datetime.fromtimestamp(Path(path).stat().st_mtime, timezone.utc).isoformat()


class IngestionPipeline:
    def __init__(self, store, gamma=None, asr=None, face=None, clip=None, geocoder=None):
        self.store = store
        self.gamma = gamma or GammaClient()
        self.asr = asr or FunASRClient()
        self.face = face or FaceAdapter()
        self.clip = clip or ClipAdapter()
        self.geocoder = geocoder or OfflineReverseGeocoder()
        from .video import VideoMemoryAdapter
        self.video_memory_adapter = VideoMemoryAdapter()

    def create_asset(self, path, file_name=None, media_type=None, mime_type=None, metadata=None):
        path = Path(path)
        asset_id = make_id("asset")
        media_type = media_type or self._media_type(path)
        # Import metadata is provenance only. Model hints and benchmark labels
        # must never enter the memory graph through the asset boundary.
        metadata = {key: value for key, value in (metadata or {}).items() if key in IMPORT_METADATA_KEYS}
        import_timings = dict(metadata.get("import_timings") or {})
        step_started = time.perf_counter()
        metadata.setdefault("content_sha256", self._sha256(path))
        import_timings["sha256_seconds"] = round(time.perf_counter() - step_started, 4)
        step_started = time.perf_counter()
        metadata.setdefault("exif", self._extract_exif(path) if media_type == "image" else {})
        import_timings["exif_seconds"] = round(time.perf_counter() - step_started, 4)
        existing = self.store.find_asset_by_hash(metadata["content_sha256"], metadata.get("scope_id"))
        if existing:
            return existing
        for key in ("captured_at", "captured_location", "source_device_id"):
            if metadata.get(key) is None and metadata["exif"].get(key):
                metadata[key] = metadata["exif"][key]
        gps = self._gps_from_metadata(metadata)
        if gps:
            # Keep the raw GPS coordinate as the event-clustering location
            # anchor (the original logic); reverse_geocode stays a display-only
            # semantic place and must not overwrite the coordinate.
            if not metadata.get("captured_location"):
                metadata["captured_location"] = f"{float(gps['latitude']):.6f},{float(gps['longitude']):.6f}"
            if "reverse_geocode" not in metadata:
                step_started = time.perf_counter()
                location_context = self.geocoder.lookup(gps)
                import_timings["reverse_geocode_seconds"] = round(time.perf_counter() - step_started, 4)
                if location_context:
                    metadata["reverse_geocode"] = location_context
        metadata["import_timings"] = import_timings
        step_started = time.perf_counter()
        created = self.store.create_asset(
            asset_id,
            file_name or path.name,
            media_type,
            str(path),
            mime_type or guess_mime_type(path),
            path.stat().st_size,
            metadata,
            scope_id=metadata.get("scope_id"),
        )
        import_timings["database_create_seconds"] = round(time.perf_counter() - step_started, 4)
        import_timings["asset_create_seconds"] = round(sum(import_timings.values()), 4)
        return self.store.update_asset(created["id"], created.get("status") or "queued", {"import_timings": import_timings})

    @staticmethod
    def _gps_from_metadata(metadata):
        exif = metadata.get("exif") or {}
        gps = metadata.get("gps") or exif.get("gps")
        if isinstance(gps, dict):
            return gps
        if isinstance(gps, (list, tuple)) and len(gps) >= 2:
            return {"latitude": gps[0], "longitude": gps[1]}
        if isinstance(gps, str):
            parts = gps.replace(",", " ").split()
            if len(parts) >= 2:
                return {"latitude": parts[0], "longitude": parts[1]}
        return None

    @staticmethod
    def _sha256(path):
        digest = hashlib.sha256()
        with path.open("rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    @staticmethod
    def _extract_exif(path):
        try:
            ensure_heif_support()
            from PIL import Image, ExifTags
            with Image.open(path) as image:
                raw = image.getexif()
                tags = {ExifTags.TAGS.get(key, str(key)): value for key, value in raw.items()}
                try:
                    gps = raw.get_ifd(ExifTags.IFD.GPSInfo) or {}
                except Exception:
                    gps = tags.get("GPSInfo") or {}
            result = {"device": tags.get("Model") or tags.get("Make")}
            if tags.get("DateTimeOriginal") or tags.get("DateTime"):
                raw_time = tags.get("DateTimeOriginal") or tags.get("DateTime")
                normalized = str(raw_time).replace(":", "-", 2)
                offset = str(tags.get("OffsetTimeOriginal") or tags.get("OffsetTime") or "").strip()
                result["captured_at"] = normalized + (offset if offset.startswith(("+", "-")) else "")
            def gps_value(value):
                try:
                    return float(value[0]) / float(value[1])
                except (TypeError, ValueError, ZeroDivisionError, IndexError):
                    return float(value)
            if gps:
                latitude = gps.get(2)
                longitude = gps.get(4)
                if latitude and longitude:
                    lat = sum(gps_value(item) / (60 ** index) for index, item in enumerate(latitude))
                    lon = sum(gps_value(item) / (60 ** index) for index, item in enumerate(longitude))
                    if str(gps.get(1, "N")).upper() == "S":
                        lat *= -1
                    if str(gps.get(3, "E")).upper() == "W":
                        lon *= -1
                    result["gps"] = {"latitude": lat, "longitude": lon}
            return {key: value for key, value in result.items() if value}
        except Exception:
            return {}

    def _media_type(self, path):
        return media_type_for_path(path)

    def process(self, asset_id, summarize_event=True, forced_event_id=None, image_analysis=None):
        asset = self.store.get_asset(asset_id)
        if not asset:
            raise KeyError(asset_id)
        if asset.get("status") == "processed":
            metadata = asset.get("metadata_json") or {}
            return asset
        if asset["media_type"] == "video":
            return self.video_memory_adapter.process(asset, self)
        self.store.update_asset(asset_id, "processing", {})
        started_at = time.perf_counter()
        try:
            if asset["media_type"] == "image":
                result = self._image_observation(asset, precomputed_analysis=image_analysis)
            elif asset["media_type"] == "audio":
                result = self._audio_observation(asset)
            else:
                result = self._text_observation(asset)
            observation = self.store.add_observation(asset_id, result)
            # Event selection needs the image vector before an event exists.
            self.store.upsert_vector(
                "visual", "asset", asset_id, result.get("clip_embedding"), self.clip.model_name,
                {"observation_id": observation["id"]},
            )
            cluster_ids = []
            for face in result.get("face_candidates", []):
                face_instance = self.store.add_face_instance(asset_id, observation["id"], face)
                if face_instance:
                    cluster_ids.append(face_instance["cluster_id"])
            # A matching confirmed face writes an EntityMention above, so it is
            # available as a real identity bridge during this event decision.
            observation = self.store.get_observation(observation["id"])
            event = (
                self.store.attach_observation_to_event(forced_event_id, observation["id"])
                if forced_event_id else self.store.merge_observation_into_event(observation)
            )
            entity_ids = [item["id"] for item in self.store.maintain_observation_entities(observation["id"], event["id"])]
            # Update the same vector with its final event link after selection.
            self.store.upsert_vector(
                "visual", "asset", asset_id, result.get("clip_embedding"), self.clip.model_name,
                {"observation_id": observation["id"], "event_id": event["id"]},
            )
            fact_text = " ".join(f"{item.get('subject', '')} {item.get('predicate', '')} {item.get('object', '')}" for item in result.get("facts", []))
            clothing_text = " ".join(str(item) for item in (observation.get("clothing") or []))
            text_embedding = self.clip.embed_text(" ".join(filter(None, [observation.get("caption"), observation.get("activity"), observation.get("place"), observation.get("ocr_text"), observation.get("transcript"), clothing_text, fact_text])))
            self.store.upsert_vector("episodic", "observation", observation["id"], text_embedding, self.clip.model_name, {"asset_id": asset_id, "event_id": event["id"]})
            self.store.upsert_vector("episodic", "event", event["id"], text_embedding, self.clip.model_name, {"observation_id": observation["id"]})
            self.store.upsert_vector("semantic", "observation", observation["id"], text_embedding, self.clip.model_name, {"asset_id": asset_id, "event_id": event["id"]})
            fact_ids = []
            scope_id = observation.get("scope_id") or (self.store.get_asset(asset_id) or {}).get("scope_id")
            for fact in result.get("facts", []):
                if all(fact.get(key) for key in ("subject", "predicate", "object")) and self.store.is_confirmed_person_name(fact.get("subject"), scope_id):
                    saved = self.store.maintain_fact(fact["subject"], fact["predicate"], fact["object"], [observation["id"]], normalize_confidence(fact.get("confidence"), result.get("confidence", 0.5)), scope_id=scope_id)
                    fact_ids.append(saved["id"])
            metadata = {"observation_id": observation["id"], "event_id": event["id"], "fact_ids": fact_ids, "cluster_ids": cluster_ids, "entity_ids": entity_ids, "model": result.get("model"), "faces": [{key: value for key, value in face.items() if key != "embedding"} for face in result.get("face_candidates", [])], "processing_timings": result.get("processing_timings", {}), "processing_seconds": round(time.perf_counter() - started_at, 4)}
            saved_asset = self.store.update_asset(asset_id, "processed", metadata)
            if asset.get("source_owner_id"):
                self.store.rebuild_person_memory(asset["source_owner_id"])
            if summarize_event:
                self.summarize_event(event["id"])
            try:
                self.store.auto_confirm_clusters(scope_id)
            except Exception:
                pass
            return saved_asset
        except Exception as error:
            self.store.cleanup_asset_derivatives(asset_id)
            return self.store.update_asset(asset_id, "failed", {"error": str(error)})

    def process_fast_image(self, asset_id):
        """Persist immediately useful image evidence without claiming semantic completion."""
        asset = self.store.get_asset(asset_id)
        if not asset:
            raise KeyError(asset_id)
        if asset.get("media_type") != "image":
            raise ValueError("fast processing is only available for images")
        if asset.get("status") in {"processed", "semantic_enriching"}:
            return asset
        started_at = time.perf_counter()
        self.store.update_asset(asset_id, "processing", {})
        try:
            prepared = self.prepare_fast_image(asset_id)
            return self.commit_fast_image(asset_id, prepared, started_at=started_at)
        except Exception as error:
            self.store.cleanup_asset_derivatives(asset_id)
            return self.store.update_asset(asset_id, "failed", {"error": str(error)})

    def prepare_fast_image(self, asset_id):
        """Run face and image embedding inference without writing clustering state."""
        asset = self.store.get_asset(asset_id)
        if not asset:
            raise KeyError(asset_id)
        if asset.get("media_type") != "image":
            raise ValueError("fast processing is only available for images")
        path = asset["path"]
        started_at = time.perf_counter()

        def timed(callable_):
            step_started = time.perf_counter()
            return callable_(), time.perf_counter() - step_started

        with ThreadPoolExecutor(max_workers=2, thread_name_prefix="sentrix-fast") as executor:
            face_future = executor.submit(timed, lambda: self.face.detect(path))
            clip_future = executor.submit(timed, lambda: self.clip.embed_image(path))
            faces, face_seconds = face_future.result()
            clip_embedding, clip_seconds = clip_future.result()
        return {
            "captured_at": asset.get("captured_at") or file_time(path),
            "faces": faces,
            "clip_embedding": clip_embedding,
            "timings": {
                "face_detection_seconds": round(face_seconds, 4),
                "image_clip_seconds": round(clip_seconds, 4),
                "fast_inference_wall_seconds": round(time.perf_counter() - started_at, 4),
                "single_image_parallelism": 2,
            },
        }

    def commit_fast_image(self, asset_id, prepared, started_at=None):
        """Commit prepared face/CLIP evidence in deterministic image order."""
        asset = self.store.get_asset(asset_id)
        if not asset:
            raise KeyError(asset_id)
        started_at = started_at or time.perf_counter()
        commit_started = time.perf_counter()
        faces = prepared["faces"]
        clip_embedding = prepared["clip_embedding"]
        captured_at = prepared["captured_at"]
        step_started = time.perf_counter()
        observation = self.store.add_observation(asset_id, {
            "source_type": "image_fast_evidence", "caption": "", "captured_at": captured_at,
            "place": asset.get("captured_location") or "", "confidence": 0.0,
            "source_owner_id": asset.get("source_owner_id"), "canonical": {"semantic_status": "pending"},
            "raw": {"fast_evidence": True, "face_candidates": [{key: value for key, value in face.items() if key != "embedding"} for face in faces]},
        })
        observation_write_seconds = time.perf_counter() - step_started
        step_started = time.perf_counter()
        self.store.upsert_vector("visual", "asset", asset_id, clip_embedding, self.clip.model_name, {"observation_id": observation["id"]})
        visual_vector_seconds = time.perf_counter() - step_started
        step_started = time.perf_counter()
        cluster_ids = []
        face_matches = []
        for face in faces:
            instance = self.store.add_face_instance(asset_id, observation["id"], face)
            if instance:
                if instance.get("cluster_id"):
                    cluster_ids.append(instance["cluster_id"])
                face_matches.append({
                    "face_instance_id": instance.get("id"),
                    "cluster_id": instance.get("cluster_id"),
                    "match_score": round(float(instance.get("score") or 0), 4),
                    "quality": round(float(face.get("quality") or 0), 4),
                    "identity_eligible": face.get("identity_eligible", True) is not False,
                })
        face_clustering_seconds = time.perf_counter() - step_started
        observation = self.store.get_observation(observation["id"])
        step_started = time.perf_counter()
        event = self.store.merge_observation_into_event(observation)
        event_clustering_seconds = time.perf_counter() - step_started
        step_started = time.perf_counter()
        entity_ids = [item["id"] for item in self.store.maintain_observation_entities(observation["id"], event["id"])]
        entity_maintenance_seconds = time.perf_counter() - step_started
        self.store.upsert_vector("visual", "asset", asset_id, clip_embedding, self.clip.model_name, {"observation_id": observation["id"], "event_id": event["id"]})
        timings = dict(prepared.get("timings") or {})
        timings.update({
            "observation_write_seconds": round(observation_write_seconds, 4),
            "visual_vector_write_seconds": round(visual_vector_seconds, 4),
            "face_clustering_seconds": round(face_clustering_seconds, 4),
            "event_clustering_seconds": round(event_clustering_seconds, 4),
            "entity_maintenance_seconds": round(entity_maintenance_seconds, 4),
            "fast_commit_seconds": round(time.perf_counter() - commit_started, 4),
            "fast_processing_seconds": round(time.perf_counter() - started_at, 4),
        })
        return self.store.update_asset(asset_id, "semantic_enriching", {
            "observation_id": observation["id"], "event_id": event["id"], "cluster_ids": cluster_ids,
            "entity_ids": entity_ids, "semantic_status": "pending", "face_matches": face_matches,
            "processing_timings": timings,
            "fast_processing_seconds": timings["fast_processing_seconds"],
        })

    def enrich_fast_image(self, asset_id, summarize_event=True):
        """Complete an explicitly pending image observation with Gemma semantics."""
        started_at = time.perf_counter()
        prepared = self.prepare_semantic_image(asset_id)
        return self.commit_semantic_image(
            asset_id, prepared, summarize_event=summarize_event, started_at=started_at)

    def prepare_semantic_image(self, asset_id):
        """Run VLM and text embedding work without mutating memory state."""
        asset = self.store.get_asset(asset_id)
        metadata = (asset or {}).get("metadata_json") or {}
        if not asset or asset.get("status") != "semantic_enriching" or not metadata.get("observation_id"):
            raise ValueError("asset is not awaiting semantic enrichment")
        started_at = time.perf_counter()
        vision_started = time.perf_counter()
        analysis = self.gamma.analyze_image(asset["path"], {
            "file_name": asset["file_name"], "captured_at": asset.get("captured_at") or file_time(asset["path"]),
            "captured_location": asset.get("captured_location") or "", "source_owner_id": asset.get("source_owner_id"),
            "location_context": metadata.get("reverse_geocode") or {},
        })
        vision_seconds = time.perf_counter() - vision_started
        analysis = normalize_semantic_analysis(analysis)
        analysis["canonical"] = {key: analysis.get(key) for key in ("caption", "activity", "place", "scene_type", "semantic", "raw_labels", "people", "objects", "clothing", "emotions", "spatial_relations", "ocr_text", "event_type")}
        analysis["location_context"] = metadata.get("reverse_geocode") or {}
        analysis["raw"] = {"gamma": {key: value for key, value in analysis.items() if key != "location_context"}, "location_context": analysis["location_context"], "semantic_status": "complete"}
        objects = analysis.get("objects") or []
        object_text = " ".join(
            item if isinstance(item, str) else " ".join(str(item.get(key, "")) for key in ("label", "primary", "details") if item.get(key))
            for item in objects
        )
        text = " ".join(filter(None, [analysis.get("caption"), analysis.get("activity"), analysis.get("place"), analysis.get("ocr_text"), object_text]))
        embedding_started = time.perf_counter()
        embedding = self.clip.embed_text(text)
        embedding_seconds = time.perf_counter() - embedding_started
        return {
            "analysis": analysis,
            "embedding": embedding,
            "timings": {
                "vlm_image_description_seconds": round(vision_seconds, 4),
                "text_embedding_seconds": round(embedding_seconds, 4),
                "semantic_inference_wall_seconds": round(time.perf_counter() - started_at, 4),
            },
        }

    def commit_semantic_image(self, asset_id, prepared, summarize_event=False, started_at=None):
        """Commit prepared semantic data after ordered face/event clustering."""
        asset = self.store.get_asset(asset_id)
        metadata = (asset or {}).get("metadata_json") or {}
        if not asset or asset.get("status") != "semantic_enriching" or not metadata.get("observation_id"):
            raise ValueError("asset is not awaiting semantic enrichment")
        started_at = started_at or time.perf_counter()
        commit_started = time.perf_counter()
        observation_id = metadata["observation_id"]
        analysis = prepared["analysis"]
        step_started = time.perf_counter()
        observation = self.store.enrich_observation(observation_id, analysis, source="deferred_vision_enrichment")
        observation_enrichment_seconds = time.perf_counter() - step_started
        event_id = metadata.get("event_id")
        step_started = time.perf_counter()
        entity_ids = [item["id"] for item in self.store.maintain_observation_entities(observation_id, event_id)]
        entity_maintenance_seconds = time.perf_counter() - step_started
        embedding = prepared["embedding"]
        step_started = time.perf_counter()
        self.store.upsert_vector("episodic", "observation", observation_id, embedding, self.clip.model_name, {"asset_id": asset_id, "event_id": event_id})
        self.store.upsert_vector("semantic", "observation", observation_id, embedding, self.clip.model_name, {"asset_id": asset_id, "event_id": event_id})
        if event_id:
            self.store.upsert_vector("episodic", "event", event_id, embedding, self.clip.model_name, {"observation_id": observation_id})
            # Fast-path events keep placeholder place ("其他或不确定"); sync the
            # visual place after enrichment so later participant refreshes do not
            # rewrite summaries with a blank location.
            display_place = self.store._event_display_place(observation)
            if display_place:
                current = self.store.get_event(event_id) or {}
                current_place = str(current.get("place") or "").strip()
                if current_place in {"", "其他或不确定", "某处", "地点未知", "未标注", "未标注地点", "待判断"}:
                    self.store.update_event(event_id, {"place": display_place})
        vector_write_seconds = time.perf_counter() - step_started
        event_summary_seconds = 0.0
        if event_id and summarize_event:
            step_started = time.perf_counter()
            self.summarize_event(event_id)
            event_summary_seconds = time.perf_counter() - step_started
        timings = dict(metadata.get("processing_timings") or {})
        timings.update(prepared.get("timings") or {})
        timings.update({
            "observation_enrichment_seconds": round(observation_enrichment_seconds, 4),
            "semantic_entity_maintenance_seconds": round(entity_maintenance_seconds, 4),
            "semantic_vector_write_seconds": round(vector_write_seconds, 4),
            "event_summary_seconds": round(event_summary_seconds, 4),
            "semantic_commit_seconds": round(time.perf_counter() - commit_started, 4),
            "semantic_enrichment_seconds": round(time.perf_counter() - started_at, 4),
        })
        return self.store.update_asset(asset_id, "processed", {
            "semantic_status": "complete", "entity_ids": entity_ids,
            "processing_timings": timings,
            "semantic_enrichment_seconds": timings["semantic_enrichment_seconds"],
        })

    def summarize_event(self, event_id):
        detail = self.store.get_event_detail(event_id)
        if not detail or not detail["observations"] or not hasattr(self.gamma, "summarize_event"):
            return self.store.get_event(event_id)

        def fallback_projection():
            event = detail["event"]
            observations = detail["observations"]
            place = str(event.get("place") or "其他或不确定").strip()
            activities = []
            captions = []
            event_types = []
            for observation in observations:
                for value, target in ((observation.get("activity"), activities), (observation.get("caption"), captions), (observation.get("event_type"), event_types)):
                    value = str(value or "").strip()
                    if value and value not in target and value not in {"待判断", "未识别", "未知"}:
                        target.append(value)
            activity = "、".join(activities[:3]) or "图片记录"
            event_type = event_types[0] if event_types else "家庭活动"
            title = f"{place}{activity}" if place != "其他或不确定" else activity
            summary = f"{place}记录了" + "；".join(captions[:4] or activities[:4])
            return {
                "title": title[:20], "event_type": event_type[:20], "activity": activity[:20],
                "summary": summary[:240], "confidence": 0.45, "model": "deterministic_event_fallback",
            }

        try:
            result = self.gamma.summarize_event(detail["event"], detail["observations"])
            if str(result.get("title") or "").strip() in {"待总结事件", "待确认的家庭记录", "待判断"}:
                result = fallback_projection()
            updated = self.store.update_event(event_id, {
                "title": result.get("title"),
                "event_type": result.get("event_type"),
                "activity": result.get("activity"),
                "summary": result.get("summary"),
            })
            event_text = " ".join(filter(None, [updated.get("title"), updated.get("event_type"), updated.get("activity"), updated.get("summary")]))
            vector = self.clip.embed_text(event_text)
            self.store.upsert_vector("episodic", "event", event_id, vector, self.clip.model_name, {"summary_model": result.get("model"), "event_summary": True})
            return updated
        except Exception:
            result = fallback_projection()
            updated = self.store.update_event(event_id, {
                "title": result["title"], "event_type": result["event_type"],
                "activity": result["activity"], "summary": result["summary"],
            })
            event_text = " ".join(filter(None, [updated.get("title"), updated.get("event_type"), updated.get("activity"), updated.get("summary")]))
            vector = self.clip.embed_text(event_text)
            self.store.upsert_vector("episodic", "event", event_id, vector, self.clip.model_name, {"summary_model": result["model"], "event_summary": True})
            return updated

    def summarize_events(self, scope_id=None):
        return [self.summarize_event(event["id"]) for event in self.store.list_events(1000, scope_id)]

    def summarize_pending_events(self, scope_id=None, limit=100):
        """Build deferred event projections without reprocessing image evidence."""
        pending = [
            event for event in self.store.list_events(max(1, limit), scope_id)
            if event.get("title") == "待总结事件"
        ]
        return [self.summarize_event(event["id"]) for event in pending]

    def finalize_ingest_batch(self, batch_id):
        """Summarize only events touched by a completed import batch."""
        batch = self.store.get_ingest_batch(batch_id)
        if not batch:
            raise KeyError(batch_id)
        if batch["status"] == "open":
            return batch
        if batch["status"] == "complete":
            if not self.store.claim_ingest_batch_summary(batch_id):
                return self.store.get_ingest_batch(batch_id)
            batch = self.store.get_ingest_batch(batch_id)
        elif batch["status"] == "summarizing":
            return batch
        else:
            return batch
        for event_id in self.store.batch_event_ids(batch_id):
            self.summarize_event(event_id)
        batch = self.store.finish_ingest_batch(batch_id)
        scope_id = (batch or {}).get("scope_id")
        if scope_id:
            self._maybe_trigger_person_insight(scope_id)
        return batch

    def _maybe_trigger_person_insight(self, scope_id):
        """Trigger an incremental person-insight run only for allowlisted scopes.

        A run only starts when new events arrived since the previous run's event
        watermark; otherwise the live portraits are left untouched.
        """
        allowlist = {
            item.strip() for item in os.getenv("SENTRIX_PERSON_INSIGHT_SCOPES", "").split(",")
            if item.strip()
        }
        if scope_id not in allowlist:
            return None
        event_count = self.store.connection.execute(
            "SELECT COUNT(*) FROM events WHERE scope_id = ?", (scope_id,)
        ).fetchone()[0]
        latest = self.store.latest_person_insight_run(scope_id)
        watermark = int(((latest or {}).get("stats") or {}).get("event_watermark", 0))
        if latest is not None and event_count <= watermark:
            return None
        run = self.store.create_person_insight_run(scope_id, {
            "max_core_people": 10, "trigger_type": "ingest",
        })

        def execute():
            from .person_insights import PersonInsightService

            worker = MemoryStore(self.store.path)
            try:
                PersonInsightService(worker, self.gamma).run(
                    run["id"], scope_id, {"max_core_people": 10, "trigger_type": "ingest"}
                )
            finally:
                worker.close()

        threading.Thread(target=execute, daemon=True).start()
        return run

    def _image_observation(self, asset, precomputed_analysis=None):
        path = asset["path"]
        captured_at = asset.get("captured_at") or file_time(path)
        metadata = {
            "file_name": asset["file_name"],
            "captured_at": captured_at,
            "captured_location": asset.get("captured_location") or "",
            "source_owner_id": asset.get("source_owner_id"),
            "source_device_id": asset.get("source_device_id"),
            "source_album_id": asset.get("source_album_id"),
            "location_context": (asset.get("metadata_json") or {}).get("reverse_geocode") or {},
        }
        started_at = time.perf_counter()
        def timed(callable_):
            step_started = time.perf_counter()
            return callable_(), time.perf_counter() - step_started
        # Model adapters do not write to MemoryStore. Keep SQLite writes and
        # event selection on this caller thread after all three complete.
        parallel = os.getenv("SENTRIX_PARALLEL_IMAGE_ANALYSIS", "true").lower() in {"1", "true", "yes"}
        if precomputed_analysis is not None:
            analysis = dict(precomputed_analysis)
            vision_seconds = float(analysis.pop("_vision_seconds", 0.0) or 0.0)
            with ThreadPoolExecutor(max_workers=2, thread_name_prefix="sentrix-image") as executor:
                face_future = executor.submit(timed, lambda: self.face.detect(path))
                clip_future = executor.submit(timed, lambda: self.clip.embed_image(path))
                faces, face_seconds = face_future.result()
                clip_embedding, clip_seconds = clip_future.result()
        elif parallel:
            with ThreadPoolExecutor(max_workers=3, thread_name_prefix="sentrix-image") as executor:
                vision_future = executor.submit(timed, lambda: self.gamma.analyze_image(path, metadata))
                face_future = executor.submit(timed, lambda: self.face.detect(path))
                clip_future = executor.submit(timed, lambda: self.clip.embed_image(path))
                analysis, vision_seconds = vision_future.result()
                faces, face_seconds = face_future.result()
                clip_embedding, clip_seconds = clip_future.result()
        else:
            analysis, vision_seconds = timed(lambda: self.gamma.analyze_image(path, metadata))
            faces, face_seconds = timed(lambda: self.face.detect(path))
            clip_embedding, clip_seconds = timed(lambda: self.clip.embed_image(path))
        analysis = normalize_semantic_analysis(analysis)
        analysis["clip_embedding"] = clip_embedding
        analysis["processing_timings"] = {
            "vision_seconds": round(vision_seconds, 4),
            "face_seconds": round(face_seconds, 4),
            "clip_seconds": round(clip_seconds, 4),
            "analysis_wall_seconds": round(time.perf_counter() - started_at, 4),
            "parallel": parallel,
        }
        analysis["captured_at"] = captured_at
        analysis["source_owner_id"] = asset.get("source_owner_id")
        analysis["location_context"] = metadata["location_context"]
        analysis["canonical"] = {
            key: analysis.get(key)
            for key in ("caption", "activity", "place", "scene_type", "semantic", "raw_labels", "people", "objects", "clothing", "spatial_relations", "emotions", "ocr_text", "event_type")
        }
        analysis["source_type"] = "video_event" if precomputed_analysis is not None else "image"
        analysis["face_candidates"] = faces
        analysis["raw"] = {"gamma": {key: value for key, value in analysis.items() if key not in {"clip_embedding", "face_candidates", "location_context"}}, "location_context": metadata["location_context"], "face_candidates": [{key: value for key, value in face.items() if key != "embedding"} for face in faces], "models": {"vision": self.gamma.model, "face_detector": "buffalo_l", "face_embedding": sorted({face.get("embedding_model", "unknown") for face in faces}), "image_embedding": self.clip.model_name}}
        return analysis

    def _audio_observation(self, asset):
        transcript = self.asr.transcribe(asset["path"])
        analysis = self.gamma.analyze_text(transcript.get("text", ""), "audio")
        analysis["captured_at"] = file_time(asset["path"])
        analysis["transcript"] = transcript.get("text", "")
        analysis["raw"] = {"funasr": transcript, "gamma": analysis.copy()}
        return analysis

    def _text_observation(self, asset):
        text = Path(asset["path"]).read_text(encoding="utf-8", errors="replace")
        analysis = self.gamma.analyze_text(text, "text")
        analysis["captured_at"] = file_time(asset["path"])
        analysis["raw"] = {"gamma": analysis.copy()}
        return analysis
