import json
import hashlib
import mimetypes
import os
from datetime import datetime, timezone
from pathlib import Path

from .db import MemoryStore, make_id
from .model_clients import ClipAdapter, FaceAdapter, FunASRClient, GammaClient, ModelError, normalize_confidence


IMPORT_METADATA_KEYS = {
    "content_sha256", "sha256", "exif", "captured_at", "captured_location",
    "source_owner_id", "source_owner_label", "source_device_id", "source_album_id",
    "source_confidence", "scope_id",
}


def file_time(path):
    return datetime.fromtimestamp(Path(path).stat().st_mtime, timezone.utc).isoformat()


class VideoMemoryAdapter:
    def reserve(self, asset):
        return {"status": "video-extraction-reserved", "asset_id": asset["id"], "adapter": "video_memory_adapter"}


class IngestionPipeline:
    def __init__(self, store, gamma=None, asr=None, face=None, clip=None):
        self.store = store
        self.gamma = gamma or GammaClient()
        self.asr = asr or FunASRClient()
        self.face = face or FaceAdapter()
        self.clip = clip or ClipAdapter()
        self.video_memory_adapter = VideoMemoryAdapter()

    def create_asset(self, path, file_name=None, media_type=None, mime_type=None, metadata=None):
        path = Path(path)
        asset_id = make_id("asset")
        media_type = media_type or self._media_type(path)
        # Import metadata is provenance only. Model hints and benchmark labels
        # must never enter the memory graph through the asset boundary.
        metadata = {key: value for key, value in (metadata or {}).items() if key in IMPORT_METADATA_KEYS}
        metadata.setdefault("content_sha256", self._sha256(path))
        metadata.setdefault("exif", self._extract_exif(path) if media_type == "image" else {})
        existing = self.store.find_asset_by_hash(metadata["content_sha256"])
        if existing:
            return existing
        for key in ("captured_at", "captured_location", "source_device_id"):
            if metadata.get(key) is None and metadata["exif"].get(key):
                metadata[key] = metadata["exif"][key]
        return self.store.create_asset(
            asset_id,
            file_name or path.name,
            media_type,
            str(path),
            mime_type or mimetypes.guess_type(path.name)[0],
            path.stat().st_size,
            metadata,
            scope_id=metadata.get("scope_id"),
        )

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
            from PIL import Image, ExifTags
            with Image.open(path) as image:
                raw = image.getexif()
                tags = {ExifTags.TAGS.get(key, str(key)): value for key, value in raw.items()}
            result = {"device": tags.get("Model") or tags.get("Make")}
            if tags.get("DateTimeOriginal"):
                result["captured_at"] = tags["DateTimeOriginal"].replace(":", "-", 2) + "+00:00"
            gps = tags.get("GPSInfo") or {}
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
        mime = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        return mime.split("/", 1)[0] if "/" in mime else "text"

    def process(self, asset_id, summarize_event=True):
        asset = self.store.get_asset(asset_id)
        if not asset:
            raise KeyError(asset_id)
        if asset.get("status") == "processed":
            metadata = asset.get("metadata_json") or {}
            return asset
        if asset["media_type"] == "video":
            result = self.video_memory_adapter.reserve(asset)
            return self.store.update_asset(asset_id, result["status"], result)
        self.store.update_asset(asset_id, "processing", {})
        try:
            if asset["media_type"] == "image":
                result = self._image_observation(asset)
            elif asset["media_type"] == "audio":
                result = self._audio_observation(asset)
            else:
                result = self._text_observation(asset)
            observation = self.store.add_observation(asset_id, result)
            # Event selection needs the image vector itself, not a later
            # projection of the selected event, so persist it before scoring.
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
            event = self.store.merge_observation_into_event(observation)
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
            for fact in result.get("facts", []):
                if all(fact.get(key) for key in ("subject", "predicate", "object")):
                    saved = self.store.maintain_fact(fact["subject"], fact["predicate"], fact["object"], [observation["id"]], normalize_confidence(fact.get("confidence"), result.get("confidence", 0.5)))
                    fact_ids.append(saved["id"])
            metadata = {"observation_id": observation["id"], "event_id": event["id"], "fact_ids": fact_ids, "cluster_ids": cluster_ids, "model": result.get("model"), "faces": [{key: value for key, value in face.items() if key != "embedding"} for face in result.get("face_candidates", [])]}
            saved_asset = self.store.update_asset(asset_id, "processed", metadata)
            if asset.get("source_owner_id"):
                self.store.rebuild_person_memory(asset["source_owner_id"])
            if summarize_event:
                self.summarize_event(event["id"])
            return saved_asset
        except Exception as error:
            self.store.cleanup_asset_derivatives(asset_id)
            return self.store.update_asset(asset_id, "failed", {"error": str(error)})

    def summarize_event(self, event_id):
        detail = self.store.get_event_detail(event_id)
        if not detail or not detail["observations"] or not hasattr(self.gamma, "summarize_event"):
            return self.store.get_event(event_id)
        try:
            result = self.gamma.summarize_event(detail["event"], detail["observations"])
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
            return self.store.get_event(event_id)

    def summarize_events(self):
        return [self.summarize_event(event["id"]) for event in self.store.list_events(1000)]

    def _image_observation(self, asset):
        path = asset["path"]
        captured_at = asset.get("captured_at") or file_time(path)
        metadata = {
            "file_name": asset["file_name"],
            "captured_at": captured_at,
            "captured_location": asset.get("captured_location") or "",
            "source_owner_id": asset.get("source_owner_id"),
            "source_device_id": asset.get("source_device_id"),
            "source_album_id": asset.get("source_album_id"),
        }
        analysis = self.gamma.analyze_image(path, metadata)
        faces = self.face.detect(path)
        analysis["clip_embedding"] = self.clip.embed_image(path)
        analysis["captured_at"] = captured_at
        analysis["source_owner_id"] = asset.get("source_owner_id")
        analysis["canonical"] = {
            key: analysis.get(key)
            for key in ("caption", "activity", "place", "people", "objects", "clothing", "spatial_relations", "ocr_text", "event_type")
        }
        analysis["source_type"] = "image"
        analysis["face_candidates"] = faces
        analysis["raw"] = {"gamma": {key: value for key, value in analysis.items() if key not in {"clip_embedding", "face_candidates"}}, "face_candidates": [{key: value for key, value in face.items() if key != "embedding"} for face in faces], "models": {"vision": self.gamma.model, "face_detector": "buffalo_l", "face_embedding": sorted({face.get("embedding_model", "unknown") for face in faces}), "image_embedding": self.clip.model_name}}
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
