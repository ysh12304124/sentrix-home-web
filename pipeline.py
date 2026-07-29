import json
import mimetypes
import os
from datetime import datetime, timezone
from pathlib import Path

from .db import MemoryStore, make_id
from .model_clients import ClipAdapter, FaceAdapter, FunASRClient, GammaClient, ModelError


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

    def create_asset(self, path, file_name=None, media_type=None, mime_type=None):
        path = Path(path)
        asset_id = make_id("asset")
        media_type = media_type or self._media_type(path)
        return self.store.create_asset(asset_id, file_name or path.name, media_type, str(path), mime_type or mimetypes.guess_type(path.name)[0], path.stat().st_size)

    def _media_type(self, path):
        mime = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        return mime.split("/", 1)[0] if "/" in mime else "text"

    def process(self, asset_id):
        asset = self.store.get_asset(asset_id)
        if not asset:
            raise KeyError(asset_id)
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
            event = self.store.merge_observation_into_event(observation)
            self.store.upsert_vector("visual", "asset", asset_id, result.get("clip_embedding"), self.clip.model_name, {"observation_id": observation["id"], "event_id": event["id"]})
            text_embedding = self.clip.embed_text(" ".join(filter(None, [observation.get("caption"), observation.get("activity"), observation.get("place"), observation.get("ocr_text"), observation.get("transcript")])))
            self.store.upsert_vector("episodic", "observation", observation["id"], text_embedding, self.clip.model_name, {"asset_id": asset_id, "event_id": event["id"]})
            self.store.upsert_vector("episodic", "event", event["id"], text_embedding, self.clip.model_name, {"observation_id": observation["id"]})
            fact_ids = []
            for fact in result.get("facts", []):
                if all(fact.get(key) for key in ("subject", "predicate", "object")):
                    saved = self.store.maintain_fact(fact["subject"], fact["predicate"], fact["object"], [observation["id"]], float(fact.get("confidence", result.get("confidence", 0.5)) or 0.5))
                    fact_ids.append(saved["id"])
                    fact_embedding = self.clip.embed_text(f"{fact['subject']} {fact['predicate']} {fact['object']}")
                    self.store.upsert_vector("semantic", "fact", saved["id"], fact_embedding, self.clip.model_name, {"observation_id": observation["id"]})
            cluster_ids = []
            for index, face in enumerate(result.get("face_candidates", [])):
                face_instance = self.store.add_face_instance(asset_id, observation["id"], face)
                cluster_ids.append(face_instance["cluster_id"])
            metadata = {"observation_id": observation["id"], "event_id": event["id"], "fact_ids": fact_ids, "cluster_ids": cluster_ids, "model": result.get("model"), "faces": [{key: value for key, value in face.items() if key != "embedding"} for face in result.get("face_candidates", [])]}
            return self.store.update_asset(asset_id, "processed", metadata)
        except Exception as error:
            return self.store.update_asset(asset_id, "failed", {"error": str(error)})

    def _image_observation(self, asset):
        path = asset["path"]
        analysis = self.gamma.analyze_image(path, {"file_name": asset["file_name"], "captured_at": file_time(path)})
        faces = self.face.detect(path)
        analysis["clip_embedding"] = self.clip.embed_image(path)
        analysis["captured_at"] = file_time(path)
        analysis["source_type"] = "image"
        analysis["face_candidates"] = faces
        analysis["raw"] = {"gamma": {key: value for key, value in analysis.items() if key not in {"clip_embedding", "face_candidates"}}, "face_candidates": [{key: value for key, value in face.items() if key != "embedding"} for face in faces], "models": {"vision": self.gamma.model, "face": "buffalo_l", "image_embedding": self.clip.model_name}}
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
