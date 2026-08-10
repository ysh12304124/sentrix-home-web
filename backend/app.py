from __future__ import annotations
import json
import os
import shutil
import hashlib
import tempfile
import threading
from datetime import datetime
from io import BytesIO
from pathlib import Path

from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .agent import MemoryAgent
from .db import MemoryStore, make_id
from .image_io import (
    encode_jpeg_preview,
    ensure_heif_support,
    guess_mime_type,
    media_type_from_upload,
    needs_browser_transcode,
)
from .model_clients import ClipAdapter, FaceAdapter, FunASRClient, GammaClient, parse_json_response
from .pipeline import IngestionPipeline
from .person_appearance import expanded_person_crop


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = Path(os.getenv("SENTRIX_DATA_DIR", ROOT / "data"))
MEDIA_DIR = DATA_DIR / "media"
PREVIEW_DIR = DATA_DIR / "previews"
DATA_DIR.mkdir(parents=True, exist_ok=True)
MEDIA_DIR.mkdir(parents=True, exist_ok=True)
PREVIEW_DIR.mkdir(parents=True, exist_ok=True)
ensure_heif_support()

store = MemoryStore(os.getenv("SENTRIX_DB_PATH", str(DATA_DIR / "sentrix.db")))
gamma = GammaClient()
gamma.bind_store(store)
pipeline = IngestionPipeline(store, gamma=gamma, asr=FunASRClient(), face=FaceAdapter(), clip=ClipAdapter())
agent = MemoryAgent(store, gamma=gamma, clip=pipeline.clip)

app = FastAPI(title="Sentrix Home Memory API", version="0.1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
maintenance_lock = threading.Lock()
runtime_lock = threading.Lock()
VLLM_MANAGER = Path(os.getenv("SENTRIX_VLLM_MANAGER", "/home/asus/sentrix-vllm/bin/sentrix_vllm_manager.py"))
VLLM_REGISTRY = Path(os.getenv("SENTRIX_VLLM_REGISTRY", "/home/asus/sentrix-vllm/registry.json"))
VLM_BACKENDS = ("ollama_12b", "e2b_lora")
SUPPORTED_IMPORT_SUFFIXES = {
    ".jpg", ".jpeg", ".png", ".webp", ".heic", ".bmp", ".gif",
    ".mp4", ".mov", ".m4v", ".avi", ".mkv", ".mp3", ".wav", ".m4a",
    ".txt", ".md", ".json",
}
MAX_REMOTE_IMPORT_FILES = int(os.getenv("SENTRIX_MAX_REMOTE_IMPORT_FILES", "500"))


def _normalized_capture_metadata(payload=None, *, captured_at=None, captured_location=None, latitude=None, longitude=None):
    payload = payload if isinstance(payload, dict) else {}
    captured_at = payload.get("capturedAt", payload.get("captured_at", captured_at))
    captured_location = payload.get("capturedLocation", payload.get("captured_location", captured_location))
    latitude = payload.get("latitude", payload.get("capturedLatitude", payload.get("captured_latitude", latitude)))
    longitude = payload.get("longitude", payload.get("capturedLongitude", payload.get("captured_longitude", longitude)))
    if (latitude is None) != (longitude is None):
        raise ValueError("latitude and longitude must be provided together")
    result = {}
    if captured_at:
        value = str(captured_at).strip()
        try:
            datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError as error:
            raise ValueError("capturedAt must be an ISO 8601 datetime") from error
        result["captured_at"] = value
    if captured_location:
        result["captured_location"] = str(captured_location).strip()
    if latitude is not None:
        try:
            latitude = float(latitude)
            longitude = float(longitude)
        except (TypeError, ValueError) as error:
            raise ValueError("latitude and longitude must be numbers") from error
        if not -90 <= latitude <= 90 or not -180 <= longitude <= 180:
            raise ValueError("latitude or longitude is outside its valid range")
        result["gps"] = {"latitude": latitude, "longitude": longitude}
    return result


def _check_ollama_health():
    try:
        import httpx
        url = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip("/")
        response = httpx.get(f"{url}/api/tags", timeout=10)
        response.raise_for_status()
        models = response.json().get("models") or []
        ollama_model = os.getenv("OLLAMA_MODEL", "gemma4:12b")
        for model in models:
            if model.get("name", "").startswith(ollama_model.replace(":12b", "")):
                return {"available": True, "model": ollama_model, "url": url}
        return {"available": False, "model": ollama_model, "url": url, "error": "model not found in /api/tags"}
    except Exception as exc:
        return {"available": False, "model": os.getenv("OLLAMA_MODEL", "gemma4:12b"), "url": os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434"), "error": str(exc)}


def _check_e2b_health():
    try:
        import httpx
        url = os.getenv("E2B_BASE_URL", "http://127.0.0.1:8100").rstrip("/")
        response = httpx.get(f"{url}/api/health", timeout=10)
        response.raise_for_status()
        data = response.json()
        return {"available": data.get("status") == "ok", "url": url, "loaded": data.get("loaded", False), "model": data.get("model", ""), "error": data.get("error")}
    except Exception as exc:
        return {"available": False, "url": os.getenv("E2B_BASE_URL", "http://127.0.0.1:8100"), "error": str(exc)}


def _fire_and_forget_post(url, payload):
    try:
        import httpx
        httpx.post(url, json=payload, timeout=5)
    except Exception:
        pass


def _schedule_backend_transition(backend_name):
    if backend_name == "e2b_lora":
        e2b_url = os.getenv("E2B_BASE_URL", "http://127.0.0.1:8100").rstrip("/")
        threading.Thread(target=_fire_and_forget_post, args=(f"{e2b_url}/admin/load", {}), daemon=True).start()


class SearchRequest(BaseModel):
    query: str
    spaceId: str = "home-default"


class ImportRequest(BaseModel):
    source_path: str
    scope_id: str = "home-default"
    batch_id: str | None = None
    recursive: bool = True
    glob: str = "*"
    copy_file: bool = Field(True, alias="copy")
    max_files: int = 5000
    source_owner_id: str | None = None
    source_owner_label: str | None = None
    source_device_id: str | None = None
    source_album_id: str | None = None
    captured_at: str | None = None
    captured_location: str | None = None
    fileName: str = "unknown"
    mediaType: str = "text"


class IngestBatchCreateRequest(BaseModel):
    scope_id: str = "home-default"
    batch_id: str | None = None
    name: str | None = None
    kind: str = "benchmark"
    source_path: str | None = None


def _allowed_import_roots():
    configured = os.getenv("SENTRIX_IMPORT_ALLOWED_ROOTS")
    defaults = [
        DATA_DIR / "imports",
        ROOT / "data" / "imports",
        # 153-only roots carried over from the hasty backend alignment.
        Path("/home/asus/data"),
        Path("/home/asus/datasets"),
        Path("/home/asus/benchmarks"),
        # 200-local staging/workspace (000Notes/family_photos, projects).
        Path("/home/sscy/lingbot-map"),
    ]
    values = configured.split(":") if configured else [str(item) for item in defaults]
    roots = []
    for value in values:
        try:
            roots.append(Path(value).expanduser().resolve())
        except OSError:
            continue
    return roots


def _assert_import_path_allowed(path: Path):
    roots = _allowed_import_roots()
    if any(path == root or root in path.parents for root in roots):
        return
    raise HTTPException(status_code=403, detail={
        "message": "source_path is outside allowed import roots",
        "source_path": str(path),
        "allowed_roots": [str(root) for root in roots],
    })


def _batch_status(batch_id: str):
    batch = store.get_ingest_batch(batch_id)
    if not batch:
        raise HTTPException(status_code=404, detail="ingest batch not found")
    rows = store._rows("SELECT status, COUNT(*) AS count FROM assets WHERE batch_id = ? GROUP BY status", (batch_id,))
    counts = {row["status"]: row["count"] for row in rows}
    total = sum(counts.values())
    return {
        "batch": batch,
        "scope_id": batch.get("scope_id"),
        "asset_counts": counts,
        "asset_total": total,
        "identity_confirmation_required": True,
        "identity_confirmation": {
            "list_clusters": f"GET /api/face-clusters?scope_id={batch.get('scope_id')}",
            "confirm_cluster": "POST /api/face-clusters/{cluster_id}/confirm",
        },
    }


def process_asset(asset_id):
    # BackgroundTasks may process uploads concurrently. Each task must own its
    # SQLite connection; sharing the request-store connection interleaves its
    # transactions and raises "cannot start a transaction within a transaction".
    task_store = MemoryStore(store.path)
    task_pipeline = IngestionPipeline(
        task_store, gamma=gamma, asr=pipeline.asr, face=pipeline.face, clip=pipeline.clip,
    )
    try:
        asset = task_store.get_asset(asset_id) or {}
        if asset.get("media_type") != "image":
            task_pipeline.process(asset_id)
            return
        fast = task_pipeline.process_fast_image(asset_id)
        if fast.get("status") == "semantic_enriching":
            # Finish the semantic observation and summarize its event in the
            # same background pipeline; imports must not require maintenance UI.
            task_pipeline.enrich_fast_image(asset_id, summarize_event=True)
    finally:
        task_store.close()



class SetVLMBackend(BaseModel):
    backend: str


class ModelSwitchRequest(BaseModel):
    profile: str
    wait_ready: bool = True
    ready_timeout: int = 900
    dry_run: bool = False
    max_model_len: int | None = None
    max_num_seqs: int | None = None
    max_num_batched_tokens: int | None = None
    gpu_memory_utilization: float | None = None
    quantization: str | None = None
    load_format: str | None = None
    dtype: str | None = None
    default_max_tokens: int | None = None
    cuda_visible_devices: str | None = None


def _read_json_file(path: Path, fallback):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return fallback


def _load_vllm_registry():
    return _read_json_file(VLLM_REGISTRY, {"profiles": {}, "default_port": 8100, "state_file": ""})


def _load_vllm_state(registry=None):
    registry = registry or _load_vllm_registry()
    state_file = Path(registry.get("state_file") or "/home/asus/sentrix-vllm/state/current.json")
    return _read_json_file(state_file, None) if state_file.exists() else None


def _profile_availability(profile):
    missing = []
    model_path = profile.get("model")
    if model_path and not Path(model_path).exists():
        missing.append(model_path)
    for module in profile.get("lora_modules") or []:
        path = module.get("path")
        if path and not Path(path).exists():
            missing.append(path)
    return {"available": not missing, "missing_paths": missing}


def _profile_summary(profile_id, profile):
    availability = _profile_availability(profile)
    return {
        "id": profile_id, "model": profile.get("model"),
        "served_model_name": profile.get("served_model_name") or profile_id,
        "dtype": profile.get("dtype"), "quantization": profile.get("quantization"),
        "load_format": profile.get("load_format"), "max_model_len": profile.get("max_model_len"),
        "max_num_seqs": profile.get("max_num_seqs"), "gpu_memory_utilization": profile.get("gpu_memory_utilization"),
        "default_max_tokens": profile.get("default_max_tokens"),
        "enable_lora": bool(profile.get("enable_lora")),
        "lora_modules": profile.get("lora_modules") or [],
        "limit_mm_per_prompt": profile.get("limit_mm_per_prompt") or {},
        "notes": profile.get("notes", ""), **availability,
    }


def _current_model_runtime():
    registry = _load_vllm_registry()
    state = _load_vllm_state(registry)
    running = bool(state and state.get("pid"))
    return {
        "backend": getattr(gamma, "backend", "unknown"),
        "base_url": gamma.base_url, "model": gamma.model,
        "profile": (state or {}).get("profile"),
        "status": "running" if running else "stopped",
        "state": state,
    }


def _apply_vllm_profile_to_runtime(profile_id, profile=None, state=None):
    global gamma, pipeline, agent
    registry = _load_vllm_registry()
    profile = profile or (registry.get("profiles") or {}).get(profile_id) or {}
    state = state or _load_vllm_state(registry) or {}
    port = int(state.get("port") or profile.get("port") or registry.get("default_port") or 8100)
    served_name = state.get("served_model_name") or profile.get("served_model_name") or profile_id
    with runtime_lock:
        new_gamma = GammaClient(base_url=f"http://127.0.0.1:{port}/v1", model=served_name, backend="openai")
        gamma = new_gamma
        pipeline = IngestionPipeline(store, gamma=gamma, asr=pipeline.asr, face=pipeline.face, clip=pipeline.clip)
        agent = MemoryAgent(store, gamma=gamma, clip=pipeline.clip)
    return _current_model_runtime()


def _run_vllm_switch(request: ModelSwitchRequest):
    if not VLLM_MANAGER.exists():
        raise HTTPException(status_code=503, detail=f"vLLM manager not found: {VLLM_MANAGER}")
    registry = _load_vllm_registry()
    profile = (registry.get("profiles") or {}).get(request.profile)
    if not profile:
        raise HTTPException(status_code=404, detail="model profile not found")
    command = [str(VLLM_MANAGER), "switch", request.profile]
    option_map = {
        "max_model_len": "--max-model-len", "max_num_seqs": "--max-num-seqs",
        "max_num_batched_tokens": "--max-num-batched-tokens",
        "gpu_memory_utilization": "--gpu-memory-utilization",
        "quantization": "--quantization", "load_format": "--load-format",
        "dtype": "--dtype", "default_max_tokens": "--default-max-tokens",
        "cuda_visible_devices": "--cuda-visible-devices",
    }
    values = request.model_dump() if hasattr(request, "model_dump") else request.dict()
    for field, flag in option_map.items():
        value = values.get(field)
        if value is not None and value != "":
            command.extend([flag, str(value)])
    if request.wait_ready:
        command.extend(["--wait-ready", "--ready-timeout", str(max(30, request.ready_timeout))])
    if request.dry_run:
        command.append("--dry-run")
    import subprocess
    completed = subprocess.run(command, cwd=str(ROOT), text=True, capture_output=True,
        timeout=max(60, int(request.ready_timeout) + 90 if request.wait_ready else 60))
    if completed.returncode != 0:
        raise HTTPException(status_code=502, detail={
            "message": "vLLM switch failed", "command": command,
            "stdout": completed.stdout[-4000:], "stderr": completed.stderr[-4000:]})
    runtime = _current_model_runtime() if request.dry_run else _apply_vllm_profile_to_runtime(request.profile, profile)
    return {"accepted": True, "profile": request.profile, "runtime": runtime,
        "stdout": completed.stdout[-4000:], "stderr": completed.stderr[-4000:]}

@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "mode": "sentrix-local-backend",
        "models": {
            "vlm": {"active": "vllm", "name": gamma.model, "endpoint": gamma.base_url},
            "llm": _current_model_runtime(),
            "gamma4_12B": {"name": gamma.model, "endpoint": gamma.base_url},
            "asr": {"name": pipeline.asr.model_name, "vad": pipeline.asr.vad_model, "punc": pipeline.asr.punc_model, "ready": pipeline.asr.error is None, "error": pipeline.asr.error},
            "face": {
                "enabled": pipeline.face.enabled,
                "ready": pipeline.face.ready,
                "detectionReady": pipeline.face.error is None,
                "identityModel": pipeline.face.identity_model,
                "identityConfigured": pipeline.face.identity_configured,
                "identityReady": pipeline.face.identity_ready,
                "error": pipeline.face.error,
                "identityError": pipeline.face.identity_runtime_error or pipeline.face.identity_error,
            },
            "clip": {"enabled": pipeline.clip.enabled, "model": pipeline.clip.model_name, "ready": pipeline.clip.evidence_ready, "evidenceReady": pipeline.clip.evidence_ready, "error": pipeline.clip.error},
        },
        "memory": {"mode": "sentrix-native", "vectorSpaces": ["episodic", "semantic", "visual"]},
        "videoExtraction": "reserved",
        "database": store.path,
    }



class MemorySpaceCreateRequest(BaseModel):
    name: str
    scope_id: str | None = None


@app.post("/api/memory-spaces", status_code=201)
def create_memory_space(request: MemorySpaceCreateRequest):
    name = request.name.strip()
    if not name:
        raise HTTPException(status_code=422, detail="name is required")
    if len(name) > 100:
        raise HTTPException(status_code=422, detail="name must be at most 100 characters")
    scope_id = (request.scope_id or make_id("album")).strip()
    if not scope_id:
        raise HTTPException(status_code=422, detail="scope_id is invalid")
    existing = store._row("SELECT id FROM memory_spaces WHERE id = ?", (scope_id,))
    if existing:
        raise HTTPException(status_code=409, detail="scope_id already exists")
    return store.create_memory_space(scope_id, name, kind="benchmark")

@app.get("/api/memory-spaces")
def memory_spaces():
    return {"spaces": store.list_memory_spaces()}


@app.get("/api/vlm-backend")
def vlm_backend():
    runtime = _current_model_runtime()
    return {
        "backend": "vllm",
        "available_backends": ["vllm"],
        "profile": runtime.get("profile"),
        "model": runtime.get("model"),
        "status": runtime.get("status"),
        "deprecated": True,
        "replacement": "/api/model-profiles",
    }


@app.post("/api/vlm-backend")
def set_vlm_backend(payload: SetVLMBackend):
    raise HTTPException(
        status_code=410,
        detail="VLM backend switching is retired; use POST /api/model-profiles/switch",
    )

@app.get("/api/model-profiles")
def model_profiles():
    registry = _load_vllm_registry()
    profiles = registry.get("profiles") or {}
    return {
        "backend": "vllm", "registry": str(VLLM_REGISTRY), "manager": str(VLLM_MANAGER),
        "current": _current_model_runtime(),
        "profiles": [_profile_summary(profile_id, profile) for profile_id, profile in profiles.items()],
    }


@app.get("/api/model-profiles/current")
def current_model_profile():
    return _current_model_runtime()


@app.post("/api/model-profiles/switch")
def switch_model_profile(request: ModelSwitchRequest):
    return _run_vllm_switch(request)




@app.get("/api/geo-places")
def geo_places(scope_id: str | None = None):
    import json as _json
    rows = store._rows(
        """SELECT a.id AS asset_id, a.file_name, a.captured_location,
                  a.captured_at, a.metadata_json,
                  o.id AS observation_id, o.caption, o.place
           FROM assets a
           LEFT JOIN observations o ON o.asset_id = a.id
           WHERE a.media_type = 'image'
           AND (a.scope_id = ? OR ? IS NULL)
           ORDER BY a.captured_location, a.captured_at""",
        (scope_id, scope_id),
    )
    cities = {}
    unknown = {"level": "unknown", "name": "无法判断地点", "count": 0, "children": [], "photos": []}
    for row in rows:
        metadata = {}
        try: metadata = _json.loads(row["metadata_json"] or "{}")
        except (TypeError, _json.JSONDecodeError): pass
        geo = metadata.get("reverse_geocode") or {}
        city = geo.get("city") or ""
        district = geo.get("district") or ""
        photo = {"asset_id": row["asset_id"], "file_name": row["file_name"], "captured_at": row["captured_at"], "caption": row["caption"] or "", "observation_id": row["observation_id"], "semantic_place": "", "observation_place": row["place"] or ""}
        if not city or not district:
            unknown["count"] += 1; unknown["photos"].append(photo); continue
        province = geo.get("province") or ""
        city_key = f"{province}{city}"
        cities.setdefault(city_key, {"level": "city", "name": city, "province": province, "count": 0, "districts": {}})
        cities[city_key]["count"] += 1
        districts = cities[city_key]["districts"]
        districts.setdefault(district, {"level": "district", "name": district, "city": city, "count": 0, "photos": []})
        districts[district]["count"] += 1
        districts[district]["photos"].append(photo)
    result = []
    for city_data in cities.values():
        city_data["children"] = sorted(city_data.pop("districts").values(), key=lambda d: -d["count"])
        result.append(city_data)
    result.sort(key=lambda c: -c["count"])
    if unknown["count"] > 0: result.append(unknown)
    return {"places": result}


@app.get("/api/dashboard")
def dashboard(scope_id: str | None = None):
    all_facts = store.list_facts(1000, scope_id=scope_id)
    return {
        "stats": {
            "assets": len(store.list_assets(limit=100000, scope_id=scope_id)),
            "observations": len(store.list_observations(100000, scope_id=scope_id)),
            "events": len(store.list_events(100000, scope_id=scope_id)),
            "facts": len(all_facts),
            "persons": len([item for item in store.list_entities(scope_id=scope_id) if item["entity_type"] == "person"]),
            "entities": len(store.list_entities(scope_id=scope_id)),
            "faceClusters": len([item for item in store.list_face_clusters() if not scope_id or item.get("scope_id") == scope_id]),
            "relationships": len(store.list_relationships(scope_id=scope_id)),
            "vectors": len(store._rows("SELECT id FROM memory_vectors" + (" WHERE scope_id = ?" if scope_id else ""), (scope_id,) if scope_id else ())),
        },
        "pendingFacts": len([item for item in all_facts if item["status"] == "pending"]),
        "events": store.list_events(8, scope_id=scope_id),
        "observations": store.list_observations(8, scope_id=scope_id),
        "facts": store.list_facts(8, scope_id=scope_id),
    }


@app.get("/api/events")
def events(scope_id: str | None = None):
    return {"events": store.list_events(100, scope_id=scope_id)}


@app.get("/api/trips")
def trips(scope_id: str | None = None, status: str | None = None):
    return {"trips": store.list_trips(scope_id, status)}


@app.get("/api/trips/{trip_id}")
def trip_detail(trip_id: str):
    value = store.get_trip_detail(trip_id)
    if not value:
        raise HTTPException(status_code=404, detail="trip not found")
    return value


@app.post("/api/trips/{trip_id}/confirm")
def confirm_trip(trip_id: str, payload: dict):
    try:
        return store.confirm_trip(trip_id, payload.get("name"), payload.get("trip_type"))
    except KeyError:
        raise HTTPException(status_code=404, detail="trip not found")
    except ValueError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error


@app.post("/api/trips/{trip_id}/reject")
def reject_trip(trip_id: str):
    try:
        return store.reject_trip(trip_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="trip not found")
    except ValueError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error


@app.get("/api/events/{event_id}")
def event_detail(event_id: str):
    value = store.get_event_detail(event_id)
    if not value:
        raise HTTPException(status_code=404, detail="event not found")
    return value


@app.post("/api/events")
def create_event(payload: dict):
    return store.create_event(payload)


@app.patch("/api/events/{event_id}")
def update_event(event_id: str, payload: dict):
    value = store.update_event(event_id, payload)
    if not value:
        raise HTTPException(status_code=404, detail="event not found")
    return value


@app.get("/api/persons")
def persons():
    rows = store.list_persons()
    return {"persons": [{**row, "display_name": row["name"], "confirmed": row["status"] == "confirmed"} for row in rows]}


@app.get("/api/persons/{person_id}")
def person_detail(person_id: str):
    value = store.get_person(person_id)
    if not value:
        raise HTTPException(status_code=404, detail="person not found")
    return value


@app.get("/api/people")
def people(status: str | None = None, scope_id: str | None = None):
    values = [item for item in store.list_entities(status, scope_id=scope_id) if item["entity_type"] == "person"]
    pending_index = 0
    for item in values:
        if item["status"] == "pending":
            pending_index += 1
            item["canonical_name"] = item["display_name"] = f"待命名成员 {pending_index}"
            item["family_role"] = None
            item["summary"] = "由人脸聚类生成，等待用户确认"
        else:
            item["display_name"] = item["canonical_name"]
        item["confirmed"] = item["status"] == "confirmed"
        item["aliases"] = store.person_aliases(item["id"])
        item["profile"] = store.get_semantic_profile(item["id"])
        item["claims"] = store.list_semantic_claims(item["id"], 100)
        item["event_memory"] = store.list_person_event_memory(item["id"], scope_id)
        item["patterns"] = store.list_person_patterns(item["id"], scope_id)
    return {"people": values}


@app.get("/api/people/{person_id}/profile")
def person_profile(person_id: str):
    person = store.get_entity(person_id)
    if not person or person["entity_type"] != "person":
        raise HTTPException(status_code=404, detail="person not found")
    detail = store.get_entity_detail(person_id)
    detail["profile"] = store.get_semantic_profile(person_id)
    detail["claims"] = store.list_semantic_claims(person_id, 500)
    detail["event_memory"] = store.list_person_event_memory(person_id)
    detail["patterns"] = store.list_person_patterns(person_id)
    if detail.get("entity"):
        detail["entity"]["aliases"] = store.person_aliases(person_id)
    entity = detail.get("entity") or {}
    if entity.get("status") == "pending":
        entity["canonical_name"] = "待命名成员"
        entity["family_role"] = None
        entity["summary"] = "由人脸聚类生成，等待用户确认"
    return detail


@app.get("/api/people/{person_id}/evidence")
def person_evidence(person_id: str, scope_id: str | None = None):
    value = store.get_person_evidence(person_id)
    if not value or value["entity"].get("entity_type") != "person":
        raise HTTPException(status_code=404, detail="person not found")
    if scope_id and value.get("scope_id") != scope_id:
        raise HTTPException(status_code=404, detail="person not found in memory space")
    entity = value.get("entity") or {}
    if entity.get("status") == "pending":
        entity["canonical_name"] = "待命名成员"
        entity["family_role"] = None
        entity["summary"] = "由人脸聚类生成，等待用户确认"
    return value


@app.get("/api/entities")
def entities(status: str | None = None, includePeople: bool = False, scope_id: str | None = None):
    values = store.list_entities(status, scope_id=scope_id)
    if not includePeople:
        values = [item for item in values if item["entity_type"] != "person"]
    for item in values:
        if item.get("entity_type") == "person" and item.get("status") == "pending":
            item["canonical_name"] = "待命名成员"
            item["family_role"] = None
            item["summary"] = "由人脸聚类生成，等待用户确认"
    return {"entities": values}


@app.get("/api/entity-groups")
def entity_groups(scope_id: str | None = None):
    return {"groups": store.list_semantic_entity_groups(scope_id)}


@app.get("/api/entity-groups/{group_id}")
def entity_group(group_id: str, scope_id: str | None = None):
    value = store.get_semantic_entity_group(group_id, scope_id)
    if not value:
        raise HTTPException(status_code=404, detail="semantic entity group not found")
    return value


@app.get("/api/entity-merge-candidates")
def entity_merge_candidates(scope_id: str | None = None, status: str | None = "pending"):
    return {"candidates": store.list_entity_merge_candidates(scope_id, status)}


@app.post("/api/maintenance/entity-merge-candidates")
def derive_entity_merge_candidates(scope_id: str | None = None):
    if not maintenance_lock.acquire(blocking=False):
        raise HTTPException(status_code=409, detail="entity merge candidate generation is already running")
    maintenance_store = MemoryStore(store.path)
    try:
        return {"candidates": maintenance_store.derive_entity_merge_candidates(scope_id)}
    finally:
        maintenance_store.close()
        maintenance_lock.release()


@app.post("/api/entity-merge-candidates/{candidate_id}/confirm")
def confirm_entity_merge_candidate(candidate_id: str, payload: dict):
    try:
        return store.confirm_entity_merge_candidate(candidate_id, payload.get("target_entity_id"))
    except KeyError:
        raise HTTPException(status_code=404, detail="entity merge candidate not found")
    except ValueError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error


@app.post("/api/entity-merge-candidates/{candidate_id}/reject")
def reject_entity_merge_candidate(candidate_id: str):
    try:
        return store.reject_entity_merge_candidate(candidate_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="entity merge candidate not found")
    except ValueError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error


@app.get("/api/knowledge")
def knowledge(person_id: str | None = None, scope_id: str | None = None):
    claims = store.list_semantic_claims(person_id, 1000)
    if scope_id:
        claims = [claim for claim in claims if claim.get("scope_id") == scope_id]
    profiles = []
    if person_id:
        profile = store.get_semantic_profile(person_id)
        if profile:
            profiles.append(profile)
    else:
        profiles = [item for item in (store.get_semantic_profile(entity["id"]) for entity in store.list_entities(scope_id=scope_id)) if item]
    return {"profiles": profiles, "claims": claims, "spaces": store.list_memory_spaces()}


@app.post("/api/maintenance/reindex-entities")
def reindex_entities(scope_id: str | None = None):
    if not maintenance_lock.acquire(blocking=False):
        raise HTTPException(status_code=409, detail="entity reindex is already running")
    maintenance_store = MemoryStore(store.path)
    try:
        return maintenance_store.reindex_observation_entities(scope_id)
    finally:
        maintenance_store.close()
        maintenance_lock.release()


@app.get("/api/entities/{entity_id}")
def entity_detail(entity_id: str):
    value = store.get_entity_detail(entity_id)
    if not value:
        raise HTTPException(status_code=404, detail="entity not found")
    entity = value.get("entity") or {}
    if entity.get("entity_type") == "person" and entity.get("status") == "pending":
        entity["canonical_name"] = "待命名成员"
    return value


@app.put("/api/entities/{entity_id}/properties/{property_key}")
def set_entity_property(entity_id: str, property_key: str, payload: dict):
    try:
        return store.set_entity_property(entity_id, property_key, payload.get("value"), payload.get("evidence_ids") or [])
    except KeyError:
        raise HTTPException(status_code=404, detail="entity not found")
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error))


@app.get("/api/face-clusters")
def face_clusters(status: str | None = None, scope_id: str | None = None):
    clusters = store.list_face_clusters(status)
    if scope_id:
        clusters = [item for item in clusters if item.get("scope_id") == scope_id]
    for cluster in clusters:
        if cluster.get("entity_status") == "pending":
            cluster["canonical_name"] = "待命名成员"
            cluster["family_role"] = None
    return {"clusters": clusters}


@app.post("/api/face-clusters/{cluster_id}/confirm")
def confirm_face_cluster(cluster_id: str, payload: dict):
    name = str(payload.get("name") or "").strip()
    if not name:
        raise HTTPException(status_code=400, detail="entity name is required")
    value = store.confirm_face_cluster(cluster_id, name, str(payload.get("family_role") or "").strip() or None)
    if not value:
        raise HTTPException(status_code=404, detail="face cluster not found")
    if value.get("merged_into"):
        refreshed = _refresh_confirmed_person(value["entity"]["id"], value.get("refresh_counts", {}))
        refreshed["merged_into"] = value["merged_into"]
        refreshed["canonical_name"] = value.get("canonical_name") or (refreshed.get("entity") or {}).get("canonical_name")
        return refreshed
    return _refresh_confirmed_person(value["entity"]["id"], value.get("refresh_counts", {}))


def _refresh_confirmed_person(person_id: str, refresh_counts: dict | None = None):
    """Rebuild the person-rooted projections and return one authoritative response."""
    _analyze_confirmed_person_appearance(person_id)
    for event_id in store.entity_event_ids(person_id):
        pipeline.summarize_event(event_id)
    memory = store.rebuild_person_memory(person_id) or {}
    refreshed = store.get_person_evidence(person_id) or store.get_entity_detail(person_id)
    refreshed["semantic_profile"] = store.get_semantic_profile(person_id)
    refreshed["semantic_claims"] = store.list_semantic_claims(person_id, 500)
    refreshed["event_memory"] = store.list_person_event_memory(person_id)
    refreshed["patterns"] = store.list_person_patterns(person_id)
    counts = dict(refresh_counts or {})
    counts.update({
        "observations": len(memory.get("observation_ids", [])),
        "events": len(memory.get("event_ids", [])),
        "patterns": len(memory.get("patterns", [])),
        "claims": len(memory.get("claims", [])),
        "appearance": len(store.list_person_appearance_evidence(person_id, include_empty=True)),
    })
    refreshed["refresh_counts"] = counts
    if refreshed.get("entity"):
        refreshed["entity"]["aliases"] = store.person_aliases(person_id)
    return refreshed


def _analyze_confirmed_person_appearance(person_id: str):
    """Analyze at most one high-quality confirmed face per event for clothing."""
    selected = []
    for event_id in store.entity_event_ids(person_id):
        candidates = store._rows(
            """SELECT fi.id FROM face_instances fi JOIN entity_mentions em ON em.face_instance_id = fi.id
            JOIN event_observations eo ON eo.observation_id = fi.observation_id
            WHERE em.entity_id = ? AND eo.event_id = ? ORDER BY fi.quality DESC, fi.detection_confidence DESC LIMIT 1""",
            (person_id, event_id),
        )
        selected.extend(candidates)
    for candidate in selected:
        instance = store.get_face_instance(candidate["id"])
        if not instance or not Path(instance["asset_path"]).is_file():
            continue
        try:
            from PIL import Image

            image = Image.open(instance["asset_path"]).convert("RGB")
            crop, crop_bbox = expanded_person_crop(image, instance.get("bbox_json") or [])
            with tempfile.NamedTemporaryFile(suffix=".jpg", dir=DATA_DIR, delete=False) as temporary:
                crop.save(temporary, format="JPEG", quality=90)
                temp_path = Path(temporary.name)
            try:
                result = gamma.analyze_person_appearance(str(temp_path), {
                    "face_instance_id": instance["id"], "target_face_bbox": instance.get("bbox_json"),
                    "capture_time": (store.get_observation(instance["observation_id"]) or {}).get("captured_at"),
                })
            finally:
                temp_path.unlink(missing_ok=True)
            store.record_person_appearance_evidence(
                person_id, instance["id"], crop_bbox, result.get("clothing", []),
                result.get("confidence", 0.0), result.get("model") or gamma.model,
            )
        except (OSError, ValueError, RuntimeError):
            continue
    store.rebuild_person_memory(person_id)


@app.post("/api/face-clusters/{cluster_id}/reject")
def reject_face_cluster(cluster_id: str):
    try:
        value = store.reject_face_cluster(cluster_id)
    except ValueError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error
    if not value:
        raise HTTPException(status_code=404, detail="face cluster not found")
    return value


@app.post("/api/face-clusters/merge")
def merge_face_clusters(payload: dict):
    target = str(payload.get("target_cluster_id") or "")
    source = str(payload.get("source_cluster_id") or "")
    if not target or not source:
        raise HTTPException(status_code=400, detail="target_cluster_id and source_cluster_id are required")
    try:
        value = store.merge_face_clusters(target, source, "user_merge")
    except ValueError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error
    if not value:
        raise HTTPException(status_code=404, detail="face cluster not found")
    return value


@app.post("/api/face-clusters/{cluster_id}/split")
def split_face_cluster(cluster_id: str, payload: dict):
    face_instance_id = str(payload.get("face_instance_id") or "")
    if not face_instance_id:
        raise HTTPException(status_code=400, detail="face_instance_id is required")
    value = store.split_face_instance(cluster_id, face_instance_id, "user_split")
    if not value:
        raise HTTPException(status_code=404, detail="face instance not found in cluster")
    return value


@app.get("/api/relationships")
def relationships(scope_id: str | None = None, kind: str | None = None):
    if kind == "person":
        entities = [entity for entity in store.list_entities(scope_id=scope_id) if entity.get("entity_type") == "person"]
        values = store.list_person_relationships(scope_id=scope_id)
    else:
        entities = store.list_entities(scope_id=scope_id)
        values = store.list_relationships(scope_id=scope_id)
    nodes = [{"id": entity["id"], "label": entity["canonical_name"], "status": entity["status"], "entity_type": entity["entity_type"]} for entity in entities]
    edges = [{"source": item["subject_entity_id"], "target": item["object_entity_id"], "label": item["predicate"], "status": item["status"], "id": item["id"]} for item in values]
    return {"nodes": nodes, "edges": edges, "relationships": values}


@app.post("/api/relationships")
def create_relationship(payload: dict):
    required = (payload.get("subject_entity_id"), payload.get("predicate"), payload.get("object_entity_id"))
    if not all(required):
        raise HTTPException(status_code=400, detail="subject_entity_id, predicate and object_entity_id are required")
    value = store.create_relationship(*required, payload.get("evidence_ids") or [], float(payload.get("confidence", 0.5) or 0.5), payload.get("status", "pending"))
    return value


@app.post("/api/relationships/{relationship_id}/confirm")
def confirm_relationship(relationship_id: str):
    value = store.confirm_relationship(relationship_id)
    if not value:
        raise HTTPException(status_code=404, detail="relationship not found")
    return value


@app.post("/api/relationships/{relationship_id}/retract")
def retract_relationship(relationship_id: str):
    value = store.retract_relationship(relationship_id)
    if not value:
        raise HTTPException(status_code=404, detail="relationship not found")
    return value


@app.post("/api/persons/{person_id}/confirm")
def confirm_person(person_id: str, payload: dict | None = None):
    name = (payload or {}).get("name", "").strip()
    if not name:
        raise HTTPException(status_code=400, detail="person name is required")
    family_role = str((payload or {}).get("family_role") or "").strip() or None
    native = store.confirm_person_entity(person_id, name, family_role)
    if native:
        if native.get("merged_into"):
            refreshed = _refresh_confirmed_person(native["entity"]["id"], native.get("refresh_counts", {}))
            refreshed["merged_into"] = native["merged_into"]
            refreshed["canonical_name"] = native.get("canonical_name") or (refreshed.get("entity") or {}).get("canonical_name")
            return refreshed
        return _refresh_confirmed_person(native["entity"]["id"], native.get("refresh_counts", {}))
    value = store.update_person(person_id, name, "confirmed")
    if not value:
        raise HTTPException(status_code=404, detail="person not found")
    return value


@app.post("/api/people/{person_id}/rename")
def rename_person(person_id: str, payload: dict | None = None):
    new_name = str((payload or {}).get("name") or "").strip()
    if not new_name:
        raise HTTPException(status_code=400, detail="person name is required")
    detail = store.rename_person(person_id, new_name)
    if not detail:
        raise HTTPException(status_code=404, detail="person not found")
    refreshed = _refresh_confirmed_person(person_id)
    refreshed["aliases"] = store.person_aliases(person_id)
    if detail.get("semantic_claims"):
        refreshed["semantic_claims"] = detail["semantic_claims"]
    return refreshed


@app.post("/api/persons/{person_id}/reject")
def reject_person(person_id: str):
    # /api/people returns native person entities; reject must follow that path.
    try:
        native = store.reject_person_entity(person_id)
    except ValueError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error
    if native:
        return native
    value = store.update_person(person_id, status="rejected")
    if not value:
        raise HTTPException(status_code=404, detail="person not found")
    return value


@app.get("/api/observations")
def observations(assetId: str | None = None, scope_id: str | None = None, limit: int = 200):
    values = store.list_observations(max(1, min(limit, 1000)), scope_id=scope_id)
    if assetId:
        values = [item for item in values if item["asset_id"] == assetId]
    return {"observations": values}


@app.get("/api/observations/{observation_id}")
def observation(observation_id: str):
    value = store.get_observation(observation_id)
    if not value:
        raise HTTPException(status_code=404, detail="observation not found")
    value["asset"] = store.get_asset(value["asset_id"])
    return value


@app.get("/api/assets/{asset_id}")
def asset(asset_id: str):
    value = store.get_asset(asset_id)
    if not value:
        raise HTTPException(status_code=404, detail="asset not found")
    return value


@app.get("/api/assets")
def assets(mediaType: str | None = None, status: str | None = None, scope_id: str | None = None, limit: int = 200):
    return {"assets": store.list_assets(mediaType, status, max(1, min(limit, 1000)), scope_id=scope_id)}


@app.get("/api/assets/{asset_id}/file")
def asset_file(asset_id: str, original: bool = False):
    value = store.get_asset(asset_id)
    path = Path(value["path"]) if value else None
    if not value or not path or not path.is_file():
        raise HTTPException(status_code=404, detail="asset file not found")
    # Keep an escape hatch for downloading the untouched HEIC source.
    if original or not needs_browser_transcode(path, value.get("mime_type")):
        return FileResponse(
            path,
            media_type=value.get("mime_type") or "application/octet-stream",
            filename=value.get("file_name"),
        )
    preview = PREVIEW_DIR / f"{asset_id}.jpg"
    try:
        source_mtime = path.stat().st_mtime
        if not preview.is_file() or preview.stat().st_mtime < source_mtime:
            preview.write_bytes(encode_jpeg_preview(path))
        return FileResponse(
            preview,
            media_type="image/jpeg",
            filename=f"{Path(value.get('file_name') or asset_id).stem}.jpg",
            headers={"Cache-Control": "private, max-age=86400"},
        )
    except OSError as error:
        raise HTTPException(status_code=422, detail=f"unable to render preview: {error}") from error


@app.get("/api/face-instances/{face_instance_id}/crop")
def face_instance_crop(face_instance_id: str):
    instance = store.get_face_instance(face_instance_id)
    if not instance or not Path(instance["asset_path"]).is_file():
        raise HTTPException(status_code=404, detail="face instance not found")
    try:
        from PIL import Image, ImageOps

        image = ImageOps.exif_transpose(Image.open(instance["asset_path"])).convert("RGB")
        left, top, right, bottom = (int(value) for value in instance.get("bbox_json") or [])
        left, top = max(0, left), max(0, top)
        right, bottom = min(image.width, right), min(image.height, bottom)
        if right <= left or bottom <= top:
            raise ValueError("invalid face bounding box")
        face = image.crop((left, top, right, bottom))
        face.thumbnail((256, 256))
        output = BytesIO()
        face.save(output, format="JPEG", quality=88)
        return Response(content=output.getvalue(), media_type="image/jpeg", headers={"Cache-Control": "private, max-age=86400"})
    except (OSError, ValueError):
        raise HTTPException(status_code=422, detail="face crop is unavailable")


@app.get("/api/facts")
def facts(status: str | None = None):
    values = store.list_facts(1000)
    return {"facts": [item for item in values if not status or item["status"] == status]}


@app.get("/api/stories")
def stories():
    return {"stories": store.list_stories()}


@app.post("/api/stories")
def create_story(payload: dict):
    event_ids = payload.get("event_ids") or []
    if event_ids and not payload.get("content"):
        evidence = []
        person_freq = {}
        self_names = set()
        place_freq = {}
        days = set()
        time_values = []
        object_set = set()
        for event_id in event_ids:
            detail = store.get_event_detail(event_id)
            if not detail:
                continue
            observations = []
            for item in detail["observations"]:
                asset = item.get("asset") or {}
                people = []
                for p in (item.get("people") or []):
                    if not p.get("entity_id"):
                        continue
                    is_self = bool(store._row(
                        "SELECT 1 FROM entity_properties WHERE entity_id=? AND property_key='is_self' AND value_json='true'",
                        (p.get("entity_id"),),
                    ))
                    people.append({"name": p.get("name"), "is_self": is_self})
                    name = p.get("name") or "未知"
                    person_freq[name] = person_freq.get(name, 0) + 1
                    if is_self:
                        self_names.add(name)
                observations.append({
                    "id": item.get("id"),
                    "caption": item.get("caption"),
                    "transcript": item.get("transcript"),
                    "captured_at": item.get("captured_at"),
                    "place": item.get("place"),
                    "objects": item.get("objects"),
                    "captured_location": asset.get("captured_location"),
                    "people": people,
                })
                captured = item.get("captured_at")
                if captured:
                    days.add(str(captured)[:10])
                    time_values.append(str(captured))
                for obj in (item.get("objects") or []):
                    label = obj if isinstance(obj, str) else (obj.get("label") or obj.get("primary") or "")
                    if label:
                        object_set.add(label)
                location = asset.get("captured_location")
                if location:
                    try:
                        lat, lon = (float(part) for part in str(location).replace(" ", "").split(","))
                        cluster = f"{round(lat, 1)},{round(lon, 1)}"
                    except Exception:
                        cluster = str(location)
                else:
                    cluster = item.get("place") or "未知"
                place_freq[cluster] = place_freq.get(cluster, 0) + 1
            event_start = (detail["event"] or {}).get("time_start")
            event_end = (detail["event"] or {}).get("time_end")
            if event_start:
                days.add(str(event_start)[:10])
            if event_end:
                days.add(str(event_end)[:10])
            evidence.append({"event": detail["event"], "observations": observations})
        if evidence:
            photo_count = sum(len(ev["observations"]) for ev in evidence)
            event_count = len(evidence)
            days_count = len(days)
            object_count = len(object_set)
            time_span = f"{min(time_values)[:10]} 至 {max(time_values)[:10]}" if len(time_values) > 1 else (time_values[0][:10] if time_values else "")
            # topPerson:排除相册主人(我)后的最高频陪伴人物;叙事主语=我+topPerson
            non_self_freq = {key: value for key, value in person_freq.items() if key not in self_names}
            top_person = max(non_self_freq, key=non_self_freq.get) if non_self_freq else None
            subject = f"我和{top_person}" if top_person else "我"
            # 地点组:GPS聚类给字母标签,附经纬度,不编地名
            cluster_list = sorted(place_freq.items(), key=lambda item: -item[1])
            cluster_labels = {}
            place_lines = []
            for index, (cluster, count) in enumerate(cluster_list):
                label = f"地点组{chr(ord('A') + index)}"
                cluster_labels[cluster] = label
                place_lines.append(f"{label}(经纬度{cluster},{count}张)")
            top_label = cluster_labels[cluster_list[0][0]] if cluster_list else ""
            other_labels = "、".join(cluster_labels[c] for c, _ in cluster_list[1:]) or "无"
            # 代表事件一句话,给叙事具体画面(防空洞)
            representative = []
            for ev in evidence:
                e = ev["event"] or {}
                one_line = (e.get("summary") or "").strip() or (e.get("title") or "").strip()
                if one_line and len(representative) < 3:
                    representative.append(one_line[:80])
            stats = (
                f"时间跨度:{time_span or '未知'};天数:{days_count}天;事件数:{event_count}个;"
                f"照片数:{photo_count}张;物件数:{object_count}件;"
                f"地点分布:{'、'.join(place_lines) or '无'};"
                f"出现最多的地点组(叙述需占约2/3篇幅):{top_label};其他地点组合计约1/3:{other_labels};"
                f"相册主人(我):{'、'.join(sorted(self_names)) or '无'};陪伴人物:{top_person or '无'}"
            )
            prompt = (
                "根据下面的真实家庭事件和证据生成故事初稿。不要补造人物、地点或时间，只能使用证据。\n"
                "统计概要(系统已算出,仅供叙事参考,不得改动或编造):\n" + stats + "\n"
                "规则:\n"
                "1. 严格返回JSON:title、content、outline(数组)。使用中文,content约400字(照片多可略长)。\n"
                "2. 叙事以" + subject + "为主语主角,第一人称\"我\"。"
                + ("陪伴人物" + top_person + "与\"我\"是并肩关系,绝不把\"我\"写成\"我与自己相伴\"。"
                   if top_person else "叙述\"我\"自身的经历,第一人称。") + "\n"
                "3. 证据中is_self=true的人物=相册主人\"我\"本人。\n"
                "4. 出现最多的地点组要占叙事约2/3篇幅,其他地点组合计约1/3。\n"
                "5. 地点可依据经纬度合理推断城市(如22.5,114.1疑似深圳)并使用,但不得编造未经证据支持的具体地名(商场/餐厅/景点等);若不确定就笼统表述。\n"
                "6. 参考这些代表性画面,让叙事有具体细节而非统计堆砌:\n"
                + "\n".join("· " + text for text in representative) + "\n"
                "证据:" + str(evidence)
            )
            try:
                generated = parse_json_response(gamma.chat(prompt))
                payload = {**payload, "title": payload.get("title") or generated.get("title"), "content": generated.get("content", ""), "outline": generated.get("outline", [])}
            except Exception:
                pass
    return store.create_story(payload)


@app.patch("/api/stories/{story_id}")
def update_story(story_id: str, payload: dict):
    value = store.update_story(story_id, payload)
    if not value:
        raise HTTPException(status_code=404, detail="story not found")
    return value


@app.delete("/api/stories/{story_id}")
def delete_story(story_id: str):
    value = store.delete_story(story_id)
    if not value:
        raise HTTPException(status_code=404, detail="story not found")
    return value


@app.post("/api/invites")
def create_invite(payload: dict):
    invite = store.create_invite(payload.get("label", "家庭成员"))
    return {**invite, "invite_url": f"sentrix://join/{invite['token']}"}


def assistant_response(result):
    """Expose stable browser names while retaining the internal contract.

    RX-6: retrieval/tool traces, the validation block and the model-call ledger
    are debug-only.  They stay out of the default API response unless the admin
    presentation switch is on; the frontend additionally hides them behind its
    own debug layer.
    """
    from .validation import full_chain_profile as _prof
    admin = _prof.admin_debug_presentation()
    result.setdefault("claims", [])
    result.setdefault("claim_verifications", [])
    result.setdefault("claim_verification_status", "not_required")
    result.setdefault("repair_count", 0)
    result.setdefault("evidence_bundles", [])
    result.setdefault("claim_evidence_index", {})
    result.setdefault("segments", [{"type": "text", "text": result.get("answer", "")}])
    result["retrievalTrace"] = result.get("retrieval_trace", [])
    result["toolTrace"] = result.get("tool_trace", [])
    result["evidencePresentation"] = result.get("evidence_presentation", {})
    result["memoryUsed"] = result.get("memory_used", False)
    result["evidenceRequired"] = result.get("evidence_required", False)
    result["evidenceStatus"] = result.get("evidence_status", "not_applicable")
    result["originalEvidenceRequested"] = result.get("original_evidence_requested", False)
    result["claimVerifications"] = result["claim_verifications"]
    result["claimVerificationStatus"] = result["claim_verification_status"]
    result["repairCount"] = result["repair_count"]
    result["evidenceBundles"] = result["evidence_bundles"]
    result["claimEvidenceIndex"] = result["claim_evidence_index"]
    if not admin:
        result["retrievalTrace"] = []
        result["toolTrace"] = []
        result.pop("validation", None)
        result.pop("model_call_ledger", None)
        # TFPE v2 structured internals are debug-only.
        result.pop("task_contract", None)
        result.pop("retrieval_strategy", None)
        result.pop("structured_result", None)
        result.pop("parser_raw", None)
    return result


@app.post("/api/search")
def search(request: SearchRequest):
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="query is required")
    return assistant_response(agent.answer_turn(request.query.strip(), scope_id=request.spaceId))


class AssistantTurnRequest(BaseModel):
    message: str
    conversation_id: str | None = None
    feedback: dict | None = None
    scope_id: str = "home-default"
    selected_entity_id: str | None = None
    viewer_id: str = "owner"


@app.post("/api/assistant/turn")
def assistant_turn(request: AssistantTurnRequest):
    if not request.message.strip():
        raise HTTPException(status_code=400, detail="message is required")
    result = agent.answer_turn(
        request.message.strip(), request.conversation_id, request.feedback, request.scope_id,
        request.selected_entity_id, request.viewer_id,
    )
    return assistant_response(result)


@app.post("/api/ingest", status_code=202)
async def ingest(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    sourceOwnerId: str | None = Form(None),
    sourceDeviceId: str | None = Form(None),
    sourceAlbumId: str | None = Form(None),
    capturedAt: str | None = Form(None),
    capturedLocation: str | None = Form(None),
):
    safe_name = Path(file.filename or "upload.bin").name
    asset_id = make_id("asset")
    destination = MEDIA_DIR / f"{asset_id}_{safe_name}"
    with destination.open("wb") as output:
        shutil.copyfileobj(file.file, output)
    media_type = media_type_from_upload(file.content_type, safe_name)
    mime_type = file.content_type or guess_mime_type(safe_name)
    metadata = {
        "source_owner_id": sourceOwnerId,
        "source_device_id": sourceDeviceId,
        "source_album_id": sourceAlbumId,
        "source_confidence": 1.0 if sourceOwnerId else 0.0,
        "captured_at": capturedAt,
        "captured_location": capturedLocation,
        "content_sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
        "exif": pipeline._extract_exif(destination) if media_type == "image" else {},
    }
    metadata["captured_at"] = metadata["captured_at"] or metadata["exif"].get("captured_at")
    metadata["source_device_id"] = metadata["source_device_id"] or metadata["exif"].get("device")
    existing = store.find_asset_by_hash(metadata["content_sha256"])
    if existing:
        destination.unlink(missing_ok=True)
        return {"accepted": True, "assetId": existing["id"], "fileName": existing["file_name"], "status": existing["status"], "mediaType": existing["media_type"], "deduplicated": True}
    created = store.create_asset(
        asset_id,
        safe_name,
        media_type,
        str(destination),
        mime_type,
        destination.stat().st_size,
        metadata,
    )
    background_tasks.add_task(process_asset, asset_id)
    return {"accepted": True, "assetId": created["id"], "fileName": safe_name, "status": "queued", "mediaType": media_type}


@app.post("/api/import", status_code=202)
async def import_remote_files(
    background_tasks: BackgroundTasks,
    files: list[UploadFile] = File(...),
    metadata: str | None = Form(None),
    sourceOwnerId: str | None = Form(None),
    sourceDeviceId: str | None = Form(None),
    sourceAlbumId: str | None = Form(None),
    sourceOwnerLabel: str | None = Form(None),
    scopeId: str | None = Form(None),
    scope_id: str | None = Form(None),
    batchId: str | None = Form(None),
    batch_id: str | None = Form(None),
):
    if not files:
        raise HTTPException(status_code=422, detail="at least one file is required")
    if len(files) > MAX_REMOTE_IMPORT_FILES:
        raise HTTPException(status_code=413, detail=f"too many files: {len(files)} > {MAX_REMOTE_IMPORT_FILES}")
    try:
        per_file = json.loads(metadata) if metadata else []
    except json.JSONDecodeError as error:
        raise HTTPException(status_code=422, detail="metadata must be a JSON array") from error
    if not isinstance(per_file, list) or len(per_file) not in {0, len(files)}:
        raise HTTPException(status_code=422, detail="metadata must contain one object per file")
    per_file = per_file or [{} for _ in files]
    scope = (scope_id or scopeId or "home-default").strip() or "home-default"
    batch = (batch_id or batchId or make_id("batch")).strip()
    store.create_memory_space(scope, scope, kind="benchmark")
    store.create_ingest_batch(batch, scope)
    items = []
    for index, upload in enumerate(files):
        safe_name = Path(upload.filename or f"upload-{index}").name
        destination = MEDIA_DIR / f"{make_id('upload')}_{safe_name}"
        try:
            with destination.open("wb") as output:
                shutil.copyfileobj(upload.file, output)
            capture = _normalized_capture_metadata(per_file[index])
            media_type = (upload.content_type or "application/octet-stream").split("/", 1)[0]
            if media_type not in {"image", "audio", "video", "text"}:
                media_type = "text"
            created = pipeline.create_asset(destination, file_name=safe_name, media_type=media_type, mime_type=upload.content_type, metadata={
                "scope_id": scope, "batch_id": batch, "source_owner_id": sourceOwnerId,
                "source_owner_label": sourceOwnerLabel, "source_device_id": sourceDeviceId,
                "source_album_id": sourceAlbumId, "source_confidence": 1.0 if sourceOwnerId else 0.0,
                **capture,
            })
            deduplicated = created.get("path") != str(destination)
            if deduplicated:
                destination.unlink(missing_ok=True)
            elif created.get("status") in {"queued", "failed"}:
                background_tasks.add_task(process_asset, created["id"])
            items.append({"accepted": True, "assetId": created["id"], "asset_id": created["id"], "fileName": created["file_name"], "status": created["status"], "scope_id": created.get("scope_id"), "batch_id": created.get("batch_id"), "deduplicated": deduplicated})
        except ValueError as error:
            destination.unlink(missing_ok=True)
            items.append({"accepted": False, "fileName": safe_name, "status": "rejected", "error": str(error)})
        except Exception as error:
            destination.unlink(missing_ok=True)
            items.append({"accepted": False, "fileName": safe_name, "status": "failed", "error": str(error)})
    return {"accepted": any(item["accepted"] for item in items), "batch_id": batch, "scope_id": scope, "items": items, "accepted_count": sum(item["accepted"] for item in items), "rejected_count": sum(not item["accepted"] for item in items)}


@app.post("/api/import/server-directory", status_code=202)
def import_assets(request: ImportRequest, background_tasks: BackgroundTasks):
    source = Path(request.source_path).expanduser().resolve()
    _assert_import_path_allowed(source)
    if not source.exists():
        raise HTTPException(status_code=404, detail="source_path not found")
    scope = (request.scope_id or "home-default").strip() or "home-default"
    batch_id = (request.batch_id or make_id("batch")).strip()
    store.create_memory_space(scope, scope, kind="benchmark", source_path=str(source))
    store.create_ingest_batch(batch_id, scope)
    if source.is_file():
        candidates = [source]
    else:
        iterator = source.rglob(request.glob) if request.recursive else source.glob(request.glob)
        candidates = [path for path in iterator if path.is_file()]
    candidates = [path for path in candidates if path.suffix.lower() in SUPPORTED_IMPORT_SUFFIXES]
    if len(candidates) > max(1, request.max_files):
        raise HTTPException(status_code=413, detail=f"too many files matched: {len(candidates)} > {request.max_files}")
    imported = []
    skipped = []
    for path in candidates:
        metadata = {
            "scope_id": scope,
            "batch_id": batch_id,
            "source_owner_id": request.source_owner_id,
            "source_owner_label": request.source_owner_label,
            "source_device_id": request.source_device_id,
            "source_album_id": request.source_album_id,
            "source_confidence": 1.0 if request.source_owner_id else 0.0,
            "captured_at": request.captured_at,
            "captured_location": request.captured_location,
        }
        target = path
        if request.copy_file:
            target = MEDIA_DIR / f"{make_id('import')}_{path.name}"
            shutil.copy2(path, target)
        try:
            created = pipeline.create_asset(target, file_name=path.name, metadata=metadata)
        except Exception as error:
            if request.copy_file:
                target.unlink(missing_ok=True)
            skipped.append({"path": str(path), "reason": str(error)})
            continue
        deduplicated = created.get("path") != str(target)
        if request.copy_file and deduplicated:
            target.unlink(missing_ok=True)
        elif created.get("status") in {"queued", "failed"}:
            background_tasks.add_task(process_asset, created["id"])
        imported.append({
            "asset_id": created["id"],
            "file_name": created["file_name"],
            "media_type": created["media_type"],
            "status": created["status"],
            "deduplicated": deduplicated,
            "source_path": str(path),
        })
    return {
        "accepted": True,
        "scope_id": scope,
        "batch_id": batch_id,
        "matched": len(candidates),
        "accepted_count": len(imported),
        "assets": imported,
        "skipped": skipped,
        "batch": _batch_status(batch_id),
        "identity_confirmation_required": True,
    }


@app.post("/api/ingest-batches", status_code=201)
def create_ingest_batch(request: IngestBatchCreateRequest):
    scope = (request.scope_id or "home-default").strip() or "home-default"
    batch_id = (request.batch_id or make_id("batch")).strip()
    store.create_memory_space(scope, request.name or scope, kind=request.kind or "benchmark", source_path=request.source_path)
    store.create_ingest_batch(batch_id, scope)
    return _batch_status(batch_id)


@app.get("/api/ingest-batches/{batch_id}")
def ingest_batch(batch_id: str):
    return _batch_status(batch_id)


@app.post("/api/ingest-batches/{batch_id}/complete")
def complete_ingest_batch(batch_id: str, background_tasks: BackgroundTasks):
    if not store.get_ingest_batch(batch_id):
        raise HTTPException(status_code=404, detail="ingest batch not found")
    store.complete_ingest_batch(batch_id)
    background_tasks.add_task(pipeline.finalize_ingest_batch, batch_id)
    return _batch_status(batch_id)


@app.post("/api/maintenance/recheck")
def recheck(background_tasks: BackgroundTasks):
    assets = [store.get_asset(row["id"]) for row in store._rows("SELECT id FROM assets WHERE status IN ('queued', 'failed', 'semantic_enriching') ORDER BY created_at")]
    for item in assets:
        background_tasks.add_task(process_asset, item["id"])
    return {"accepted": len(assets), "status": "recheck-queued"}


@app.post("/api/maintenance/summarize-events")
def summarize_pending_events(background_tasks: BackgroundTasks, scope_id: str | None = None, limit: int = 100):
    accepted = len([
        event for event in store.list_events(max(1, limit), scope_id)
        if event.get("title") == "待总结事件"
    ])
    background_tasks.add_task(pipeline.summarize_pending_events, scope_id, max(1, limit))
    return {"accepted": accepted, "status": "event-summary-queued"}


@app.get("/api/query-gaps")
def query_gaps(status: str | None = None):
    return {"queryGaps": store.list_query_gaps(status)}


@app.post("/api/query-gaps/{gap_id}/feedback")
def query_gap_feedback(gap_id: str, payload: dict):
    if not store.get_query_gap(gap_id):
        raise HTTPException(status_code=404, detail="query gap not found")
    return store.add_memory_feedback(
        gap_id,
        payload.get("user_id"),
        payload.get("accepted_answer"),
        payload.get("correction"),
        payload.get("target_claim_id"),
        payload.get("target_entity_id"),
        payload.get("target_event_id"),
        payload.get("target_property_key"),
    )


@app.post("/api/facts/{fact_id}/confirm")
def confirm_fact(fact_id: str):
    try:
        return store.confirm_fact(fact_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="fact not found")


@app.post("/api/facts/{fact_id}/reject")
def reject_fact(fact_id: str):
    if not store.get_fact(fact_id):
        raise HTTPException(status_code=404, detail="fact not found")
    return store.reject_fact(fact_id)
