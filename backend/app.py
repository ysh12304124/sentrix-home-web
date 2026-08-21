import asyncio
import contextlib
import json
import os
import shutil
import hashlib
import tempfile
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from io import BytesIO
from pathlib import Path

import httpx
from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, Response, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .agent_conversation import ConversationStore
from .db import MemoryStore, make_id
from .image_io import (
    encode_jpeg_preview,
    ensure_heif_support,
    guess_mime_type,
    media_type_from_upload,
    needs_browser_transcode,
)
from .model_clients import ClipAdapter, FaceAdapter, FunASRClient, GammaClient, align_face_crop, parse_json_response
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
conversation_store = ConversationStore(store)
CONVERSATION_STORE_ENABLED = os.getenv("SENTRIX_CONVERSATION_STORE_V1", "0").lower() in {"1", "true", "on"}

app = FastAPI(title="Sentrix Home Memory API", version="0.1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
maintenance_lock = threading.Lock()
runtime_lock = threading.Lock()
batch_worker_lock = threading.Lock()
db_write_lock = threading.RLock()
active_batch_workers = set()
VLLM_MANAGER = Path(os.getenv("SENTRIX_VLLM_MANAGER", "/home/asus/sentrix-vllm/bin/sentrix_vllm_manager.py"))
VLLM_REGISTRY = Path(os.getenv("SENTRIX_VLLM_REGISTRY", "/home/asus/sentrix-vllm/registry.json"))
VLLM_API_URL = os.getenv("SENTRIX_VLLM_API_URL", "").strip()
RUNTIME_VLLM_API_URL = None
RUNTIME_VLLM_BASE_URL = None
SUPPORTED_IMPORT_SUFFIXES = {
    ".jpg", ".jpeg", ".png", ".webp", ".heic", ".bmp", ".gif",
    ".mp4", ".mov", ".m4v", ".avi", ".mkv", ".mp3", ".wav", ".m4a",
    ".txt", ".md", ".json",
}
MAX_REMOTE_IMPORT_FILES = int(os.getenv("SENTRIX_MAX_REMOTE_IMPORT_FILES", "500"))


@contextlib.contextmanager
def db_write_guard(label: str = "db-write", timeout: float = 120.0):
    """Serialize SQLite writes across the request store and background worker stores.

    Background pipeline workers own separate sqlite3 connections (see process_asset),
    while import endpoints write through the request-scoped ``store``.  SQLite WAL
    allows only one writer, so all write transactions must be serialized in-process.
    A timed acquire turns a silent deadlock into an actionable failure instead of
    hanging the request forever.
    """
    started = time.perf_counter()
    if not db_write_lock.acquire(timeout=timeout):
        raise RuntimeError(
            f"timed out acquiring SQLite write lock ({label}) after {timeout:.0f}s; "
            "a concurrent writer is stuck"
        )
    try:
        waited = time.perf_counter() - started
        if waited > 0.5:
            print(f"[db-write-lock] {label} waited {waited:.2f}s", flush=True)
        yield
    finally:
        db_write_lock.release()


def _upload_destination(identifier, safe_name, media_type):
    if media_type == "video":
        directory = MEDIA_DIR / "videos" / identifier
        directory.mkdir(parents=True, exist_ok=True)
        return directory / f"original{Path(safe_name).suffix or '.mp4'}"
    return MEDIA_DIR / f"{identifier}_{safe_name}"


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
        Path("/home/asus/data"),
        Path("/home/asus/datasets"),
        Path("/home/asus/benchmarks"),
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
        "pipeline_metrics": (batch.get("metadata_json") or {}).get("pipeline_metrics") or {},
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


def _pipeline_worker_limits():
    configured = max(1, int(os.getenv("SENTRIX_PIPELINE_MAX_WORKERS", "2")))
    state = _load_vllm_state() or {}
    service_limit = max(1, int(state.get("max_num_seqs") or 1))
    summary_configured = max(1, int(os.getenv("SENTRIX_EVENT_SUMMARY_MAX_WORKERS", "2")))
    return {
        "configured_workers": configured,
        "vllm_max_num_seqs": service_limit,
        "effective_workers": min(configured, service_limit),
        "event_summary_workers": min(summary_configured, service_limit),
    }


def _prepare_asset_stage(asset_id, stage):
    worker_store = MemoryStore(store.path)
    worker_pipeline = IngestionPipeline(
        worker_store, gamma=gamma, asr=pipeline.asr, face=pipeline.face, clip=pipeline.clip,
    )
    try:
        if stage == "fast":
            return worker_pipeline.prepare_fast_image(asset_id)
        if stage == "semantic":
            return worker_pipeline.prepare_semantic_image(asset_id)
        raise ValueError(f"unknown pipeline stage: {stage}")
    finally:
        worker_store.close()


def _summarize_event_worker(event_id):
    worker_store = MemoryStore(store.path)
    worker_pipeline = IngestionPipeline(
        worker_store, gamma=gamma, asr=pipeline.asr, face=pipeline.face, clip=pipeline.clip,
    )
    started_at = time.perf_counter()
    try:
        worker_pipeline.summarize_event(event_id)
        return {"event_id": event_id, "seconds": round(time.perf_counter() - started_at, 4), "status": "completed"}
    except Exception as error:
        return {"event_id": event_id, "seconds": round(time.perf_counter() - started_at, 4), "status": "failed", "error": str(error)}
    finally:
        worker_store.close()


def _pipeline_timing_summary(task_store, asset_ids):
    values = {}
    for asset_id in asset_ids:
        asset = task_store.get_asset(asset_id) or {}
        asset_metadata = asset.get("metadata_json") or {}
        timings = {
            **(asset_metadata.get("import_timings") or {}),
            **(asset_metadata.get("processing_timings") or {}),
        }
        for key, value in timings.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                values.setdefault(key, []).append(float(value))

    def percentile(items, ratio):
        ordered = sorted(items)
        if not ordered:
            return None
        index = min(len(ordered) - 1, max(0, int(round((len(ordered) - 1) * ratio))))
        return ordered[index]

    return {
        key: {
            "count": len(items),
            "sum_seconds": round(sum(items), 4),
            "mean_seconds": round(sum(items) / len(items), 4),
            "p50_seconds": round(percentile(items, 0.50), 4),
            "p95_seconds": round(percentile(items, 0.95), 4),
            "max_seconds": round(max(items), 4),
        }
        for key, items in values.items() if items
    }


def _process_ingest_asset_group(asset_ids, batch_id, finalize_batch=False):
    """Parallelize model work while committing clustering state in upload order."""
    asset_ids = list(dict.fromkeys(asset_ids or []))
    task_store = MemoryStore(store.path)
    task_pipeline = IngestionPipeline(
        task_store, gamma=gamma, asr=pipeline.asr, face=pipeline.face, clip=pipeline.clip,
    )
    limits = _pipeline_worker_limits()
    started_at = time.perf_counter()
    with db_write_guard("ingest-group-start"):
        task_store.update_ingest_batch_metadata(batch_id, {
            "pipeline_metrics": {**limits, "status": "processing", "asset_count": len(asset_ids)}
        })
    try:
        image_ids = []
        for asset_id in asset_ids:
            asset = task_store.get_asset(asset_id) or {}
            if asset.get("media_type") == "image" and asset.get("status") in {"queued", "failed"}:
                with db_write_guard("ingest-group-mark-processing"):
                    task_store.update_asset(asset_id, "processing", {
                        "processing_timings": {"queue_wait_seconds": round(time.perf_counter() - started_at, 4)}
                    })
                image_ids.append(asset_id)
            elif asset.get("status") in {"queued", "failed", "video-queued"}:
                process_asset(asset_id)

        with ThreadPoolExecutor(max_workers=limits["effective_workers"], thread_name_prefix="sentrix-batch-fast") as executor:
            futures = {asset_id: executor.submit(_prepare_asset_stage, asset_id, "fast") for asset_id in image_ids}
            for asset_id in image_ids:
                try:
                    prepared = futures[asset_id].result()
                    with db_write_guard("ingest-commit-fast"):
                        task_pipeline.commit_fast_image(asset_id, prepared)
                except Exception as error:
                    with db_write_guard("ingest-commit-fast-error"):
                        task_store.cleanup_asset_derivatives(asset_id)
                        task_store.update_asset(asset_id, "failed", {"error": str(error), "failed_stage": "fast"})

        semantic_ids = [
            asset_id for asset_id in image_ids
            if (task_store.get_asset(asset_id) or {}).get("status") == "semantic_enriching"
        ]
        with ThreadPoolExecutor(max_workers=limits["effective_workers"], thread_name_prefix="sentrix-batch-semantic") as executor:
            futures = {asset_id: executor.submit(_prepare_asset_stage, asset_id, "semantic") for asset_id in semantic_ids}
            for asset_id in semantic_ids:
                try:
                    prepared = futures[asset_id].result()
                    with db_write_guard("ingest-commit-semantic"):
                        task_pipeline.commit_semantic_image(asset_id, prepared, summarize_event=False)
                except Exception as error:
                    with db_write_guard("ingest-commit-semantic-error"):
                        task_store.update_asset(asset_id, "failed", {"error": str(error), "failed_stage": "semantic"})

        if finalize_batch:
            task_store.complete_ingest_batch(batch_id)
        if finalize_batch and task_store.claim_ingest_batch_summary(batch_id):
            event_ids = task_store.batch_event_ids(batch_id)
            summary_started = time.perf_counter()
            with ThreadPoolExecutor(max_workers=limits["event_summary_workers"], thread_name_prefix="sentrix-event-summary") as executor:
                event_results = list(executor.map(_summarize_event_worker, event_ids))
            summary_wall_seconds = round(time.perf_counter() - summary_started, 4)
            task_store.finish_ingest_batch(batch_id)
        else:
            event_ids = []
            event_results = []
            summary_wall_seconds = 0.0

        metrics = {
            **limits,
            "status": "completed" if finalize_batch else "processing",
            "asset_count": len(asset_ids),
            "image_count": len(image_ids),
            "event_count": len(event_ids),
            "event_summary_call_count": len(event_results),
            "event_summary_wall_seconds": summary_wall_seconds,
            "event_summaries": event_results,
            "stage_timings": _pipeline_timing_summary(task_store, asset_ids),
            "total_wall_seconds": round(time.perf_counter() - started_at, 4),
        }
        with db_write_guard("ingest-group-metrics"):
            task_store.update_ingest_batch_metadata(batch_id, {"pipeline_metrics": metrics})
    except Exception as error:
        with db_write_guard("ingest-group-metrics-error"):
            task_store.update_ingest_batch_metadata(batch_id, {
                "pipeline_metrics": {
                    **limits, "status": "failed", "asset_count": len(asset_ids),
                    "total_wall_seconds": round(time.perf_counter() - started_at, 4), "error": str(error),
                }
            })
        raise
    finally:
        task_store.close()


def process_ingest_batch(asset_ids, batch_id):
    """Consume a batch incrementally while uploads continue, then finalize once."""
    with batch_worker_lock:
        if batch_id in active_batch_workers:
            return
        active_batch_workers.add(batch_id)
    task_store = MemoryStore(store.path)
    all_asset_ids = list(dict.fromkeys(asset_ids or []))
    pipeline_started_at = time.perf_counter()
    try:
        first = True
        while True:
            rows = task_store._rows(
                "SELECT id FROM assets WHERE batch_id = ? AND status IN ('queued', 'failed', 'video-queued') ORDER BY created_at, id",
                (batch_id,),
            )
            queued_ids = [row["id"] for row in rows]
            if queued_ids:
                all_asset_ids.extend(item for item in queued_ids if item not in all_asset_ids)
                _process_ingest_asset_group(queued_ids, batch_id, finalize_batch=False)
                first = False
                continue
            batch = task_store.get_ingest_batch(batch_id) or {}
            pending_row = task_store._row(
                "SELECT COUNT(*) AS count FROM assets WHERE batch_id = ? AND status IN ('queued', 'processing', 'semantic_enriching')",
                (batch_id,),
            )
            if batch.get("status") == "complete" and not (pending_row and pending_row["count"]):
                break
            time.sleep(0.5)

        with db_write_guard("ingest-batch-complete"):
            task_store.complete_ingest_batch(batch_id)
        with db_write_guard("ingest-batch-claim"):
            claimed = task_store.claim_ingest_batch_summary(batch_id)
        if claimed:
            event_ids = task_store.batch_event_ids(batch_id)
            limits = _pipeline_worker_limits()
            summary_started = time.perf_counter()
            with ThreadPoolExecutor(max_workers=limits["event_summary_workers"], thread_name_prefix="sentrix-event-summary") as executor:
                event_results = list(executor.map(_summarize_event_worker, event_ids))
            summary_wall_seconds = round(time.perf_counter() - summary_started, 4)
            with db_write_guard("ingest-batch-finish"):
                task_store.finish_ingest_batch(batch_id)
            metrics = {
                **limits, "status": "completed", "asset_count": len(all_asset_ids),
                "image_count": len(all_asset_ids), "event_count": len(event_ids),
                "event_summary_call_count": len(event_results),
                "event_summary_wall_seconds": summary_wall_seconds,
                "event_summaries": event_results,
                "stage_timings": _pipeline_timing_summary(task_store, all_asset_ids),
                "total_wall_seconds": round(time.perf_counter() - pipeline_started_at, 4),
            }
            with db_write_guard("ingest-batch-metrics"):
                task_store.update_ingest_batch_metadata(batch_id, {"pipeline_metrics": metrics})
    except Exception as error:
        with db_write_guard("ingest-batch-metrics-error"):
            task_store.update_ingest_batch_metadata(batch_id, {"pipeline_metrics": {"status": "failed", "error": str(error)}})
        raise
    finally:
        task_store.close()
        with batch_worker_lock:
            active_batch_workers.discard(batch_id)



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


def _vllm_api(path: str, method: str = "GET", json_body=None, timeout=30):
    """Call the selected remote vLLM Manager HTTP API."""
    manager_url = RUNTIME_VLLM_API_URL or VLLM_API_URL
    if not manager_url:
        return None
    import httpx as _httpx
    url = manager_url.rstrip("/") + path
    try:
        resp = _httpx.request(method, url, json=json_body, timeout=timeout)
        if resp.status_code >= 400:
            return None
        return resp.json()
    except Exception:
        return None


def _load_vllm_state(registry=None):
    if VLLM_API_URL:
        return _vllm_api("/state")
    registry = registry or _load_vllm_registry()
    state_file = Path(registry.get("state_file") or "/home/asus/sentrix-vllm/state/current.json")
    return _read_json_file(state_file, None) if state_file.exists() else None


def _profile_availability(profile):
    missing = []
    model_path = profile.get("model")
    if model_path and not Path(model_path).exists():
        missing.append(model_path)
    for module in profile.get("lora_modules") or []:
        lora_path = module.get("path")
        if lora_path and not Path(lora_path).exists():
            missing.append(lora_path)
    return {"available": not missing, "missing_paths": missing}


_remote_profiles_cache = None

def _remote_profile_availability(profile_id):
    """Get availability from remote vLLM API, with a simple cache."""
    global _remote_profiles_cache
    if not VLLM_API_URL:
        return None
    if _remote_profiles_cache is None:
        _remote_profiles_cache = _vllm_api("/profiles") or []
    for p in _remote_profiles_cache:
        if p.get("id") == profile_id:
            return {"available": p.get("available", False), "missing_paths": p.get("missing_paths", [])}
    return None


def _profile_summary(profile_id, profile):
    availability = _remote_profile_availability(profile_id) or _profile_availability(profile)
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
    global gamma, pipeline
    registry = _load_vllm_registry()
    profile = profile or (registry.get("profiles") or {}).get(profile_id) or {}
    state = state or _load_vllm_state(registry) or {}
    port = int(state.get("port") or profile.get("port") or registry.get("default_port") or 8100)
    served_name = state.get("served_model_name") or profile.get("served_model_name") or profile_id
    with runtime_lock:
        base_url = (state.get("external_url_hint") if state else None) or gamma.base_url
        new_gamma = GammaClient(base_url=base_url, model=served_name, backend="openai", manager_url=RUNTIME_VLLM_API_URL or VLLM_API_URL)
        gamma = new_gamma
        pipeline = IngestionPipeline(store, gamma=gamma, asr=pipeline.asr, face=pipeline.face, clip=pipeline.clip)
    return _current_model_runtime()


def _run_vllm_switch(request: ModelSwitchRequest):
    if VLLM_API_URL:
        values = request.model_dump() if hasattr(request, "model_dump") else request.dict()
        payload = {k: v for k, v in values.items() if v is not None and v != ""}
        timeout = max(60, int(request.ready_timeout) + 90) if request.wait_ready else 60
        result = _vllm_api("/switch", method="POST", json_body=payload, timeout=timeout)
        if not result:
            raise HTTPException(status_code=502, detail="vLLM switch failed: no response from remote API")
        runtime = _apply_vllm_profile_to_runtime(request.profile, state=result.get("state"))
        return {"accepted": True, "profile": request.profile, "runtime": runtime,
            "stdout": result.get("stdout", ""), "stderr": result.get("stderr", "")}
    if not VLLM_MANAGER.exists():
        raise HTTPException(status_code=503, detail=f"vLLM manager not found: {VLLM_MANAGER}")
    registry = _load_vllm_registry()
    profile = (registry.get("profiles") or {}).get(request.profile)
    if not profile:
        raise HTTPException(status_code=404, detail="model profile not found")
    # 统一走 vLLM Manager 服务（HTTP），不再由后端直接拉起子进程 CLI。
    manager_api = os.getenv("SENTRIX_VLLM_MANAGER_API", "http://127.0.0.1:8500").rstrip("/")
    values = request.model_dump() if hasattr(request, "model_dump") else request.dict()
    body = {k: values.get(k) for k in (
        "profile", "wait_ready", "ready_timeout", "dry_run",
        "max_model_len", "max_num_seqs", "max_num_batched_tokens",
        "gpu_memory_utilization", "quantization", "load_format", "dtype",
        "default_max_tokens", "cuda_visible_devices",
    )}
    timeout = max(60, int(request.ready_timeout) + 90 if request.wait_ready else 60)
    try:
        response = httpx.post(f"{manager_api}/switch", json=body, timeout=timeout)
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=503, detail={
            "message": f"vLLM manager API unreachable: {manager_api}", "error": str(exc)})
    if response.status_code != 200:
        try:
            detail = response.json()
        except Exception:
            detail = {"stdout": response.text[-4000:]}
        raise HTTPException(status_code=502, detail={
            "message": "vLLM switch failed", "manager_api": manager_api, **detail})
    runtime = _current_model_runtime() if request.dry_run else _apply_vllm_profile_to_runtime(request.profile, profile)
    payload = response.json()
    return {"accepted": True, "profile": request.profile, "runtime": runtime,
        "stdout": (payload.get("stdout") or "")[-4000:],
        "stderr": (payload.get("stderr") or "")[-4000:]}


@app.on_event("startup")
def _sync_vllm_state_on_startup():
    """Sync gamma client with remote vLLM state on startup."""
    try:
        state = _load_vllm_state()
        if state and state.get("pid"):
            _apply_vllm_profile_to_runtime(state.get("profile", ""), state=state)
    except Exception:
        pass

@app.get("/api/health")
def health():
    # Phase C C12：profile manifest 作为运维真实来源
    agent = {
        "profile": os.getenv("SENTRIX_AGENT_PROFILE", "goal_driven_shadow").strip().lower(),
        "runtime": "tool_loop",
    }
    try:
        from .agent_runtime import tools as _runtime_tools
        _runtime_tools.register_tools()  # 幂等：确保工具注册表在首个 turn 前也可查
        from .agent_runtime.tool_registry import list_tools
        agent["tools"] = [
            {"name": s.name, "readiness": s.readiness}
            for s in list_tools() if s.readiness != "blocked"
        ]
        matrix = {t["name"]: t for t in agent["tools"]}
        # D10：Capability Manifest — 前端按真实 readiness 渲染能力
        agent["capabilities"] = {
            "conversation_management": CONVERSATION_STORE_ENABLED,
            "photo_inspector": matrix.get("inspect_photo", {}).get("readiness") == "ready",
            "long_term_memory": matrix.get("get_core_memory", {}).get("readiness") != "blocked",
            "person_memory": matrix.get("get_person_memory", {}).get("readiness") != "blocked",
            "conversation_search": matrix.get("search_conversation_history", {}).get("readiness") != "blocked",
            "memory_write": False,
        }
        agent["ui_actions"] = {
            "new_conversation": CONVERSATION_STORE_ENABLED,
            "delete_conversation": CONVERSATION_STORE_ENABLED,
            "view_original": matrix.get("get_original_photos", {}).get("readiness") in {"ready", "limited"},
        }
        from pathlib import Path as _Path
        _matrix_path = _Path(__file__).resolve().parent.parent / "configs" / "tool_capability_matrix.json"
        if _matrix_path.is_file():
            import json as _json
            try:
                agent["capability_matrix"] = _json.loads(_matrix_path.read_text(encoding="utf-8"))
            except Exception:
                agent["capability_matrix"] = {}
    except Exception:
        agent["tools"] = []
    active_vlm_backend = getattr(gamma, "backend", "vllm")
    if not isinstance(active_vlm_backend, str):
        active_vlm_backend = "vllm"
    return {
        "status": "ok",
        "mode": "sentrix-local-backend",
        "agent": agent,
        "models": {
            "vlm": {"active": active_vlm_backend, "name": gamma.model, "endpoint": gamma.base_url},
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
                "identityFallback": pipeline.face.identity_fallback,
                "identityFallbackModel": pipeline.face.identity_fallback_model,
                "identityFallbackError": pipeline.face.identity_fallback_error,
                "error": pipeline.face.error,
                "identityError": pipeline.face.identity_runtime_error or pipeline.face.identity_error,
            },
            "clip": {"enabled": pipeline.clip.enabled, "model": pipeline.clip.model_name, "ready": pipeline.clip.evidence_ready, "evidenceReady": pipeline.clip.evidence_ready, "error": pipeline.clip.error},
        },
        "memory": {"mode": "sentrix-native", "vectorSpaces": ["episodic", "semantic", "visual"]},
        "videoExtraction": {
            "adapter": "hybrid_webp_memory", "status": "available",
            "package": "tools/video_keyframe/katna/run_yolo_prefilter_event_webp.py",
            "sampleFps": 10, "yoloBatch": 16, "targetDecode": "NVDEC",
            "memoryMerge": True, "duplicateFrameRemoval": True,
        },
        "database": store.path,
    }

@app.get("/api/hardware")
def hardware():
    """QA 硬件参数采集：GPU / CPU / 内存 + 当前模型快照（全容错）。"""
    from .hardware import collect_hardware
    hw = collect_hardware()
    try:
        hw["models"] = {
            "vlm": {"active": "vllm", "name": gamma.model, "endpoint": gamma.base_url},
            "llm": _current_model_runtime(),
            "asr": {"name": pipeline.asr.model_name, "ready": pipeline.asr.error is None},
        }
    except Exception:
        hw["models"] = {}
    return hw



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

@app.delete("/api/memory-spaces/{scope_id}")
def delete_memory_space(scope_id: str):
    scope_id = (scope_id or "").strip()
    if not scope_id:
        raise HTTPException(status_code=422, detail="scope_id required")
    if scope_id == "home-default":
        raise HTTPException(status_code=403, detail="home-default 是系统默认相册,不允许删除")
    if not store._row("SELECT id FROM memory_spaces WHERE id = ?", (scope_id,)):
        raise HTTPException(status_code=404, detail=f"相册 {scope_id} 不存在")
    try:
        stats = store.delete_memory_space(scope_id)
    except ValueError as exc:
        raise HTTPException(status_code=403, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"删除失败: {exc}")
    return {"ok": True, "scope_id": scope_id, "removed": stats}



_OCR_SETTING_KEY = "ocr.small_enabled"


def _ocr_settings():
    from .agent_runtime.tools import small_ocr_available
    enabled = store.get_setting(_OCR_SETTING_KEY, "false").lower() in {"1", "true", "on"}
    available = small_ocr_available()
    return {
        "small_ocr_enabled": enabled,
        "small_ocr_available": available,
        "readiness": "ready" if available else "unavailable",
        "model": "paddleocr" if available else None,
    }


class OCRSettingsPayload(BaseModel):
    small_ocr_enabled: bool


@app.get("/api/settings/ocr")
def get_ocr_settings():
    return _ocr_settings()


@app.put("/api/settings/ocr")
def put_ocr_settings(payload: OCRSettingsPayload):
    store.set_setting(_OCR_SETTING_KEY, "true" if payload.small_ocr_enabled else "false")
    return _ocr_settings()


@app.get("/api/telemetry/ocr")
def ocr_telemetry():
    """Phase H H6：OCR provider 使用率/延迟/置信度聚合（dashboard 用）。"""
    from .agent_runtime.tools import ocr_telemetry_snapshot
    return {
        "providers": ocr_telemetry_snapshot(),
        "small_enabled": _ocr_settings().get("small_ocr_enabled"),
    }


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


class RuntimeBindRequest(BaseModel):
    manager_url: str
    model_base_url: str | None = None

@app.post("/api/model-profiles/bind-runtime")
def bind_model_runtime(request: RuntimeBindRequest):
    """Bind Agent runtime to one fixed Manager/model-service pair for this process."""
    global RUNTIME_VLLM_API_URL, RUNTIME_VLLM_BASE_URL
    if not request.manager_url.startswith(("http://", "https://")):
        raise HTTPException(status_code=400, detail="invalid vLLM manager URL")
    if request.model_base_url and not request.model_base_url.startswith(("http://", "https://")):
        raise HTTPException(status_code=400, detail="invalid vLLM model URL")
    previous = (RUNTIME_VLLM_API_URL, RUNTIME_VLLM_BASE_URL)
    RUNTIME_VLLM_API_URL = request.manager_url.rstrip("/")
    RUNTIME_VLLM_BASE_URL = request.model_base_url.rstrip("/") if request.model_base_url else None
    state = _load_vllm_state()
    if not state or not state.get("pid"):
        RUNTIME_VLLM_API_URL, RUNTIME_VLLM_BASE_URL = previous
        raise HTTPException(status_code=502, detail="selected vLLM Manager has no active model")
    runtime = _apply_vllm_profile_to_runtime_from_state()
    return {"accepted": True, "manager_url": RUNTIME_VLLM_API_URL,
            "model_base_url": RUNTIME_VLLM_BASE_URL, "runtime": runtime}

@app.post("/api/model-profiles/sync-runtime")
def sync_model_runtime():
    """Sync gamma client to the currently bound vLLM model without restarting it."""
    runtime = _apply_vllm_profile_to_runtime_from_state()
    return {"accepted": True, "runtime": runtime}


def _apply_vllm_profile_to_runtime_from_state():
    """Read vLLM state and update gamma client. No model lifecycle changes."""
    registry = _load_vllm_registry()
    state = _load_vllm_state(registry) or {}
    profile_id = state.get("profile") or ""
    return _apply_vllm_profile_to_runtime(profile_id, state=state)




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
def events(scope_id: str | None = None, limit: int = 1000):
    # The timeline needs older events when the user selects a historical date.
    # Keep a server-side ceiling while avoiding the previous silent 100-event
    # truncation that hid older video scenes.
    limit = min(max(int(limit or 1000), 1), 5000)
    return {"events": store.list_events(limit, scope_id=scope_id)}


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


@app.get("/api/videos/{asset_id}")
def video_detail(asset_id: str):
    value = store.get_asset(asset_id)
    if not value or value.get("media_type") != "video":
        raise HTTPException(status_code=404, detail="video asset not found")
    return {"video": value, "scenes": store.list_video_scene_events(asset_id)}


@app.get("/api/videos/{asset_id}/scenes")
def video_scenes(asset_id: str):
    value = store.get_asset(asset_id)
    if not value or value.get("media_type") != "video":
        raise HTTPException(status_code=404, detail="video asset not found")
    return {"video_asset_id": asset_id, "scenes": store.list_video_scene_events(asset_id)}


@app.get("/api/video-scenes/{scene_id}")
def video_scene_detail(scene_id: str):
    value = store.get_event_detail(scene_id)
    if not value or value["event"].get("source_type") != "video_scene":
        raise HTTPException(status_code=404, detail="video scene not found")
    return value


@app.post("/api/videos/{asset_id}/reprocess", status_code=202)
def reprocess_video(asset_id: str, background_tasks: BackgroundTasks):
    value = store.get_asset(asset_id)
    if not value or value.get("media_type") != "video":
        raise HTTPException(status_code=404, detail="video asset not found")
    store.cleanup_video_derivatives(asset_id)
    store.update_asset(asset_id, "video-queued", {
        "video_stage": "video-queued", "error": None, "error_stage": None,
    })
    background_tasks.add_task(process_asset, asset_id)
    return {"accepted": True, "asset_id": asset_id, "status": "video-queued"}


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


@app.post("/api/relationships/batch")
def create_relationships_batch(payload: dict):
    scope_id = str(payload.get("scope_id") or "")
    entity_by_name = {str(k): str(v) for k, v in (payload.get("entity_by_name") or {}).items()}
    relationships = payload.get("relationships") or []

    entities = {}
    for entity in store.list_entities(scope_id=scope_id):
        name = str(entity.get("canonical_name") or "").strip()
        if name:
            entities.setdefault(name, entity)
        role = str(entity.get("family_role") or "").strip()
        if role:
            entities.setdefault(role, entity)

    def resolve_entity(name: str) -> str | None:
        key = str(name or "").strip()
        if not key:
            return None
        if key in entity_by_name:
            return entity_by_name[key]
        entity = entities.get(key)
        return entity.get("id") if entity else None

    results = []
    for rel in relationships:
        subject_name = str(rel.get("subject") or "")
        object_name = str(rel.get("object") or "")
        predicate = str(rel.get("predicate") or "")
        subject_id = resolve_entity(subject_name)
        object_id = resolve_entity(object_name)
        if not subject_id or not object_id or not predicate:
            results.append({"subject": subject_name, "predicate": predicate,
                            "object": object_name, "error": "unresolved entity"})
            continue
        try:
            value = store.create_relationship(
                subject_id, predicate, object_id, [],
                float(rel.get("confidence") or 0.5), "pending")
            results.append({"subject": subject_name, "predicate": predicate,
                            "object": object_name,
                            "id": value.get("id") if isinstance(value, dict) else None})
        except Exception as exc:
            results.append({"subject": subject_name, "predicate": predicate,
                            "object": object_name, "error": str(exc)[:200]})
    imported = sum(1 for r in results if not r.get("error"))
    return {"requested": len(relationships), "imported": imported, "results": results}


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



def _seed_face_detect(path: str) -> list[dict]:
    """Standardised face extraction for identity seeding.

    Flow:
      1. buffalo_l detection (handles full-body / scene photos)
      2. If miss: upscale 2x and retry (handles small faces in large images)
      3. If still miss: whole-image embedding fallback (handles tight face crops)

    Returns a face dict list compatible with seed_person_identity.
    """
    faces = pipeline.face.detect(path)
    if faces:
        return faces

    if not pipeline.face.identity_configured:
        return []

    import cv2 as _cv2
    _img = _cv2.imread(path)
    if _img is None:
        return []
    _h, _w = _img.shape[:2]

    # Step 2: upscale and retry detection for small faces in large images
    _max_dim = max(_w, _h)
    if _max_dim > 600:
        _scale = min(2.0, 1280.0 / _max_dim)
        if _scale > 1.05:
            _img_big = _cv2.resize(_img, (int(_w * _scale), int(_h * _scale)),
                                   interpolation=_cv2.INTER_CUBIC)
            _tmp_path = path + ".upscaled.jpg"
            _cv2.imwrite(_tmp_path, _img_big)
            try:
                faces = pipeline.face.detect(_tmp_path)
            finally:
                import os as _os
                _os.unlink(_tmp_path)
            if faces:
                # Scale bbox coordinates back to original image
                for f in faces:
                    f["bbox"] = [v / _scale for v in f["bbox"]]
                return faces

    # Step 3: whole-image fallback for pre-cropped face photos
    try:
        _crop = align_face_crop(_img, [0, 0, _w, _h])
        _emb = pipeline.face.identity_adapter.embed(_crop)
        return [{
            "bbox": [0, 0, float(_w), float(_h)],
            "confidence": 0.99, "quality": 0.8,
            "area_ratio": 1.0, "sharpness": 0.0, "pose": [],
            "landmarks": [], "embedding": _emb.embedding,
            "embedding_model": pipeline.face.identity_model,
            "embedding_version": _emb.model_version,
            "quality_signal": _emb.quality_signal,
            "pose_bucket": "frontal", "identity_ready": True,
        }]
    except Exception:
        return []


@app.post("/api/people/seed", status_code=201)
async def seed_person_identity(
    background_tasks: BackgroundTasks,
    name: str = Form(...),
    familyRole: str | None = Form(None),
    aliases: str | None = Form(None),
    scopeId: str | None = Form(None),
    scope_id: str | None = Form(None),
    files: list[UploadFile] = File(...),
):
    name = name.strip()
    if not name:
        raise HTTPException(status_code=400, detail="name is required")
    scope = (scope_id or scopeId or "home-default").strip() or "home-default"
    alias_list = [a.strip() for a in (aliases or "").replace("、", ",").split(",") if a.strip()]
    face_photos = []
    for upload in files:
        safe_name = Path(upload.filename or "identity.bin").name
        asset_id = make_id("asset")
        destination = MEDIA_DIR / f"{asset_id}_{safe_name}"
        with destination.open("wb") as output:
            shutil.copyfileobj(upload.file, output)
        sha = hashlib.sha256(destination.read_bytes()).hexdigest()
        meta = {"scope_id": scope, "source_type": "identity_seed", "content_sha256": sha}
        store.create_asset(asset_id, safe_name, "image", str(destination), upload.content_type, destination.stat().st_size, meta, scope_id=scope)
        observation = store.add_observation(asset_id, {"source_type": "identity_seed", "caption": "", "confidence": 0.0}, scope_id=scope)
        store.update_asset(asset_id, "processed", {"observation_id": observation["id"]})
        faces = _seed_face_detect(str(destination))
        if not faces:
            continue
        best = max(faces, key=lambda f: f.get("quality", f.get("confidence", 0)))
        face_photos.append({**best, "asset_id": asset_id, "observation_id": observation["id"]})
    if not face_photos:
        raise HTTPException(status_code=422, detail="no detectable faces in uploaded photos")
    result = store.seed_person_identity(scope, name, (familyRole or "").strip() or None, alias_list, face_photos)
    return {"entity_id": result["entity"]["id"], "cluster_id": result["cluster_id"], "name": result["name"], "face_count": result["face_count"], "family_role": m_role, "aliases": result["aliases"]}


@app.post("/api/people/seed-batch", status_code=201)
async def seed_persons_batch(
    background_tasks: BackgroundTasks,
    manifest: str = Form(...),
    scopeId: str | None = Form(None),
    scope_id: str | None = Form(None),
    files: list[UploadFile] = File(...),
):
    import json as _json
    scope = (scope_id or scopeId or "home-default").strip() or "home-default"
    members = _json.loads(manifest)
    results = []
    for member in members:
        m_name = (member.get("name") or "").strip()
        if not m_name:
            results.append({"name": "", "error": "name is required"})
            continue
        _raw_aliases = member.get("aliases") or []
        if isinstance(_raw_aliases, str):
            _raw_aliases = _raw_aliases.replace("、", ",").split(",")
        m_aliases = [a.strip() for a in _raw_aliases if isinstance(a, str) and a.strip()]
        m_role = (member.get("family_role") or "").strip() or None
        file_indices = member.get("file_indices") or []
        member_files = [files[i] for i in file_indices if 0 <= i < len(files)]
        if not member_files:
            results.append({"name": m_name, "error": "no photos provided"})
            continue
        face_photos = []
        for upload in member_files:
            safe_name = Path(upload.filename or "identity.bin").name
            asset_id = make_id("asset")
            destination = MEDIA_DIR / f"{asset_id}_{safe_name}"
            with destination.open("wb") as output:
                shutil.copyfileobj(upload.file, output)
            sha = hashlib.sha256(destination.read_bytes()).hexdigest()
            meta = {"scope_id": scope, "source_type": "identity_seed", "content_sha256": sha}
            store.create_asset(asset_id, safe_name, "image", str(destination), upload.content_type, destination.stat().st_size, meta, scope_id=scope)
            observation = store.add_observation(asset_id, {"source_type": "identity_seed", "caption": "", "confidence": 0.0}, scope_id=scope)
            store.update_asset(asset_id, "processed", {"observation_id": observation["id"]})
            faces = _seed_face_detect(str(destination))
            if not faces:
                continue
            best = max(faces, key=lambda f: f.get("quality", f.get("confidence", 0)))
            face_photos.append({**best, "asset_id": asset_id, "observation_id": observation["id"]})
        if not face_photos:
            results.append({"name": m_name, "error": "no detectable faces"})
            continue
        result = store.seed_person_identity(scope, m_name, m_role, m_aliases, face_photos)
        results.append({"entity_id": result["entity"]["id"], "cluster_id": result["cluster_id"], "name": result["name"], "face_count": result["face_count"], "family_role": m_role, "aliases": result["aliases"]})
    return {"results": results}


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
    preview_path = Path((value.get("metadata_json") or {}).get("browser_preview_path") or "")
    if value.get("media_type") == "video" and not original and preview_path.is_file():
        return FileResponse(
            preview_path, media_type="video/mp4",
            filename=f"{Path(value.get('file_name') or asset_id).stem}-preview.mp4",
            headers={"Cache-Control": "private, max-age=86400", "X-Sentrix-Source": "browser-preview"},
        )
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


@app.get("/api/assistant/result-set/{result_set_id}/photo")
def result_set_photo(result_set_id: str, handle: str = "", scope_id: str = "home-default",
                     original: bool = False):
    """B3.2：ResultSet 授权原图交付。handle 必须属于该结果集且 scope 匹配。"""
    from .agent_runtime import tools as runtime_tools
    rs_store = runtime_tools.get_result_set_store()
    if rs_store is None:
        raise HTTPException(status_code=404, detail="result set service unavailable")
    rs = rs_store.get(result_set_id)
    if rs is None:
        raise HTTPException(status_code=404, detail="result set not found or expired")
    if rs.scope_id != scope_id:
        raise HTTPException(status_code=403, detail="scope mismatch")
    if handle:
        asset_id = rs_store.resolve_handle(result_set_id, handle)
        if not asset_id:
            raise HTTPException(status_code=404, detail="handle not in result set")
    else:
        asset_id = rs.asset_ids[0] if rs.asset_ids else None
        if not asset_id:
            raise HTTPException(status_code=404, detail="empty result set")
    return asset_file(asset_id, original=original)


@app.get("/api/face-instances/{face_instance_id}/crop")
def face_instance_crop(face_instance_id: str):
    instance = store.get_face_instance(face_instance_id)
    asset_path = (instance or {}).get("asset_path")
    if not instance or not asset_path or not Path(asset_path).is_file():
        raise HTTPException(status_code=404, detail="face instance not found")
    try:
        from PIL import Image

        ensure_heif_support()
        with Image.open(asset_path) as source:
            image = source.convert("RGB")
        bbox = instance.get("bbox_json") or []
        if not isinstance(bbox, (list, tuple)) or len(bbox) < 4:
            raise ValueError("invalid face bounding box")
        left, top, right, bottom = (int(value) for value in bbox[:4])
        left, top = max(0, left), max(0, top)
        right, bottom = min(image.width, right), min(image.height, bottom)
        if right <= left or bottom <= top:
            raise ValueError("invalid face bounding box")
        face = image.crop((left, top, right, bottom))
        face.thumbnail((256, 256))
        output = BytesIO()
        face.save(output, format="JPEG", quality=88)
        return Response(content=output.getvalue(), media_type="image/jpeg", headers={"Cache-Control": "private, max-age=86400"})
    except (OSError, ValueError, TypeError) as error:
        raise HTTPException(status_code=422, detail=f"face crop is unavailable: {error}") from error


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
        for event_id in event_ids:
            detail = store.get_event_detail(event_id)
            if detail:
                evidence.append({"event": detail["event"], "observations": [{"id": item["id"], "caption": item.get("caption"), "transcript": item.get("transcript"), "asset_id": item.get("asset_id")} for item in detail["observations"]]})
        if evidence:
            prompt = """根据下面的真实家庭事件和证据生成故事初稿。不要补造人物、地点或时间，只能使用证据。严格返回 JSON：title、content、outline（数组）。使用中文，content 300 字以内。证据：""" + str(evidence)
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


class AssistantTurnRequest(BaseModel):
    message: str
    conversation_id: str | None = None
    feedback: dict | None = None
    scope_id: str = "home-default"
    include_debug: bool = False
    selected_entity_id: str | None = None
    selected_asset_handle: str | None = None
    selected_result_set_id: str | None = None
    viewer_id: str = "owner"


_TOOL_LOOP_TASK_STATE: dict[str, dict] = {}  # conversation_id -> task_state（B3.1 跨 turn 结果集续接）
_TURN_JOBS: dict[str, dict] = {}  # turn_id -> job（B3.4 异步轮询）
_TURN_EXECUTOR = None  # 惰性初始化 ThreadPoolExecutor


def _turn_executor():
    global _TURN_EXECUTOR
    if _TURN_EXECUTOR is None:
        from concurrent.futures import ThreadPoolExecutor
        _TURN_EXECUTOR = ThreadPoolExecutor(max_workers=2)
    return _TURN_EXECUTOR


def _tool_loop_turn(message, conversation_id, scope_id, viewer_id, recent_turns="",
                   progress_callback=None, selected_asset_handle=None,
                   selected_result_set_id=None, conversation_summary="",
                   profile_name=None, include_debug=False):
    """SENTRIX_AGENT_PROFILE=tool_loop* 时走 AgentRuntime（模型自主 Tool-Loop）。"""
    from .agent_runtime import tools as runtime_tools
    from .agent_runtime.runtime import AgentRuntime, public_agent2_trace

    try:
        from .embeddings import EmbeddingRouter
        from .model_clients import ClipAdapter
        from .retrieval import RetrievalConfig
        embedding_router = EmbeddingRouter.from_clip(ClipAdapter())
        retrieval_config = RetrievalConfig()
    except Exception:
        embedding_router = None
        retrieval_config = None
    runtime_tools.bind_runtime(store, gamma=gamma, embedding_router=embedding_router,
                               retrieval_config=retrieval_config)
    runtime_tools.set_conversation_id(conversation_id)
    runtime_tools.register_tools()
    profile_name = (profile_name or os.getenv("SENTRIX_AGENT_PROFILE", "goal_driven_shadow")).strip().lower()
    model_call_metrics = []
    gamma.get_and_clear_call_metrics()

    def _estimate_prompt_tokens(messages):
        # 估算：中文约 0.7 token/字，ASCII 约 0.25 token/字符；加 300 结构开销。
        # 系数按实测校准（system 提示以 ASCII JSON schema 为主，0.6 统一系数会高估 2 倍）。
        zh = 0
        total = 0
        for m in messages:
            c = m.get("content") or ""
            if isinstance(c, str):
                total += len(c)
                zh += sum(1 for ch in c if "\u4e00" <= ch <= "\u9fff")
            elif isinstance(c, list):
                for part in c:
                    if isinstance(part, dict):
                        t = part.get("text") or ""
                        total += len(t)
                        zh += sum(1 for ch in t if "\u4e00" <= ch <= "\u9fff")
        return int(zh * 0.7 + (total - zh) * 0.25) + 400

    def chat_fn(messages, *, call_type="agent", step_id=None):
        max_tokens = max(1, min(1501, int(os.getenv("SENTRIX_TOOL_LOOP_MAX_TOKENS", "384"))))
        # Phase H H4：max_model_len=4501 的硬上限保护——prompt 越长输出预算越小，
        # 避免 400（此前 guard recovery 时 prompt 4118 + 384 = 4502 恰好超限）
        try:
            room = 4400 - _estimate_prompt_tokens(messages)
            if room < max_tokens:
                max_tokens = max(64, room)
        except Exception:
            pass
        try:
            text = gamma.chat_messages(
                messages, role="tool_loop", temperature=0.0, max_tokens=max_tokens)
        finally:
            metrics = gamma.get_and_clear_call_metrics()
            for metric in metrics:
                metric["call_type"] = call_type
                metric["step_id"] = step_id
            model_call_metrics.extend(metrics)
        return text

    _ocr_setting = store.get_setting("ocr.small_enabled", "false").lower() in {"1", "true", "on"}
    runtime = AgentRuntime(chat_fn=chat_fn, profile_name=profile_name,
                           ocr_settings={"small_ocr_enabled": _ocr_setting},
                           scope_id=scope_id, viewer_id=viewer_id,
                           conversation_id=conversation_id,
                           include_debug=include_debug)
    prev_state = _TOOL_LOOP_TASK_STATE.get(conversation_id) if conversation_id else None
    # Phase C C15：用户点选的照片写入本轮 task_state（selected handle 稳定跨轮可用）
    if selected_asset_handle:
        prev_state = dict(prev_state or {})
        if selected_result_set_id:
            prev_state["current_result_set"] = selected_result_set_id
        prev_state["selected_asset_handle"] = selected_asset_handle
    turn = runtime.run(message, history=recent_turns, task_state=prev_state,
                       progress_callback=progress_callback,
                       selected_handle=selected_asset_handle,
                       selected_result_set_id=selected_result_set_id,
                       conversation_summary=conversation_summary)
    model_call_metrics.extend(gamma.get_and_clear_call_metrics())
    if conversation_id:
        _TOOL_LOOP_TASK_STATE[conversation_id] = turn.task_state
    trace = []
    for s in turn.steps:
        item = {"stage": s.get("type", "step"), "status": s.get("status", "complete"),
                "reason": s.get("reason") or "",
                "detail": s.get("tool") or s.get("raw") or s.get("observation") or {}}
        if s.get("step_id"):
            item["step_id"] = s.get("step_id")
        if s.get("parent_step_id"):
            item["parent_step_id"] = s.get("parent_step_id")
        if s.get("call_type"):
            item["call_type"] = s.get("call_type")
        for field in ("trigger", "action", "tool", "answer_preview", "attempt",
                      "turn_outcome", "parse_status", "next_step"):
            if s.get(field) is not None:
                item[field] = s.get(field)
        if s.get("type") == "judge":
            item["detail"] = {"faithful": s.get("faithful"),
                              "problems": list(s.get("problems") or [])}
        if s.get("type") == "guard":
            item["detail"] = {"l1_codes": list(s.get("codes") or []),
                              "attempt": s.get("attempt", 1)}
        if isinstance(s.get("arguments"), dict):
            item["args"] = s.get("arguments")
        trace.append(item)

    metrics_by_step = {}
    unassigned_metrics = []
    for metric in model_call_metrics:
        step_id = metric.get("step_id")
        if step_id:
            metrics_by_step.setdefault(step_id, []).append(metric)
        else:
            unassigned_metrics.append(metric)
    ordered_metrics = []
    for step in turn.steps:
        step_id = step.get("step_id")
        step_type = step.get("type")
        for metric in metrics_by_step.pop(step_id, []):
            if step_type == "model":
                metric["turn_outcome"] = step.get("turn_outcome")
                metric["parse_status"] = step.get("parse_status")
                metric["next_step"] = step.get("next_step")
            metric["call_observation"] = {
                "kind": metric.get("call_type") or "unknown",
                "label": {
                    "planner": "Agent 2.0 目标分解与规划",
                    "agent": "Agent 决策 / 回答",
                    "recovery": "Agent 恢复调用",
                    "writer": "最终回答重写",
                    "faithfulness_judge": "L2 事实一致性检查",
                }.get(metric.get("call_type"), "模型调用"),
                "purpose": {
                    "planner": "解析用户目标并声明最小充分证据需求（TaskState/EvidenceLedger）",
                    "agent": "选择下一步工具或生成候选回答",
                    "recovery": "根据运行时纠正信息重新决策或修正回答",
                    "writer": "基于受控事实调整最终回答的结构和措辞",
                    "faithfulness_judge": "检查候选回答是否与工具事实一致",
                }.get(metric.get("call_type"), "执行模型推理"),
                "trigger": step.get("trigger") or "",
                "outcome": (
                    f"已解析，准备调用工具 {step.get('tool')}" if step.get("turn_outcome") == "tool_call"
                    else "已解析，正常输出回答" if step.get("turn_outcome") == "final_answer"
                    else "JSON 解析失败，触发格式恢复" if step.get("turn_outcome") == "parse_failure"
                    else "上下文或 token 预检拦截" if step.get("turn_outcome") == "context_blocked"
                    else "模型请求失败" if step.get("turn_outcome") == "model_error"
                    else f"判定 {'通过' if step.get('faithful') else '未通过'}"
                    if step_type == "judge" and step.get("faithful") is not None
                    else f"重写状态：{step.get('status')}" if step_type == "writer"
                    else f"调用失败：{step.get('reason')}" if step.get("status") == "error"
                    else "完成模型推理"
                ),
                "source": "backend_recorded",
                "related_tool": step.get("tool") or None,
            }
            ordered_metrics.append(metric)
        if step_type == "tool":
            internal_metrics = step.get("internal_model_call_metrics") or []
            for index, metric in enumerate(internal_metrics, 1):
                metric["call_type"] = "tool_internal"
                metric["step_id"] = f"{step_id}:model_{index}" if step_id else None
                metric["parent_step_id"] = step_id
                subtask = metric.get("tool_subtask")
                metric["call_observation"] = {
                    "kind": "tool_internal",
                    "label": "工具内部模型调用",
                    "purpose": (
                        "读取照片中的文字" if step.get("tool") == "read_photo_text"
                        else "识别照片中的视觉细节"
                    ),
                    "trigger": f"工具 {step.get('tool')} 执行内部推理",
                    "outcome": (
                        f"调用失败：{metric.get('error') or '未返回结果'}"
                        if metric.get("status") == "error" else
                        f"完成 {subtask} 子任务" if subtask else
                        f"完成 {step.get('tool')} 的模型处理"
                    ),
                    "source": "backend_recorded",
                    "related_tool": step.get("tool") or None,
                    "parent_step_id": step_id,
                }
                ordered_metrics.append(metric)
    for metrics in metrics_by_step.values():
        ordered_metrics.extend(metrics)
    ordered_metrics.extend(unassigned_metrics)

    guard_debug = {
        "status": turn.status,
        "reason": turn.reason or "",
        "termination_reason": turn.termination_reason or "",
        "recovery_attempts": sum(1 for p in turn.public_progress
                                 if p.get("stage") == "recovering"),
        "l1_codes": [c for s in turn.steps if s.get("type") == "guard"
                     for c in (s.get("codes") or [])],
        "judge": [{"faithful": s.get("faithful"),
                   "problems": list(s.get("problems") or [])}
                  for s in turn.steps if s.get("type") == "judge"],
    }
    tool_trace = [
        {"tool": s.get("tool", ""), "status": s.get("status", ""),
         "latency_s": s.get("latency_s"), "reason": s.get("reason") or "",
         "error": s.get("error") or "",
         "retrieval_timing": (s.get("observation") or {}).get("retrieval_timing")}
        for s in turn.steps if s.get("type") == "tool"
    ]
    return {
        "answer": turn.final_answer,
        "model_call_metrics": ordered_metrics,
        "conversation_id": conversation_id or f"conversation_{uuid.uuid4().hex[:12]}",
        "intent": "tool_loop",
        "evidence_status": "tool_loop",
        "retrieval_trace": trace,
        "public_progress": turn.public_progress,
        "tool_trace": tool_trace,
        "tool_loop_status": turn.status,
        "tool_loop_reason": turn.reason,
        "task_state": turn.task_state,
        "agent2_trace": turn.agent2_trace if include_debug else public_agent2_trace(turn.agent2_trace),
        "guard_debug": guard_debug,
        "answer_grounding": turn.answer_grounding,
        "termination_reason": turn.termination_reason,
        "debug_trace": turn.steps if include_debug else None,
    }


@app.post("/api/assistant/turn")
def assistant_turn(request: AssistantTurnRequest):
    if not request.message.strip():
        raise HTTPException(status_code=400, detail="message is required")
    message = request.message.strip()
    conversation_id = request.conversation_id
    recent_turns = ""
    conversation_summary = ""
    if CONVERSATION_STORE_ENABLED:
        # D2：无会话时自动创建；恢复旧会话时按 active 校验
        if not conversation_id:
            conversation_id = conversation_store.create_conversation(scope_id=request.scope_id)
        try:
            if conversation_store.get_conversation(conversation_id):
                history = conversation_store.last_messages(conversation_id, limit=8)
                recent_turns = "\n".join(
                    f"{'用户' if m['role'] == 'user' else '助手'}：{m['content'].get('text', '')}"
                    for m in history if m.get("content", {}).get("text")
                )
                conversation_summary = conversation_store.get_summary(conversation_id)
        except Exception:
            recent_turns = ""
    # tool_loop 是唯一 agent 路径：异步执行，立即返回 turn_id 供前端轮询实时进度
    turn_id = make_id("turn")
    _TURN_JOBS[turn_id] = {"status": "running", "public_progress": [],
                           "progress_events": [], "result": None,
                           "created_at": time.time()}
    _turn_executor().submit(
        _execute_turn_job, turn_id, message, conversation_id,
        request.scope_id, request.viewer_id, recent_turns,
        request.selected_asset_handle, request.selected_result_set_id,
        conversation_summary, request.include_debug)
    return {
        "turn_id": turn_id,
        "status": "running",
        "conversation_id": conversation_id or f"conversation_{uuid.uuid4().hex[:12]}",
        "intent": "tool_loop",
    }


@app.get("/api/assistant/turn/{turn_id}")
def assistant_turn_status(turn_id: str):
    """B3.4：轮询异步 turn 的实时进度与最终结果。"""
    job = _TURN_JOBS.get(turn_id)
    if job is None:
        raise HTTPException(status_code=404, detail="turn not found")
    if job["status"] in {"running", "pending"}:
        return {"turn_id": turn_id, "status": job["status"],
                "public_progress": job.get("public_progress") or []}
    if job["status"] == "error":
        return {"turn_id": turn_id, "status": "error", "error": job.get("error")}
    result = job.get("result") or {}
    return {
        "turn_id": turn_id,
        "status": "complete",
        "public_progress": job.get("public_progress") or result.get("public_progress") or [],
        "result": result,
    }


@app.get("/api/assistant/turn/{turn_id}/events")
async def assistant_turn_events(turn_id: str):
    """Phase C C13：SSE 实时进度事件流（GET EventSource）。

    事件契约：progress {text,status,stage,step_index,timestamp}；结束时 complete {result}。
    前端断开/失败可回退轮询（/api/assistant/turn/{turn_id} 仍保留快照）。
    """
    job = _TURN_JOBS.get(turn_id)
    if job is None:
        raise HTTPException(status_code=404, detail="turn not found")

    async def _stream():
        sent = 0
        while True:
            events = job.get("progress_events") or []
            for ev in events[sent:]:
                sent += 1
                yield f"event: progress\ndata: {json.dumps(ev, ensure_ascii=False)}\n\n"
            if job["status"] in {"complete", "error"}:
                payload = {
                    "type": "complete", "turn_id": turn_id,
                    "status": job["status"],
                    "public_progress": job.get("progress_events") or job.get("public_progress") or [],
                }
                if job["status"] == "complete" and job.get("result"):
                    payload["result"] = job["result"]
                if job["status"] == "error":
                    payload["error"] = job.get("error")
                yield f"event: complete\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"
                return
            yield ": keep-alive\n\n"
            await asyncio.sleep(0.3)

    return StreamingResponse(
        _stream(), media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


def _record_turn_conversation(message, request, result, turn_id=""):
    """把一轮对话写入 conversation store + trajectory（同步/异步两条路径共用）。"""
    if not (CONVERSATION_STORE_ENABLED and result.get("conversation_id")):
        return
    try:
        cid = result["conversation_id"]
        turn_id = turn_id or make_id("turn")
        result["turn_id"] = turn_id
        scope_id = request.scope_id or "home-default"
        # D2：生命周期维护（自动建会话/标题/更新时间）
        is_photo_thread = str(cid).startswith("photothread_")
        if not conversation_store.get_conversation(cid) and not is_photo_thread:
            conversation_store.create_conversation(scope_id=scope_id)
        if not is_photo_thread:
            conversation_store.ensure_title(cid, message)
            conversation_store.touch_conversation(cid)
        conversation_store.add_message(cid, "user", {"text": message},
                                       scope_id=scope_id, turn_id=turn_id)
        conversation_store.add_message(cid, "assistant", {
            "text": result.get("answer", ""),
            "intent": result.get("intent"),
            "evidence_status": result.get("evidence_status"),
        }, scope_id=scope_id, turn_id=turn_id)
        trace = result.get("retrieval_trace") or result.get("trace") or []
        steps = []
        for item in trace:
            if not isinstance(item, dict):
                continue
            steps.append({
                "stage": item.get("stage") or item.get("gate") or "unknown",
                "status": item.get("status", "complete"),
                "detail": item.get("reason") or item.get("counts") or {},
            })
        public_progress = result.get("public_progress") or [
            {"text": _public_progress_text(s), "status": s.get("status", "complete")}
            for s in steps if _public_progress_text(s)
        ]
        conversation_store.save_trajectory(
            turn_id, cid, profile=os.getenv("SENTRIX_AGENT_PROFILE", "goal_driven_shadow"),
            steps=steps, result={"answer": result.get("answer", ""), "intent": result.get("intent"),
                                 "telemetry": result.get("telemetry") or {}},
            public_progress=public_progress, scope_id=scope_id,
        )
        result["public_progress"] = public_progress
    except Exception:
        # 记录失败不影响回答
        pass


# ---- D2: Conversation Lifecycle API ----
class ConversationCreateRequest(BaseModel):
    scope_id: str = "home-default"
    title: str | None = None


class ConversationRenameRequest(BaseModel):
    title: str


@app.post("/api/conversations")
def create_conversation_api(request: ConversationCreateRequest):
    if not CONVERSATION_STORE_ENABLED:
        raise HTTPException(status_code=404, detail="conversation store disabled")
    cid = conversation_store.create_conversation(scope_id=request.scope_id, title=request.title)
    return {"conversation_id": cid, "conversation": conversation_store.get_conversation(cid)}


@app.get("/api/conversations")
def list_conversations_api(scope_id: str | None = None, limit: int = 50):
    if not CONVERSATION_STORE_ENABLED:
        raise HTTPException(status_code=404, detail="conversation store disabled")
    return {"conversations": conversation_store.list_conversations(scope_id=scope_id, limit=limit)}


@app.get("/api/conversations/{conversation_id}")
def get_conversation_api(conversation_id: str, limit: int = 50):
    if not CONVERSATION_STORE_ENABLED:
        raise HTTPException(status_code=404, detail="conversation store disabled")
    conv = conversation_store.get_conversation(conversation_id)
    if conv is None:
        raise HTTPException(status_code=404, detail="conversation not found")
    messages = conversation_store.list_messages(conversation_id, limit=limit)
    return {"conversation": conv, "messages": messages}


@app.patch("/api/conversations/{conversation_id}")
def rename_conversation_api(conversation_id: str, request: ConversationRenameRequest):
    if not CONVERSATION_STORE_ENABLED:
        raise HTTPException(status_code=404, detail="conversation store disabled")
    conv = conversation_store.rename_conversation(conversation_id, request.title)
    if conv is None:
        raise HTTPException(status_code=404, detail="conversation not found")
    return {"conversation": conv}


@app.delete("/api/conversations/{conversation_id}")
def delete_conversation_api(conversation_id: str):
    """删除会话：只删聊天历史/轨迹/摘要/临时 ResultSet 引用，不删家庭长期记忆与 Asset。"""
    if not CONVERSATION_STORE_ENABLED:
        raise HTTPException(status_code=404, detail="conversation store disabled")
    conv = conversation_store.get_conversation(conversation_id)
    if conv is None:
        raise HTTPException(status_code=404, detail="conversation not found")
    conversation_store.delete_conversation(conversation_id)
    _TOOL_LOOP_TASK_STATE.pop(conversation_id, None)
    return {"deleted": True, "conversation_id": conversation_id,
            "note": "已删除聊天记录与处理过程；家庭照片和长期记忆不受影响。"}


# ---- D8: Photo Inspector（照片子会话）----
class PhotoThreadCreateRequest(BaseModel):
    asset_handle: str | None = None
    result_set_id: str | None = None
    asset_id: str | None = None
    parent_conversation_id: str | None = None
    scope_id: str = "home-default"


class PhotoThreadMessageRequest(BaseModel):
    message: str
    viewer_id: str = "owner"


def _photo_thread_meta(thread_id: str):
    try:
        row = store.connection.execute(
            "SELECT * FROM agent_photo_threads WHERE thread_id = ?", (thread_id,)).fetchone()
        return dict(row) if row else None
    except Exception:
        return None


@app.post("/api/photo-threads")
def create_photo_thread(request: PhotoThreadCreateRequest):
    from .agent_runtime import tools as runtime_tools
    runtime_tools.register_tools()
    scope_id = request.scope_id or "home-default"
    handle = request.asset_handle or ""
    asset_id = request.asset_id
    if not asset_id:
        asset_id = runtime_tools.resolve_handle_asset_id(
            handle, request.result_set_id, scope_id)
    if not asset_id:
        raise HTTPException(status_code=404, detail="无法定位照片")
    row = store.connection.execute(
        "SELECT scope_id FROM assets WHERE id = ?", (asset_id,)).fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail="照片不存在")
    if scope_id and row["scope_id"] != scope_id:
        raise HTTPException(status_code=404, detail="照片不在当前相册范围")
    thread_id = make_id("photothread")
    now = datetime.now().astimezone().isoformat(timespec="seconds")
    store.connection.execute(
        """INSERT INTO agent_photo_threads
           (thread_id, parent_conversation_id, scope_id, asset_handle, asset_id,
            result_set_id, created_at, updated_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        (thread_id, request.parent_conversation_id, scope_id, handle or "photo_1",
         asset_id, request.result_set_id or "", now, now))
    store.connection.commit()
    return {"thread_id": thread_id, "asset_handle": handle or "photo_1",
            "asset_id": asset_id, "result_set_id": request.result_set_id or "",
            "scope_id": scope_id, "parent_conversation_id": request.parent_conversation_id}


@app.post("/api/photo-threads/{thread_id}/turn")
def photo_thread_turn(thread_id: str, request: PhotoThreadMessageRequest):
    meta = _photo_thread_meta(thread_id)
    if meta is None:
        raise HTTPException(status_code=404, detail="photo thread not found")
    message = request.message.strip()
    if not message:
        raise HTTPException(status_code=400, detail="message is required")
    recent_turns = ""
    try:
        history = conversation_store.last_messages(thread_id, limit=8)
        recent_turns = "\n".join(
            f"{'用户' if m['role'] == 'user' else '助手'}：{m['content'].get('text', '')}"
            for m in history if m.get("content", {}).get("text"))
    except Exception:
        recent_turns = ""
    turn_id = make_id("turn")
    _TURN_JOBS[turn_id] = {"status": "running", "public_progress": [],
                           "progress_events": [], "result": None,
                           "created_at": time.time()}
    _turn_executor().submit(
        _execute_photo_thread_job, turn_id, thread_id, message,
        meta["scope_id"], request.viewer_id, recent_turns, meta)
    return {"turn_id": turn_id, "status": "running", "thread_id": thread_id}


def _execute_photo_thread_job(turn_id, thread_id, message, scope_id, viewer_id,
                              recent_turns, meta):
    """D8：photo thread 后台 turn（复用 tool-loop，profile=photo_inspector，绑定当前照片）。"""
    job = _TURN_JOBS.get(turn_id)
    try:
        def on_progress(event):
            if job is not None:
                job.setdefault("progress_events", []).append(event)
                job["public_progress"] = job.get("progress_events")

        result = _tool_loop_turn(
            message, conversation_id=thread_id, scope_id=scope_id,
            viewer_id=viewer_id, recent_turns=recent_turns,
            progress_callback=on_progress,
            selected_asset_handle=meta["asset_handle"],
            selected_result_set_id=meta["result_set_id"] or None,
            profile_name="photo_inspector")
        result["photo_thread_id"] = thread_id
        result = assistant_response(result)
        _record_turn_conversation(message, _AssistantTurnLike(
            conversation_id=thread_id, scope_id=scope_id), result, turn_id=turn_id)
        if job is not None:
            job.update({"status": "complete", "result": result,
                        "public_progress": result.get("public_progress") or job.get("public_progress") or []})
    except Exception as exc:
        if job is not None:
            job.update({"status": "error", "error": str(exc)})


@app.get("/api/photo-threads/{thread_id}/messages")
def photo_thread_messages(thread_id: str, limit: int = 20):
    meta = _photo_thread_meta(thread_id)
    if meta is None:
        raise HTTPException(status_code=404, detail="photo thread not found")
    return {"thread_id": thread_id, "asset_handle": meta["asset_handle"],
            "asset_id": meta["asset_id"], "result_set_id": meta["result_set_id"],
            "messages": conversation_store.list_messages(thread_id, limit=limit)}


@app.delete("/api/photo-threads/{thread_id}")
def delete_photo_thread(thread_id: str):
    meta = _photo_thread_meta(thread_id)
    if meta is None:
        raise HTTPException(status_code=404, detail="photo thread not found")
    store.connection.execute(
        "DELETE FROM agent_photo_threads WHERE thread_id = ?", (thread_id,))
    conversation_store.delete_conversation(thread_id)
    _TOOL_LOOP_TASK_STATE.pop(thread_id, None)
    return {"deleted": True, "thread_id": thread_id}


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
    result["answerGrounding"] = result.get("answer_grounding", {})
    result["terminationReason"] = result.get("termination_reason", "")
    result["claimVerifications"] = result["claim_verifications"]
    result["claimVerificationStatus"] = result["claim_verification_status"]
    result["repairCount"] = result["repair_count"]
    result["evidenceBundles"] = result["evidence_bundles"]
    result["claimEvidenceIndex"] = result["claim_evidence_index"]
    if not admin:
        result["retrievalTrace"] = []
        # 思考过程对普通用户可见工具名/状态/耗时；参数与 observation 等明细仍仅管理员可见。
        result["toolTrace"] = [
            {k: v for k, v in (t or {}).items()
             if k in ("tool", "status", "latency_s", "retrieval_timing")}
            for t in (result.get("tool_trace") or [])
        ]
        result.pop("validation", None)
        result.pop("model_call_ledger", None)
        result.pop("task_contract", None)
        result.pop("retrieval_strategy", None)
        result.pop("structured_result", None)
        result.pop("parser_raw", None)
    return result


def _execute_turn_job(turn_id, message, conversation_id, scope_id, viewer_id, recent_turns,
                     selected_asset_handle=None, selected_result_set_id=None,
                     conversation_summary="", include_debug=False):
    """B3.4：后台执行 tool-loop turn，progress 增量写入 job。"""
    job = _TURN_JOBS.get(turn_id)
    try:
        def on_progress(event):
            if job is not None:
                job.setdefault("progress_events", []).append(event)
                job["public_progress"] = job.get("progress_events")

        started = time.time()
        result = _tool_loop_turn(message, conversation_id, scope_id, viewer_id,
                                 recent_turns=recent_turns, progress_callback=on_progress,
                                 selected_asset_handle=selected_asset_handle,
                                 selected_result_set_id=selected_result_set_id,
                                 conversation_summary=conversation_summary,
                                 include_debug=include_debug)
        # B4 canary telemetry：profile / 工具序列 / guard / 延迟 / fallback 标记
        try:
            trace = result.get("retrieval_trace") or []
            result["telemetry"] = {
                "profile": os.getenv("SENTRIX_AGENT_PROFILE", "goal_driven_shadow"),
                "status": result.get("tool_loop_status"),
                "reason": result.get("tool_loop_reason"),
                "termination_reason": result.get("termination_reason"),
                "tools": [s.get("tool") for s in trace if s.get("stage") == "tool" and s.get("tool")],
                "latency_s": round(time.time() - started, 2),
                "fallback": False,
                "guard_blocked": result.get("tool_loop_status") in {"blocked_by_guard", "partial", "timeout", "error"},
                "public_progress_count": len(result.get("public_progress") or []),
            }
        except Exception:
            pass
        residual_metrics = gamma.get_and_clear_call_metrics()
        if residual_metrics:
            result.setdefault("model_call_metrics", []).extend(residual_metrics)
        if conversation_id:
            _TOOL_LOOP_TASK_STATE[conversation_id] = result.get("task_state") or {}
        result = assistant_response(result)
        _record_turn_conversation(message, _AssistantTurnLike(
            conversation_id=conversation_id, scope_id=scope_id), result, turn_id=turn_id)
        if job is not None:
            job.update({"status": "complete", "result": result,
                        "public_progress": result.get("public_progress") or job.get("public_progress") or []})
        # D3：后台生成会话摘要（不阻塞回答交付）
        if CONVERSATION_STORE_ENABLED and conversation_id and result.get("tool_loop_status") == "complete":
            threading.Thread(target=_background_conversation_summary,
                             args=(conversation_id, scope_id), daemon=True).start()
    except Exception as exc:
        if job is not None:
            job.update({"status": "error", "error": str(exc)})


def _background_conversation_summary(conversation_id, scope_id):
    """D3：对话轮次足够后，用 12B 生成/更新会话摘要并保存。"""
    try:
        from .agent_runtime.conversation_summary import summarize_with_model
        messages = conversation_store.list_messages(conversation_id, limit=60)
        count = sum(1 for m in messages if m.get("role") in {"user", "assistant"})
        if count < 6:
            return

        def _chat_fn(msgs):
            payload = {
                "model": getattr(gamma, "model", "gemma4-12b-it"),
                "messages": msgs, "temperature": 0.0, "max_tokens": 700,
            }
            resp = httpx.post(f"{gamma.base_url}/chat/completions", json=payload, timeout=60)
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"]

        summary = summarize_with_model(_chat_fn, messages)
        if summary.strip():
            conversation_store.save_summary(conversation_id, summary)
    except Exception:
        pass


class _AssistantTurnLike:
    """异步路径的 request 占位（只暴露 _record_turn_conversation 需要的字段）。"""
    def __init__(self, *, conversation_id, scope_id):
        self.conversation_id = conversation_id
        self.scope_id = scope_id


@app.get("/api/conversation/{conversation_id}/messages")
def conversation_messages(conversation_id: str, limit: int = 20):
    if not CONVERSATION_STORE_ENABLED:
        raise HTTPException(status_code=404, detail="conversation store disabled")
    try:
        return {"conversation_id": conversation_id,
                "messages": conversation_store.list_messages(conversation_id, limit=limit)}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/api/conversation/{conversation_id}/trajectory")
def conversation_trajectory(conversation_id: str, limit: int = 20):
    if not CONVERSATION_STORE_ENABLED:
        raise HTTPException(status_code=404, detail="conversation store disabled")
    try:
        return {"conversation_id": conversation_id,
                "trajectories": conversation_store.list_trajectories(conversation_id, limit=limit)}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


def _public_progress_text(item):
    stage = item.get("stage") or ""
    status = item.get("status") or ""
    if stage == "gate":
        return "正在判断你的意图。"
    if stage == "retrieval":
        counts = item.get("counts") or {}
        n = counts.get("assets", counts.get("exact", 0))
        return f"已找到 {n} 条相关记录。" if n else "正在检索相关记录。"
    if stage == "answer":
        return "正在组织回答。"
    if stage == "channels":
        return "已合并多路检索结果。"
    if status == "contextual":
        return "正在读取记忆上下文。"
    if status == "gap":
        return "当前记录中没有找到足够匹配的证据。"
    return ""


@app.post("/api/ingest", status_code=202)
async def ingest(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    sourceOwnerId: str | None = Form(None),
    sourceDeviceId: str | None = Form(None),
    sourceAlbumId: str | None = Form(None),
    capturedAt: str | None = Form(None),
    capturedLocation: str | None = Form(None),
    scopeId: str | None = Form(None),
    scope_id: str | None = Form(None),
):
    safe_name = Path(file.filename or "upload.bin").name
    asset_id = make_id("asset")
    media_type = media_type_from_upload(file.content_type, safe_name)
    destination = _upload_destination(asset_id, safe_name, media_type)
    with destination.open("wb") as output:
        shutil.copyfileobj(file.file, output)
    mime_type = file.content_type or guess_mime_type(safe_name)
    scope = (scope_id or scopeId or "home-default").strip() or "home-default"
    metadata = {
        "scope_id": scope,
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
    # Deduplication is scoped to the album: the same photo may legitimately
    # appear in a different memory space without being treated as a duplicate.
    existing = store.find_asset_by_hash(metadata["content_sha256"], scope)
    if existing:
        destination.unlink(missing_ok=True)
        return {"accepted": True, "assetId": existing["id"], "fileName": existing["file_name"], "status": existing["status"], "mediaType": existing["media_type"], "scope_id": scope, "deduplicated": True}
    created = store.create_asset(
        asset_id,
        safe_name,
        media_type,
        str(destination),
        mime_type,
        destination.stat().st_size,
        metadata,
    )
    if media_type == "video":
        created = store.update_asset(asset_id, "video-queued", {"video_stage": "video-queued"})
    background_tasks.add_task(process_asset, asset_id)
    return {"accepted": True, "assetId": created["id"], "fileName": safe_name, "status": created["status"], "mediaType": media_type, "scope_id": scope}


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
    deferBatchComplete: bool = Form(False),
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
    with db_write_guard("import-remote-init"):
        store.create_memory_space(scope, scope, kind="benchmark")
        store.create_ingest_batch(batch, scope)
    items = []
    queued_asset_ids = []
    for index, upload in enumerate(files):
        safe_name = Path(upload.filename or f"upload-{index}").name
        media_type = (upload.content_type or "application/octet-stream").split("/", 1)[0]
        if media_type not in {"image", "audio", "video", "text"}:
            media_type = media_type_from_upload(upload.content_type, safe_name)
        destination = _upload_destination(make_id("upload"), safe_name, media_type)
        try:
            save_started = time.perf_counter()
            with destination.open("wb") as output:
                shutil.copyfileobj(upload.file, output)
            file_save_seconds = round(time.perf_counter() - save_started, 4)
            capture = _normalized_capture_metadata(per_file[index])
            with db_write_guard("import-remote-file"):
                created = pipeline.create_asset(destination, file_name=safe_name, media_type=media_type, mime_type=upload.content_type, metadata={
                    "scope_id": scope, "batch_id": batch, "source_owner_id": sourceOwnerId,
                    "source_owner_label": sourceOwnerLabel, "source_device_id": sourceDeviceId,
                    "source_album_id": sourceAlbumId, "source_confidence": 1.0 if sourceOwnerId else 0.0,
                    **capture,
                })
                if media_type == "video" and created.get("path") == str(destination):
                    created = store.update_asset(created["id"], "video-queued", {"video_stage": "video-queued"})
                created = store.update_asset(created["id"], created.get("status") or "queued", {
                    "import_timings": {**((created.get("metadata_json") or {}).get("import_timings") or {}), "file_save_seconds": file_save_seconds}
                })
            deduplicated = created.get("path") != str(destination)
            if deduplicated:
                destination.unlink(missing_ok=True)
            elif created.get("status") in {"queued", "failed", "video-queued", "video-processing-failed"}:
                queued_asset_ids.append(created["id"])
            items.append({"accepted": True, "assetId": created["id"], "asset_id": created["id"], "fileName": created["file_name"], "status": created["status"], "scope_id": created.get("scope_id"), "batch_id": created.get("batch_id"), "deduplicated": deduplicated})
        except ValueError as error:
            destination.unlink(missing_ok=True)
            items.append({"accepted": False, "fileName": safe_name, "status": "rejected", "error": str(error)})
        except Exception as error:
            destination.unlink(missing_ok=True)
            items.append({"accepted": False, "fileName": safe_name, "status": "failed", "error": str(error)})
    if queued_asset_ids:
        if not deferBatchComplete:
            store.complete_ingest_batch(batch)
        background_tasks.add_task(process_ingest_batch, queued_asset_ids, batch)
    else:
        store.complete_ingest_batch(batch)
        background_tasks.add_task(pipeline.finalize_ingest_batch, batch)
    return {"accepted": any(item["accepted"] for item in items), "batch_id": batch, "scope_id": scope, "items": items, "accepted_count": sum(item["accepted"] for item in items), "rejected_count": sum(not item["accepted"] for item in items)}


@app.post("/api/import/server-directory", status_code=202)
def import_assets(request: ImportRequest, background_tasks: BackgroundTasks):
    source = Path(request.source_path).expanduser().resolve()
    _assert_import_path_allowed(source)
    if not source.exists():
        raise HTTPException(status_code=404, detail="source_path not found")
    scope = (request.scope_id or "home-default").strip() or "home-default"
    batch_id = (request.batch_id or make_id("batch")).strip()
    with db_write_guard("import-directory-init"):
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
    queued_asset_ids = []
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
            copy_started = time.perf_counter()
            with db_write_guard("import-directory-file"):
                created = pipeline.create_asset(target, file_name=path.name, metadata=metadata)
                if created.get("media_type") == "video" and created.get("path") == str(target):
                    created = store.update_asset(created["id"], "video-queued", {"video_stage": "video-queued"})
                created = store.update_asset(created["id"], created.get("status") or "queued", {
                    "import_timings": {**((created.get("metadata_json") or {}).get("import_timings") or {}), "file_save_seconds": round(time.perf_counter() - copy_started, 4) if request.copy_file else 0.0}
                })
        except Exception as error:
            if request.copy_file:
                target.unlink(missing_ok=True)
            skipped.append({"path": str(path), "reason": str(error)})
            continue
        deduplicated = created.get("path") != str(target)
        if request.copy_file and deduplicated:
            target.unlink(missing_ok=True)
        elif created.get("status") in {"queued", "failed", "video-queued", "video-processing-failed"}:
            queued_asset_ids.append(created["id"])
        imported.append({
            "asset_id": created["id"],
            "file_name": created["file_name"],
            "media_type": created["media_type"],
            "status": created["status"],
            "deduplicated": deduplicated,
            "source_path": str(path),
        })
    if queued_asset_ids:
        store.complete_ingest_batch(batch_id)
        background_tasks.add_task(process_ingest_batch, queued_asset_ids, batch_id)
    else:
        store.complete_ingest_batch(batch_id)
        background_tasks.add_task(pipeline.finalize_ingest_batch, batch_id)
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
    with db_write_guard("ingest-batch-create"):
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
    with db_write_guard("ingest-batch-complete-endpoint"):
        store.complete_ingest_batch(batch_id)
    with batch_worker_lock:
        worker_active = batch_id in active_batch_workers
    if not worker_active:
        background_tasks.add_task(pipeline.finalize_ingest_batch, batch_id)
    return _batch_status(batch_id)


@app.post("/api/maintenance/recheck")
def recheck(background_tasks: BackgroundTasks):
    assets = [store.get_asset(row["id"]) for row in store._rows("SELECT id FROM assets WHERE status IN ('queued', 'failed', 'semantic_enriching', 'video-queued', 'video-processing-failed') ORDER BY created_at")]
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
