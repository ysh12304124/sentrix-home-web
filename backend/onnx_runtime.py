"""Shared ONNX Runtime settings for the optional GPU face pipeline."""

from __future__ import annotations

import contextlib
import os
import threading


def _cuda_requested(provider_env: str) -> bool:
    return any(
        item.strip() == "CUDAExecutionProvider"
        for item in os.getenv(provider_env, "CPUExecutionProvider").split(",")
    )


def face_onnx_providers(provider_env: str):
    """Build bounded providers for one RetinaFace/InsightFace ONNX session.

    The limit is per ONNX session. The default 256 MiB budget is deliberately
    conservative because FaceAdapter may own several sessions at once.
    """
    requested = [item.strip() for item in os.getenv(provider_env, "CPUExecutionProvider").split(",") if item.strip()]
    if not _cuda_requested(provider_env):
        return requested

    limit_env = (
        "SENTRIX_RETINAFACE_GPU_SESSION_LIMIT_MIB"
        if provider_env == "RETINAFACE_PROVIDERS"
        else "SENTRIX_FACE_GPU_SESSION_LIMIT_MIB"
    )
    try:
        limit_mib = max(128, int(os.getenv(limit_env, "256")))
    except ValueError:
        limit_mib = 256
    options = {
        "device_id": int(os.getenv("SENTRIX_FACE_GPU_DEVICE_ID", "0")),
        "gpu_mem_limit": limit_mib * 1024 * 1024,
        "arena_extend_strategy": os.getenv("SENTRIX_FACE_GPU_ARENA_EXTEND_STRATEGY", "kSameAsRequested"),
        "cudnn_conv_algo_search": "HEURISTIC",
        "cudnn_conv_use_max_workspace": "0",
    }
    providers = []
    for item in requested:
        providers.append((item, options.copy()) if item == "CUDAExecutionProvider" else item)
    return providers


def face_onnx_provider_options(provider_env: str):
    """Return InsightFace-compatible provider options for the same budget."""
    providers = face_onnx_providers(provider_env)
    options = []
    for provider in providers:
        if isinstance(provider, tuple):
            options.append(provider[1])
        else:
            options.append({})
    return options


_FACE_GPU_GATE = threading.BoundedSemaphore(
    max(1, int(os.getenv("SENTRIX_FACE_GPU_MAX_CONCURRENCY", "1")))
)


@contextlib.contextmanager
def face_gpu_inference_gate():
    """Bound face GPU temporary allocations while ASR/VLM remain concurrent."""
    if _cuda_requested("FACE_PROVIDERS") or _cuda_requested("RETINAFACE_PROVIDERS"):
        _FACE_GPU_GATE.acquire()
        try:
            yield
        finally:
            _FACE_GPU_GATE.release()
    else:
        yield
