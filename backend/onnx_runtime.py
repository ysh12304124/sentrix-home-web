"""Shared ONNX Runtime settings for the optional GPU face pipeline."""

from __future__ import annotations

import contextlib
import os
import threading
from .platform_profile import profile


def _effective_provider_spec(provider_env: str) -> str:
    """该 provider 链的实际取值：环境变量优先，否则由能力档案探测。

    为什么默认不能写死 CPUExecutionProvider：在 46/118 上不配任何变量就会
    退化成纯 CPU，视觉编码从 0.72 秒变成 21~36 秒（实测），而这条退化在日志里
    看不出来。
    """
    override = (os.getenv(provider_env) or "").strip()
    if override:
        return override
    if provider_env == "RETINAFACE_PROVIDERS":
        return profile.retinaface_providers()
    return profile.face_providers()


def _cuda_requested(provider_env: str) -> bool:
    return any(
        item.strip() == "CUDAExecutionProvider"
        for item in _effective_provider_spec(provider_env).split(",")
    )


def face_onnx_providers(provider_env: str):
    """Build bounded providers for one RetinaFace/InsightFace ONNX session.

    The limit is per ONNX session. The default 256 MiB budget is deliberately
    conservative because FaceAdapter may own several sessions at once.
    """
    requested = [item.strip() for item in _effective_provider_spec(provider_env).split(",") if item.strip()]
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
