#!/usr/bin/env python3
"""HTTP sidecar for Qwen3-VL visual embeddings or multimodal reranking.

The official Qwen3-VL-Embedding repository ships custom model wrappers rather
than SentenceTransformers classes.  Keeping them in a sidecar isolates their
large Torch state from the Sentrix API process and lets each service unload
after an idle period.
"""

from __future__ import annotations

import argparse
import base64
import gc
import io
import json
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


MODE = os.getenv("SENTRIX_QWEN3_VL_MODE", "embedding").strip().lower()
MODEL_PATH = os.getenv("SENTRIX_QWEN3_VL_MODEL_PATH", "").strip()
MODEL_ID = os.getenv("SENTRIX_QWEN3_VL_MODEL_ID", Path(MODEL_PATH).name or MODE)
CODE_PATH = os.getenv("SENTRIX_QWEN3_VL_CODE_PATH", "").strip()
DEVICE = os.getenv("SENTRIX_QWEN3_VL_DEVICE", "cuda").strip().lower()
MAX_LENGTH = max(256, int(os.getenv("SENTRIX_QWEN3_VL_MAX_LENGTH", "4096")))
MAX_PIXELS = max(4096, int(os.getenv("SENTRIX_QWEN3_VL_MAX_PIXELS", "786432")))
MIN_PIXELS = max(1024, int(os.getenv("SENTRIX_QWEN3_VL_MIN_PIXELS", "4096")))
MAX_FRAMES = max(1, int(os.getenv("SENTRIX_QWEN3_VL_MAX_FRAMES", "16")))
IDLE_UNLOAD_SECONDS = max(
    0.0, float(os.getenv("SENTRIX_QWEN3_VL_IDLE_UNLOAD_SECONDS", "60")),
)

if DEVICE == "cpu":
    # Must be set before importing torch/the official wrapper.  The upstream
    # classes choose their device from torch.cuda.is_available().
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

_model = None
_lock = threading.RLock()
_last_used = 0.0
_timer: threading.Timer | None = None
_load_ms = 0.0
_error = ""


def _model_kwargs():
    import torch

    if DEVICE.startswith("cuda") and torch.cuda.is_available():
        return {"torch_dtype": torch.float16, "low_cpu_mem_usage": True}
    return {"torch_dtype": torch.float32, "low_cpu_mem_usage": True}


def _load():
    global _model, _load_ms, _error
    if _model is not None:
        return _model
    if MODE not in {"embedding", "reranker"}:
        raise ValueError("mode must be embedding or reranker")
    if not Path(MODEL_PATH).is_dir():
        raise FileNotFoundError(f"model directory not found: {MODEL_PATH}")
    if not Path(CODE_PATH).is_dir():
        raise FileNotFoundError(f"official Qwen3-VL code directory not found: {CODE_PATH}")
    if CODE_PATH not in sys.path:
        sys.path.insert(0, CODE_PATH)
    started = time.monotonic()
    try:
        if MODE == "embedding":
            from src.models.qwen3_vl_embedding import Qwen3VLEmbedder

            _model = Qwen3VLEmbedder(
                MODEL_PATH, max_length=MAX_LENGTH, min_pixels=MIN_PIXELS,
                max_pixels=MAX_PIXELS, max_frames=MAX_FRAMES, **_model_kwargs(),
            )
        else:
            from src.models.qwen3_vl_reranker import Qwen3VLReranker

            _model = Qwen3VLReranker(
                MODEL_PATH, max_length=MAX_LENGTH, min_pixels=MIN_PIXELS,
                max_pixels=MAX_PIXELS, max_frames=MAX_FRAMES, **_model_kwargs(),
            )
        _load_ms = round((time.monotonic() - started) * 1000, 1)
        _error = ""
        return _model
    except Exception as exc:
        _error = f"{type(exc).__name__}: {exc}"
        raise


def _unload():
    global _model, _timer
    with _lock:
        _model = None
        if _timer is not None:
            _timer.cancel()
            _timer = None
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass


def _schedule_unload():
    global _timer
    if _timer is not None:
        _timer.cancel()
    if IDLE_UNLOAD_SECONDS <= 0:
        _timer = None
        return
    expected = _last_used

    def unload_if_idle():
        if _last_used == expected and time.monotonic() - expected >= IDLE_UNLOAD_SECONDS:
            _unload()

    _timer = threading.Timer(IDLE_UNLOAD_SECONDS + 0.05, unload_if_idle)
    _timer.daemon = True
    _timer.start()


def _read_video_frames(path: str, max_frames: int):
    """Decode evenly-spaced local frames before the upstream file:// shim.

    qwen-vl-utils prepends ``file://`` to local video strings.  On Windows
    that produces ``file://D:\\...``, which torchvision accepts as an empty
    video and the official reranker then silently falls back to text-only.
    Passing PIL frames avoids that URI ambiguity entirely.
    """
    import cv2
    from PIL import Image

    capture = cv2.VideoCapture(path)
    if not capture.isOpened():
        raise ValueError(f"unable to open local video: {path}")
    try:
        total = max(1, int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 1))
        count = max(1, min(int(max_frames), total))
        indices = ([0] if count == 1 else [
            round(index * (total - 1) / (count - 1)) for index in range(count)
        ])
        frames = []
        for frame_index in indices:
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame = capture.read()
            if ok and frame is not None:
                frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
        if not frames:
            raise ValueError(f"no frames decoded from local video: {path}")
        return frames
    finally:
        capture.release()


def _normalise_input(value: dict, *, video_max_frames: int | None = None) -> dict:
    result = {}
    for key in ("text", "image", "video", "instruction", "fps", "max_frames"):
        if value.get(key) not in (None, "", []):
            result[key] = value[key]
    local_image = result.get("image")
    if isinstance(local_image, str) and Path(local_image).is_file():
        from PIL import Image

        # PIL avoids ambiguous ``file://D:\...`` handling in qwen-vl-utils on
        # Windows while keeping HTTP image URLs untouched.
        with Image.open(local_image) as image:
            result["image"] = image.convert("RGB").copy()
    local_video = result.get("video")
    if isinstance(local_video, str) and Path(local_video).is_file():
        try:
            result["video"] = _read_video_frames(
                local_video, video_max_frames or result.get("max_frames") or MAX_FRAMES,
            )
        except Exception:
            # A native FFmpeg allocation/codec failure must not invalidate the
            # whole query.  Keep the candidate's structured text and rerank it
            # text-only; the caller still gets one score per document.
            result.pop("video", None)
    encoded_image = value.get("image_base64")
    if encoded_image:
        from PIL import Image

        result["image"] = Image.open(io.BytesIO(base64.b64decode(encoded_image))).convert("RGB")
    return result


def _close_media(value) -> None:
    """Release decoded PIL frames immediately after one sidecar request.

    The VL reranker decodes many videos in a long benchmark.  Waiting for a
    later cyclic-GC pass can retain hundreds of full-resolution frame buffers
    and eventually make OpenCV/FFmpeg fail a small native allocation on
    Windows.  Model inputs are no longer needed once ``process`` returns.
    """
    if isinstance(value, dict):
        for item in value.values():
            _close_media(item)
        return
    if isinstance(value, (list, tuple)):
        for item in value:
            _close_media(item)
        return
    close = getattr(value, "close", None)
    if callable(close):
        try:
            close()
        except Exception:
            pass


def _embed(payload: dict) -> dict:
    global _last_used
    values = payload.get("inputs")
    if not isinstance(values, list):
        values = [payload]
    values = [_normalise_input(value if isinstance(value, dict) else {}) for value in values]
    with _lock:
        model = _load()
        vectors = model.process(values)
        if hasattr(vectors, "detach"):
            vectors = vectors.detach().float().cpu().tolist()
        else:
            vectors = vectors.tolist()
        _last_used = time.monotonic()
        _schedule_unload()
    return {
        "vectors": vectors, "vector": vectors[0] if len(vectors) == 1 else None,
        "model": MODEL_ID, "dimension": len(vectors[0]) if vectors else 0,
        "load_ms": _load_ms,
    }


def _rerank(payload: dict) -> dict:
    global _last_used
    query = payload.get("query") or {}
    if isinstance(query, str):
        query = {"text": query}
    documents = payload.get("documents") or []
    if not isinstance(documents, list):
        raise ValueError("documents must be an array")
    requested_frames = int(payload.get("max_frames") or MAX_FRAMES)
    request = {
        "instruction": payload.get("instruction") or (
            "Given a personal memory question, rank media containing evidence "
            "relevant to answering the question."
        ),
        "query": _normalise_input(query, video_max_frames=requested_frames),
        "documents": [
            _normalise_input(item, video_max_frames=requested_frames)
            for item in documents if isinstance(item, dict)
        ],
        "fps": float(payload.get("fps") or 1.0),
        "max_frames": requested_frames,
    }
    with _lock:
        try:
            model = _load()
            scores = [float(score) for score in model.process(request)]
            _last_used = time.monotonic()
            _schedule_unload()
        finally:
            _close_media(request)
            gc.collect()
    return {"scores": scores, "model": MODEL_ID, "load_ms": _load_ms}


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def _send(self, code: int, payload: dict):
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path != "/health":
            self._send(404, {"error": "not found"})
            return
        self._send(200, {
            "status": "ready" if _model is not None else "idle",
            "mode": MODE, "model": MODEL_ID, "model_path": MODEL_PATH,
            "device": DEVICE, "dimension": 2048 if MODE == "embedding" else None,
            "loaded": _model is not None, "load_ms": _load_ms, "error": _error or None,
        })

    def do_POST(self):
        try:
            length = int(self.headers.get("Content-Length", 0) or 0)
            payload = json.loads(self.rfile.read(length) or b"{}")
            if self.path == "/embed" and MODE == "embedding":
                self._send(200, _embed(payload))
            elif self.path == "/rerank" and MODE == "reranker":
                self._send(200, _rerank(payload))
            elif self.path == "/unload":
                _unload()
                self._send(200, {"status": "unloaded"})
            else:
                self._send(404, {"error": "endpoint unavailable for this mode"})
        except Exception as exc:
            self._send(500, {"error": f"{type(exc).__name__}: {exc}"})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    args = parser.parse_args()
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(
        f"Qwen3-VL {MODE} sidecar ready on {args.host}:{args.port}; "
        f"model={MODEL_ID} device={DEVICE}", flush=True,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        _unload()


if __name__ == "__main__":
    main()
