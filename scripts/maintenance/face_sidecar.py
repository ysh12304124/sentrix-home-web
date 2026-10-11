"""CPU face detection/embedding sidecar for the Windows Python 3.12 runtime."""
from __future__ import annotations

import base64
import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import cv2
import numpy as np
import onnxruntime

from backend.face_detector import RetinaFaceTiledDetector

ROOT = Path(os.getenv("FACE_MODEL_ROOT", "data/face-models")).resolve()
RETINA = RetinaFaceTiledDetector(str(ROOT / "retinaface_r50.onnx"))
ARC = onnxruntime.InferenceSession(
    str(ROOT / os.getenv("FACE_MODEL_NAME", "buffalo_l") / "w600k_r50.onnx"),
    providers=["CPUExecutionProvider"],
)
ARC_INPUT = ARC.get_inputs()[0].name


def embedding(image: np.ndarray, bbox: list[float]) -> list[float]:
    x1, y1, x2, y2 = [int(round(x)) for x in bbox]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(image.shape[1], x2), min(image.shape[0], y2)
    crop = image[y1:y2, x1:x2]
    if crop.size == 0:
        return []
    crop = cv2.resize(crop, (112, 112), interpolation=cv2.INTER_LINEAR)
    crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB).astype(np.float32)
    crop = ((crop - 127.5) / 128.0).transpose(2, 0, 1)[None, ...]
    out = ARC.run(None, {ARC_INPUT: crop})[0][0]
    return [float(x) for x in out]


def detect(path: str) -> list[dict]:
    image = cv2.imread(path)
    if image is None:
        return []
    detections = RETINA.detect(image)
    result = []
    for item in detections:
        bbox = [float(x) for x in item["bbox"]]
        emb = embedding(image, bbox)
        if not emb:
            continue
        result.append({
            "bbox": bbox,
            "confidence": float(item.get("confidence", 0.0)),
            "landmarks": item.get("landmarks") or [],
            "embedding": emb,
            "embedding_model": "buffalo_l",
            "embedding_version": os.getenv("FACE_MODEL_NAME", "buffalo_l"),
            "identity_ready": True,
            "face_validity": "verified",
            "identity_eligible": float(item.get("confidence", 0.0)) >= 0.7,
        })
    return result


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *_args):
        pass

    def do_GET(self):
        if self.path == "/health":
            self._reply(200, {"status": "ok", "detector": "retinaface-r50", "recognizer": "w600k_r50"})
        else:
            self._reply(404, {"error": "not found"})

    def do_POST(self):
        if self.path != "/detect":
            self._reply(404, {"error": "not found"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            body = json.loads(self.rfile.read(length) or b"{}")
            path = str(body.get("path") or "")
            self._reply(200, {"faces": detect(path)})
        except Exception as exc:
            self._reply(500, {"error": str(exc), "faces": []})

    def _reply(self, code: int, payload: dict):
        raw = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)


if __name__ == "__main__":
    host = os.getenv("FACE_SIDECAR_HOST", "127.0.0.1")
    port = int(os.getenv("FACE_SIDECAR_PORT", "8102"))
    ThreadingHTTPServer((host, port), Handler).serve_forever()
