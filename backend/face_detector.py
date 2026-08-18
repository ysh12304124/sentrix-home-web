"""Optional RetinaFace-R50 tiled face detector for FaceAdapter.

SCRFD-10G misses small faces in high-resolution family photos because the whole
image is downscaled to 640 before detection. This detector tiles the image and
detects each tile (so a small face is larger relative to the tile), then maps
results back to image coordinates and merges with NMS. A two-pass strategy runs a
fast coarse pass (1024px tiles) and only falls back to a fine pass (640px tiles,
lower threshold) for images where the coarse pass found nothing.

The model is biubug6/Pytorch_Retinaface (MIT), BGR + mean 104/117/123, decode via
priors (min_sizes) + variance, not InsightFace's SCRFD format.
"""

from __future__ import annotations

import os
import threading

import numpy as np

STEPS = [8, 16, 32]
MIN_SIZES = [[16, 32], [64, 128], [256, 512]]
INPUT_SIZE = (640, 640)
VARIANCES = (0.1, 0.2)

DEFAULT_MODEL_PATH = "/home/asus/benchmarks/retinaface/retinaface_r50.onnx"


def _generate_priors(input_size):
    priors = []
    for step, min_size in zip(STEPS, MIN_SIZES):
        height = input_size[1] // step
        width = input_size[0] // step
        for i in range(height):
            for j in range(width):
                cx = (j + 0.5) * step
                cy = (i + 0.5) * step
                for ms in min_size:
                    priors.append([cx, cy, ms, ms])
    return np.array(priors, dtype=np.float32)


def _decode(loc, priors):
    boxes = np.concatenate((
        priors[:, :2] + loc[:, :2] * VARIANCES[0] * priors[:, 2:],
        priors[:, 2:] * np.exp(loc[:, 2:] * VARIANCES[0]),
    ), 1)
    boxes[:, :2] -= boxes[:, 2:] / 2
    boxes[:, 2:] += boxes[:, :2]
    return boxes


def _decode_landms(landms, priors):
    parts = []
    for index in range(5):
        parts.append(priors[:, :2] + landms[:, index * 2:index * 2 + 2] * VARIANCES[1] * priors[:, 2:])
    return np.concatenate(parts, 1).reshape(-1, 5, 2)


def _nms(boxes, threshold=0.4):
    x1, y1, x2, y2, scores = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3], boxes[:, 4]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
        union = areas[i] + areas[order[1:]] - inter
        iou = inter / np.maximum(union, 1e-9)
        order = order[1:][iou <= threshold]
    return np.array(keep, dtype=int)


class RetinaFaceTiledDetector:
    """Tiled RetinaFace-R50 detection returning image-coordinate faces."""

    def __init__(self, model_path=None):
        self.model_path = os.getenv("RETINAFACE_MODEL_PATH", "") or model_path or DEFAULT_MODEL_PATH
        self._session = None
        self._input_name = None
        self._lock = threading.Lock()
        self.error = None

    def _load(self):
        if self._session is not None:
            return
        with self._lock:
            if self._session is None:
                try:
                    import onnxruntime
                    providers = [
                        item for item in os.getenv(
                            "RETINAFACE_PROVIDERS", "CUDAExecutionProvider,CPUExecutionProvider"
                        ).split(",") if item
                    ]
                    self._session = onnxruntime.InferenceSession(self.model_path, providers=providers)
                    self._input_name = self._session.get_inputs()[0].name
                except Exception as error:
                    self.error = str(error)
                    raise

    def _detect_tile(self, image):
        import cv2
        image_height, image_width = image.shape[:2]
        ratio = image_height / image_width
        model_ratio = INPUT_SIZE[1] / INPUT_SIZE[0]
        if ratio > model_ratio:
            new_height = INPUT_SIZE[1]
            new_width = int(new_height / ratio)
        else:
            new_width = INPUT_SIZE[0]
            new_height = int(new_width * ratio)
        scale = new_height / image_height
        resized = cv2.resize(image, (new_width, new_height))
        canvas = np.zeros((INPUT_SIZE[1], INPUT_SIZE[0], 3), dtype=np.uint8)
        canvas[:new_height, :new_width] = resized
        blob = cv2.dnn.blobFromImage(canvas, 1.0, INPUT_SIZE, (104, 117, 123), swapRB=False)
        loc, conf, landms = self._session.run(None, {self._input_name: blob})
        priors = _generate_priors(INPUT_SIZE)
        boxes = _decode(loc[0], priors)
        scores = conf[0][:, 1]
        kps = _decode_landms(landms[0], priors)
        positions = np.where(scores >= 0.05)[0]
        detections = []
        for index in positions:
            x1, y1, x2, y2 = boxes[index] / scale
            face_kps = kps[index] / scale  # retinaface order already matches arcface
            detections.append({
                "bbox": [float(x1), float(y1), float(x2), float(y2)],
                "confidence": float(scores[index]),
                "landmarks": [[float(x), float(y)] for x, y in face_kps],
            })
        return detections

    def _detect_tiled(self, image, tile, overlap, threshold):
        import cv2
        image_height, image_width = image.shape[:2]
        step = tile - overlap
        all_faces = []
        for y in range(0, image_height, step):
            for x in range(0, image_width, step):
                x2 = min(x + tile, image_width)
                y2 = min(y + tile, image_height)
                patch = image[y:y2, x:x2]
                if patch.shape[0] < 16 or patch.shape[1] < 16:
                    continue
                for face in self._detect_tile(patch):
                    if face["confidence"] < threshold:
                        continue
                    bbox = face["bbox"]
                    all_faces.append((
                        bbox[0] + x, bbox[1] + y, bbox[2] + x, bbox[3] + y, face["confidence"],
                        face["landmarks"],
                    ))
        if not all_faces:
            return []
        boxes = np.array([[f[0], f[1], f[2], f[3], f[4]] for f in all_faces])
        keep = _nms(boxes, 0.4)
        return [
            {
                "bbox": [float(all_faces[i][0]), float(all_faces[i][1]), float(all_faces[i][2]), float(all_faces[i][3])],
                "confidence": float(all_faces[i][4]),
                "landmarks": [[float(v) for v in point] for point in all_faces[i][5]],
            }
            for i in keep
        ]

    def detect(self, image, threshold=0.3):
        """Two-pass tiled detection: coarse 1024px, then fine 640px only if empty."""
        self._load()
        faces = self._detect_tiled(image, tile=1024, overlap=128, threshold=threshold)
        if not faces:
            faces = self._detect_tiled(image, tile=640, overlap=256, threshold=0.1)
        return faces
