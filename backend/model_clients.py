import base64
from io import BytesIO
import json
import mimetypes
import os
import re
import threading
import time
from pathlib import Path

from .face_embeddings import FaceEmbeddingUnavailable, compute_face_quality
from .geocoding import format_gps_prefix
from .onnx_runtime import face_gpu_inference_gate, face_onnx_provider_options, face_onnx_providers
from .runtime_providers import OpenAICompatibleInferenceProvider, normalize_openai_base_url


def align_face_crop(image, bbox, landmarks=None):
    """Return an RGB AdaFace input, using InsightFace's five-point alignment when available."""
    from PIL import Image

    if landmarks:
        try:
            import numpy as np
            from insightface.utils import face_align

            aligned_bgr = face_align.norm_crop(
                image,
                np.asarray(landmarks, dtype="float32"),
                image_size=112,
            )
            return Image.fromarray(aligned_bgr[:, :, ::-1])
        except (ImportError, TypeError, ValueError):
            pass
    left, top, right, bottom = (int(round(value)) for value in bbox)
    image_height, image_width = image.shape[:2]
    left, top = max(0, left), max(0, top)
    right, bottom = min(image_width, right), min(image_height, bottom)
    if right <= left or bottom <= top:
        raise FaceEmbeddingUnavailable("invalid face bounding box")
    return Image.fromarray(image[top:bottom, left:right][:, :, ::-1])


def _laplacian_variance(pil_image):
    """Grayscale Laplacian variance of an aligned face crop as a sharpness signal."""
    import cv2
    import numpy as np

    try:
        gray = cv2.cvtColor(np.asarray(pil_image.convert("RGB")), cv2.COLOR_RGB2GRAY)
        return float(cv2.Laplacian(gray, cv2.CV_64F).var())
    except Exception:
        return 0.0


def _normalize_sharpness(raw, low=2.5, high=7.0):
    """Log-compress raw Laplacian variance into 0..1, clipping at the reference bounds.

    low/high are on the log1p scale and should be calibrated against the real
    face-crop distribution; the defaults assume sharp aligned 112x112 crops.
    """
    import math

    value = (math.log1p(max(0.0, raw)) - low) / (high - low)
    return max(0.0, min(1.0, value))

try:
    import httpx
except ImportError:  # Keep pure parsing and SQLite tests runnable without optional runtime deps.
    class _HttpxUnavailable:
        class HTTPError(Exception):
            pass

        @staticmethod
        def post(*args, **kwargs):
            raise _HttpxUnavailable.HTTPError("httpx is not installed")

    # Keep a patchable module-shaped object.  The production dependency is
    # still declared in requirements; this fallback preserves model-client
    # contract tests and produces a normal ModelError at runtime.
    httpx = _HttpxUnavailable()

from .semantic_taxonomy import (
    ATMOSPHERE_PRIMARY_TYPES,
    OBJECT_PRIMARY_TYPES,
    PLACE_PRIMARY_TYPES,
    normalize_semantic_analysis,
)


class ModelError(RuntimeError):
    pass


class ContextBudgetExceeded(ModelError):
    pass


def _http_error_detail(error, limit=2000):
    response = getattr(error, "response", None)
    if response is None:
        return str(error)
    try:
        body = response.text.strip()
    except Exception:
        body = ""
    return f"{error}: {body[:limit]}" if body else str(error)


def parse_json_response(value):
    if isinstance(value, dict):
        return value
    text = str(value or "").strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.IGNORECASE | re.DOTALL).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            try:
                return json.loads(text[start:end + 1])
            except json.JSONDecodeError:
                pass
    return {}


def as_list(value):
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def as_text(value):
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (int, float, bool)):
        return str(value)
    if isinstance(value, list):
        return "、".join(as_text(item) for item in value)
    return json.dumps(value, ensure_ascii=False)


COORDINATE_PHRASE_RE = re.compile(
    r"(?:GPS(?:坐标)?|坐标|经纬度)?\s*[+-]?\d{1,3}(?:\.\d+)?\s*[,，]\s*[+-]?\d{1,3}(?:\.\d+)?"
)


def _event_place(event, observations):
    """Choose a semantic place label; asset GPS is never a display place."""
    for item in observations:
        place = as_text(item.get("place")).strip()
        if place and not COORDINATE_PHRASE_RE.fullmatch(place):
            return place
        canonical = item.get("canonical") if isinstance(item.get("canonical"), dict) else {}
        semantic = canonical.get("semantic") if isinstance(canonical.get("semantic"), dict) else {}
        semantic_place = semantic.get("place") if isinstance(semantic.get("place"), dict) else {}
        primary = as_text(semantic_place.get("primary")).strip()
        if primary and primary != "其他或不确定":
            return primary
    place = as_text(event.get("place")).strip()
    return place if place and not COORDINATE_PHRASE_RE.fullmatch(place) else "某处"


def _strip_event_coordinates(value, place):
    text = as_text(value).strip()
    if not text:
        return ""
    text = COORDINATE_PHRASE_RE.sub(place, text)
    text = re.sub(r"(?:GPS|坐标|经纬度)(?:位置)?", "", text)
    return re.sub(r"\s{2,}", " ", text).strip()


def normalize_confidence(value, default=0.5):
    """Accept numeric, percentage, and Chinese confidence labels from local models."""
    if isinstance(value, bool):
        return default
    if isinstance(value, (int, float)):
        return max(0.0, min(1.0, float(value)))
    text = str(value or "").strip().lower()
    labels = {
        "很高": 0.95,
        "高": 0.85,
        "中高": 0.72,
        "中等": 0.6,
        "中": 0.6,
        "一般": 0.5,
        "中低": 0.42,
        "低": 0.35,
        "很低": 0.2,
    }
    if text in labels:
        return labels[text]
    try:
        number = float(text[:-1]) / 100 if text.endswith("%") else float(text)
        return max(0.0, min(1.0, number))
    except (TypeError, ValueError):
        return default


def normalize_analysis_fields(parsed):
    for key in ("caption", "activity", "place", "event_type", "transcript", "ocr_text"):
        parsed[key] = as_text(parsed.get(key))
    parsed["scene_type"] = normalize_scene_type(parsed.get("scene_type"))
    return parsed


SCENE_TYPE_OPTIONS = PLACE_PRIMARY_TYPES


def normalize_scene_type(value):
    text = as_text(value).strip()
    return text if text in SCENE_TYPE_OPTIONS else "其他或不确定"


def normalize_fact_confidences(facts, default):
    values = []
    for fact in as_list(facts):
        if isinstance(fact, dict):
            values.append({**fact, "confidence": normalize_confidence(fact.get("confidence"), default)})
    return values


def contains_latin_text(value):
    text = as_text(value)
    letters = sum(character.isascii() and character.isalpha() for character in text)
    chinese = sum("\u4e00" <= character <= "\u9fff" for character in text)
    return letters > 8 and letters > chinese


# R9-3: per-role inference parameters for the same 12B model.  Each role gets a
# distinct temperature / think / generation bound so behaviour stays testable and
# isolated.  ``writer`` / ``claim`` / ``repair`` are aliases used by the complex
# answer and repair paths.
ROLE_INFERENCE = {
    "parser": {"temperature": 0.0, "think": False, "num_ctx": 4096, "num_predict": 512},
    # Search validation is a bounded classification pass, not free-form
    # reasoning. Keep the JSON response short so one vision batch does not
    # consume the Agent wall-time budget.
    # Validation emits one short JSON row per candidate. Keep the completion
    # budget below the 12B server context ceiling; tools.py also splits a
    # batch if a deployment reports a tighter prompt budget.
    "search_validation": {"temperature": 0.0, "think": False, "num_ctx": 4096, "num_predict": 192},
    "answer": {"temperature": 0.3, "think": False, "num_ctx": 8192, "num_predict": 800},
    "writer": {"temperature": 0.3, "think": False, "num_ctx": 8192, "num_predict": 800},
    "verify": {"temperature": 0.0, "think": False, "num_ctx": 4096, "num_predict": 512},
    "claim": {"temperature": 0.0, "think": False, "num_ctx": 4096, "num_predict": 512},
    "repair": {"temperature": 0.0, "think": False, "num_ctx": 4096, "num_predict": 512},
}


def _openai_thinking_kwargs():
    """Return the vLLM chat-template switch; reasoning is disabled by default."""
    value = os.getenv("SENTRIX_ENABLE_THINKING", "0").strip().lower()
    return {"enable_thinking": value in {"1", "true", "yes", "on"}}


def build_image_prompt(metadata=None):
    prompt = """你是家庭记忆观察器。仅根据图片和元数据抽取可验证的核心观察，不猜测姓名。
严格返回简体中文 JSON 对象。place 和 semantic.place.primary 必须只依据图片视觉证据，不能用 GPS 或地点上下文覆盖；地点上下文只能作为候选背景。
地点主类只能从："""
    prompt += "、".join(PLACE_PRIMARY_TYPES)
    prompt += "；物品主类只能从："
    prompt += "、".join(OBJECT_PRIMARY_TYPES)
    prompt += "；氛围主类只能从："
    prompt += "、".join(ATMOSPHERE_PRIMARY_TYPES)
    prompt += "\nmetadata: " + json.dumps(metadata or {}, ensure_ascii=False)
    return prompt


class OllamaBackend:
    """Ollama 12B backend — local GPU via Ollama server."""

    name = "ollama_12b"

    def __init__(self, base_url, model, timeout, keep_alive):
        self._base_url = base_url.rstrip("/")
        self._model = model
        self.timeout = timeout
        self.keep_alive = keep_alive

    @property
    def endpoint(self):
        return self._base_url

    @property
    def model_name(self):
        return self._model

    def chat(self, prompt, images=None, vision_options=None, json_mode=True, role=None):
        message = {"role": "user", "content": prompt}
        if images:
            message["images"] = [img["base64"] for img in images]
        payload = {
            "model": self._model,
            "messages": [message],
            "stream": False,
            "keep_alive": self.keep_alive,
            "options": {"temperature": 0},
        }
        if json_mode:
            payload["format"] = "json"
        params = ROLE_INFERENCE.get(role)
        if params is not None:
            payload["options"].update({
                "temperature": params["temperature"],
                "num_ctx": params["num_ctx"],
                "num_predict": params["num_predict"],
            })
            payload["think"] = bool(params.get("think", False))
        if vision_options:
            payload["think"] = vision_options.get("think", False)
            payload["options"].update({
                "num_ctx": vision_options["num_ctx"],
                "num_predict": vision_options["num_predict"],
            })
        try:
            response = httpx.post(f"{self._base_url}/api/chat", json=payload, timeout=self.timeout)
            response.raise_for_status()
            data = response.json()
            return data.get("message", {}).get("content", "")
        except (httpx.HTTPError, ValueError) as error:
            raise ModelError(f"ollama request failed: {error}") from error

    def embed_text(self, text):
        """Ollama embedding endpoint."""
        if not str(text or "").strip():
            return []
        model = os.getenv("SENTRIX_TEXT_EMBED_MODEL", self._model)
        try:
            response = httpx.post(f"{self._base_url}/api/embed", json={"model": model, "input": [str(text)]}, timeout=self.timeout)
            response.raise_for_status()
            payload = response.json()
            embeddings = payload.get("embeddings") or []
            return embeddings[0] if embeddings else []
        except (httpx.HTTPError, ValueError, KeyError):
            return []


class LocalQwen3VLBackend:
    """Lazy in-process Qwen3-VL fallback for hosts without a healthy VLM API."""

    name = "qwen3-vl"
    _load_lock = threading.Lock()
    _generation_lock = threading.Lock()
    _shared = {}

    def __init__(self, model_path, device="cpu"):
        self.model_path = str(Path(model_path).expanduser())
        self.device = str(device or "cpu").strip().lower()

    @property
    def endpoint(self):
        return self.model_path

    @property
    def model_name(self):
        return Path(self.model_path).name or "Qwen3-VL"

    def _load(self):
        key = (self.model_path, self.device)
        if key in self._shared:
            return self._shared[key]
        with self._load_lock:
            if key in self._shared:
                return self._shared[key]
            if not Path(self.model_path).is_dir():
                raise ModelError(f"Qwen3-VL model path is unavailable: {self.model_path}")
            try:
                import torch
                from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

                load_options = {
                    "dtype": torch.bfloat16,
                    "low_cpu_mem_usage": True,
                }
                if self.device.startswith("cuda"):
                    gpu_index = int(self.device.split(":", 1)[1]) if ":" in self.device else 0
                    load_options.update({
                        "device_map": "auto",
                        "max_memory": {
                            gpu_index: os.getenv("SENTRIX_QWEN3_VL_GPU_MEMORY", "7GiB"),
                            "cpu": os.getenv("SENTRIX_QWEN3_VL_CPU_MEMORY", "48GiB"),
                        },
                    })
                model = Qwen3VLForConditionalGeneration.from_pretrained(
                    self.model_path, **load_options,
                )
                if not self.device.startswith("cuda"):
                    model.to(self.device)
                model.eval()
                processor = AutoProcessor.from_pretrained(self.model_path)
            except Exception as error:
                raise ModelError(f"Qwen3-VL load failed: {error}") from error
            self._shared[key] = (model, processor)
            return model, processor

    def chat(self, prompt, images=None, vision_options=None, json_mode=True, role=None):
        try:
            import torch
            from PIL import Image

            model, processor = self._load()
            content = []
            for item in images or []:
                image = Image.open(BytesIO(base64.b64decode(item["base64"]))).convert("RGB")
                content.append({"type": "image", "image": image})
            content.append({"type": "text", "text": prompt})
            messages = [{"role": "user", "content": content}]
            inputs = processor.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True,
                return_dict=True, return_tensors="pt",
            ).to(model.device)
            requested = (vision_options or {}).get("num_predict")
            if requested is None and role in ROLE_INFERENCE:
                requested = ROLE_INFERENCE[role]["num_predict"]
            max_new_tokens = min(1024, max(64, int(requested or 768)))
            with self._generation_lock, torch.inference_mode():
                generated = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
            trimmed = [output[len(source):] for source, output in zip(inputs.input_ids, generated)]
            return processor.batch_decode(trimmed, skip_special_tokens=True)[0].strip()
        except ModelError:
            raise
        except Exception as error:
            raise ModelError(f"Qwen3-VL inference failed: {error}") from error

    def embed_text(self, text):
        return []


class E2BBackend:
    """E2B LoRA backend — local 2B model via E2B server on :8100."""

    name = "e2b_lora"
    model_name = "gemma-4-e2b-it+lora-v2"

    def __init__(self, base_url=None, timeout=None):
        self._base_url = (base_url or os.getenv("E2B_BASE_URL", "http://127.0.0.1:8101")).rstrip("/")
        self.timeout = timeout or float(os.getenv("E2B_TIMEOUT_SECONDS", "300"))
        self.model_name = os.getenv("E2B_MODEL_NAME", "gemma-4-e2b-it+lora-v2")

    @property
    def endpoint(self):
        return self._base_url

    def chat(self, prompt, images=None, vision_options=None, json_mode=True, role=None):
        message = {"role": "user", "content": prompt}
        if images:
            message["images"] = [img["base64"] for img in images]
        payload = {
            "model": self.model_name,
            "messages": [message],
            "stream": False,
            "options": {"temperature": 0},
        }
        if json_mode:
            payload["format"] = "json"
        params = ROLE_INFERENCE.get(role)
        if params is not None:
            payload["options"].update({
                "temperature": params["temperature"],
                "num_ctx": params["num_ctx"],
                "num_predict": params["num_predict"],
            })
            payload["think"] = bool(params.get("think", False))
        if vision_options:
            payload["think"] = vision_options.get("think", False)
            payload["options"].update({
                "num_ctx": vision_options["num_ctx"],
                "num_predict": vision_options["num_predict"],
            })
        try:
            response = httpx.post(f"{self._base_url}/api/chat", json=payload, timeout=self.timeout)
            response.raise_for_status()
            data = response.json()
            return data.get("message", {}).get("content", "")
        except (httpx.HTTPError, ValueError) as error:
            raise ModelError(f"e2b request failed: {error}") from error

    def embed_text(self, text):
        """E2B server does not support embeddings. Always raises NotImplementedError."""
        raise NotImplementedError("E2B backend does not support text embeddings")

    def health(self):
        try:
            response = httpx.get(f"{self._base_url}/api/health", timeout=min(10, self.timeout))
            response.raise_for_status()
            return response.json()
        except Exception:
            return {}


class GammaClient:
    def __init__(self, base_url=None, model=None, timeout=None, keep_alive=None,
                 parse_model=None, answer_model=None, verify_model=None,
                 parse_backend=None, parse_base_url=None, claim_model=None,
                 repair_model=None, backend=None, api_key=None, manager_url=None,
                 runtime_source=None, api_mode=None):
        self.backend = self._normalize_backend(backend or os.getenv("SENTRIX_LLM_BACKEND", "vllm"))
        self.runtime_source = str(
            runtime_source or os.getenv("SENTRIX_RUNTIME_SOURCE", "managed")
        ).strip().lower()
        self.api_mode = str(
            api_mode or os.getenv("SENTRIX_OPENAI_API_MODE", "vllm")
        ).strip().lower()
        ollama_fallback_url = (base_url or os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")).rstrip("/")
        if self.backend == "openai":
            self._base_url_setting = self._normalize_openai_base_url(
                base_url
                or os.getenv("SENTRIX_VLLM_BASE_URL")
                or os.getenv("VLLM_BASE_URL")
                or os.getenv("OPENAI_BASE_URL")
                or "http://127.0.0.1:8100/v1"
            )
            self._model_setting = model or os.getenv("SENTRIX_VLLM_MODEL") or os.getenv("VLLM_MODEL") or os.getenv("OPENAI_MODEL") or "gemma4-12b-it"
        elif self.backend == "qwen3-vl":
            self._base_url_setting = str(
                base_url or os.getenv("SENTRIX_QWEN3_VL_MODEL_PATH", "")
            ).strip()
            self._model_setting = model or Path(self._base_url_setting).name or "Qwen3-VL"
        else:
            self._base_url_setting = (base_url or os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")).rstrip("/")
            self._model_setting = model or os.getenv("OLLAMA_MODEL", "gemma4:12b")
        self.api_key = api_key or os.getenv("SENTRIX_VLLM_API_KEY") or os.getenv("OPENAI_API_KEY") or ""
        manager_setting = (
            manager_url if manager_url is not None
            else os.getenv("SENTRIX_VLLM_MANAGER_API")
            or os.getenv("SENTRIX_VLLM_API_URL")
            or ""
        )
        self.manager_url = str(manager_setting or "").strip().rstrip("/")
        self._call_metrics_local = threading.local()
        # --- E2B facade wiring (before per-role setup) ---
        _init_timeout = timeout or float(os.getenv("OLLAMA_TIMEOUT_SECONDS", "180"))
        _init_keep_alive = keep_alive if keep_alive is not None else os.getenv("OLLAMA_KEEP_ALIVE", "0")
        self._ollama = OllamaBackend(
            base_url=ollama_fallback_url,
            model=os.getenv("OLLAMA_MODEL", "gemma4:12b"),
            timeout=_init_timeout,
            keep_alive=-1 if str(_init_keep_alive).strip() == "-1" else _init_keep_alive,
        )
        self._e2b = E2BBackend(os.getenv("E2B_BASE_URL"), _init_timeout)
        self._local_qwen = LocalQwen3VLBackend(
            self._base_url_setting,
            os.getenv("SENTRIX_QWEN3_VL_DEVICE", "cpu"),
        ) if self.backend == "qwen3-vl" else None
        self._store = None
        self._active_cache = None
        self._cache_ts = 0.0
        # Phase R R5 + R9-3: per-role model separation.  Explicit per-role env
        # wins; otherwise SENTRIX_AGENT_MODEL_PROFILE decides the default —
        # quality_12b (all roles on the 12B main endpoint) or experimental_2b
        # (2B parser via the e2b backend, answer/verify stay 12B).  Without
        # either, every role uses the main model (backward compatible).
        model_split = os.getenv("SENTRIX_MODEL_SPLIT_V1", "0").strip().lower() in {"1", "true", "yes", "on"}
        profile = os.getenv("SENTRIX_AGENT_MODEL_PROFILE", "quality_12b").strip().lower()
        explicit_any = any((parse_model, answer_model, verify_model, parse_backend,
                            os.getenv("SENTRIX_PARSE_MODEL"), os.getenv("SENTRIX_PARSE_BACKEND")))
        if self.backend == "openai":
            self.parse_model = parse_model or os.getenv("SENTRIX_PARSE_MODEL", self.model)
            self.answer_model = answer_model or os.getenv("SENTRIX_ANSWER_MODEL", self.model)
            self.verify_model = verify_model or os.getenv("SENTRIX_VERIFY_MODEL", self.model)
            self.parse_backend = parse_backend or os.getenv("SENTRIX_PARSE_BACKEND", self.backend)
            self.parse_base_url = self._normalize_openai_base_url(parse_base_url or os.getenv("SENTRIX_PARSE_BASE_URL", self.base_url))
        elif model_split or explicit_any:
            self.parse_model = parse_model or os.getenv("SENTRIX_PARSE_MODEL", self.model)
            self.answer_model = answer_model or os.getenv("SENTRIX_ANSWER_MODEL", self.model)
            self.verify_model = verify_model or os.getenv("SENTRIX_VERIFY_MODEL", self.model)
            self.parse_backend = parse_backend or os.getenv("SENTRIX_PARSE_BACKEND", "ollama_local")
            self.parse_base_url = (parse_base_url or os.getenv("SENTRIX_PARSE_BASE_URL", "")).rstrip("/")
        elif profile == "experimental_2b":
            self.parse_model = os.getenv("SENTRIX_PARSE_MODEL", "gemma-4-e2b-it+lora-v2")
            self.answer_model = os.getenv("SENTRIX_ANSWER_MODEL", self.model)
            self.verify_model = os.getenv("SENTRIX_VERIFY_MODEL", self.model)
            self.parse_backend = os.getenv("SENTRIX_PARSE_BACKEND", "e2b")
            self.parse_base_url = os.getenv("SENTRIX_PARSE_BASE_URL", "http://127.0.0.1:8100").rstrip("/")
        else:  # quality_12b (default)
            self.parse_model = self.answer_model = self.verify_model = self.model
            self.parse_backend = "ollama_local"
            self.parse_base_url = ""
        self.claim_model = claim_model or os.getenv("SENTRIX_CLAIM_MODEL", self.verify_model)
        self.repair_model = repair_model or os.getenv("SENTRIX_REPAIR_MODEL", self.parse_model)
        self.timeout = timeout or float(os.getenv("OLLAMA_TIMEOUT_SECONDS", "180"))
        configured_keep_alive = keep_alive if keep_alive is not None else os.getenv("OLLAMA_KEEP_ALIVE", "0")
        # Ollama expects numeric -1 for indefinite residency; the string "-1"
        # is rejected by its request schema.
        self.keep_alive = -1 if str(configured_keep_alive).strip() == "-1" else configured_keep_alive
        self.inference_provider = (
            OpenAICompatibleInferenceProvider(
                self._base_url_setting,
                api_key=self.api_key,
                api_mode=self.api_mode,
                manager_url=self.manager_url,
                timeout=self.timeout,
            )
            if self.backend == "openai" else None
        )

    @staticmethod
    def _normalize_backend(value):
        value = str(value or "vllm").strip().lower()
        if value in {"vllm", "openai", "openai_compatible", "openai-compatible"}:
            return "openai"
        if value in {"ollama", "ollama_local"}:
            return "ollama"
        if value in {"qwen3-vl", "qwen3_vl", "local_qwen3_vl"}:
            return "qwen3-vl"
        raise ModelError(f"unsupported llm backend: {value}")

    @staticmethod
    def _normalize_openai_base_url(value):
        return normalize_openai_base_url(value or "http://127.0.0.1:8100/v1")

    def _inference_for(self, endpoint_base):
        endpoint_base = normalize_openai_base_url(endpoint_base)
        if self.inference_provider and self.inference_provider.base_url == endpoint_base:
            return self.inference_provider
        return OpenAICompatibleInferenceProvider(
            endpoint_base,
            api_key=self.api_key,
            api_mode=self.api_mode,
            manager_url=self.manager_url,
            timeout=self.timeout,
        )

    _CACHE_TTL_SECONDS = 5.0

    def bind_store(self, store):
        self._store = store

    def invalidate_backend_cache(self):
        self._active_cache = None
        self._cache_ts = 0.0

    def _read_active_name(self):
        if self._store is not None:
            name = self._store.get_setting("vlm_backend")
            if name in ("ollama_12b", "e2b_lora"):
                return name
        return "ollama_12b"

    def _active(self):
        if self.backend == "qwen3-vl":
            return self._local_qwen
        import time
        now = time.monotonic()
        if self._active_cache is not None and (now - self._cache_ts) < self._CACHE_TTL_SECONDS:
            return self._active_cache
        name = self._read_active_name()
        if name == "e2b_lora":
            self._active_cache = self._e2b
        else:
            self._active_cache = getattr(self, '_ollama', None)
        self._cache_ts = now
        return self._active_cache

    @property
    def active_name(self):
        return self._read_active_name()

    @property
    def base_url(self):
        if self.backend == "openai":
            return self._base_url_setting
        return self._active().endpoint

    @property
    def model(self):
        if self.backend == "openai":
            return self._model_setting
        return self._active().model_name

    def _endpoint_for(self, role):
        """Resolve (base_url, model) for a model role.

        The parser role can be split to a separate backend (153 e2b 2B, D6).
        When ``parse_backend=e2b`` but no e2b base_url is configured, we fall
        back to the main endpoint rather than hard-failing — the 153 wiring
        sets SENTRIX_PARSE_BASE_URL.
        """
        if role == "parser" and self.parse_backend in {"e2b", "e2b_lora"}:
            if self.parse_base_url:
                return self.parse_base_url, self.parse_model
            return self.base_url, self.parse_model
        if role == "parser":
            return self.base_url, self.parse_model
        if role == "answer":
            return self.base_url, self.answer_model
        if role == "verify":
            return self.base_url, self.verify_model
        if role == "claim":
            return self.base_url, self.claim_model
        if role == "repair":
            return self.base_url, self.repair_model
        return self.base_url, self.model

    def chat(self, prompt, images=None, vision_options=None, json_mode=True, role=None):
        if httpx is None:
            raise ModelError("httpx is not installed")
        if self.backend == "openai":
            endpoint_base, model = self._endpoint_for(role)
            return self._chat_openai(endpoint_base, model, prompt, images, vision_options, json_mode, role)
        backend = self._active()
        if backend is None:
            raise ModelError("no active VLM backend")
        text = backend.chat(prompt, images, vision_options, json_mode, role)
        self._record_validation_call(role, backend.endpoint, backend.model_name, json_mode, text)
        return text

    def chat_messages(self, messages, *, role=None, temperature=0.0, max_tokens=None):
        """Stream a complete OpenAI messages array and record per-call metrics.

        AgentRuntime owns a multi-message conversation, so routing it through
        ``chat(prompt)`` would collapse the role structure.  This entry point
        preserves the original messages while reusing the same TTFT/usage
        instrumentation as normal GammaClient calls.
        """
        if httpx is None:
            raise ModelError("httpx is not installed")
        if not isinstance(messages, list) or not messages:
            raise ModelError("messages must be a non-empty list")
        if self.backend != "openai":
            prompt = "\n\n".join(
                f"[{str(message.get('role') or 'user')}] {message.get('content') or ''}"
                for message in messages if isinstance(message, dict)
            )
            return self.chat(prompt, json_mode=False, role=role)

        endpoint_base, model = self._endpoint_for(role)
        requested_max_tokens = int(max_tokens) if max_tokens is not None else None
        is_cloud_api = self.runtime_source == "cloud_api"
        budget = None if is_cloud_api else self._tokenize_for_budget(endpoint_base, messages)
        if is_cloud_api:
            budget_source = "provider_managed"
            preflight_status = "not_requested"
            preflight_reason = "cloud_api_context_managed_by_provider"
        else:
            budget_source = str((budget or {}).get("token_count_source") or "vllm_tokenize")
            preflight_status = str((budget or {}).get("preflight_status") or "ok")
            preflight_reason = str((budget or {}).get("preflight_fallback_reason") or "")
        if budget:
            prompt_tokens = int(budget["prompt_tokens"])
            max_model_len = int(budget["max_model_len"])
            available_output_tokens = max_model_len - prompt_tokens
            if available_output_tokens < 1:
                metrics = {
                    "status": "context_budget_exceeded",
                    "error": "prompt leaves no room for generation",
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": None,
                    "requested_max_tokens": requested_max_tokens,
                    "effective_max_tokens": 0,
                    "available_output_tokens": max(0, available_output_tokens),
                    "max_model_len": max_model_len,
                    "estimated_total_tokens": prompt_tokens + (requested_max_tokens or 0),
                    "token_count_source": budget_source,
                    "preflight_status": preflight_status,
                    "preflight_fallback_reason": preflight_reason,
                    "ttft_ms": None,
                    "total_ms": None,
                    "tokens_per_second": None,
                    "streamed": False,
                }
                self._record_call_metrics(role, model, endpoint_base, metrics)
                raise ContextBudgetExceeded(
                    f"context budget exceeded: prompt_tokens={prompt_tokens}, "
                    f"max_model_len={max_model_len}")
            if requested_max_tokens is None:
                max_tokens = available_output_tokens
            else:
                max_tokens = min(requested_max_tokens, available_output_tokens)

        payload = {
            "model": model,
            "messages": messages,
            "stream": True,
            "stream_options": {"include_usage": True},
            "temperature": temperature,
        }
        if self.api_mode != "generic":
            payload["chat_template_kwargs"] = _openai_thinking_kwargs()
        if max_tokens is not None:
            payload["max_tokens"] = int(max_tokens)
        headers = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        request_started = time.perf_counter()
        try:
            return self._chat_openai_stream(
                endpoint_base, payload, headers, role, model, json_mode=False,
                budget_metrics={
                    "requested_max_tokens": requested_max_tokens,
                    "effective_max_tokens": int(max_tokens) if max_tokens is not None else None,
                    "available_output_tokens": (
                        int(budget["max_model_len"]) - int(budget["prompt_tokens"])
                        if budget else None
                    ),
                    "max_model_len": int(budget["max_model_len"]) if budget else None,
                    "preflight_prompt_tokens": int(budget["prompt_tokens"]) if budget else None,
                    "estimated_total_tokens": (
                        int(budget["prompt_tokens"]) + int(max_tokens)
                        if budget and max_tokens is not None else None
                    ),
                    "token_count_source": budget_source if (budget or is_cloud_api) else "response_usage",
                    "preflight_status": preflight_status if (budget or is_cloud_api) else "not_configured",
                    "preflight_fallback_reason": preflight_reason,
                })
        except (httpx.HTTPError, ValueError, KeyError, IndexError, TypeError) as error:
            error_detail = _http_error_detail(error)
            self._record_call_metrics(role, model, endpoint_base, {
                "status": "error",
                "error": error_detail,
                "ttft_ms": None,
                "total_ms": round((time.perf_counter() - request_started) * 1000, 1),
                "prompt_tokens": int(budget["prompt_tokens"]) if budget else None,
                "completion_tokens": None,
                "tokens_per_second": None,
                "streamed": True,
                "requested_max_tokens": requested_max_tokens,
                "effective_max_tokens": int(max_tokens) if max_tokens is not None else None,
                "max_model_len": int(budget["max_model_len"]) if budget else None,
                "token_count_source": budget_source if (budget or is_cloud_api) else None,
                "preflight_status": preflight_status if (budget or is_cloud_api) else "not_configured",
                "preflight_fallback_reason": preflight_reason,
            })
            raise ModelError(f"model request failed: {error_detail}") from error

    def _tokenize_for_budget(self, endpoint_base, messages):
        """Ask the Manager bound to this endpoint to tokenize with the active model."""
        provider = self._inference_for(endpoint_base)
        if not provider.manager_url:
            return None
        last_error = None
        for attempt in range(2):
            try:
                value = provider.token_count(messages, timeout=min(15, self.timeout))
                if value is None:
                    return None
                value["token_count_source"] = "vllm_tokenize"
                value["preflight_status"] = "ok"
                return value
            except (httpx.HTTPError, ValueError, TypeError) as error:
                last_error = error
                response = getattr(error, "response", None)
                status = int(getattr(response, "status_code", 0) or 0)
                transient = status >= 500 or status == 0
                if transient and attempt == 0:
                    time.sleep(0.1)
                    continue
                if transient:
                    return self._local_token_budget(messages, reason=_http_error_detail(error))
                if os.getenv("SENTRIX_TOKEN_BUDGET_REQUIRED", "1").strip().lower() in {"1", "true", "yes", "on"}:
                    raise ModelError(f"token budget preflight failed: {error}") from error
                return None
        return self._local_token_budget(messages, reason=str(last_error or "manager_unavailable"))

    @staticmethod
    def _local_token_budget(messages, *, reason: str = "manager_unavailable"):
        """Conservative fallback when the manager tokenizer is unavailable."""
        text = json.dumps(messages or [], ensure_ascii=False, separators=(",", ":"))
        chinese = sum(1 for char in text if "\u4e00" <= char <= "\u9fff")
        estimate = max(1, int(chinese * 0.7 + (len(text) - chinese) * 0.25) + 400)
        max_model_len = int(
            os.getenv("SENTRIX_TOKEN_BUDGET_MAX_MODEL_LEN")
            or os.getenv("SENTRIX_MAX_MODEL_LEN")
            or os.getenv("VLLM_MAX_MODEL_LEN")
            or "4501"
        )
        return {
            "prompt_tokens": estimate,
            "max_model_len": max_model_len,
            "token_count_source": "local_estimate",
            "preflight_status": "fallback",
            "preflight_fallback_reason": reason[:500],
        }

    def _chat_ollama(self, endpoint_base, model, prompt, images=None, vision_options=None, json_mode=True, role=None):
        message = {"role": "user", "content": prompt}
        if images:
            message["images"] = [image["base64"] for image in images]
        payload = {
            "model": model,
            "messages": [message],
            "stream": False,
            "keep_alive": self.keep_alive,
            "options": {"temperature": 0},
        }
        if json_mode:
            payload["format"] = "json"
        # R9-3: apply per-role inference parameters (temperature / context /
        # generation bound / think).  role=None keeps the legacy bare options.
        params = ROLE_INFERENCE.get(role)
        if params is not None:
            payload["options"].update({
                "temperature": params["temperature"],
                "num_ctx": params["num_ctx"],
                "num_predict": params["num_predict"],
            })
            payload["think"] = bool(params.get("think", False))
        if vision_options:
            payload["think"] = vision_options.get("think", False)
            payload["options"].update({
                "num_ctx": vision_options["num_ctx"],
                "num_predict": vision_options["num_predict"],
            })
        try:
            response = httpx.post(f"{endpoint_base}/api/chat", json=payload, timeout=self.timeout)
            response.raise_for_status()
            data = response.json()
            text = data.get("message", {}).get("content", "")
            self._record_validation_call(role, endpoint_base, model, json_mode, text)
            return text
        except (httpx.HTTPError, ValueError) as error:
            raise ModelError(f"model request failed: {_http_error_detail(error)}") from error


    def _record_call_metrics(self, role, model, endpoint, metrics):
        """Record per-call LLM metrics (thread-local)."""
        if not hasattr(self._call_metrics_local, "calls"):
            self._call_metrics_local.calls = []
        entry = {"role": role or "unknown", "model": model, "endpoint": endpoint, **metrics}
        self._call_metrics_local.calls.append(entry)

    def get_and_clear_call_metrics(self):
        """Retrieve and clear accumulated per-call metrics for this thread."""
        calls = getattr(self._call_metrics_local, "calls", [])
        self._call_metrics_local.calls = []
        return calls

    def _chat_openai(self, endpoint_base, model, prompt, images=None, vision_options=None, json_mode=True, role=None):
        if images:
            content = [{"type": "text", "text": prompt}]
            for image in images:
                mime_type = image.get("mime_type") or "image/jpeg"
                content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:{mime_type};base64,{image['base64']}"},
                })
            message = {"role": "user", "content": content}
        else:
            message = {"role": "user", "content": prompt}
        use_stream = not json_mode  # only stream non-JSON calls (final answer gen)
        payload = {
            "model": model,
            "messages": [message],
            "stream": use_stream,
            "stream_options": {"include_usage": True} if use_stream else None,
            "temperature": 0,
        }
        if self.api_mode != "generic":
            payload["chat_template_kwargs"] = _openai_thinking_kwargs()
        if json_mode and os.getenv("SENTRIX_OPENAI_RESPONSE_FORMAT", "1").strip().lower() in {"1", "true", "yes", "on"}:
            payload["response_format"] = {"type": "json_object"}
        params = ROLE_INFERENCE.get(role)
        if params is not None:
            payload["temperature"] = params["temperature"]
            payload["max_tokens"] = params["num_predict"]
        if vision_options:
            payload["max_tokens"] = int(vision_options["num_predict"])
        headers = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        request_started = time.perf_counter()
        try:
            if use_stream:
                return self._chat_openai_stream(endpoint_base, payload, headers, role, model, json_mode)
            response = self._inference_for(endpoint_base).chat(payload, timeout=self.timeout)
            data = response.json()
            text = (((data.get("choices") or [{}])[0].get("message") or {}).get("content") or "")
            if isinstance(text, list):
                text = "".join(part.get("text", "") if isinstance(part, dict) else str(part) for part in text)
            usage = data.get("usage") or {}
            self._record_call_metrics(role, model, endpoint_base, {
                "ttft_ms": None, "total_ms": None,
                "prompt_tokens": usage.get("prompt_tokens"),
                "completion_tokens": usage.get("completion_tokens"),
                "tokens_per_second": None, "streamed": False,
            })
            self._record_validation_call(role, endpoint_base, model, json_mode, text)
            return text
        except (httpx.HTTPError, ValueError, KeyError, IndexError, TypeError) as error:
            error_detail = _http_error_detail(error)
            self._record_call_metrics(role, model, endpoint_base, {
                "status": "error",
                "error": error_detail,
                "ttft_ms": None,
                "total_ms": round((time.perf_counter() - request_started) * 1000, 1),
                "prompt_tokens": None,
                "completion_tokens": None,
                "tokens_per_second": None,
                "streamed": use_stream,
            })
            raise ModelError(f"model request failed: {error_detail}") from error

    def _chat_openai_stream(self, endpoint_base, payload, headers, role, model, json_mode=False,
                            budget_metrics=None):
        """Streaming variant: buffer chunks, capture TTFT/tokens/throughput."""
        t0 = time.perf_counter()
        first_token_t = None
        chunks = []
        prompt_tokens = None
        completion_tokens = None
        with self._inference_for(endpoint_base).chat_stream(payload, timeout=self.timeout) as response:
            if response.status_code >= 400:
                try:
                    import sys as _sys
                    _body = response.read().decode("utf-8", errors="replace")[:2000]
                    print(f"[gamma] HTTP {response.status_code} body: {_body}", file=_sys.stderr)
                except Exception:
                    pass
            response.raise_for_status()
            for line in response.iter_lines():
                if not line or not line.startswith("data: "):
                    continue
                data_str = line[6:]
                if data_str.strip() == "[DONE]":
                    break
                try:
                    chunk = json.loads(data_str)
                except (json.JSONDecodeError, ValueError):
                    continue
                delta = ((chunk.get("choices") or [{}])[0].get("delta") or {})
                text_part = delta.get("content") or ""
                if text_part:
                    if first_token_t is None:
                        first_token_t = time.perf_counter()
                    chunks.append(text_part)
                usage = chunk.get("usage")
                if usage:
                    prompt_tokens = usage.get("prompt_tokens", prompt_tokens)
                    completion_tokens = usage.get("completion_tokens", completion_tokens)
        total_t = time.perf_counter()
        text = "".join(chunks)
        ttft_ms = round((first_token_t - t0) * 1000, 1) if first_token_t else None
        total_ms = round((total_t - t0) * 1000, 1)
        gen_ms = round((total_t - (first_token_t or t0)) * 1000, 1) if first_token_t else None
        tps = round(completion_tokens / (gen_ms / 1000), 1) if completion_tokens and gen_ms and gen_ms > 0 else None
        self._record_call_metrics(role, model, endpoint_base, {
            "ttft_ms": ttft_ms, "total_ms": total_ms,
            "prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens,
            "tokens_per_second": tps, "streamed": True,
            **(budget_metrics or {}),
        })
        self._record_validation_call(role, endpoint_base, model, json_mode, text)
        return text

    def _record_validation_call(self, role, endpoint_base, model, json_mode, text):
        """Write the actual model into the ModelCallLedger (12B-FC V2).

        Handles both ModelRouter-mediated calls (a record already exists) and
        direct gamma.chat calls (writer/claim/verify) by creating the record.
        """
        from .validation import full_chain_profile as _prof
        from .validation import model_call_ledger as _ledger
        if not (_prof.validation_active() and _prof.require_model_trace()):
            return
        expected = {
            "parser": self.parse_model, "answer": self.answer_model,
            "verify": self.verify_model, "claim": self.claim_model,
            "repair": self.repair_model,
        }.get(role, self.model)
        record = _ledger.active_record()
        if record is None:
            record = _ledger.new_call(role or "unknown", expected, endpoint_base)
            record["input_size"] = 0
        _ledger.record_response(text, actual_model=model, endpoint=endpoint_base,
                                json_mode=json_mode)

    def _core_vision_options(self):
        return {
            "think": False,
            "num_ctx": int(os.getenv("VISION_CORE_NUM_CTX", "4096")),
            # The full observation contract contains caption, people, objects,
            # clothing, relations and detail arrays.  320 tokens truncates this
            # JSON before it can be parsed, silently producing an empty memory.
            "num_predict": int(os.getenv("VISION_CORE_NUM_PREDICT", "800")),
        }

    def encode_vision_image(self, path):
        """Downsample only the model input; the source asset remains untouched."""
        file_path = Path(path)
        max_dimension = int(os.getenv("VISION_CORE_MAX_DIMENSION", "896"))
        try:
            from .image_io import ensure_heif_support, guess_mime_type
            from PIL import Image

            ensure_heif_support()
            with Image.open(file_path) as source:
                image = source.convert("RGB")
                image.thumbnail((max_dimension, max_dimension))
                output = BytesIO()
                image.save(output, format="JPEG", quality=90, optimize=True)
                return base64.b64encode(output.getvalue()).decode("ascii"), "image/jpeg"
        except Exception:
            encoded = base64.b64encode(file_path.read_bytes()).decode("ascii")
            mime_type = guess_mime_type(file_path)
            return encoded, mime_type

    def _encode_core_image(self, path):
        return self.encode_vision_image(path)

    def analyze_image(self, path, metadata=None):
        file_path = Path(path)
        encoded, mime_type = self._encode_core_image(file_path)
        prompt = """你是家庭记忆观察器。仅根据图片和元数据抽取可验证的核心观察，不猜测姓名。
严格返回简体中文 JSON 对象，不要解释。caption、activity、place、event_type 是必须同时输出的自然语言观察字段；即使能够选择 semantic，也不能只输出 semantic 选择。画面能判断时不要留空，caption 不超过160字；activity、place、event_type 各不超过40字；people、objects、clothing、emotions、spatial_relations 尽量完整记录（分别最多12、40、12、12、40项），每项可包含不超过80字的可见细节；facts 最多8项；ocr_text 不超过1000字；确实看不清才用空数组或空字符串。
字段固定为：caption、activity、place、scene_type、semantic、people、objects、clothing、emotions、spatial_relations、ocr_text、event_type、facts、detail。detail 用于保存不应被短摘要丢弃的可验证细节，包含 visible_details、regions、text_blocks、uncertainties 四个数组，每项写清可见内容和 confidence，不要猜测。semantic.place.primary 只能选择地点主类，details 从图片可观察的地点细节中多选；semantic.objects 是物品记录数组，每项包含 primary、label、details；semantic.atmosphere.labels 和 details 都是可观察画面氛围的多选值，不描述人物心理。
地点主类只能从："""
        prompt += "、".join(PLACE_PRIMARY_TYPES)
        prompt += "；物品主类只能从："
        prompt += "、".join(OBJECT_PRIMARY_TYPES)
        prompt += "；氛围主类只能从："
        prompt += "、".join(ATMOSPHERE_PRIMARY_TYPES)
        prompt += "。facts 项仅含 subject、predicate、object、confidence。\n不要把来源成员当成画面人物，也不要推测拍摄者姓名；source_owner 只作为事件来源候选。\nmetadata: "
        prompt += json.dumps(metadata or {}, ensure_ascii=False)
        parsed = parse_json_response(self.chat(prompt, [{"base64": encoded, "mime_type": mime_type}], self._core_vision_options()))
        if not any(str(parsed.get(key) or "").strip() for key in ("caption", "activity", "place", "event_type", "ocr_text")) and not parsed.get("people") and not parsed.get("objects"):
            recovery_prompt = """首轮图片结果只有分类或为空，请补齐可验证的自然语言观察。只根据图片，不猜测姓名，不输出坐标。
严格返回简体中文 JSON：caption（图片中看到什么，160字内）、activity（正在发生什么，40字内）、place（语义地点描述，如家中客厅/餐厅/公园，不要GPS，40字内）、event_type（40字内）、people（最多12项）、objects（最多40项）、ocr_text（1000字内）、detail（visible_details/regions/text_blocks/uncertainties）。画面确实看不清才留空；不要只返回分类字段。"""
            recovered = parse_json_response(self.chat(recovery_prompt, [{"base64": encoded, "mime_type": mime_type}], self._core_vision_options()))
            for key in ("caption", "activity", "place", "event_type", "people", "objects", "ocr_text"):
                if recovered.get(key) not in (None, "", []):
                    parsed[key] = recovered[key]
        scalar_text = " ".join(as_text(parsed.get(key)) for key in ("caption", "activity", "place", "event_type", "ocr_text"))
        if contains_latin_text(scalar_text):
            canonical_prompt = "把下面的家庭图片观察规范化为简体中文 JSON。只翻译和整理已有内容，不新增人物、物体、活动或事实，不猜测姓名。保留字段 caption、activity、place、scene_type、semantic、people、objects、clothing、spatial_relations、ocr_text、event_type、facts、detail。detail 必须保留 visible_details、regions、text_blocks、uncertainties。semantic 必须保留地点主类、地点细节、物品记录和可观察画面氛围。scene_type 必须保留为下列之一："
            canonical_prompt += "、".join(SCENE_TYPE_OPTIONS)
            canonical_prompt += "。\n原始观察：" + json.dumps(parsed, ensure_ascii=False)
            parsed = parse_json_response(self.chat(canonical_prompt))
        parsed["people"] = as_list(parsed.get("people"))
        parsed["objects"] = as_list(parsed.get("objects"))
        parsed["clothing"] = as_list(parsed.get("clothing"))
        parsed["emotions"] = as_list(parsed.get("emotions"))
        parsed["spatial_relations"] = as_list(parsed.get("spatial_relations"))
        parsed["facts"] = normalize_fact_confidences(parsed.get("facts"), 0.65)
        detail = parsed.get("detail") if isinstance(parsed.get("detail"), dict) else {}
        parsed["detail"] = {
            "schema_version": 1,
            "visible_details": as_list(detail.get("visible_details")),
            "regions": as_list(detail.get("regions")),
            "text_blocks": as_list(detail.get("text_blocks")),
            "uncertainties": as_list(detail.get("uncertainties")),
            **{key: value for key, value in detail.items()
               if key not in {"visible_details", "regions", "text_blocks", "uncertainties"}},
        }
        normalize_analysis_fields(parsed)
        parsed = normalize_semantic_analysis(parsed)
        parsed["confidence"] = normalize_confidence(parsed.get("confidence"), 0.65)
        parsed["model"] = self.model
        return parsed

    def analyze_video_event(self, paths, metadata=None, yolo_semantics=None):
        """Describe one ordered video event from transient evidence images."""
        images = []
        for path in list(paths or [])[:5]:
            encoded, mime_type = self._encode_core_image(Path(path))
            images.append({"base64": encoded, "mime_type": mime_type})
        if not images:
            raise ValueError("video event analysis requires at least one evidence image")
        prompt = """你是家庭视频事件观察器。输入是同一连续事件中按时间顺序排列的3至5张临时证据图。
综合全部图片和YOLO时间序列语义，描述事件期间可验证的人物、物品、环境与活动变化；不能只描述第一张或最后一张，不能猜测姓名或关系。忽略单纯的站立、坐着、抬手等低信息动作，除非它们对事件变化不可缺少。
caption 和 activity 必须由选中的证据图片直接支持，不得描述已经离开画面的活动。返回 representative_indices：能够覆盖 caption、activity 和事件中不同阶段的最小图片序号集合，从0开始，最多3张。单一活动或相似画面只能选1张；只有出现不同地点、不同活动阶段且单图无法覆盖时才选2至3张，例如“泳池环境”和“烧烤操作”应各选一张。禁止选择重复画面。
严格返回简体中文 JSON：caption（160字内）、activity（60字内）、place（40字内）、scene_type、semantic、people（最多20项）、objects（最多60项）、clothing（最多20项）、emotions（最多20项）、spatial_relations（最多60项）、ocr_text（1000字内）、event_type、facts（最多12项）、detail（visible_details/regions/text_blocks/uncertainties）、representative_indices（整数数组，1至3项）。
图片顺序和事件上下文：""" + json.dumps({
            "metadata": metadata or {}, "yolo_timeline": yolo_semantics or {},
        }, ensure_ascii=False)
        parsed = parse_json_response(self.chat(prompt, images, self._core_vision_options()))
        parsed["people"] = as_list(parsed.get("people"))
        parsed["objects"] = as_list(parsed.get("objects"))
        parsed["clothing"] = as_list(parsed.get("clothing"))
        parsed["emotions"] = as_list(parsed.get("emotions"))
        parsed["spatial_relations"] = as_list(parsed.get("spatial_relations"))
        parsed["facts"] = normalize_fact_confidences(parsed.get("facts"), 0.65)
        detail = parsed.get("detail") if isinstance(parsed.get("detail"), dict) else {}
        parsed["detail"] = {
            "schema_version": 1,
            "visible_details": as_list(detail.get("visible_details")),
            "regions": as_list(detail.get("regions")),
            "text_blocks": as_list(detail.get("text_blocks")),
            "uncertainties": as_list(detail.get("uncertainties")),
            **{key: value for key, value in detail.items()
               if key not in {"visible_details", "regions", "text_blocks", "uncertainties"}},
        }
        normalize_analysis_fields(parsed)
        parsed = normalize_semantic_analysis(parsed)
        raw_indices = parsed.get("representative_indices")
        if not isinstance(raw_indices, list):
            raw_indices = [parsed.get("representative_index", 0)]
        representative_indices = []
        for value in raw_indices:
            try:
                index = max(0, min(len(images) - 1, int(value)))
            except (TypeError, ValueError):
                continue
            if index not in representative_indices:
                representative_indices.append(index)
        parsed["representative_indices"] = (representative_indices or [0])[:3]
        parsed["confidence"] = normalize_confidence(parsed.get("confidence"), 0.65)
        parsed["model"] = self.model
        parsed["video_event_evidence_count"] = len(images)
        return parsed

    def analyze_image_focus(self, path, dimension, metadata=None):
        file_path = Path(path)
        encoded = base64.b64encode(file_path.read_bytes()).decode("ascii")
        mime_type = mimetypes.guess_type(file_path.name)[0] or "image/jpeg"
        fields = {
            "clothing": "clothing（衣物、颜色、款式和配饰数组）",
            "object": "objects（与问题相关的物体、颜色、形状和状态数组）",
            "spatial_relation": "spatial_relations（人物、物体和空间位置关系数组）",
        }.get(dimension, "visual_details（与问题相关的可验证视觉细节数组）")
        prompt = f"""你是家庭记忆视觉补全器。只根据图片和元数据分析 {dimension}，不要猜测姓名和关系。
严格只返回 JSON，所有值使用简体中文，字段为：{fields}、confidence。
没有看清或无法确认时返回空数组，不要编造。
metadata: {json.dumps(metadata or {}, ensure_ascii=False)}"""
        parsed = parse_json_response(self.chat(prompt, [{"base64": encoded, "mime_type": mime_type}]))
        parsed["clothing"] = as_list(parsed.get("clothing"))
        parsed["objects"] = as_list(parsed.get("objects"))
        parsed["spatial_relations"] = as_list(parsed.get("spatial_relations"))
        parsed["confidence"] = normalize_confidence(parsed.get("confidence"), 0.55)
        parsed["model"] = self.model
        return parsed

    def write_person_portrait(self, pack, role="writer"):
        """Generate a hedged, evidence-bound living portrait from a bounded pack."""
        from .person_portraits import PERSON_PORTRAIT_PROMPT, normalize_writer_output

        prompt = PERSON_PORTRAIT_PROMPT + "\n证据包：" + json.dumps(pack, ensure_ascii=False)
        parsed = parse_json_response(self.chat(prompt, json_mode=True, role=role))
        return normalize_writer_output(parsed)

    def infer_person_graph(self, paths, graph_payload, role="verify"):
        """Infer album owner, roles and relationships from anonymized person refs."""
        from .person_graph import PERSON_GRAPH_PROMPT, normalize_person_graph

        images = []
        for path in list(paths or [])[:12]:
            encoded, mime_type = self.encode_vision_image(Path(path))
            images.append({"base64": encoded, "mime_type": mime_type})
        prompt = PERSON_GRAPH_PROMPT + "\n匿名人物与证据：" + json.dumps(
            graph_payload or {}, ensure_ascii=False
        )
        parsed = parse_json_response(self.chat(
            prompt, images, self._core_vision_options(), role=role,
        ))
        people = list(graph_payload.get("people") or []) if isinstance(graph_payload, dict) else []
        return normalize_person_graph(parsed, people)

    def analyze_person_moments(self, path, labels, context=None):
        """Extract evidence-bound person moments from a numbered preview image."""
        from .person_moments import PERSON_MOMENT_PROMPT, normalize_person_moments

        encoded, mime_type = self.encode_vision_image(path)
        prompt = PERSON_MOMENT_PROMPT
        if context:
            prompt += "\n图片上下文：" + json.dumps(context, ensure_ascii=False)
        parsed = parse_json_response(self.chat(
            prompt,
            [{"base64": encoded, "mime_type": mime_type}],
            vision_options=self._core_vision_options(),
            role="verify",
        ))
        return {"moments": normalize_person_moments(parsed, labels)}

    def analyze_person_appearance(self, path, metadata=None):
        """Extract clothing only for the person represented by a body crop."""
        file_path = Path(path)
        encoded = base64.b64encode(file_path.read_bytes()).decode("ascii")
        mime_type = mimetypes.guess_type(file_path.name)[0] or "image/jpeg"
        prompt = """你是家庭记忆人物外观观察器。输入图片是从一个已检测人脸向下扩展得到的同一目标人物裁剪。
只描述这个目标人物能够明确看见的衣物、颜色、款式和配饰；不得描述背景或其他人物，不得猜测姓名、性别、关系或看不清的细节。
严格只返回简体中文 JSON：clothing（数组）、confidence。无法可靠归属给目标人物时 clothing 返回空数组。
目标人物裁剪元数据：""" + json.dumps(metadata or {}, ensure_ascii=False)
        parsed = parse_json_response(self.chat(prompt, [{"base64": encoded, "mime_type": mime_type}]))
        clothing = []
        for item in as_list(parsed.get("clothing")):
            value = as_text(item).strip()
            if value:
                clothing.append(value)
        return {
            "clothing": list(dict.fromkeys(clothing)),
            "confidence": normalize_confidence(parsed.get("confidence"), 0.55),
            "model": self.model,
        }

    def analyze_text(self, text, source_type="text"):
        prompt = """从下面的家庭文本或音频转写中抽取事件观察和可维护事实。只返回 JSON，不要添加 Markdown。
字段：caption、activity、place、people（数组）、objects（数组）、event_type、facts（数组）。
facts 每项为 subject、predicate、object、confidence；没有明确证据时 facts 返回空数组。
文本：""" + text
        parsed = parse_json_response(self.chat(prompt))
        parsed["people"] = as_list(parsed.get("people"))
        parsed["objects"] = as_list(parsed.get("objects"))
        parsed["facts"] = normalize_fact_confidences(parsed.get("facts"), 0.6)
        normalize_analysis_fields(parsed)
        parsed["source_type"] = source_type
        parsed["confidence"] = normalize_confidence(parsed.get("confidence"), 0.6)
        parsed["model"] = self.model
        return parsed

    def _event_gps_prefix(self, event, observations):
        """Extract GPS-derived location from observation assets via the store."""
        observation_ids = [item.get("id") for item in observations if item.get("id")]
        if not observation_ids or not self._store:
            return ""
        placeholders = ",".join("?" for _ in observation_ids)
        try:
            rows = self._store._rows(
                f"""SELECT DISTINCT json_extract(a.metadata_json, '$.reverse_geocode') as geo
                    FROM observations o JOIN assets a ON a.id = o.asset_id
                    WHERE o.id IN ({placeholders})""",
                observation_ids,
            )
        except Exception:
            return ""
        prefixes = []
        for row in rows:
            try:
                geo = json.loads(row["geo"] or "{}")
            except (TypeError, json.JSONDecodeError):
                continue
            prefix = format_gps_prefix(geo)
            if prefix and prefix not in prefixes:
                prefixes.append(prefix)
        return prefixes[0] if prefixes else ""

    def summarize_event(self, event, observations):
        semantic_place = _event_place(event, observations)
        # Extract GPS location BEFORE model call as factual context (not model input)
        gps_prefix = self._event_gps_prefix(event, observations)
        evidence = [{
            "observation_id": item.get("id"),
            "caption": item.get("caption"),
            "activity": item.get("activity"),
            "place": item.get("place"),
            "semantic": (item.get("canonical") or {}).get("semantic", {}) if isinstance(item.get("canonical"), dict) else {},
            "people": item.get("people", []),
            "objects": item.get("objects", []),
            "ocr_text": item.get("ocr_text"),
            "clothing": item.get("clothing", []),
            "spatial_relations": item.get("spatial_relations", []),
        } for item in observations]
        prompt = """你是家庭事件总结器。下面是一组已经按拍摄时间和地点聚类的图片观察。
只能使用给定观察，不得把元数据地点以外的信息当作事实，不得猜测未确认人物姓名；如果观察彼此不足以支持具体事件，使用保守、描述性的标题。
地点必须使用观察中的语义地点（例如餐厅、家中厨房、湖边、商场或语义主类），禁止输出 GPS 坐标、经纬度、文件名或路径。事件总结必须综合全部 observations，不能只依据其中一张图片。
严格返回简体中文 JSON：title（不超过20字）、event_type、activity、summary（包含时间范围、语义地点和可验证活动）、confidence。
事件：""" + json.dumps({
            "time_start": event.get("time_start"), "time_end": event.get("time_end"), "place": semantic_place, "observations": evidence,
        }, ensure_ascii=False)
        parsed = parse_json_response(self.chat(prompt))
        fallback_place = semantic_place
        title = _strip_event_coordinates(as_text(parsed.get("title")) or "家庭图片记录", fallback_place)
        summary = _strip_event_coordinates(as_text(parsed.get("summary")) or "该事件的图片证据尚不足以生成更具体的总结。", fallback_place)
        # Inject GPS location as factual prefix into summary (post-model, never in prompt)
        if gps_prefix:
            summary = f"在{gps_prefix}，{summary}"
            if not any(gps_prefix in t for t in [title, fallback_place]):
                pass  # Title stays visual to avoid model confusion
        return {
            "title": title,
            "event_type": _strip_event_coordinates(as_text(parsed.get("event_type")) or "家庭记录", fallback_place),
            "activity": _strip_event_coordinates(as_text(parsed.get("activity")) or "家庭活动", fallback_place),
            "summary": summary,
            "confidence": normalize_confidence(parsed.get("confidence"), 0.5),
            "model": self.model,
        }

    def embed_text(self, text):
        """Embedding is hard-pinned to Ollama. E2B does not support embeddings."""
        if getattr(self, '_ollama', None) is not None:
            return self._ollama.embed_text(text)
        return []

    def answer(self, query, context):
        prompt = f"""你是 Sentrix 家庭记忆 Agent。
问题：{query}
证据上下文：
{context}

只能使用上下文回答。请严格返回 JSON：{{"answer":"...","confidence":0.0,"evidence":[{{"id":"...","summary":"..."}}],"insufficient_evidence":false}}。
没有足够证据时 answer 必须说明证据不足，confidence 不得高于 0.35，insufficient_evidence 必须为 true。不要编造证据 ID。"""
        parsed = parse_json_response(self.chat(prompt))
        parsed.setdefault("answer", "证据不足，暂时无法可靠回答这个问题。")
        parsed.setdefault("confidence", 0.25)
        parsed.setdefault("evidence", [])
        parsed.setdefault("insufficient_evidence", not bool(parsed.get("evidence")))
        parsed["model"] = self.model
        return parsed


class FunASRClient:
    """Native Sentrix FunASR adapter; it never calls the legacy FMA Whisper service."""

    def __init__(self, timeout=None):
        self.model_name = os.getenv("FUNASR_MODEL", "paraformer-zh")
        self.vad_model = os.getenv("FUNASR_VAD_MODEL", "fsmn-vad")
        self.punc_model = os.getenv("FUNASR_PUNC_MODEL", "ct-punc")
        self.device = os.getenv("FUNASR_DEVICE", "cpu")
        self.timeout = timeout or float(os.getenv("FUNASR_TIMEOUT_SECONDS", "300"))
        self._model = None
        self.error = None

    def _load(self):
        if self._model is not None:
            return self._model
        try:
            from funasr import AutoModel
            self._model = AutoModel(
                model=self.model_name,
                vad_model=self.vad_model,
                punc_model=self.punc_model,
                device=self.device,
                disable_update=True,
            )
            return self._model
        except Exception as error:
            self.error = str(error)
            raise ModelError(f"FunASR unavailable: {error}") from error

    def transcribe(self, path):
        result = self._load().generate(input=str(path), batch_size_s=300)
        if isinstance(result, dict):
            result = [result]
        first = result[0] if result else {}
        return {
            "text": first.get("text", ""),
            "segments": first.get("sentence_info") or first.get("timestamp") or [],
            "model": self.model_name,
            "vad_model": self.vad_model,
            "punc_model": self.punc_model,
        }


class ClipAdapter:
    """Optional local CLIP ViT-B/32 adapter for native visual memory."""

    def __init__(self):
        self.enabled = os.getenv("CLIP_ENABLED", "true").lower() in {"1", "true", "yes"}
        self.model_name = os.getenv("CLIP_MODEL_NAME", "ViT-B-32")
        configured_checkpoint = os.getenv("CLIP_CHECKPOINT", "")
        project_checkpoint = Path(__file__).resolve().parents[1] / "data" / "models" / "clip" / f"{self.model_name}.bin"
        self.checkpoint = configured_checkpoint or (str(project_checkpoint) if project_checkpoint.is_file() else "")
        self._model = None
        self._preprocess = None
        self._tokenizer = None
        self._load_lock = threading.Lock()
        self.error = None
        self.device = os.getenv("CLIP_DEVICE", "auto")
        # A randomly initialized model must never be used as retrieval evidence.
        self.weights_ready = bool(self.checkpoint) or os.getenv("CLIP_ALLOW_DOWNLOAD", "false").lower() in {"1", "true", "yes"}

    @property
    def evidence_ready(self):
        return self.enabled and self.weights_ready and self.error is None

    @property
    def embedding_dimension(self):
        configured = os.getenv("CLIP_EMBED_DIM")
        if configured:
            return int(configured)
        normalized = str(self.model_name or "").lower().replace("_", "-")
        if "vit-h-14" in normalized:
            return 1024
        if "vit-l-14" in normalized:
            return 768
        return 512

    def _device(self, torch):
        requested = str(self.device or "auto").strip().lower()
        if requested == "auto":
            return "cuda:0" if torch.cuda.is_available() else "cpu"
        return requested

    def _load(self):
        if self._model is not None:
            return self._model, self._preprocess
        with self._load_lock:
            if self._model is not None:
                return self._model, self._preprocess
            if not self.enabled:
                return None, None
            if not self.checkpoint and os.getenv("CLIP_ALLOW_DOWNLOAD", "false").lower() not in {"1", "true", "yes"}:
                self.error = "CLIP_CHECKPOINT is not configured"
                self.weights_ready = False
                return None, None
            try:
                import open_clip
                import torch
                kwargs = {"model_name": self.model_name, "pretrained": "openai" if not self.checkpoint else None, "load_weights": not bool(self.checkpoint)}
                self._model, _, self._preprocess = open_clip.create_model_and_transforms(**kwargs)
                if self.checkpoint:
                    state = torch.load(self.checkpoint, map_location="cpu", weights_only=True)
                    self._model.load_state_dict(state, strict=False)
                self._tokenizer = open_clip.get_tokenizer(self.model_name)
                self.device = self._device(torch)
                self._model.to(self.device)
                self._model.eval()
                return self._model, self._preprocess
            except Exception as error:
                self.error = str(error)
                self.weights_ready = False
                return None, None

    def embed_image(self, path):
        model, preprocess = self._load()
        if model is None:
            return []
        try:
            import torch
            from PIL import Image
            from .image_io import ensure_heif_support

            ensure_heif_support()
            image = preprocess(Image.open(path).convert("RGB")).unsqueeze(0).to(self.device)
            with torch.no_grad():
                embedding = model.encode_image(image)
            return embedding[0].cpu().tolist()
        except Exception as error:
            self.error = str(error)
            return []

    def embed_text(self, text):
        model, _ = self._load()
        if model is None or not str(text or "").strip():
            return []
        try:
            import torch
            tokens = self._tokenizer([str(text)]).to(self.device)
            with torch.no_grad():
                embedding = model.encode_text(tokens)
            return embedding[0].cpu().tolist()
        except Exception as error:
            self.error = str(error)
            return []


class FaceAdapter:
    """Optional independent InsightFace adapter; Sentrix owns this boundary."""

    def __init__(self):
        self.enabled = os.getenv("FACE_ENABLED", "true").lower() in {"1", "true", "yes"}
        self._app = None
        self._load_lock = threading.Lock()
        self._face_analysis_lock = threading.Lock()
        self._recognition_lock = threading.Lock()
        self._retina = None
        self._recognition_session = None
        self.error = None
        self.identity_model = "legacy"
        self.identity_adapter = None
        self.identity_error = None
        self.identity_runtime_error = None
        self.identity_fallback = False
        self.identity_fallback_model = None
        self.identity_fallback_error = None

    @property
    def identity_configured(self):
        return True

    @property
    def identity_ready(self):
        return True

    @property
    def ready(self):
        return self.enabled and self.error is None

    @staticmethod
    def _configure_onnx_runtime_libraries():
        """Expose pip-installed NVIDIA runtime libraries before ONNX imports."""
        try:
            import site

            directories = []
            for root in site.getsitepackages():
                nvidia_root = Path(root) / "nvidia"
                if nvidia_root.is_dir():
                    directories.extend(str(path) for path in nvidia_root.glob("*/lib") if path.is_dir())
            if directories:
                existing = os.environ.get("LD_LIBRARY_PATH", "")
                merged = ":".join(dict.fromkeys(directories + ([existing] if existing else [])))
                os.environ["LD_LIBRARY_PATH"] = merged
        except Exception:
            pass

    def detect(self, path):
        if not self.enabled:
            return []
        try:
            import cv2
            import numpy as np
            image = cv2.imread(str(path))
            if image is None:
                # Apple HEIC/HEIF and some PNGs are unreadable by OpenCV alone.
                from .image_io import ensure_heif_support
                from PIL import Image, ImageOps

                ensure_heif_support()
                with Image.open(path) as pil_image:
                    # cv2.imread applies EXIF orientation on the JPEG path; transpose
                    # here so HEIC/HEIF detections share the same oriented bbox space.
                    image = cv2.cvtColor(np.array(ImageOps.exif_transpose(pil_image).convert("RGB")), cv2.COLOR_RGB2BGR)
            if image is None:
                return []
            return self._detect_retina_tiled(image)
        except Exception as error:  # Optional model should not block image memory.
            self.error = str(error)
            return []

    def _load_buffalo_recognition(self):
        """Load buffalo_l recognition (w600k_r50) for RetinaFace-only fallback."""
        try:
            import onnxruntime
            model_root = os.getenv("FACE_MODEL_ROOT", os.path.expanduser("~/.insightface/models"))
            model_path = os.path.join(model_root, os.getenv("FACE_MODEL_NAME", "buffalo_l"), "w600k_r50.onnx")
            if not os.path.isfile(model_path):
                return None
            providers = face_onnx_providers("FACE_PROVIDERS")
            return onnxruntime.InferenceSession(model_path, providers=providers)
        except Exception:
            return None

    def _recognition_embed(self, crop):
        try:
            import numpy as np
            crop = crop.resize((112, 112))
            image = np.asarray(crop.convert("RGB"), dtype=np.float32)
            blob = ((image - 127.5) / 128.0).transpose(2, 0, 1)[None, ...]
            with self._recognition_lock:
                output = self._recognition_session.run(
                    None, {self._recognition_session.get_inputs()[0].name: blob}
                )[0]
            return [float(value) for value in output[0]]
        except Exception:
            return []

    def _detect_retina_tiled(self, image):
        with face_gpu_inference_gate():
            return self._detect_retina_tiled_unlocked(image)

    def _detect_retina_tiled_unlocked(self, image):
        """RetinaFace tiled detection + SCRFD validity gate.

        RetinaFace finds face candidates (high recall), then buffalo_l SCRFD on
        an expanded sub-crop is the secondary verifier. Only candidates that are
        confirmed by SCRFD with a high score AND a consistent bbox AND sane
        landmark geometry become VERIFIED (eligible to seed a person cluster).
        Everything else is kept as evidence only (UNCERTAIN) or dropped.
        """
        try:
            from .face_detector import RetinaFaceTiledDetector
            if self._retina is None:
                self._retina = RetinaFaceTiledDetector()
                self._ensure_face_analysis()
                if self._recognition_session is None:
                    self._recognition_session = self._load_buffalo_recognition()
            if self._app is None:
                return []
            detections = self._retina.detect(image)
            image_height, image_width = image.shape[:2]
            min_size = int(os.getenv("FACE_MIN_SIZE", "64"))
            verified_scrfd_score = float(os.getenv("FACE_VERIFIED_SCRFD_SCORE", "0.7"))
            verified_min_area = float(os.getenv("FACE_VERIFIED_MIN_AREA", "0.003"))
            results = []
            for det in detections:
                bbox = [float(value) for value in det["bbox"]]
                width = max(0.0, bbox[2] - bbox[0])
                height = max(0.0, bbox[3] - bbox[1])
                if min(width, height) < min_size:
                    continue
                score = float(det["confidence"])
                sub, sub_x, sub_y = self._expand_crop(image, bbox)
                if sub is None:
                    continue
                with self._face_analysis_lock:
                    sub_faces = self._app.get(sub)
                if not sub_faces:
                    # SCRFD finds no face here -> UNCERTAIN evidence only, never a
                    # cluster seed. RetinaFace landmark alignment is too unreliable
                    # to hand this candidate clustering rights.
                    try:
                        crop = align_face_crop(image, bbox, det["landmarks"])
                        raw_sharpness = _laplacian_variance(crop)
                    except Exception:
                        crop = None
                        raw_sharpness = 0.0
                    sharpness = _normalize_sharpness(raw_sharpness)
                    embedding = self._recognition_embed(crop) if crop is not None and self._recognition_session is not None else []
                    if not embedding:
                        continue
                    area_ratio = min(1.0, (width * height) / max(1.0, image_width * image_height))
                    quality = compute_face_quality(score, area_ratio, sharpness, [])
                    results.append({
                        "bbox": bbox,
                        "confidence": score,
                        "quality": quality,
                        "area_ratio": area_ratio,
                        "sharpness": sharpness,
                        "raw_sharpness": raw_sharpness,
                        "pose": [],
                        "landmarks": det["landmarks"],
                        "embedding": embedding,
                        "embedding_model": "buffalo_l",
                        "embedding_version": os.getenv("FACE_MODEL_NAME", "buffalo_l"),
                        "identity_ready": bool(embedding),
                        "face_validity": "uncertain",
                        "identity_eligible": False,
                    })
                    continue
                best = max(sub_faces, key=lambda f: float(f.det_score))
                sub_bbox = [float(value) for value in best.bbox]
                scrfd_bbox = [
                    sub_bbox[0] + sub_x, sub_bbox[1] + sub_y,
                    sub_bbox[2] + sub_x, sub_bbox[3] + sub_y,
                ]
                landmarks = [[float(value) for value in point] for point in best.kps] if getattr(best, "kps", None) is not None else []
                embedding = best.embedding.tolist() if getattr(best, "embedding", None) is not None else []
                score = float(best.det_score)
                agreed = self._bbox_agreement(bbox, scrfd_bbox)
                # Use SCRFD's own (reliable) landmarks, not RetinaFace's, whose
                # geometry is unreliable even on large faces.
                scrfd_landmarks = [
                    [float(point[0]) + sub_x, float(point[1]) + sub_y]
                    for point in best.kps
                ] if getattr(best, "kps", None) is not None else []
                sane = self._landmark_sanity(scrfd_landmarks, bbox) if scrfd_landmarks else False
                area_ratio = min(1.0, (width * height) / max(1.0, image_width * image_height))
                validity = "verified" if (
                    score >= verified_scrfd_score and agreed and sane and area_ratio >= verified_min_area
                ) else "uncertain"
                area_ratio = min(1.0, (width * height) / max(1.0, image_width * image_height))
                pose = [float(value) for value in best.pose] if getattr(best, "pose", None) is not None else []
                try:
                    crop = align_face_crop(sub, sub_bbox, landmarks)
                    raw_sharpness = _laplacian_variance(crop)
                except Exception:
                    raw_sharpness = 0.0
                sharpness = _normalize_sharpness(raw_sharpness)
                quality = compute_face_quality(score, area_ratio, sharpness, pose)
                results.append({
                    "bbox": bbox,
                    "confidence": score,
                    "quality": quality,
                    "area_ratio": area_ratio,
                    "sharpness": sharpness,
                    "raw_sharpness": raw_sharpness,
                    "pose": pose,
                    "landmarks": landmarks,
                    "embedding": embedding,
                    "embedding_model": "buffalo_l",
                    "embedding_version": os.getenv("FACE_MODEL_NAME", "buffalo_l"),
                    "identity_ready": bool(embedding),
                    "face_validity": validity,
                    "identity_eligible": (
                        validity == "verified"
                        and quality >= float(os.getenv("FACE_IDENTITY_MIN_QUALITY", "0.55"))
                    ),
                })
            return results
        except Exception as error:
            self.error = str(error)
            return []

    @staticmethod
    def _bbox_agreement(retina_bbox, scrfd_bbox, center_ratio=1.0, iou_threshold=0.05):
        """RetinaFace and SCRFD must roughly agree on where the face is.

        Center-only check: the SCRFD face centre must lie within one retina box
        diagonal of the retina centre. IoU is a weak floor. This is deliberately
        loose because on small faces the two detectors produce differently-sized
        boxes (SCRFD re-runs on an upscaled sub-crop), so strict IoU kills real
        small faces.
        """
        rw = max(1.0, retina_bbox[2] - retina_bbox[0])
        rh = max(1.0, retina_bbox[3] - retina_bbox[1])
        rcx = (retina_bbox[0] + retina_bbox[2]) / 2.0
        rcy = (retina_bbox[1] + retina_bbox[3]) / 2.0
        scx = (scrfd_bbox[0] + scrfd_bbox[2]) / 2.0
        scy = (scrfd_bbox[1] + scrfd_bbox[3]) / 2.0
        center_dist = ((rcx - scx) ** 2 + (rcy - scy) ** 2) ** 0.5
        if center_dist > (rw ** 2 + rh ** 2) ** 0.5 * center_ratio:
            return False
        ix1 = max(retina_bbox[0], scrfd_bbox[0])
        iy1 = max(retina_bbox[1], scrfd_bbox[1])
        ix2 = min(retina_bbox[2], scrfd_bbox[2])
        iy2 = min(retina_bbox[3], scrfd_bbox[3])
        inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
        area_r = rw * rh
        area_s = max(1.0, scrfd_bbox[2] - scrfd_bbox[0]) * max(1.0, scrfd_bbox[3] - scrfd_bbox[1])
        iou = inter / max(1e-9, area_r + area_s - inter)
        return iou >= iou_threshold

    @staticmethod
    def _landmark_sanity(landmarks, bbox, min_eye_ratio=0.1):
        """Loose sanity that the landmarks look like a face, not a pretty face."""
        if not landmarks or len(landmarks) < 5:
            return False
        bbox_w = max(1.0, bbox[2] - bbox[0])
        bbox_h = max(1.0, bbox[3] - bbox[1])
        if bbox_w < 64 or bbox_h < 64:
            return True  # too small for geometry checks; trust the detectors
        eye_dist = ((landmarks[0][0] - landmarks[1][0]) ** 2 + (landmarks[0][1] - landmarks[1][1]) ** 2) ** 0.5
        if eye_dist < bbox_w * min_eye_ratio:
            return False
        pad_x = bbox_w * 0.2
        pad_y = bbox_h * 0.2
        inside = sum(
            1 for x, y in landmarks
            if (bbox[0] - pad_x) <= x <= (bbox[2] + pad_x) and (bbox[1] - pad_y) <= y <= (bbox[3] + pad_y)
        )
        return inside >= 3

    def _ensure_face_analysis(self):
        """Load the buffalo_l FaceAnalysis used for sub-crop alignment/embedding."""
        if self._app is not None:
            return
        with getattr(self, "_load_lock", threading.Lock()):
            if self._app is None:
                self._configure_onnx_runtime_libraries()
                from insightface.app import FaceAnalysis
                providers = [item for item in os.getenv("FACE_PROVIDERS", "CUDAExecutionProvider,CPUExecutionProvider").split(",") if item]
                kwargs = {
                    "name": os.getenv("FACE_MODEL_NAME", "buffalo_l"),
                    "providers": providers,
                    "provider_options": face_onnx_provider_options("FACE_PROVIDERS"),
                    # This adapter only consumes SCRFD landmarks and ArcFace
                    # embeddings; avoid loading unused 3D/106-point/gender models.
                    "allowed_modules": ["detection", "recognition"],
                }
                if os.getenv("FACE_MODEL_ROOT"):
                    kwargs["root"] = os.getenv("FACE_MODEL_ROOT")
                self._app = FaceAnalysis(**kwargs)
                det_size = int(os.getenv("FACE_DET_SIZE", "640"))
                use_cuda = any(item.strip() == "CUDAExecutionProvider" for item in providers)
                self._app.prepare(ctx_id=0 if use_cuda else -1, det_size=(det_size, det_size))

    @staticmethod
    def _expand_crop(image, bbox, margin=0.75, min_side=256):
        image_height, image_width = image.shape[:2]
        x1, y1, x2, y2 = (int(round(value)) for value in bbox)
        width, height = x2 - x1, y2 - y1
        if width <= 0 or height <= 0:
            return None, 0, 0
        mx, my = int(width * margin), int(height * margin)
        x1, y1 = x1 - mx, y1 - my
        x2, y2 = x2 + mx, y2 + my
        sub_w, sub_h = x2 - x1, y2 - y1
        if sub_w < min_side or sub_h < min_side:
            scale = min_side / min(sub_w, sub_h)
            cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
            nw, nh = int(sub_w * scale), int(sub_h * scale)
            x1, y1 = int(cx - nw / 2.0), int(cy - nh / 2.0)
            x2, y2 = x1 + nw, y1 + nh
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(image_width, x2), min(image_height, y2)
        if x2 <= x1 or y2 <= y1:
            return None, 0, 0
        return image[y1:y2, x1:x2], x1, y1
