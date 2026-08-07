import base64
from io import BytesIO
import json
import mimetypes
import os
import re
import threading
from pathlib import Path

from .face_embeddings import AdaFaceAdapter, FaceEmbeddingUnavailable, MagFaceAdapter, compute_face_quality


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
    "answer": {"temperature": 0.3, "think": False, "num_ctx": 8192, "num_predict": 800},
    "writer": {"temperature": 0.3, "think": False, "num_ctx": 8192, "num_predict": 800},
    "verify": {"temperature": 0.0, "think": False, "num_ctx": 4096, "num_predict": 512},
    "claim": {"temperature": 0.0, "think": False, "num_ctx": 4096, "num_predict": 512},
    "repair": {"temperature": 0.0, "think": False, "num_ctx": 4096, "num_predict": 512},
}


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


class E2BBackend:
    """E2B LoRA backend — local 2B model via E2B server on :8100."""

    name = "e2b_lora"
    model_name = "gemma-4-e2b-it+lora-v2"

    def __init__(self, base_url=None, timeout=None):
        self._base_url = (base_url or os.getenv("E2B_BASE_URL", "http://127.0.0.1:8100")).rstrip("/")
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
                 repair_model=None, backend=None, api_key=None):
        self.backend = self._normalize_backend(backend or os.getenv("SENTRIX_LLM_BACKEND", "vllm"))
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
        else:
            self._base_url_setting = (base_url or os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")).rstrip("/")
            self._model_setting = model or os.getenv("OLLAMA_MODEL", "gemma4:12b")
        self.api_key = api_key or os.getenv("SENTRIX_VLLM_API_KEY") or os.getenv("OPENAI_API_KEY") or ""
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

    @staticmethod
    def _normalize_backend(value):
        value = str(value or "vllm").strip().lower()
        if value in {"vllm", "openai", "openai_compatible", "openai-compatible"}:
            return "openai"
        if value in {"ollama", "ollama_local"}:
            return "ollama"
        raise ModelError(f"unsupported llm backend: {value}")

    @staticmethod
    def _normalize_openai_base_url(value):
        value = str(value or "http://127.0.0.1:8100/v1").rstrip("/")
        return value if value.endswith("/v1") else f"{value}/v1"

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
            raise ModelError(f"gamma request failed: {error}") from error

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
        payload = {
            "model": model,
            "messages": [message],
            "stream": False,
            "temperature": 0,
        }
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
        try:
            response = httpx.post(f"{endpoint_base}/chat/completions", json=payload, headers=headers, timeout=self.timeout)
            response.raise_for_status()
            data = response.json()
            text = (((data.get("choices") or [{}])[0].get("message") or {}).get("content") or "")
            if isinstance(text, list):
                text = "".join(part.get("text", "") if isinstance(part, dict) else str(part) for part in text)
            self._record_validation_call(role, endpoint_base, model, json_mode, text)
            return text
        except (httpx.HTTPError, ValueError, KeyError, IndexError, TypeError) as error:
            raise ModelError(f"gamma request failed: {error}") from error

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
            "num_predict": int(os.getenv("VISION_CORE_NUM_PREDICT", "320")),
        }

    def _encode_core_image(self, path):
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

    def analyze_image(self, path, metadata=None):
        file_path = Path(path)
        encoded, mime_type = self._encode_core_image(file_path)
        prompt = """你是家庭记忆观察器。仅根据图片和元数据抽取可验证的核心观察，不猜测姓名。
严格返回简体中文 JSON 对象，不要解释。caption、activity、place、event_type 是必须同时输出的自然语言观察字段；即使能够选择 semantic，也不能只输出 semantic 选择。画面能判断时不要留空，caption 不超过20字；activity、place、event_type 各不超过10字；people、objects、clothing、emotions、spatial_relations 各最多2项，每项不超过10字；facts 最多1项；ocr_text 不超过20字；确实看不清才用空数组或空字符串。
字段固定为：caption、activity、place、scene_type、semantic、people、objects、clothing、emotions、spatial_relations、ocr_text、event_type、facts。semantic.place.primary 只能选择地点主类，details 从图片可观察的地点细节中多选；semantic.objects 是物品记录数组，每项包含 primary、label、details；semantic.atmosphere.labels 和 details 都是可观察画面氛围的多选值，不描述人物心理。
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
严格返回简体中文 JSON：caption（图片中看到什么，20字内）、activity（正在发生什么，10字内）、place（语义地点描述，如家中客厅/餐厅/公园，不要GPS，10字内）、event_type（10字内）、people（最多2项）、objects（最多4项）、ocr_text（20字内）。画面确实看不清才留空；不要只返回分类字段。"""
            recovered = parse_json_response(self.chat(recovery_prompt, [{"base64": encoded, "mime_type": mime_type}], self._core_vision_options()))
            for key in ("caption", "activity", "place", "event_type", "people", "objects", "ocr_text"):
                if recovered.get(key) not in (None, "", []):
                    parsed[key] = recovered[key]
        scalar_text = " ".join(as_text(parsed.get(key)) for key in ("caption", "activity", "place", "event_type", "ocr_text"))
        if contains_latin_text(scalar_text):
            canonical_prompt = "把下面的家庭图片观察规范化为简体中文 JSON。只翻译和整理已有内容，不新增人物、物体、活动或事实，不猜测姓名。保留字段 caption、activity、place、scene_type、semantic、people、objects、clothing、spatial_relations、ocr_text、event_type、facts。semantic 必须保留地点主类、地点细节、物品记录和可观察画面氛围。scene_type 必须保留为下列之一："
            canonical_prompt += "、".join(SCENE_TYPE_OPTIONS)
            canonical_prompt += "。\n原始观察：" + json.dumps(parsed, ensure_ascii=False)
            parsed = parse_json_response(self.chat(canonical_prompt))
        parsed["people"] = as_list(parsed.get("people"))
        parsed["objects"] = as_list(parsed.get("objects"))
        parsed["clothing"] = as_list(parsed.get("clothing"))
        parsed["emotions"] = as_list(parsed.get("emotions"))
        parsed["spatial_relations"] = as_list(parsed.get("spatial_relations"))
        parsed["facts"] = normalize_fact_confidences(parsed.get("facts"), 0.65)
        normalize_analysis_fields(parsed)
        parsed = normalize_semantic_analysis(parsed)
        parsed["confidence"] = normalize_confidence(parsed.get("confidence"), 0.65)
        parsed["model"] = self.model
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

    def summarize_event(self, event, observations):
        semantic_place = _event_place(event, observations)
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
        return {
            "title": _strip_event_coordinates(as_text(parsed.get("title")) or "家庭图片记录", fallback_place),
            "event_type": _strip_event_coordinates(as_text(parsed.get("event_type")) or "家庭记录", fallback_place),
            "activity": _strip_event_coordinates(as_text(parsed.get("activity")) or "家庭活动", fallback_place),
            "summary": _strip_event_coordinates(as_text(parsed.get("summary")) or "该事件的图片证据尚不足以生成更具体的总结。", fallback_place),
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

    def _device(self, torch):
        requested = str(self.device or "auto").strip().lower()
        return "cuda:0" if requested == "auto" and torch.cuda.is_available() else requested

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
        self.error = None
        self.identity_model = os.getenv("FACE_EMBEDDING_MODE", "adaface").lower()
        self.identity_adapter = self._build_identity_adapter()
        self.identity_error = None
        self.identity_runtime_error = None
        if self.identity_model not in {"none", "legacy", "adaface", "magface"}:
            self.identity_error = f"unsupported face embedding mode: {self.identity_model}"
        elif self.identity_model in {"adaface", "magface"} and not self.identity_adapter.available:
            self.identity_error = f"{self.identity_model} checkpoint is unavailable"

    def _build_identity_adapter(self):
        if self.identity_model == "adaface":
            return AdaFaceAdapter()
        if self.identity_model == "magface":
            return MagFaceAdapter(
                model_version=os.getenv("MAGFACE_MODEL_VERSION", "unconfigured"),
                backend=None,
            )
        return None

    @property
    def identity_configured(self):
        return self.identity_model == "legacy" or bool(self.identity_adapter and self.identity_adapter.available)

    @property
    def identity_ready(self):
        return self.identity_configured and self.identity_runtime_error is None

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
            if self._app is None:
                with getattr(self, "_load_lock", threading.Lock()):
                    if self._app is None:
                        self._configure_onnx_runtime_libraries()
                        from insightface.app import FaceAnalysis
                        providers = [item for item in os.getenv("FACE_PROVIDERS", "CUDAExecutionProvider,CPUExecutionProvider").split(",") if item]
                        kwargs = {"name": os.getenv("FACE_MODEL_NAME", "buffalo_l"), "providers": providers}
                        if self.identity_model in {"adaface", "magface"}:
                            # AdaFace/MagFace produce the only identity vector. Avoid
                            # loading buffalo_l recognition and demographic models.
                            kwargs["allowed_modules"] = ["detection", "landmark_2d_106"]
                        if os.getenv("FACE_MODEL_ROOT"):
                            kwargs["root"] = os.getenv("FACE_MODEL_ROOT")
                        self._app = FaceAnalysis(**kwargs)
                        self._app.prepare(ctx_id=-1, det_size=(640, 640))
            import cv2
            import numpy as np
            image = cv2.imread(str(path))
            if image is None:
                # Apple HEIC/HEIF and some PNGs are unreadable by OpenCV alone.
                from .image_io import ensure_heif_support
                from PIL import Image

                ensure_heif_support()
                with Image.open(path) as pil_image:
                    image = cv2.cvtColor(np.array(pil_image.convert("RGB")), cv2.COLOR_RGB2BGR)
            if image is None:
                return []
            faces = self._app.get(image)
            image_height, image_width = image.shape[:2]
            min_size = int(os.getenv("FACE_MIN_SIZE", "64"))
            min_score = float(os.getenv("FACE_MIN_DETECTION_SCORE", "0.72"))
            results = []
            for face in faces:
                bbox = [float(item) for item in face.bbox]
                width = max(0.0, bbox[2] - bbox[0])
                height = max(0.0, bbox[3] - bbox[1])
                score = float(face.det_score)
                if score < min_score or min(width, height) < min_size:
                    continue
                area_ratio = min(1.0, (width * height) / max(1.0, image_width * image_height))
                # Identity candidacy is deliberately stricter than face evidence.
                # A weak/small face stays attached to the observation but cannot
                # create a noisy pending person cluster.
                sharpness = 0.0
                quality = compute_face_quality(score, area_ratio, sharpness, getattr(face, "pose", []))
                results.append({
                    "bbox": bbox,
                    "confidence": score,
                    "quality": quality,
                    "area_ratio": area_ratio,
                    "sharpness": sharpness,
                    "pose": [float(item) for item in getattr(face, "pose", [])] if getattr(face, "pose", None) is not None else [],
                    "landmarks": [[float(value) for value in point] for point in getattr(face, "kps", [])] if getattr(face, "kps", None) is not None else [],
                    "embedding": face.embedding.tolist() if getattr(face, "embedding", None) is not None else [],
                })
                result = results[-1]
                if self.identity_model in {"adaface", "magface"}:
                    result["embedding_model"] = self.identity_model
                    result["embedding_version"] = self.identity_adapter.model_version
                    result["identity_ready"] = self.identity_configured
                    if not self.identity_configured:
                        result["embedding"] = []
                        result["identity_error"] = self.identity_error
                    else:
                        try:
                            crop = align_face_crop(image, bbox, result["landmarks"])
                            embedded = self.identity_adapter.embed(crop)
                            result["embedding"] = embedded.embedding
                            result["embedding_version"] = embedded.model_version
                            result["quality_signal"] = embedded.quality_signal
                            # AdaFace norm is stored as provenance, not treated as
                            # a 0..10 score. It must not saturate all sample quality.
                            result["identity_eligible"] = result["quality"] >= float(
                                os.getenv("FACE_IDENTITY_MIN_QUALITY", "0.55")
                            )
                        except FaceEmbeddingUnavailable as error:
                            result["embedding"] = []
                            result["identity_ready"] = False
                            result["identity_error"] = str(error)
                            self.identity_runtime_error = str(error)
                elif self.identity_model == "legacy":
                    result["embedding_model"] = "buffalo_l"
                    result["embedding_version"] = os.getenv("FACE_MODEL_NAME", "buffalo_l")
                    result["identity_ready"] = True
            return results
        except Exception as error:  # Optional model should not block image memory.
            self.error = str(error)
            return []
