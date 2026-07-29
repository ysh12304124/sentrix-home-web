import base64
import json
import mimetypes
import os
import re
from pathlib import Path

try:
    import httpx
except ImportError:  # Keep pure parsing and SQLite tests runnable without optional runtime deps.
    httpx = None


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


def normalize_analysis_fields(parsed):
    for key in ("caption", "activity", "place", "event_type", "transcript", "ocr_text"):
        parsed[key] = as_text(parsed.get(key))
    return parsed


class GammaClient:
    def __init__(self, base_url=None, model=None, timeout=None):
        self.base_url = (base_url or os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")).rstrip("/")
        self.model = model or os.getenv("OLLAMA_MODEL", "gemma4:12b")
        self.timeout = timeout or float(os.getenv("OLLAMA_TIMEOUT_SECONDS", "180"))

    def chat(self, prompt, images=None):
        if httpx is None:
            raise ModelError("httpx is not installed")
        message = {"role": "user", "content": prompt}
        if images:
            message["images"] = [image["base64"] for image in images]
        payload = {
            "model": self.model,
            "messages": [message],
            "stream": False,
            "format": "json",
            "options": {"temperature": 0},
        }
        try:
            response = httpx.post(f"{self.base_url}/api/chat", json=payload, timeout=self.timeout)
            response.raise_for_status()
            data = response.json()
            return data.get("message", {}).get("content", "")
        except (httpx.HTTPError, ValueError) as error:
            raise ModelError(f"gamma request failed: {error}") from error

    def analyze_image(self, path, metadata=None):
        file_path = Path(path)
        encoded = base64.b64encode(file_path.read_bytes()).decode("ascii")
        mime_type = mimetypes.guess_type(file_path.name)[0] or "image/jpeg"
        prompt = """你是家庭记忆观察器。只根据图片和给定元数据抽取可验证观察，不要猜测姓名。
严格只返回 JSON 对象，字段必须为：
caption（图片内容简述）、activity（活动）、place（地点，不确定为空字符串）、people（人物描述数组）、objects（物体数组）、ocr_text（图片中可读文字，没有则为空字符串）、event_type（事件类型）、facts（可维护事实数组）。
facts 每项字段为 subject、predicate、object、confidence；不确定的事实不要放入 facts。
metadata: """ + json.dumps(metadata or {}, ensure_ascii=False)
        parsed = parse_json_response(self.chat(prompt, [{"base64": encoded, "mime_type": mime_type}]))
        parsed["people"] = as_list(parsed.get("people"))
        parsed["objects"] = as_list(parsed.get("objects"))
        parsed["facts"] = as_list(parsed.get("facts"))
        normalize_analysis_fields(parsed)
        parsed["confidence"] = float(parsed.get("confidence", 0.65) or 0.65)
        parsed["model"] = self.model
        return parsed

    def analyze_text(self, text, source_type="text"):
        prompt = """从下面的家庭文本或音频转写中抽取事件观察和可维护事实。只返回 JSON，不要添加 Markdown。
字段：caption、activity、place、people（数组）、objects（数组）、event_type、facts（数组）。
facts 每项为 subject、predicate、object、confidence；没有明确证据时 facts 返回空数组。
文本：""" + text
        parsed = parse_json_response(self.chat(prompt))
        parsed["people"] = as_list(parsed.get("people"))
        parsed["objects"] = as_list(parsed.get("objects"))
        parsed["facts"] = as_list(parsed.get("facts"))
        normalize_analysis_fields(parsed)
        parsed["source_type"] = source_type
        parsed["confidence"] = float(parsed.get("confidence", 0.6) or 0.6)
        parsed["model"] = self.model
        return parsed

    def embed_text(self, text):
        """Use the local Ollama embedding endpoint when the configured model supports it."""
        if httpx is None or not str(text or "").strip():
            return []
        model = os.getenv("SENTRIX_TEXT_EMBED_MODEL", self.model)
        try:
            response = httpx.post(f"{self.base_url}/api/embed", json={"model": model, "input": [str(text)]}, timeout=self.timeout)
            response.raise_for_status()
            payload = response.json()
            embeddings = payload.get("embeddings") or []
            return embeddings[0] if embeddings else []
        except (httpx.HTTPError, ValueError, KeyError):
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
        self.checkpoint = os.getenv("CLIP_CHECKPOINT", "")
        self._model = None
        self._preprocess = None
        self._tokenizer = None
        self.error = None

    def _load(self):
        if self._model is not None:
            return self._model, self._preprocess
        if not self.enabled:
            return None, None
        if not self.checkpoint and os.getenv("CLIP_ALLOW_DOWNLOAD", "false").lower() not in {"1", "true", "yes"}:
            self.error = "CLIP_CHECKPOINT is not configured"
            return None, None
        try:
            import open_clip
            kwargs = {"model_name": self.model_name, "pretrained": "openai" if not self.checkpoint else None, "load_weights": not bool(self.checkpoint)}
            self._model, _, self._preprocess = open_clip.create_model_and_transforms(**kwargs)
            if self.checkpoint:
                import torch
                state = torch.load(self.checkpoint, map_location="cpu", weights_only=True)
                self._model.load_state_dict(state, strict=False)
            self._tokenizer = open_clip.get_tokenizer(self.model_name)
            self._model.eval()
            return self._model, self._preprocess
        except Exception as error:
            self.error = str(error)
            return None, None

    def embed_image(self, path):
        model, preprocess = self._load()
        if model is None:
            return []
        try:
            import torch
            from PIL import Image
            image = preprocess(Image.open(path).convert("RGB")).unsqueeze(0)
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
            tokens = self._tokenizer([str(text)])
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
        self.error = None

    def detect(self, path):
        if not self.enabled:
            return []
        try:
            if self._app is None:
                from insightface.app import FaceAnalysis
                providers = [item for item in os.getenv("FACE_PROVIDERS", "CPUExecutionProvider").split(",") if item]
                kwargs = {"name": os.getenv("FACE_MODEL_NAME", "buffalo_l"), "providers": providers}
                if os.getenv("FACE_MODEL_ROOT"):
                    kwargs["root"] = os.getenv("FACE_MODEL_ROOT")
                self._app = FaceAnalysis(**kwargs)
                self._app.prepare(ctx_id=-1, det_size=(640, 640))
            import cv2
            image = cv2.imread(str(path))
            if image is None:
                return []
            faces = self._app.get(image)
            return [{"bbox": [float(item) for item in face.bbox], "confidence": float(face.det_score), "embedding": face.embedding.tolist() if getattr(face, "embedding", None) is not None else []} for face in faces]
        except Exception as error:  # Optional model should not block image memory.
            self.error = str(error)
            return []
