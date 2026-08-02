import base64
import json
import mimetypes
import os
import re
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
    return parsed


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
            "keep_alive": os.getenv("OLLAMA_KEEP_ALIVE", "0"),
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
严格只返回 JSON 对象，所有字段值必须使用简体中文；画面中没有人物时 people 返回空数组。
字段必须为：
caption（图片内容简述）、activity（活动）、place（地点，不确定为空字符串）、people（人物外观描述数组，不包含姓名）、objects（物体数组）、clothing（衣物和配饰数组）、emotions（画面中可明确观察到的情感或氛围数组）、spatial_relations（空间关系数组）、ocr_text（图片中可读文字，没有则为空字符串）、event_type（事件类型）、facts（可维护事实数组）。
facts 每项字段为 subject、predicate、object、confidence；不确定的事实不要放入 facts。
不要把来源成员当成画面人物，也不要推测拍摄者姓名；source_owner 只作为事件来源候选。
metadata: """ + json.dumps(metadata or {}, ensure_ascii=False)
        parsed = parse_json_response(self.chat(prompt, [{"base64": encoded, "mime_type": mime_type}]))
        scalar_text = " ".join(as_text(parsed.get(key)) for key in ("caption", "activity", "place", "event_type", "ocr_text"))
        if contains_latin_text(scalar_text):
            canonical_prompt = """把下面的家庭图片观察规范化为简体中文 JSON。只翻译和整理已有内容，不新增人物、物体、活动或事实，不猜测姓名。保留字段 caption、activity、place、people、objects、clothing、spatial_relations、ocr_text、event_type、facts。
原始观察：""" + json.dumps(parsed, ensure_ascii=False)
            parsed = parse_json_response(self.chat(canonical_prompt))
        parsed["people"] = as_list(parsed.get("people"))
        parsed["objects"] = as_list(parsed.get("objects"))
        parsed["clothing"] = as_list(parsed.get("clothing"))
        parsed["emotions"] = as_list(parsed.get("emotions"))
        parsed["spatial_relations"] = as_list(parsed.get("spatial_relations"))
        parsed["facts"] = normalize_fact_confidences(parsed.get("facts"), 0.65)
        normalize_analysis_fields(parsed)
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
        evidence = [{
            "observation_id": item.get("id"),
            "caption": item.get("caption"),
            "activity": item.get("activity"),
            "people": item.get("people", []),
            "objects": item.get("objects", []),
            "ocr_text": item.get("ocr_text"),
            "clothing": item.get("clothing", []),
            "spatial_relations": item.get("spatial_relations", []),
        } for item in observations]
        prompt = """你是家庭事件总结器。下面是一组已经按拍摄时间和地点聚类的图片观察。
只能使用给定观察，不得把元数据地点以外的信息当作事实，不得猜测未确认人物姓名；如果观察彼此不足以支持具体事件，使用保守、描述性的标题。
严格返回简体中文 JSON：title（不超过20字）、event_type、activity、summary（包含时间地点范围和可验证活动）、confidence。
事件：""" + json.dumps({
            "time_start": event.get("time_start"), "time_end": event.get("time_end"), "place": event.get("place"), "observations": evidence,
        }, ensure_ascii=False)
        parsed = parse_json_response(self.chat(prompt))
        return {
            "title": as_text(parsed.get("title")) or "待确认的家庭记录",
            "event_type": as_text(parsed.get("event_type")) or "家庭记录",
            "activity": as_text(parsed.get("activity")) or "家庭活动",
            "summary": as_text(parsed.get("summary")) or "该事件的图片证据尚不足以生成更具体的总结。",
            "confidence": normalize_confidence(parsed.get("confidence"), 0.5),
            "model": self.model,
        }

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
        configured_checkpoint = os.getenv("CLIP_CHECKPOINT", "")
        project_checkpoint = Path(__file__).resolve().parents[1] / "data" / "models" / "clip" / f"{self.model_name}.bin"
        self.checkpoint = configured_checkpoint or (str(project_checkpoint) if project_checkpoint.is_file() else "")
        self._model = None
        self._preprocess = None
        self._tokenizer = None
        self.error = None
        self.device = os.getenv("CLIP_DEVICE", "auto")

    def _device(self, torch):
        requested = str(self.device or "auto").strip().lower()
        return "cuda:0" if requested == "auto" and torch.cuda.is_available() else requested

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
            return None, None

    def embed_image(self, path):
        model, preprocess = self._load()
        if model is None:
            return []
        try:
            import torch
            from PIL import Image
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

    def detect(self, path):
        if not self.enabled:
            return []
        try:
            if self._app is None:
                from insightface.app import FaceAnalysis
                providers = [item for item in os.getenv("FACE_PROVIDERS", "CPUExecutionProvider").split(",") if item]
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
            image = cv2.imread(str(path))
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
