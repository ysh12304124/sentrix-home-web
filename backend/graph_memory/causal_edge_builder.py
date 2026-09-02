# -*- coding: utf-8 -*-
"""
Causal Edge Builder

Infers CAUSAL links (LEADS_TO / ENABLES) between keyframes.

Three inference methods, selected via CAUSAL_METHOD env var:
  - "rules"     (default): pure rule-based, no LLM calls. Fastest.
  - "rules_vlm": rules extract structural signals (bbox displacement,
                 relation change, count change), LLM judges causality
                 from text only. No images. Fast + semantic.
  - "vlm"       : VLM sees both frame images. Slowest, most detailed.

The inferred edges are written back into the graph as LinkType.CAUSAL
with sub_type LEADS_TO or ENABLES, then persisted to SQLite.
"""

import json
import logging
import os
import re
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Optional, Tuple

logger = logging.getLogger(__name__)

# max frame pairs per VLM batch call (each pair adds 2 images).  Vision
# requests are much larger than text-only signal requests, so the default is
# reduced to bound image-token and KV-cache memory under concurrency.
DEFAULT_BATCH_SIZE = 4
CAUSAL_DEFAULT_MAX_GAP = 2
CAUSAL_DEFAULT_MAX_SECONDS = 30.0
DEFAULT_KEEP_ALIVE = "5m"
DEFAULT_NUM_PREDICT = 2048
# Concurrent requests each allocate an Ollama KV cache.  Keep the text-only
# causal path intentionally small so 4 workers do not recreate the VRAM
# pressure of the old single-request 4096-token budget.
DEFAULT_SIGNALS_NUM_PREDICT = 768
DEFAULT_SIGNALS_NUM_CTX = 3072
DEFAULT_SIGNALS_CONTEXT_RESERVE = 256
DEFAULT_VLM_NUM_PREDICT = 512
DEFAULT_VLM_NUM_CTX = 2048
DEFAULT_VLM_WORKERS = 2

# Static background labels that often flicker between frames; changes in
# these alone are not treated as causal candidates.
GENERIC_OBJECT_LABELS = frozenset({
    "floor", "wall", "ceiling", "sky", "ground", "pavement", "road",
    "floor-wood", "wall-tile", "ceiling-tile", "door", "window",
})

NON_SPECIFIC_OBJECT_LABELS = GENERIC_OBJECT_LABELS | frozenset({
    "person", "people", "human", "hand", "body", "camera",
})

EMPTY_CAUSAL_FIELDS = frozenset({
    "", "none", "null", "n/a", "na", "unknown", "not visible",
})

# Relation predicates that can directly describe a person-object action.
# They are much stronger causal evidence than static spatial relations such
# as "bowl on dining table" or "person beside counter".
ACTION_RELATION_WORDS = frozenset({
    "hold", "holding", "carry", "carrying", "touch", "touching",
    "open", "opening", "close", "closing", "cut", "cutting",
    "put", "putting", "take", "taking", "pick", "picking",
    "pour", "pouring", "use", "using", "eat", "eating",
    "drink", "drinking", "wear", "wearing", "wash", "washing",
    "remove", "removing", "insert", "inserting", "pull", "pulling",
    "push", "pushing", "turn", "turning", "flip", "flipping",
    "throw", "throwing",
})

# Posture can describe a visible human state transition, but is weaker than
# an explicit manipulation verb. It is a separate profile so we can benchmark
# recall/time without mixing it into the original 28h baseline.
POSTURE_RELATION_WORDS = frozenset({
    "sit", "sitting", "lie", "lying", "lean", "leaning",
    "kneel", "kneeling", "crouch", "crouching",
})

# IoU threshold below which a shared object is considered to have moved
# significantly between frames (strong causal signal).
BBOX_DISPLACEMENT_IOU = 0.3
# Larger batch for text-only mode (no images to encode).  A larger batch
# amortizes prompt/model overhead while still returning structured JSON.
# Measured on the current DB: 12 pairs give a ~986-token median / ~1107-token
# max prompt.  After removing unused output fields, a 512-token response and
# 256-token reserve still fit inside a 2048-token context.
SIGNALS_BATCH_SIZE = 12
DEFAULT_CAUSAL_WORKERS = 4


class CausalEdgeBuilder:
    """Causal inference with Ollama or OpenAI-compatible vLLM backend."""

    def __init__(self, ollama_url=None, model=None):
        # CAUSAL_API is a complete chat endpoint and therefore supports both:
        #   Ollama: http://127.0.0.1:11434/api/chat
        #   vLLM:   http://127.0.0.1:8000/v1/chat/completions
        # Leaving it unset preserves the original Ollama behavior.
        configured_api = os.getenv("CAUSAL_API", "").strip()
        if configured_api:
            self.api = configured_api.rstrip("/")
        else:
            base_url = (
                ollama_url
                or os.getenv("OLLAMA_BASE_URL", "").strip()
                or os.getenv("OPENAI_BASE_URL", "").strip()
                or "http://127.0.0.1:11434"
            )
            self.api = (
                base_url.replace("/v1", "").rstrip("/") + "/api/chat"
            )

        configured_backend = os.getenv("CAUSAL_BACKEND", "auto").strip().lower()
        path = self.api.lower()
        if configured_backend in {"vllm", "openai"}:
            self.backend = "openai"
        elif configured_backend == "ollama":
            self.backend = "ollama"
        elif path.endswith("/api/chat"):
            self.backend = "ollama"
        elif path.endswith("/chat/completions"):
            self.backend = "openai"
        else:
            # Backward-compatible default for callers that pass a bare URL.
            self.backend = "ollama"
            self.api = self.api.replace("/v1", "").rstrip("/") + "/api/chat"

        self.api_key = os.getenv("CAUSAL_API_KEY", "EMPTY").strip() or "EMPTY"
        # The causal path is text-only (rules_vlm sends structural signals, not
        # images), so it can use a smaller text model without changing QA/VLM.
        self.model = (
            model
            or os.getenv("CAUSAL_MODEL")
            or os.getenv("LLM_MODEL", "qwen3-vl:4b-instruct")
        )
        self.keep_alive = os.getenv("CAUSAL_KEEP_ALIVE", DEFAULT_KEEP_ALIVE)
        logger.info("Causal LLM backend: %s (%s)", self.backend, self.api)

    @staticmethod
    def _llm_options(num_predict: int, num_ctx=None) -> dict:
        """Build bounded Ollama generation options.

        ``num_predict`` caps generated tokens.  ``num_ctx`` caps the KV cache
        (prompt + generated tokens) and is the more important VRAM control
        when several requests run concurrently.
        """
        options = {
            "temperature": 0.0,
            "num_predict": max(64, int(num_predict)),
        }
        if num_ctx is not None:
            options["num_ctx"] = max(512, int(num_ctx))
        return options

    @staticmethod
    def _fit_output_budget(prompt: str, num_predict: int, num_ctx: int) -> int:
        """Reduce output tokens when the prompt already fills the context."""
        # Mixed English/JSON prompt: 3 characters/token is a conservative
        # estimate that avoids allocating a much larger context.
        prompt_tokens = max(1, len(prompt) // 3)
        reserve = max(128, int(os.getenv(
            "CAUSAL_CONTEXT_RESERVE", str(DEFAULT_SIGNALS_CONTEXT_RESERVE))))
        available = num_ctx - prompt_tokens - reserve
        if available <= 0:
            logger.warning(
                "Causal prompt may exceed num_ctx=%d; reduce CAUSAL_BATCH_SIZE",
                num_ctx)
            return 64
        return max(64, min(int(num_predict), available))

    @staticmethod
    def _openai_image_content(prompt: str, images=None):
        """Build OpenAI-compatible multimodal content for vLLM."""
        content = [{"type": "text", "text": prompt}]
        for image_b64 in images or []:
            # Qwen VL accepts standard data URLs.  Keyframes are JPEG in this
            # project; the data URL is required even when the bytes are passed.
            content.append({
                "type": "image_url",
                "image_url": {"url": "data:image/jpeg;base64," + image_b64},
            })
        return content

    @staticmethod
    def _extract_openai_content(data: dict) -> str:
        choices = data.get("choices") or []
        if not choices:
            raise ValueError("vLLM/OpenAI response has no choices")
        message = choices[0].get("message") or {}
        content = message.get("content", "")
        if isinstance(content, list):
            return "".join(
                str(part.get("text", "")) if isinstance(part, dict) else str(part)
                for part in content
            )
        return str(content or "")

    def _ollama_request_body(self, prompt, images, num_predict,
                             num_ctx, format_schema) -> dict:
        user_msg = {"role": "user", "content": prompt}
        if images:
            user_msg["images"] = images
        return {
            "model": self.model,
            "messages": [user_msg],
            "stream": False,
            "think": False,
            "keep_alive": self.keep_alive,
            "format": format_schema,
            "options": self._llm_options(num_predict, num_ctx),
        }

    def _openai_request_body(self, prompt, images, num_predict,
                             format_schema) -> dict:
        content = self._openai_image_content(prompt, images)
        body = {
            "model": self.model,
            "messages": [{"role": "user", "content": content}],
            "temperature": 0.0,
            "top_p": 1.0,
            "max_tokens": max(64, int(num_predict)),
            "stream": False,
        }

        # vLLM's OpenAI server supports JSON Schema through response_format.
        # The guided_json request form is kept selectable for older versions.
        guided = os.getenv("CAUSAL_VLLM_GUIDED", "response_format").strip().lower()
        if guided in {"off", "none", "0", "false"} or format_schema is None:
            return body
        schema = dict(format_schema)
        if guided in {"guided_json", "extra"}:
            body["guided_json"] = schema
        else:
            body["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "causal_edge_result",
                    "schema": schema,
                },
            }
        return body

    @staticmethod
    def _default_vlm_format_schema() -> dict:
        return {
            "type": "object",
            "properties": {
                "results": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "pair_id": {"type": "integer"},
                            "action_a": {"type": "string"},
                            "object_involved": {"type": "string"},
                            "state_before": {"type": "string"},
                            "state_after": {"type": "string"},
                            "cause_evidence": {"type": "string"},
                            "has_causal": {"type": "boolean"},
                            "edge_type": {"type": "string"},
                            "confidence": {"type": "number"},
                            "reason": {"type": "string"}
                        },
                        "required": [
                            "pair_id", "action_a", "object_involved",
                            "state_before", "state_after", "cause_evidence",
                            "has_causal", "edge_type", "confidence", "reason"
                        ],
                        "additionalProperties": False
                    }
                }
            },
            "required": ["results"],
            "additionalProperties": False
        }

    def _call_llm(self, prompt: str, images=None,
                  num_predict=DEFAULT_NUM_PREDICT, format_schema=None,
                  num_ctx=None, return_metadata=False):
        """Call Ollama or vLLM and optionally expose finish metadata.

        finish_reason=length proves malformed JSON came from truncation.
        """
        if format_schema is None:
            format_schema = self._default_vlm_format_schema()
        if self.backend == "openai":
            body_dict = self._openai_request_body(
                prompt, images, num_predict, format_schema)
        else:
            body_dict = self._ollama_request_body(
                prompt, images, num_predict, num_ctx, format_schema)

        body = json.dumps(body_dict).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self.backend == "openai":
            headers["Authorization"] = "Bearer " + self.api_key
        req = urllib.request.Request(self.api, data=body, headers=headers)
        timeout = max(30, int(os.getenv("CAUSAL_TIMEOUT", "120")))
        with urllib.request.urlopen(req, timeout=timeout) as r:
            data = json.loads(r.read().decode("utf-8"))

        if self.backend == "openai":
            content = self._extract_openai_content(data)
            choice = (data.get("choices") or [{}])[0]
            metadata = {"finish_reason": choice.get("finish_reason")}
        else:
            content = data.get("message", {}).get("content", "")
            metadata = {
                "finish_reason": data.get("done_reason"),
                "eval_count": data.get("eval_count"),
            }
        return (content, metadata) if return_metadata else content

    def _build_batch_prompt(self, pairs: List[dict], with_images=False,
                            image_refs=None) -> str:
        lines = ["You are a video understanding expert. Below are keyframe pairs from a video."]
        lines.append("For each pair, decide whether the state/action in frame A CAUSES or ENABLES the state in frame B.")
        lines.append("")
        lines.append("Rules:")
        lines.append("- Start from the actual images, not from the object/relation labels.")
        lines.append("- Identify a concrete physical action visible in frame A and the exact object it affects.")
        lines.append("- has_causal=true only if the same object changes from state_before to state_after because of action_a.")
        lines.append("- Use ENABLES if A creates a condition for B (e.g. opening fridge -> food visible).")
        lines.append("- Use LEADS_TO if A is a direct action causing B (e.g. cutting -> pieces appear).")
        lines.append("- has_causal=false if changes are just camera movement, viewpoint, detector flicker, or unrelated.")
        lines.append("- Object label changes alone are NOT evidence of causality. Judge from the actual images.")
        lines.append("- has_causal=false if the only change is an object appearing/disappearing, a person walking or moving to another place, or a scene change.")
        lines.append("- confidence is only a summary, not a substitute for visible evidence. If you cannot see action_a and the object state change, set has_causal=false.")
        lines.append("- Be conservative: if unsure, answer false.")
        lines.append("")
        lines.append("For each pair fill these fields:")
        lines.append("- action_a: the concrete physical action visible in frame A, e.g. \"hand lifts bottle\"; use \"none\" if no object manipulation is visible.")
        lines.append("- object_involved: the specific object whose state/position changes between A and B, e.g. \"bottle\" or \"fridge door\".")
        lines.append("- state_before: the object's visible state/position in frame A.")
        lines.append("- state_after: the object's visible state/position in frame B.")
        lines.append("- cause_evidence: the visible evidence connecting action_a to state_after.")
        lines.append("")
        if with_images:
            if image_refs:
                lines.append("Unique images are attached once in this order. [image k] refers to the k-th attached image (0-based).")
            else:
                lines.append("Images are attached in this order: [pair 0 A], [pair 0 B], [pair 1 A], [pair 1 B], ...")
            lines.append("")
        lines.append("=== Frame Pairs ===")
        for i, p in enumerate(pairs):
            if image_refs:
                ia, ib = image_refs[i]
                lines.append("[pair %d] time %.0fs -> %.0fs; images [%d] and [%d]"
                             % (i, p["t_a"], p["t_b"], ia, ib))
            else:
                lines.append("[pair %d] time %.0fs -> %.0fs" % (i, p["t_a"], p["t_b"]))
            if p["obj_added"]:
                lines.append("  A->B added objects: %s" % ", ".join(p["obj_added"]))
            if p["obj_removed"]:
                lines.append("  A->B removed objects: %s" % ", ".join(p["obj_removed"]))
            if p["rel_added"]:
                lines.append("  A->B new relations: %s" % ", ".join(p["rel_added"]))
            if p["rel_removed"]:
                lines.append("  A->B gone relations: %s" % ", ".join(p["rel_removed"]))
            lines.append("")
        lines.append("Return JSON with a results array. Each entry has pair_id (0-based), action_a, object_involved, state_before, state_after, cause_evidence, has_causal (bool), edge_type (\"LEADS_TO\" or \"ENABLES\" or \"NONE\"), confidence (0-1), reason.")
        lines.append("reason must be in Chinese: a concise sentence stating the causal relationship, e.g. 人打开冰箱门后拿出了瓶子.")
        lines.append("All other fields must be in English.")
        return "\n".join(lines)

    def _load_batch_images(self, pairs: List[dict]):
        """Return (base64_image_list, image_refs, all_images_available).

        Frames shared by consecutive pairs are attached only once; the prompt
        refers to them by index to avoid re-sending the same JPEG.
        """
        import base64
        paths = []
        path_to_index = {}
        image_refs = []
        for pair in pairs:
            refs = []
            for key in ("frame_path_a", "frame_path_b"):
                fp = pair.get(key)
                if not fp or not os.path.exists(fp):
                    return [], [], False
                if fp not in path_to_index:
                    path_to_index[fp] = len(paths)
                    paths.append(fp)
                refs.append(path_to_index[fp])
            image_refs.append(refs)

        images = []
        for fp in paths:
            try:
                with open(fp, "rb") as fh:
                    images.append(base64.b64encode(fh.read()).decode("utf-8"))
            except OSError:
                return [], [], False
        return images, image_refs, True

    def infer_batch(self, pairs: List[dict]) -> List[dict]:
        """Send a batch of frame pairs to the VLM, return verdicts."""
        images, image_refs, with_images = self._load_batch_images(pairs)
        prompt = self._build_batch_prompt(
            pairs, with_images=with_images, image_refs=image_refs)
        num_predict = max(128, int(os.getenv(
            "CAUSAL_VLM_NUM_PREDICT", str(DEFAULT_VLM_NUM_PREDICT))))
        num_ctx = max(512, int(os.getenv(
            "CAUSAL_VLM_NUM_CTX", str(DEFAULT_VLM_NUM_CTX))))
        num_predict = self._fit_output_budget(prompt, num_predict, num_ctx)
        try:
            raw = self._call_llm(
                prompt, images=images if with_images else None,
                num_predict=num_predict, num_ctx=num_ctx)
        except Exception as e:
            logger.warning("Causal LLM batch failed: %s", e)
            return []
        cleaned = re.sub(r"<think>.*?</think>", "", raw, flags=re.S).strip()
        try:
            obj = json.loads(cleaned)
            return obj.get("results", [])
        except (ValueError, TypeError) as e:
            logger.warning("Causal LLM JSON parse failed: %s", e)
            logger.debug("raw: %s", cleaned[:300])
            return []

    @staticmethod
    def _compute_iou(bbox_a, bbox_b):
        """IoU between two [x1, y1, x2, y2] bounding boxes."""
        if not bbox_a or not bbox_b or len(bbox_a) < 4 or len(bbox_b) < 4:
            return 1.0
        ax1, ay1, ax2, ay2 = bbox_a[:4]
        bx1, by1, bx2, by2 = bbox_b[:4]
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
        inter = iw * ih
        area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
        area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
        union = area_a + area_b - inter
        return inter / union if union > 0 else 0.0

    @staticmethod
    def _compute_scene_iou(raw_objects_a, raw_objects_b):
        """Compute IoU across ALL shared objects (including background).

        This is the core scene-stability check. When a camera moves, most
        objects shift simultaneously, producing low IoU across the board.
        When a person manipulates one object, the background stays put
        (high IoU) while only that object shifts.

        Returns dict with: total_shared, stable (IoU>=0.5), displaced
        (IoU<0.3), displacement_ratio, stable_ratio, per_object {label: iou}.
        """
        bboxes_a = {}
        for obj in raw_objects_a:
            label = (obj.get("label") or "").lower().strip()
            bbox = obj.get("bbox")
            if label and bbox:
                bboxes_a.setdefault(label, []).append(bbox)
        bboxes_b = {}
        for obj in raw_objects_b:
            label = (obj.get("label") or "").lower().strip()
            bbox = obj.get("bbox")
            if label and bbox:
                bboxes_b.setdefault(label, []).append(bbox)

        shared = set(bboxes_a) & set(bboxes_b)
        if not shared:
            return {"total_shared": 0, "stable": 0, "displaced": 0,
                    "displacement_ratio": 1.0, "stable_ratio": 0.0,
                    "per_object": {}}

        per_object = {}
        stable_count = 0
        displaced_count = 0
        for label in shared:
            best = 0.0
            for ba in bboxes_a[label]:
                for bb in bboxes_b[label]:
                    iou = CausalEdgeBuilder._compute_iou(ba, bb)
                    if iou > best:
                        best = iou
            per_object[label] = round(best, 2)
            if best >= 0.5:
                stable_count += 1
            elif best < 0.3:
                displaced_count += 1

        total = len(shared)
        return {
            "total_shared": total,
            "stable": stable_count,
            "displaced": displaced_count,
            "displacement_ratio": displaced_count / total,
            "stable_ratio": stable_count / total,
            "per_object": per_object,
        }

    def _extract_pair_signals(self, pair: dict) -> dict:
        """Extract structured causal signals with scene-stability gating.

        Key principle: displacement and count changes are only meaningful
        when the scene is stable (background objects maintain high IoU).
        Relation changes are always valid regardless of camera movement.
        """
        raw_a = pair.get("raw_objects_a") or []
        raw_b = pair.get("raw_objects_b") or []
        triples_a = pair.get("relation_triples_a") or []
        triples_b = pair.get("relation_triples_b") or []

        # Scene stability across ALL objects (background included)
        scene = self._compute_scene_iou(raw_a, raw_b)
        # Many imported observations contain labels/relations but no bounding
        # boxes.  Treating those pairs as an unstable scene discarded every
        # possible causal transition before rules/VLM could judge it.  Use a
        # conservative label-only fallback: at least two shared specific
        # objects anchor the scene and no more than four objects change.
        has_bbox = any(obj.get("bbox") for obj in (raw_a + raw_b)
                       if isinstance(obj, dict))
        if scene["total_shared"] < 2 and not has_bbox:
            labels_a = {
                str(obj.get("label") or obj).strip().lower()
                for obj in raw_a
            } or {str(value).strip().lower()
                  for value in (pair.get("objects_a") or [])}
            labels_b = {
                str(obj.get("label") or obj).strip().lower()
                for obj in raw_b
            } or {str(value).strip().lower()
                  for value in (pair.get("objects_b") or [])}
            labels_a -= GENERIC_OBJECT_LABELS
            labels_b -= GENERIC_OBJECT_LABELS
            shared_count = len(labels_a & labels_b)
            changed_count = len(labels_a.symmetric_difference(labels_b))
            scene = {
                "total_shared": shared_count,
                "stable": shared_count,
                "displaced": 0,
                "displacement_ratio": 0.0 if shared_count >= 2 and changed_count <= 4 else 1.0,
                "stable_ratio": 1.0 if shared_count >= 2 else 0.0,
                "per_object": {},
            }
        scene_is_stable = (
            scene["displacement_ratio"] <= 0.4 and scene["stable"] >= 2
        )

        # Relation state changes (strongest causal signal)
        rel_str_a = set()
        for t in triples_a:
            rel_str_a.add(("%s %s %s" % (
                t.get("subject") or "", t.get("predicate") or "",
                t.get("object") or "")).strip().lower())
        rel_str_b = set()
        for t in triples_b:
            rel_str_b.add(("%s %s %s" % (
                t.get("subject") or "", t.get("predicate") or "",
                t.get("object") or "")).strip().lower())
        rel_gone = sorted(rel_str_a - rel_str_b)
        rel_new = sorted(rel_str_b - rel_str_a)

        # Object displacement (only trusted when scene is stable)
        displaced = []
        if scene_is_stable:
            for label, iou in scene["per_object"].items():
                if iou < 0.3 and label not in GENERIC_OBJECT_LABELS:
                    displaced.append({"label": label, "iou": iou})

        # Count changes for ALL objects (no hardcoded exclusions).
        # Corroboration is enforced later: only count changes for objects
        # that ALSO have displacement or relation changes are causal.
        count_changes = []
        if scene_is_stable:
            counts_a = {}
            for obj in raw_a:
                label = (obj.get("label") or "").lower()
                if label:
                    counts_a[label] = counts_a.get(label, 0) + 1
            counts_b = {}
            for obj in raw_b:
                label = (obj.get("label") or "").lower()
                if label:
                    counts_b[label] = counts_b.get(label, 0) + 1
            for label in sorted(set(counts_a) & set(counts_b)):
                ca, cb = counts_a[label], counts_b[label]
                if ca != cb and label not in GENERIC_OBJECT_LABELS:
                    count_changes.append({"label": label, "a": ca, "b": cb})

        # Specific object appearance/disappearance (filtered)
        obj_new = [o for o in (pair.get("obj_added") or [])
                   if o.lower() not in GENERIC_OBJECT_LABELS]
        obj_gone = [o for o in (pair.get("obj_removed") or [])
                    if o.lower() not in GENERIC_OBJECT_LABELS]

        return {
            "displaced_objects": displaced,
            "rel_gone": rel_gone,
            "rel_new": rel_new,
            "count_changes": count_changes,
            "obj_new": obj_new,
            "obj_gone": obj_gone,
            "scene_stable": scene["stable"],
            "scene_total": scene["total_shared"],
            "scene_displacement_ratio": round(scene["displacement_ratio"], 2),
            "scene_is_stable": scene_is_stable,
        }

    @staticmethod
    def _corroborated_counts(signals: dict) -> list:
        """Only count changes for objects that also have displacement or
        relation changes.  Pure count changes (no corroboration) are treated
        as detection noise, regardless of the object type.
        """
        evidence_labels = set()
        for d in signals.get("displaced_objects", []):
            evidence_labels.add(d["label"])
        for r in signals.get("rel_gone", []) + signals.get("rel_new", []):
            for tok in r.replace("_", " ").split():
                evidence_labels.add(tok.lower())
        return [c for c in signals.get("count_changes", [])
                if c["label"] in evidence_labels]

    def _has_signals(self, signals: dict) -> bool:
        """True only when scene is stable AND there is a corroborated signal.

        Count changes alone never qualify. Only relation state changes and
        object displacement (gated by scene stability) are independent
        causal signals.
        """
        if not signals.get("scene_is_stable", False):
            return False
        return bool(
            signals["displaced_objects"]
            or signals["rel_gone"]
            or signals["rel_new"]
        )

    def _build_signals_prompt(self, pairs, signals_list) -> str:
        lines = ["You are a video understanding expert. For each keyframe pair, decide if there is a real causal relationship."]
        lines.append("")
        lines.append("Causality means: a human action in frame A directly causes an object state change in frame B.")
        lines.append("")
        lines.append("REAL causal signals:")
        lines.append("- Relation state change: e.g. person was holding X, now not holding X (or vice versa)")
        lines.append("- Object manipulation: person picks up, puts down, opens, closes, touches an object")
        lines.append("- Scene is stable (most objects unchanged) while 1-2 objects clearly change state")
        lines.append("")
        lines.append("NOT causal (always answer false):")
        lines.append("- Camera movement (many objects shift simultaneously, scene unstable)")
        lines.append("- Scene switch (many objects appear or disappear at once)")
        lines.append("- Person walking to a different location")
        lines.append("- Object detection flicker (same scene, slightly different labels)")
        lines.append("- Large furniture or appliances appearing/disappearing")
        lines.append("")
        lines.append("=== Frame Pair Signals ===")
        for i, (p, sig) in enumerate(zip(pairs, signals_list)):
            lines.append("[pair %d] time %.0fs -> %.0fs" % (i, p["t_a"], p["t_b"]))
            lines.append("  scene: %d/%d objects stable, displacement=%d%%" % (
                sig["scene_stable"], sig["scene_total"],
                int(sig["scene_displacement_ratio"] * 100)))
            if sig["rel_gone"]:
                lines.append("  relations ended: %s" % "; ".join(sig["rel_gone"]))
            if sig["rel_new"]:
                lines.append("  relations started: %s" % "; ".join(sig["rel_new"]))
            if sig["displaced_objects"]:
                items = ", ".join("%s(IoU=%.2f)" % (d["label"], d["iou"]) for d in sig["displaced_objects"])
                lines.append("  moved objects: %s" % items)
            if sig["count_changes"]:
                items = ", ".join("%s(%d->%d)" % (c["label"], c["a"], c["b"]) for c in sig["count_changes"])
                lines.append("  count changes: %s" % items)
            if sig["obj_new"]:
                lines.append("  appeared: %s" % ", ".join(sig["obj_new"]))
            if sig["obj_gone"]:
                lines.append("  disappeared: %s" % ", ".join(sig["obj_gone"]))
            if not self._has_signals(sig):
                lines.append("  (no significant change)")
            lines.append("")
        lines.append("Return exactly one compact JSON object per pair:")
        lines.append("{results: [{pair_id, has_causal, edge_type, confidence, reason}]}")
        lines.append("edge_type must be exactly LEADS_TO, ENABLES, or NONE.")
        lines.append("reason: Chinese, at most 12 characters; use 无 when has_causal=false.")
        lines.append("Do not add explanations, markdown, or missing fields.")
        return "\n".join(lines)

    @staticmethod
    def _parse_signals_response(raw: str, expected_count: int):
        """Parse complete JSON; otherwise salvage complete result rows."""
        text = re.sub(r"<think>.*?</think>", "", raw or "", flags=re.S).strip()
        fence = re.search(r"```(?:json)?\s*(.*?)```", text, flags=re.S)
        if fence:
            text = fence.group(1).strip()

        results = []
        parse_error = None
        try:
            obj = json.loads(text)
            results = obj.get("results", []) if isinstance(obj, dict) else obj
        except (ValueError, TypeError) as exc:
            parse_error = str(exc)
            # Rows are flat schema objects. Skip any malformed/truncated tail.
            for match in re.finditer(r"\{[^{}]*\}", text):
                try:
                    row = json.loads(match.group(0))
                except (ValueError, TypeError):
                    continue
                if isinstance(row, dict):
                    results.append(row)

        verdicts = []
        seen = set()
        for row in results:
            if not isinstance(row, dict):
                continue
            try:
                pair_id = int(row.get("pair_id", -1))
            except (TypeError, ValueError):
                continue
            if pair_id in seen or pair_id < 0 or pair_id >= expected_count:
                continue
            seen.add(pair_id)
            verdicts.append(row)
        missing = [i for i in range(expected_count) if i not in seen]
        return verdicts, missing, parse_error

    def infer_signals_batch(self, pairs) -> list:
        """Judge pairs without discarding a batch when JSON is truncated."""
        if not pairs:
            return []

        # Keep only fields consumed by the builder; enum + compact reason
        # greatly reduce output tokens compared with the old prompt/schema.
        signals_format = {
            "type": "object",
            "properties": {
                "results": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "pair_id": {"type": "integer"},
                            "has_causal": {"type": "boolean"},
                            "edge_type": {
                                "type": "string",
                                "enum": ["LEADS_TO", "ENABLES", "NONE"],
                            },
                            "confidence": {"type": "number"},
                            "reason": {"type": "string"}
                        },
                        "required": [
                            "pair_id", "has_causal", "edge_type",
                            "confidence", "reason"
                        ],
                        "additionalProperties": False
                    }
                }
            },
            "required": ["results"],
            "additionalProperties": False
        }

        num_predict = max(128, int(os.getenv(
            "CAUSAL_NUM_PREDICT", str(DEFAULT_SIGNALS_NUM_PREDICT))))
        num_ctx = max(512, int(os.getenv(
            "CAUSAL_NUM_CTX", str(DEFAULT_SIGNALS_NUM_CTX))))

        def ask(batch, output_limit, context_limit):
            signals_list = [
                p.get("_signals") or self._extract_pair_signals(p)
                for p in batch
            ]
            prompt = self._build_signals_prompt(batch, signals_list)
            fitted = self._fit_output_budget(prompt, output_limit, context_limit)
            return self._call_llm(
                prompt, images=None, num_predict=fitted,
                format_schema=signals_format, num_ctx=context_limit,
                return_metadata=True,
            )

        try:
            raw, metadata = ask(pairs, num_predict, num_ctx)
        except Exception as e:
            logger.warning("Causal signals LLM batch failed: %s", e)
            return []

        verdicts, missing, parse_error = self._parse_signals_response(
            raw, len(pairs))
        if missing:
            retry_predict = max(768, int(num_predict * 1.5))
            retry_ctx = max(3072, int(num_ctx))
            retry_size = max(1, int(os.getenv("CAUSAL_RETRY_BATCH_SIZE", "4")))
            for start in range(0, len(missing), retry_size):
                ids = missing[start:start + retry_size]
                retry_pairs = [pairs[i] for i in ids]
                try:
                    retry_raw, retry_meta = ask(
                        retry_pairs, retry_predict, retry_ctx)
                except Exception as e:
                    logger.warning(
                        "Causal signals LLM retry failed (%d pairs): %s",
                        len(retry_pairs), e)
                    continue

                retry_verdicts, retry_missing, retry_error = \
                    self._parse_signals_response(retry_raw, len(retry_pairs))
                for local_id, original_id in enumerate(ids):
                    for verdict in retry_verdicts:
                        if verdict.get("pair_id") == local_id:
                            verdict["pair_id"] = original_id
                            verdicts.append(verdict)
                            break
                if retry_missing:
                    logger.warning(
                        "Causal signals retry still missing pairs %s (%s, finish=%s)",
                        [ids[i] for i in retry_missing], retry_error,
                        retry_meta.get("finish_reason"))

            logger.warning(
                "Causal signals recovered %d/%d rows; finish=%s; parse_error=%s",
                len(verdicts), len(pairs), metadata.get("finish_reason"),
                parse_error)

        by_id = {}
        for verdict in verdicts:
            try:
                pair_id = int(verdict.get("pair_id", -1))
            except (TypeError, ValueError):
                continue
            if 0 <= pair_id < len(pairs):
                by_id.setdefault(pair_id, verdict)
        return [by_id[i] for i in sorted(by_id)]

    def _build_rule_reason(self, signals) -> tuple:
        """Generate causal reason from relation changes and displacement.

        Count changes are never included -- they are detection side
        effects, not causal evidence.
        """
        parts = []
        for rel in signals.get("rel_gone", []):
            parts.append(rel + " 结束")
        for rel in signals.get("rel_new", []):
            parts.append(rel + " 开始")
        for d in signals.get("displaced_objects", []):
            parts.append(d["label"] + " 被移动")
        reason = "，".join(parts[:2]) if parts else "状态变化"
        return "LEADS_TO", reason

    @staticmethod
    def _classify_relation_evidence(pair: dict) -> str:
        """Classify the strongest changed relation in a candidate pair."""
        import re
        relations = list(pair.get("rel_added") or [])
        relations += list(pair.get("rel_removed") or [])
        for relation in relations:
            words = set(re.findall(r"[a-z]+", str(relation).lower()))
            if words & ACTION_RELATION_WORDS:
                return "action"
        for relation in relations:
            words = set(re.findall(r"[a-z]+", str(relation).lower()))
            if words & POSTURE_RELATION_WORDS:
                return "posture"
        return "spatial_other"

    @staticmethod
    def _nearest_relation_effect_pairs(frame_pairs: list) -> list:
        """Keep the nearest target for repeated source+relation-delta effects.

        With CAUSAL_MAX_GAP=3 the same source and the same relation delta can
        be emitted against up to three later frames. Keeping the nearest one
        preserves the unique transition while removing lag duplicates. This
        is deliberately optional: it is not part of the original 28h baseline.
        """
        from collections import defaultdict
        groups = defaultdict(list)
        for index, pair in enumerate(frame_pairs):
            key = (
                pair.get("node_a"),
                tuple(pair.get("rel_added") or []),
                tuple(pair.get("rel_removed") or []),
            )
            groups[key].append((index, pair))

        selected = []
        for candidates in groups.values():
            _, pair = min(
                candidates,
                key=lambda item: (
                    float(item[1].get("t_b") or 0)
                    - float(item[1].get("t_a") or 0),
                    item[0],
                ),
            )
            selected.append(pair)
        selected.sort(key=lambda p: int(p.get("_candidate_index") or 0))
        return selected

    def _prefilter_causal_pair(self, pair: dict, mode: str,
                               relation_profile: str = "legacy") -> bool:
        """Pre-filter using scene-stability from bbox IoU.

        The key insight: when a camera moves, most shared objects show low
        IoU simultaneously. When a person manipulates an object, the
        background stays put while only that object shifts.

        Rejects pairs where:
        - fewer than 2 shared objects have usable bbox data
        - more than 40% of shared objects are displaced (camera movement)
        - fewer than 2 objects remain stable (scene not anchored)

        Falls back to label-based filtering when bbox data is unavailable.
        """
        if mode in ("off", "none", "all"):
            return True

        # The original 28h build used CAUSAL_PREFILTER=relation_evidence.
        # It keeps a pair only when an explicit subject-predicate-object
        # relation appears or disappears between the two frames.
        if mode in ("relation_evidence", "relation"):
            if not ((pair.get("rel_added") or [])
                    or (pair.get("rel_removed") or [])):
                return False
            if relation_profile in {"legacy", "all", "28h"}:
                return True

            tier = self._classify_relation_evidence(pair)
            pair["_relation_tier"] = tier
            if relation_profile in {"action", "action_only"}:
                return tier == "action"
            if relation_profile in {"action_posture", "human_state"}:
                return tier in {"action", "posture"}
            if relation_profile in {
                "action_corroborated", "corroborated", "hybrid"
            }:
                if tier == "action":
                    return True
                signals = self._extract_pair_signals(pair)
                pair["_signals"] = signals
                return bool(
                    signals.get("scene_is_stable")
                    and signals.get("displaced_objects")
                )
            # Unknown profiles preserve the legacy behavior.
            return True

        raw_a = pair.get("raw_objects_a") or []
        raw_b = pair.get("raw_objects_b") or []

        if raw_a and raw_b:
            scene = CausalEdgeBuilder._compute_scene_iou(raw_a, raw_b)
            if scene["total_shared"] < 2:
                return False
            if scene["displacement_ratio"] > 0.4:
                return False
            if scene["stable"] < 2:
                return False
            return True

        # Fallback: label-based stability
        objects_a = pair.get("objects_a") or list(pair.get("obj_removed") or [])
        objects_b = pair.get("objects_b") or list(pair.get("obj_added") or [])
        specific_a = {o.lower() for o in objects_a
                      if o.lower() not in GENERIC_OBJECT_LABELS}
        specific_b = {o.lower() for o in objects_b
                      if o.lower() not in GENERIC_OBJECT_LABELS}
        shared = specific_a & specific_b
        changed = specific_a.symmetric_difference(specific_b)
        if len(shared) < 2:
            return False
        if len(changed) > 4:
            return False
        return True

    @staticmethod
    def _is_credible_causal_pair(pair: dict, verdict: dict) -> bool:
        """Validate VLM verdict via structured fields, not keyword matching.

        Precision comes from the structured output itself: the VLM must
        name a specific object, describe its state transition, and cite
        visible evidence. Language-agnostic and far more robust than
        matching keywords in free-text reason strings.
        """
        fields = ("action_a", "object_involved", "state_before",
                  "state_after", "cause_evidence")
        values = {}
        for field in fields:
            value = (verdict.get(field) or "").strip()
            if value.lower() in EMPTY_CAUSAL_FIELDS:
                return False
            values[field] = value

        if values["state_before"].lower() == values["state_after"].lower():
            return False

        object_low = values["object_involved"].lower()
        object_tokens = set(re.findall(r"[a-z0-9]+|[\u4e00-\u9fff]+", object_low))
        if not object_tokens or not (object_tokens - NON_SPECIFIC_OBJECT_LABELS):
            return False

        objects_a = pair.get("objects_a") or list(pair.get("obj_removed") or [])
        objects_b = pair.get("objects_b") or list(pair.get("obj_added") or [])
        relations_a = pair.get("relations_a") or list(pair.get("rel_removed") or [])
        relations_b = pair.get("relations_b") or list(pair.get("rel_added") or [])

        data_a = " ".join(objects_a + relations_a).lower()
        data_b = " ".join(objects_b + relations_b).lower()
        tokens_a = set(re.findall(r"[a-z0-9]+|[\u4e00-\u9fff]+", data_a))
        tokens_b = set(re.findall(r"[a-z0-9]+|[\u4e00-\u9fff]+", data_b))
        return bool(object_tokens & tokens_a and object_tokens & tokens_b)

    def build_causal_edges(self, frame_pairs: List[dict], graph_db,
                           batch_size=None) -> dict:
        """Infer and insert causal edges into the graph.

        CAUSAL_METHOD env var selects the inference engine:
          - "rules_vlm" (default): rules extract signals, LLM judges (no images)
          - "vlm": VLM sees frame images (slowest)
          - "rules": pure rule-based, no LLM (fastest)

        Args:
            frame_pairs: list of dicts from extract_causal_pairs()
            graph_db: the NetworkXGraphDB / SQLiteGraphDB
            batch_size: pairs per LLM call (auto-selected if None)

        Returns stats dict.
        """
        from .graph_db import Link, LinkType

        method = os.getenv("CAUSAL_METHOD", "rules_vlm").strip().lower()
        if batch_size is None:
            default_batch = SIGNALS_BATCH_SIZE if method != "vlm" else DEFAULT_BATCH_SIZE
            batch_size = max(1, int(os.getenv("CAUSAL_BATCH_SIZE", str(default_batch))))

        candidate_pairs = len(frame_pairs)
        prefilter_mode = os.getenv("CAUSAL_PREFILTER", "continuity").strip().lower()
        relation_profile = os.getenv(
            "CAUSAL_RELATION_EVIDENCE", "legacy").strip().lower()
        effect_dedup = os.getenv(
            "CAUSAL_EFFECT_DEDUP", "off").strip().lower()
        filter_profile = os.getenv(
            "CAUSAL_FILTER_PROFILE", "strict").strip().lower()
        legacy_filter = filter_profile in {
            "legacy", "legacy_28h", "28h", "relation_evidence"
        }
        no_signal_pairs = 0
        signal_filter_started = time.perf_counter()

        if method == "vlm" or legacy_filter:
            retained = []
            for index, pair in enumerate(frame_pairs):
                pair["_candidate_index"] = index
                if self._prefilter_causal_pair(
                        pair, prefilter_mode, relation_profile):
                    retained.append(pair)
            frame_pairs = retained
            prefiltered_out = candidate_pairs - len(frame_pairs)
            no_signal_pairs = prefiltered_out
        else:
            # rules / rules_vlm only need pairs that already contain a causal
            # signal.  Calling the LLM for scene-stable but signal-less pairs
            # was one of the major full-build time sinks.
            retained = []
            for pair in frame_pairs:
                signals = self._extract_pair_signals(pair)
                pair["_signals"] = signals
                if self._has_signals(signals):
                    retained.append(pair)
                else:
                    no_signal_pairs += 1
            frame_pairs = retained
            prefiltered_out = candidate_pairs - len(frame_pairs)

        effect_duplicates = 0
        if effect_dedup in {"nearest", "1", "true", "yes", "on"}:
            before_dedup = len(frame_pairs)
            frame_pairs = self._nearest_relation_effect_pairs(frame_pairs)
            effect_duplicates = before_dedup - len(frame_pairs)

        stats = {"total_pairs": len(frame_pairs), "causal_found": 0,
                 "batches": 0, "edges_added": 0, "rejected_non_causal": 0,
                 "llm_judged_pairs": 0, "llm_true_pairs": 0,
                 "llm_false_pairs": 0, "llm_missing_pairs": 0,
                 "candidate_pairs": candidate_pairs,
                 "prefiltered_pairs": prefiltered_out,
                 "prefilter_mode": prefilter_mode,
                 "relation_evidence_profile": relation_profile,
                 "effect_dedup": effect_dedup,
                 "effect_duplicates": effect_duplicates,
                 "filter_profile": (
                     "legacy_28h" if legacy_filter else "strict"),
                 "method": method, "no_signal_pairs": no_signal_pairs,
                 "signal_filter_seconds": round(time.perf_counter() - signal_filter_started, 3)}
        if prefiltered_out:
            logger.info("Causal prefilter %s: %d -> %d pairs (%d skipped, %d no signal)",
                        prefilter_mode, candidate_pairs, len(frame_pairs),
                        prefiltered_out, no_signal_pairs)
        default_workers = DEFAULT_VLM_WORKERS if method == "vlm" else DEFAULT_CAUSAL_WORKERS
        workers = max(1, int(os.getenv("CAUSAL_WORKERS", str(default_workers))))
        logger.info("Causal method: %s, batch_size: %d, workers: %d",
                    method, batch_size, workers)

        if method == "rules":
            return self._build_rules(frame_pairs, graph_db, stats)
        elif method == "vlm":
            return self._build_vlm(frame_pairs, graph_db, batch_size, stats, workers)
        else:
            return self._build_rules_vlm(
                frame_pairs, graph_db, batch_size, stats, workers)

    def _build_rules_vlm(self, frame_pairs, graph_db, batch_size, stats,
                         workers=1):
        """Rules extract signals -> concurrent LLM causality judgment."""
        from .graph_db import Link, LinkType

        batches = [
            frame_pairs[start:start + batch_size]
            for start in range(0, len(frame_pairs), batch_size)
        ]
        total_batches = len(batches)
        llm_started = time.perf_counter()

        def run_batch(batch):
            return batch, self.infer_signals_batch(batch)

        if workers <= 1:
            iterator = map(run_batch, batches)
            futures = None
        else:
            executor = ThreadPoolExecutor(max_workers=workers,
                                          thread_name_prefix="causal")
            future_map = {executor.submit(run_batch, batch): batch
                          for batch in batches}
            futures = as_completed(future_map)

        try:
            items = iterator if futures is None else futures
            for item in items:
                if futures is None:
                    batch, verdicts = item
                else:
                    future = item
                    batch = future_map[future]
                    batch, verdicts = future.result()
                stats["batches"] += 1

                stats["llm_judged_pairs"] += len(verdicts)
                stats["llm_missing_pairs"] += len(batch) - len(verdicts)
                for v in verdicts:
                    pid = v.get("pair_id", -1)
                    if pid < 0 or pid >= len(batch):
                        continue
                    if v.get("has_causal", False):
                        stats["llm_true_pairs"] += 1
                    else:
                        stats["llm_false_pairs"] += 1
                    if not v.get("has_causal", False):
                        continue
                    pair = batch[pid]
                    edge_type_raw = (v.get("edge_type") or "LEADS_TO").upper()
                    if "ENABLE" in edge_type_raw:
                        edge_type = "ENABLES"
                    elif "NONE" in edge_type_raw or "FALSE" in edge_type_raw:
                        continue
                    else:
                        edge_type = "LEADS_TO"
                    try:
                        confidence = float(v.get("confidence") or 0.0)
                    except (TypeError, ValueError):
                        confidence = 0.0

                    graph_db.add_link(Link(
                        source_node_id=pair["node_a"],
                        target_node_id=pair["node_b"],
                        link_type=LinkType.CAUSAL,
                        properties={
                            "sub_type": edge_type,
                            "confidence": confidence,
                            "reason": v.get("reason", ""),
                        },
                    ))
                    stats["causal_found"] += 1
                    stats["edges_added"] += 1

                if stats["batches"] % 10 == 0 or stats["batches"] == total_batches:
                    logger.info("Causal batch %d/%d (%s): %d edges so far",
                                stats["batches"], total_batches,
                                "rules_vlm", stats["edges_added"])
        finally:
            if futures is not None:
                executor.shutdown(wait=True, cancel_futures=True)
        stats["llm_seconds"] = round(time.perf_counter() - llm_started, 3)
        if stats["llm_seconds"] > 0:
            stats["pairs_per_second"] = round(
                len(frame_pairs) / stats["llm_seconds"], 3)
            stats["batches_per_second"] = round(
                total_batches / stats["llm_seconds"], 3)
        return stats

    def _build_vlm(self, frame_pairs, graph_db, batch_size, stats,
                   workers=1):
        """VLM sees frame images, with concurrent batch execution."""
        from .graph_db import Link, LinkType

        batches = [
            frame_pairs[start:start + batch_size]
            for start in range(0, len(frame_pairs), batch_size)
        ]
        total_batches = len(batches)
        if workers > 1:
            executor = ThreadPoolExecutor(max_workers=workers,
                                          thread_name_prefix="causal-vlm")
            future_map = {executor.submit(self.infer_batch, batch): batch
                          for batch in batches}
            iterator = ((future_map[f], f.result()) for f in as_completed(future_map))
        else:
            iterator = ((batch, self.infer_batch(batch)) for batch in batches)

        try:
            for batch, verdicts in iterator:
                stats["batches"] += 1

                stats["llm_judged_pairs"] += len(verdicts)
                stats["llm_missing_pairs"] += len(batch) - len(verdicts)
                for v in verdicts:
                    pid = v.get("pair_id", -1)
                    if pid < 0 or pid >= len(batch):
                        continue
                    if v.get("has_causal", False):
                        stats["llm_true_pairs"] += 1
                    else:
                        stats["llm_false_pairs"] += 1
                    if not v.get("has_causal", False):
                        continue
                    pair = batch[pid]
                    if not self._is_credible_causal_pair(pair, v):
                        stats["rejected_non_causal"] += 1
                        continue
                    edge_type_raw = (v.get("edge_type") or "LEADS_TO").upper()
                    if "ENABLE" in edge_type_raw:
                        edge_type = "ENABLES"
                    elif "NONE" in edge_type_raw or "FALSE" in edge_type_raw:
                        continue
                    else:
                        edge_type = "LEADS_TO"
                    try:
                        confidence = float(v.get("confidence") or 0.0)
                    except (TypeError, ValueError):
                        confidence = 0.0

                    graph_db.add_link(Link(
                        source_node_id=pair["node_a"],
                        target_node_id=pair["node_b"],
                        link_type=LinkType.CAUSAL,
                        properties={
                            "sub_type": edge_type,
                            "confidence": confidence,
                            "reason": v.get("reason", ""),
                        },
                    ))
                    stats["causal_found"] += 1
                    stats["edges_added"] += 1

                if stats["batches"] % 10 == 0 or stats["batches"] == total_batches:
                    logger.info("Causal batch %d/%d (vlm): %d edges so far",
                                stats["batches"], total_batches,
                                stats["edges_added"])
        finally:
            if workers > 1:
                executor.shutdown(wait=True, cancel_futures=True)
        return stats

    def _build_rules(self, frame_pairs, graph_db, stats):
        """Pure rule-based causal edges, no LLM calls."""
        from .graph_db import Link, LinkType

        for pair in frame_pairs:
            signals = pair.get("_signals") or self._extract_pair_signals(pair)
            if not self._has_signals(signals):
                continue
            # Require at least bbox displacement or relation change
            if not signals["displaced_objects"] and not (signals["rel_gone"] or signals["rel_new"]):
                continue
            edge_type, reason = self._build_rule_reason(signals)
            graph_db.add_link(Link(
                source_node_id=pair["node_a"],
                target_node_id=pair["node_b"],
                link_type=LinkType.CAUSAL,
                properties={
                    "sub_type": edge_type,
                    "confidence": 0.8,
                    "reason": reason,
                },
            ))
            stats["causal_found"] += 1
            stats["edges_added"] += 1

        logger.info("Causal rules: %d edges from %d pairs",
                    stats["edges_added"], stats["total_pairs"])
        return stats


def _is_meaningful_change(obj_added, obj_removed, rel_added, rel_removed) -> bool:
    """True when a relation changed or a non-background object changed."""
    if rel_added or rel_removed:
        return True
    specific_added = [o for o in obj_added if o not in GENERIC_OBJECT_LABELS]
    specific_removed = [o for o in obj_removed if o not in GENERIC_OBJECT_LABELS]
    return bool(specific_added or specific_removed)


def extract_causal_pairs(graph_db, max_gap=None, max_seconds=None) -> List[dict]:
    """Extract candidate causal pairs within the same video.

    By default this looks up to CAUSAL_DEFAULT_MAX_GAP frames ahead and up to
    CAUSAL_DEFAULT_MAX_SECONDS seconds ahead, automatically crossing clip
    boundaries inside the same video. It never creates cross-video pairs.
    """
    from collections import defaultdict

    if max_gap is None:
        max_gap = max(1, int(os.getenv("CAUSAL_MAX_GAP", str(CAUSAL_DEFAULT_MAX_GAP))))
    if max_seconds is None:
        max_seconds = float(os.getenv("CAUSAL_MAX_SECONDS", str(CAUSAL_DEFAULT_MAX_SECONDS)))

    clips = defaultdict(list)
    for nid, node in graph_db.nodes.items():
        if node.node_type.value != "EVENT":
            continue
        video_uid = node.attributes.get("video_uid")
        if video_uid:
            clips[video_uid].append(node)

    pairs = []
    for video_uid, clist in clips.items():
        clist.sort(key=lambda n: (
            n.attributes.get("video_time_sec")
            if n.attributes.get("video_time_sec") is not None
            else (n.attributes.get("clip_time_sec") or 0),
            n.attributes.get("clip_uid") or "",
            n.attributes.get("frame_seq") or 0,
        ))
        for i in range(len(clist) - 1):
            for gap in range(1, max_gap + 1):
                j = i + gap
                if j >= len(clist):
                    break
                a, b = clist[i], clist[j]
                t_a = (a.attributes.get("video_time_sec")
                       if a.attributes.get("video_time_sec") is not None
                       else (a.attributes.get("clip_time_sec") or 0))
                t_b = (b.attributes.get("video_time_sec")
                       if b.attributes.get("video_time_sec") is not None
                       else (b.attributes.get("clip_time_sec") or 0))
                if t_b - t_a > max_seconds:
                    break
                oa = set(a.attributes.get("object_labels") or [])
                ob = set(b.attributes.get("object_labels") or [])
                ra = set(a.attributes.get("relation_labels") or [])
                rb = set(b.attributes.get("relation_labels") or [])
                if oa == ob and ra == rb:
                    continue
                obj_added = sorted(ob - oa)
                obj_removed = sorted(oa - ob)
                rel_added = sorted(rb - ra)
                rel_removed = sorted(ra - rb)
                if not _is_meaningful_change(
                        obj_added, obj_removed, rel_added, rel_removed):
                    continue
                pairs.append({
                    "node_a": a.node_id,
                    "node_b": b.node_id,
                    "objects_a": sorted(oa),
                    "objects_b": sorted(ob),
                    "raw_objects_a": list(a.attributes.get("objects") or []),
                    "raw_objects_b": list(b.attributes.get("objects") or []),
                    "relations_a": sorted(ra),
                    "relations_b": sorted(rb),
                    "relation_triples_a": list(a.attributes.get("relations") or []),
                    "relation_triples_b": list(b.attributes.get("relations") or []),
                    "t_a": t_a,
                    "t_b": t_b,
                    "obj_added": obj_added,
                    "obj_removed": obj_removed,
                    "rel_added": rel_added,
                    "rel_removed": rel_removed,
                    "frame_path_a": a.attributes.get("frame_path"),
                    "frame_path_b": b.attributes.get("frame_path"),
                })
    return pairs
