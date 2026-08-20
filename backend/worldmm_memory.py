"""WorldMM-style memory construction and retrieval for precomputed keyframes.

The keyframe extractor remains upstream.  This module only consumes its WebP
representatives and sidecars, then maintains separate caption/audio, episodic,
semantic and visual memories with late fusion at QA time.
"""

from __future__ import annotations

import base64
import json
import re
import time
from pathlib import Path


TOKEN_RE = re.compile(r"[a-z0-9]+|[\u4e00-\u9fff]")
STOP = {
    "the", "a", "an", "what", "when", "where", "which", "who", "how",
    "was", "were", "does", "did", "is", "are", "in", "on", "to", "of",
    "and", "or", "with", "for", "during", "while", "main", "character",
    "visually", "shown", "scene", "video", "背景", "什么", "哪个", "什么", "时候",
}


def _tokens(value):
    return {item for item in TOKEN_RE.findall(str(value or "").lower()) if item not in STOP and len(item) > 1}


def _text(item):
    return " ".join(str(item.get(key) or "") for key in (
        "text", "caption", "summary", "title", "activity", "objects", "actions",
        "transcript", "semantic", "labels", "question_context",
    ))


def _overlap(query, item):
    q = _tokens(query)
    t = _tokens(_text(item))
    if not q or not t:
        return 0.0
    exact = len(q & t)
    # A small exact phrase bonus helps options such as "green copper dome".
    raw_q = str(query or "").lower()
    raw_t = _text(item).lower()
    phrase = sum(1 for token in re.findall(r"[a-z][a-z -]{3,}", raw_q) if token.strip() in raw_t)
    return exact / max(1.0, len(q) ** 0.5) + 0.9 * phrase


def _mime(path):
    return "image/webp" if str(path).lower().endswith(".webp") else "image/jpeg"


def _image_payload(path):
    try:
        data = Path(path).read_bytes()
    except OSError:
        return None
    return {"base64": base64.b64encode(data).decode("ascii"), "mime_type": _mime(path), "path": str(path)}


class WorldMMMemory:
    """Loads a persisted WorldMM-style artifact and performs late-fusion QA."""

    def __init__(self, artifact_path):
        self.artifact_path = Path(artifact_path)
        self.data = json.loads(self.artifact_path.read_text(encoding="utf-8"))
        self.caption = list(self.data.get("caption_memory") or [])
        self.audio = list(self.data.get("audio_memory") or [])
        self.episodic = list(self.data.get("episodic_memory") or [])
        self.semantic = list(self.data.get("semantic_memory") or [])
        self.visual = list(self.data.get("visual_memory") or [])

    @property
    def manifest(self):
        return self.data.get("manifest") or {}

    def retrieve(self, question, options=None, limit=5):
        started = time.perf_counter()
        options = options if isinstance(options, dict) else {}
        option_text = " ".join(f"{key} {value}" for key, value in options.items())
        query = f"{question} {option_text}".strip()
        qlow = query.lower()
        wants_audio = any(word in qlow for word in ("hear", "heard", "sound", "audible", "audio", "say", "said", "mention", "discuss", "reaction", "what did"))
        wants_visual = any(word in qlow for word in ("visual", "shown", "visible", "wear", "wearing", "color", "colour", "look", "depicted", "placing", "using", "architectural", "what is"))
        wants_summary = any(word in qlow for word in ("overall", "plan", "hobby", "breakfast", "trip", "challenge", "experience", "enjoy", "activity", "summarization"))
        weights = {
            "caption": 1.0, "audio": 1.15 if wants_audio else 0.65,
            "episodic": 1.15 if wants_summary else 0.9,
            "semantic": 1.0 if wants_summary else 0.85,
            "visual": 1.2 if wants_visual else 0.65,
        }
        channels = {
            "caption": self.caption, "audio": self.audio, "episodic": self.episodic,
            "semantic": self.semantic, "visual": self.visual,
        }
        hits = {}
        for name, rows in channels.items():
            ranked = []
            for row in rows:
                score = _overlap(query, row) * weights[name]
                if score > 0:
                    ranked.append((score, row))
            ranked.sort(key=lambda pair: pair[0], reverse=True)
            # A visual memory must still contribute candidates when its text
            # caption is sparse; the attached image is the retrieval payload.
            if not ranked:
                fallback = rows[: (max(limit, 8) if name == "visual" and wants_visual else limit)]
                ranked = [(0.01, row) for row in fallback]
            hits[name] = [{**row, "retrieval_score": round(score, 4), "memory_type": name}
                          for score, row in ranked[: (max(limit, 8) if name == "visual" and wants_visual else limit)]]
        # Add the best time-neighbouring visual frame to a retrieved episode.
        visual_by_id = {str(item.get("id")): item for item in self.visual}
        selected_visual = []
        for row in hits["caption"] + hits["episodic"] + hits["audio"]:
            for candidate in self.visual:
                if row.get("event_id") and row.get("event_id") == candidate.get("event_id"):
                    selected_visual.append(candidate)
        selected_visual.extend(hits["visual"])
        seen = set()
        images = []
        visual_hits = []
        for row in selected_visual:
            key = str(row.get("id"))
            if key in seen:
                continue
            seen.add(key)
            visual_hits.append(row)
            payload = _image_payload(row.get("image_path"))
            if payload:
                images.append(payload)
            if len(images) >= 4:
                break
        hits["visual_context"] = visual_hits[:limit]
        hits["images"] = images
        hits["latency_ms"] = round((time.perf_counter() - started) * 1000, 2)
        return hits

    def context_text(self, hits, max_chars=12000):
        sections = []
        for name in ("audio", "caption", "episodic", "semantic", "visual_context"):
            rows = hits.get(name) or []
            if not rows:
                continue
            lines = []
            for row in rows:
                stamp = f"{row.get('start_sec', row.get('timestamp_sec', ''))}-{row.get('end_sec', '')}s"
                lines.append(f"[{name} {row.get('id')} {stamp}] {_text(row)}")
            sections.append("\n".join(lines))
        return "\n\n".join(sections)[:max_chars]

    def answer(self, gamma, question, options=None):
        options = options if isinstance(options, dict) else {}
        hits = self.retrieve(question, options=options)
        context = self.context_text(hits)
        if options:
            option_lines = "\n".join(f"{key}: {value}" for key, value in options.items())
            prompt = (
                "You are the WorldMM response agent. Answer the multiple-choice question using only the "
                "retrieved caption/audio/episodic/semantic/visual evidence. Images are attached when available. "
                "Return JSON only with answer_letter, answer, explanation, confidence. Do not guess beyond evidence.\n"
                f"Question: {question}\nOptions:\n{option_lines}\nRetrieved memory:\n{context}"
            )
        else:
            prompt = (
                "你是 WorldMM 视频记忆回答器。只能根据检索到的记忆和图片回答，不确定就明确说明证据不足。"
                "请返回 JSON：answer、explanation、confidence。\n"
                f"问题：{question}\n检索记忆：\n{context}"
            )
        try:
            raw = gamma.chat(prompt, images=hits.get("images") or None, json_mode=True, role="answer")
            from .model_clients import parse_json_response
            parsed = parse_json_response(raw)
        except Exception as error:
            parsed = {"answer": "证据不足，暂时无法可靠回答。", "explanation": str(error), "confidence": 0.2}
        answer = str(parsed.get("answer") or parsed.get("answer_letter") or "").strip()
        letter = str(parsed.get("answer_letter") or "").strip().upper()
        if options and letter not in options:
            match = re.search(r"\b([A-Z])\b", answer.upper())
            letter = match.group(1) if match and match.group(1) in options else ""
        if options and letter:
            answer = letter
        return {
            "answer": answer or "证据不足，暂时无法可靠回答。",
            "explanation": str(parsed.get("explanation") or ""),
            "confidence": parsed.get("confidence", 0.35),
            "evidence": [
                {"id": row.get("id"), "memory_type": row.get("memory_type"),
                 "start_sec": row.get("start_sec", row.get("timestamp_sec")),
                 "end_sec": row.get("end_sec"), "text": _text(row)[:500],
                 "image_path": row.get("image_path")}
                for name in ("audio", "caption", "episodic", "semantic", "visual_context")
                for row in (hits.get(name) or [])[:3]
            ],
            "retrieval": {name: len(rows) for name, rows in hits.items() if isinstance(rows, list)},
            "retrieval_timing": {"fusion_ms": hits.get("latency_ms", 0)},
            "raw_model_response": raw if 'raw' in locals() else "",
        }


def worldmm_artifact_path(data_dir, scope_id):
    return Path(data_dir) / "derived" / "worldmm-memory" / str(scope_id) / "memory.json"
