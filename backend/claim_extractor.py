"""Independent claim scan performed after the Writer returns text.

Two extractors are exposed:

* :class:`ClaimExtractor` — regex-based span slicer used for structural
  fallback and simple-path answers.  It must never carry semantic
  classification — sentence type stays a best-effort hint.
* :class:`LLMClaimExtractor` — model-driven scanner following plan §7.7.
  Used by complex paths (person summary, cross-event patterns, comparisons).
"""

import json
import re

from .model_clients import parse_json_response


_CLAIM_EXTRACTOR_MARKER = "Claim Extractor"


_LLM_CLAIM_PROMPT = """你是独立的家庭事实 Claim Extractor。
扫描完整回答正文和 follow_up，不信任 Writer 提供的 claim 列表。
抽取所有可能涉及家庭人物、时间、地点、事件、物品、衣着、关系、偏好或否定证据的主张。
"没有证据""记录不足""没有找到"也要抽取，因为它们可能错误忽略已有证据。
不要验证，不要补充事实，只返回原文 span 和主张类型。

完整回答：{{answer_text}}
Writer 候选：{{writer_candidates}}

只输出 JSON：
{
  "claims": [
    {
      "claim_id": "claim_1",
      "text": "完整原文片段",
      "start": 0,
      "end": 0,
      "intended_type": "fact|derived_pattern|inference|negative|uncertainty"
    }
  ]
}"""


class ClaimExtractor:
    def scan(self, text, writer_candidates=None):
        value = str(text or "")
        candidates = list(writer_candidates or [])
        claims = []
        start = 0
        for match in re.finditer(r"[^。！？!?；;\n]+", value):
            sentence = match.group(0).strip()
            if not sentence:
                continue
            sentence_start = match.start() + len(match.group(0)) - len(match.group(0).lstrip())
            sentence_end = sentence_start + len(sentence)
            candidate = next((item for item in candidates if str(item.get("text") or "").strip() in sentence), None)
            claims.append({
                "claim_id": f"claim_{len(claims) + 1}",
                "start": sentence_start,
                "end": sentence_end,
                "text": sentence,
                # ``intended_type`` from the regex extractor stays a hint only.
                # Semantic classification lives in :class:`LLMClaimExtractor`.
                "intended_type": "family_fact",
                "candidate_evidence_ids": list((candidate or {}).get("candidate_evidence_ids") or []),
            })
        return claims


class LLMClaimExtractor:
    """Model-driven claim scan for complex Thin Agent answers (plan §7.7).

    The extractor never verifies claims and never invents evidence IDs.  When
    the model is unavailable or returns invalid JSON, it returns an empty
    claim list — callers must then fall back to a structured safe answer,
    never to unverified free text.
    """

    def __init__(self, gamma=None):
        self.gamma = gamma

    def scan(self, answer_text, writer_candidates=None):
        text = str(answer_text or "")
        candidates = list(writer_candidates or [])
        if not self.gamma or not hasattr(self.gamma, "chat"):
            return []
        prompt = _LLM_CLAIM_PROMPT.replace("{{answer_text}}", text).replace(
            "{{writer_candidates}}",
            json.dumps([{"text": item.get("text"), "candidate_evidence_ids": list(item.get("candidate_evidence_ids") or [])} for item in candidates], ensure_ascii=False),
        )
        try:
            raw = self.gamma.chat(prompt, json_mode=True)
        except Exception:
            return []
        parsed = parse_json_response(raw)
        claims = parsed.get("claims") if isinstance(parsed, dict) else None
        if not isinstance(claims, list):
            return []
        cleaned = []
        for index, claim in enumerate(claims, 1):
            if not isinstance(claim, dict):
                continue
            claim_text = str(claim.get("text") or "").strip()
            if not claim_text:
                continue
            cleaned.append({
                "claim_id": claim.get("claim_id") or f"claim_{index}",
                "text": claim_text,
                "start": int(claim.get("start", 0) or 0),
                "end": int(claim.get("end", 0) or 0),
                "intended_type": claim.get("intended_type") or "fact",
                "candidate_evidence_ids": [],
            })
        return cleaned
