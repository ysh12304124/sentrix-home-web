"""Complex-path answer generation for Thin Agent (Phase 4).

Chain (plan §4.2):

    NarrativeContextPacket -> Writer -> LLMClaimExtractor -> verify_claims
    -> repair_answer (<=1) -> re-verify -> deterministic fallback if failed

Every claim in the final answer must be either verified against the canonical
evidence bundle (from :func:`evidence_retrieval.build_verifier_evidence_bundle`)
or explicitly downgraded / removed.  Free-text without evidence support is
never returned.
"""

from __future__ import annotations

import json

from .agent_contracts import repair_answer, verify_claims
from .claim_extractor import LLMClaimExtractor
from .evidence_retrieval import build_verifier_evidence_bundle
from .model_clients import parse_json_response
from .narrative_context import build_narrative_context_packet


_WRITER_PROMPT = """你是 Sentrix 的人物叙事 Writer。
请根据 NarrativeContextPacket 写一段自然的人物介绍，不要罗列检索日志。

必须区分：
- confirmed_fact：可以直接陈述；
- user_assertion：说明"你之前确认/告诉我"；
- observed_pattern：说明"现有多次记录中"；
- agent_inference：使用"给人的印象/可能"；
- unknown/contradicted：不要当作事实。

人物介绍应优先归纳：家庭角色、常见活动、反复出现的地点/场景、可观察外观和穿着模式、可能的性格印象。
没有足够证据的性格、偏好和关系必须明确保留未知。
不要新增 Evidence ID，不要把一个 Event 的内容推广成稳定特征。

用户问题：{{message}}
NarrativeContextPacket：{{context_packet}}

只输出 JSON：
{
  "text": "自然回答",
  "candidate_claims": [
    {"text": "主张原文", "intended_type": "confirmed_fact|observed_pattern|agent_inference|unknown", "candidate_evidence_ids": ["..."]}
  ],
  "unknowns": ["..."]
}"""


class ComplexAnswerBuilder:
    """Produce a verified natural-language answer for person/event queries."""

    def __init__(self, gamma=None):
        self.gamma = gamma
        self.claim_extractor = LLMClaimExtractor(gamma=gamma)

    def build(self, message, spec, packet):
        """Return ``{answer, statements, claims, verification, fallback}``.

        If the model is unavailable or every claim fails verification even
        after one repair, the caller receives a structured safe answer.
        """
        writer_output = self._call_writer(message, spec, packet)
        if not writer_output:
            return self._safe_fallback(spec, packet, reason="writer_unavailable")
        text = writer_output.get("text") or ""
        candidate_claims = writer_output.get("candidate_claims") or []
        scanned = self.claim_extractor.scan(text, candidate_claims)
        if not scanned:
            # Model did not produce a scannable claim list; keep the safe path
            # rather than surface unverified free text.
            return self._safe_fallback(spec, packet, reason="claim_extractor_unavailable")
        bundles = [build_verifier_evidence_bundle(packet, claim["claim_id"]) for claim in scanned]
        verifications = verify_claims(scanned, bundles,
                                       scope_id=spec.scope_id, viewer_id=spec.viewer_id)
        failing_ids = {item.get("claim_id") for item in verifications
                        if item.get("status") in {"unsupported", "overstated", "contradicted", "privacy_blocked"}}
        if failing_ids:
            repaired = repair_answer(text, scanned, verifications, max_repairs=1)
            text = repaired["text"]
            # Recompute claims on the repaired text; failing spans replaced with
            # deterministic downgrade phrases.
            scanned = self.claim_extractor.scan(text, candidate_claims) or scanned
            verifications = verify_claims(scanned, bundles,
                                           scope_id=spec.scope_id, viewer_id=spec.viewer_id)
            failing_ids = {item.get("claim_id") for item in verifications
                            if item.get("status") in {"unsupported", "overstated", "contradicted", "privacy_blocked"}}
        if failing_ids:
            return self._safe_fallback(spec, packet, reason="verification_failed")
        statements = [
            {
                "text": claim.get("text"),
                "status": verification_map(verifications).get(claim.get("claim_id"), {}).get("status", "reasonable_summary"),
                "evidence_ids": _evidence_ids_for_claim(claim, packet),
            }
            for claim in scanned
        ]
        return {
            "answer": text,
            "statements": statements,
            "claims": scanned,
            "verifications": verifications,
            "fallback": False,
        }

    def _call_writer(self, message, spec, packet):
        if not self.gamma or not hasattr(self.gamma, "chat"):
            return None
        context = build_narrative_context_packet(packet, spec)
        prompt = _WRITER_PROMPT.replace("{{message}}", str(message or "")).replace(
            "{{context_packet}}", json.dumps(context, ensure_ascii=False),
        )
        try:
            raw = self.gamma.chat(prompt, json_mode=True, role="writer")
        except Exception:
            return None
        parsed = parse_json_response(raw)
        return parsed if isinstance(parsed, dict) else None

    @staticmethod
    def _safe_fallback(spec, packet, reason):
        name = next((item.value for item in spec.constraints if item.dimension == "person"), "这个人")
        if packet.assets:
            text = f"关于{name}的自然总结暂时无法完成，但相关证据已按时间和活动整理，可以按需展开查看。"
        else:
            text = f"目前没有关于{name}的可靠证据支持一个完整总结。"
        return {
            "answer": text,
            "statements": [{"text": text, "status": "unknown", "evidence_ids": []}],
            "claims": [],
            "verifications": [],
            "fallback": True,
            "fallback_reason": reason,
        }


def verification_map(verifications):
    return {item.get("claim_id"): item for item in verifications or ()}


def _evidence_ids_for_claim(claim, packet):
    """Anchor the claim on every asset/observation from the packet.

    The Verifier already checked bundle membership; here we just expose the
    pointers so downstream callers can render evidence entrances.
    """
    ids = []
    for item in packet.assets:
        ids.extend(item.get("evidence_ids") or [])
    return ids
