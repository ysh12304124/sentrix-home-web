"""The correctness-first Thin Agent runtime.

This module deliberately keeps model use at the language boundary.  Query
identity, evidence selection and the final evidence envelope remain code-owned.
"""

import uuid

from .answer_composer import compose_answer
from .claim_extractor import ClaimExtractor
from .evidence_retrieval import EvidenceRetrievalKernel
from .memory_gate import MemoryGate
from .query_contracts import build_query_spec
from .query_parser import QueryParser


class ThinAgentRuntime:
    def __init__(self, store, gamma=None):
        self.store = store
        self.gamma = gamma
        self.gate = MemoryGate()
        self.kernel = EvidenceRetrievalKernel(store)
        self.parser = QueryParser(gamma=gamma)

    def answer_turn(self, message, conversation_id=None, feedback=None, scope_id=None, viewer_id=None, recent_turns="", selected_entity_id=None):
        conversation_id = conversation_id or f"conversation_{uuid.uuid4().hex[:12]}"
        viewer_id = viewer_id or "owner"
        scope_id = scope_id if scope_id is not None else "home-default"
        api_signals = {"feedback": feedback, "selected_entity_id": selected_entity_id}
        # Fast-path first — writing prompts and explicit API signals never call
        # the parser (plan §2.5: normal chat QuerySpec/Gate calls = 0).
        fast_decision = self.gate.fast_path(message, api_signals=api_signals)
        if fast_decision is not None and fast_decision.mode == "none":
            empty_draft = self.parser._safe_fallback()  # deterministic empty draft, no LLM call
            return self._normal_chat(message, recent_turns, conversation_id, scope_id, viewer_id, fast_decision, empty_draft)
        draft = self.parser.parse(message, recent_turns=recent_turns)
        decision = fast_decision or self.gate.classify(message, recent_turns, draft=draft, api_signals=api_signals)
        if decision.mode == "none":
            return self._normal_chat(message, recent_turns, conversation_id, scope_id, viewer_id, decision, draft)
        if decision.mode == "contextual":
            return self._contextual(message, conversation_id, scope_id, viewer_id, decision, draft)
        spec = build_query_spec(
            draft,
            scope_id=scope_id,
            viewer_id=viewer_id,
            conversation_id=conversation_id,
            entity_resolver=lambda name: self._resolve_person(name, scope_id),
            query_id=f"query_{uuid.uuid4().hex[:12]}",
        )
        packet = self.kernel.retrieve(spec)
        if spec.answer_target == "person" and not spec.entity_ids:
            packet.assets = []
            packet.exact_results = []
            packet.strong_results = []
            packet.approximate_results = []
            packet.gaps = [{"condition": "confirmed_person", "reason": "没有找到当前 scope 内已确认的人物"}]
        return self._evidence_answer(message, conversation_id, scope_id, viewer_id, decision, spec, packet, draft)

    def _normal_chat(self, message, recent_turns, conversation_id, scope_id, viewer_id, decision, draft):
        answer = "我在听。"
        if self.gamma and hasattr(self.gamma, "chat"):
            prompt = (
                "你是 Sentrix，一个自然、克制的家庭数字助手。本轮不是家庭记忆查询，"
                "不要读取或猜测具体家庭事实。直接自然回答用户，不要提到数据库、检索或工具。\n"
                f"最近对话：{str(recent_turns or '')[-1200:]}\n用户：{message}"
            )
            try:
                text = self.gamma.chat(prompt, json_mode=False)
                answer = str(text or "").strip() or answer
            except Exception:
                pass
        trace = [{"stage": "gate", "status": "none", "counts": {"memory_tools": 0, "evidence": 0, "query_parse": decision.query_parse_calls}}]
        return self._envelope(answer, conversation_id, scope_id, viewer_id, decision, [], [], "not_applicable", trace, draft=draft)

    def _contextual(self, message, conversation_id, scope_id, viewer_id, decision, draft):
        cards = []
        try:
            people = self.store.list_entities(status="confirmed", scope_id=scope_id)
            cards = [{"subject_id": item.get("id"), "display_name": item.get("canonical_name"),
                      "epistemic_type": "confirmed_fact",
                      "text": item.get("summary") or "已确认人物"} for item in people[:5]]
        except Exception:
            cards = []
        answer = "我明白，这种时候会特别想起熟悉的人。"
        if cards:
            answer += "我只保留了关于已确认人物的轻量记忆，不展开具体照片。"
        trace = [{"stage": "gate", "status": "contextual",
                  "counts": {"core_memory": len(cards), "concrete_memory": 0, "assets": 0,
                             "query_parse": decision.query_parse_calls}}]
        return self._envelope(answer, conversation_id, scope_id, viewer_id, decision, [], [], "not_applicable", trace, core_cards=cards, draft=draft)

    def _resolve_person(self, name, scope_id):
        try:
            for entity in self.store.list_entities(status="confirmed", scope_id=scope_id):
                if entity.get("canonical_name") == name:
                    return entity.get("id")
        except Exception:
            pass
        return None

    def _evidence_answer(self, message, conversation_id, scope_id, viewer_id, decision, spec, packet, draft):
        evidence = []
        for item in packet.assets:
            evidence.append({
                "kind": "observation",
                "id": item["observation_ids"][0] if item["observation_ids"] else item["asset_id"],
                "asset_id": item["asset_id"],
                "observation_id": item["observation_ids"][0] if item["observation_ids"] else None,
                "file_name": item.get("file_name"), "media_type": item.get("media_type"),
                "captured_at": item.get("captured_at"),
                "condition_results": item.get("condition_results", {}),
                "level": item.get("level"), "evidence_ids": item.get("evidence_ids", []),
            })
        person_summary = spec.answer_target == "person" and bool(spec.entity_ids)
        clothing_gap = spec.answer_target == "clothing" and bool(spec.entity_ids) and not self._has_subject_clothing(packet)
        if spec.answer_target == "person" and not spec.entity_ids:
            answer, statements = ("目前没有找到当前范围内已确认的人物，不能把待确认人物簇直接当作人物介绍。", [])
        elif person_summary:
            answer, statements = self._person_summary(spec, packet)
        elif clothing_gap:
            name = next((item.value for item in spec.constraints if item.dimension == "person"), "这个人")
            answer = f"现有记录没有把衣物字段可靠绑定到{name}，无法确认这件衣服属于他。"
            statements = []
        elif not evidence:
            answer, statements = ("没有找到满足这些条件的可靠家庭证据。", [])
        else:
            answer, statements = self._simple_answer(packet)
        allowed = self._allowed_facts(packet)
        composed = (
            {"answer": answer, "statements": statements, "valid": True}
            if person_summary or clothing_gap
            else compose_answer({"answer": answer, "statements": statements}, allowed)
            if statements
            else {"answer": answer, "statements": [], "valid": True}
        )
        answer = composed["answer"]
        claims, index = [], {}
        extracted_claims = ClaimExtractor().scan(composed["answer"], composed.get("statements", []))
        for number, claim in enumerate(extracted_claims, 1):
            claim_id = f"claim_{number}"
            claim["claim_id"] = claim_id
            statement = next((item for item in composed.get("statements", []) if item.get("text") == claim["text"]), {})
            claim["status"] = statement.get("status", "reasonable_summary")
            claim["evidence_ids"] = statement.get("evidence_ids", [])
            claims.append(claim)
            index[claim_id] = {"evidence_ids": claim["evidence_ids"], "status": claim["status"]}
        trace = [
            {"stage": "gate", "status": "evidence", "counts": {"query_parse": decision.query_parse_calls, "evidence_search": 1}},
            {"stage": "retrieval", "status": "complete",
             "counts": {"assets": len(packet.assets), "excluded": packet.excluded_count,
                        "exact": len(packet.exact_results), "approximate": len(packet.approximate_results)}},
        ]
        result = self._envelope(answer, conversation_id, scope_id, viewer_id, decision, evidence, packet.gaps,
                                 "anchored" if evidence else "gap", trace, spec=spec, packet=packet, draft=draft)
        return_assets_requested = any(action.type == "return_assets" for action in spec.actions) or bool(spec.result_requirement.get("return_original_assets"))
        result.update({
            "claims": claims, "claim_evidence_index": index, "statement_plan": statements,
            "memory_intensity": "targeted", "original_evidence_requested": return_assets_requested,
            "image_results": self._image_results(evidence) if return_assets_requested else [],
        })
        result["evidence_presentation"] = {
            "required": True, "available": bool(evidence or packet.gaps),
            "default_collapsed": not return_assets_requested,
            "direct_original_evidence": return_assets_requested,
        }
        result["tool_trace"] = [{"tool": "search_evidence", "permission": "read", "status": "complete", "asset_count": len(evidence)}]
        if spec.answer_target == "person" and spec.entity_ids:
            result["tool_trace"].append({"tool": "summarize_person", "permission": "read",
                                          "status": "complete" if evidence else "requires_anchor"})
        return result

    def _person_summary(self, spec, packet):
        name = next((item.value for item in spec.constraints if item.dimension == "person"), "这个人")
        places, activities, clothes = [], [], []
        for item in packet.assets:
            observed = item.get("observation_fields") or {}
            if observed.get("place"):
                places.append(observed["place"])
            if observed.get("activity"):
                activities.append(observed["activity"])
            # Only formation-provided subject bindings enter the person summary.
            clothes.extend(observed.get("subject_clothing") or [])
        parts = [f"从现有照片记录看，{name}在这些记录中多次出现"]
        if places:
            parts.append("，出现过的地点包括" + "、".join(dict.fromkeys(places)))
        if activities:
            parts.append("，记录到的活动包括" + "、".join(dict.fromkeys(activities)))
        if clothes:
            parts.append("，记录中出现过" + "、".join(dict.fromkeys(str(item) for item in clothes)) + "等衣着描述")
        parts.append("。这些是记录里的可观察内容；仅凭现有照片还不能确定他的性格、长期偏好或未被确认的家庭关系。")
        text = "".join(parts)
        ids = [evidence_id for item in packet.assets for evidence_id in item.get("evidence_ids", [])]
        return text, [{"text": text, "status": "reasonable_summary", "evidence_ids": ids}]

    @staticmethod
    def _has_subject_clothing(packet):
        return any((item.get("observation_fields") or {}).get("subject_clothing") for item in packet.assets)

    def _simple_answer(self, packet):
        exact = len(packet.exact_results)
        approximate = len(packet.approximate_results)
        statements = []
        if exact:
            text = f"找到 {exact} 条完全符合确定条件的照片记录。"
            statements.append({
                "text": text, "status": "matched",
                "evidence_ids": [item["asset_id"] for item in packet.exact_results],
                "condition_keys": [key for item in packet.exact_results for key, value in item["condition_results"].items() if value.get("status") == "matched"],
            })
        if approximate:
            text = f"另外有 {approximate} 条近似记录，但有条件目前无法从直接观察中确认。"
            statements.append({
                "text": text, "status": "possible",
                "evidence_ids": [item["asset_id"] for item in packet.approximate_results],
                "condition_keys": [key for item in packet.approximate_results for key, value in item["condition_results"].items() if value.get("status") in {"matched", "possible", "unknown"}],
            })
        if packet.gaps:
            statements.append({"text": "目前无法确认其中的关键活动或视觉细节。", "status": "unknown", "evidence_ids": []})
        return "".join(item["text"] for item in statements), statements

    @staticmethod
    def _allowed_facts(packet):
        facts, possibilities, unknowns = [], [], []
        for item in packet.assets:
            for key, condition in item.get("condition_results", {}).items():
                text = f"记录支持{key.split(':', 1)[1]}"
                target = facts if condition.get("status") == "matched" else possibilities if condition.get("status") == "possible" else unknowns
                target.append({"text": text, "status": condition.get("status"), "condition_key": key, "evidence_ids": item.get("evidence_ids", [])})
        unknowns.extend({"text": "目前无法确认其中的关键活动或视觉细节。", "evidence_ids": []} for _ in packet.gaps)
        return {"allowed_answer_facts": facts, "allowed_possibilities": possibilities, "required_unknowns": unknowns}

    @staticmethod
    def _image_results(evidence):
        return [
            {"asset_id": item["asset_id"], "file_name": item.get("file_name"),
             "media_type": item.get("media_type"), "media_url": f"/api/assets/{item['asset_id']}/file"}
            for item in evidence if item.get("media_type") == "image"
        ]

    @staticmethod
    def _envelope(answer, conversation_id, scope_id, viewer_id, decision, evidence, gaps, status, trace, *, draft=None, **extra):
        result = {
            "intent": decision.reason, "conversation_id": conversation_id, "answer": answer,
            "confidence": 0.75 if evidence else 0.0,
            "insufficient_evidence": not bool(evidence), "evidence": evidence, "image_results": [],
            "retrieval_trace": trace,
            "memory_used": decision.mode != "none",
            "evidence_required": decision.mode != "none",
            "evidence_status": status,
            "memory_actually_referenced": decision.mode != "none",
            "scope_id": scope_id, "viewer_id": viewer_id,
            "evidence_layers": {
                "answers": [{"id": None, "text": answer}],
                "people": [], "events": [], "claims": [], "appearance": [],
                "observations": evidence,
                "assets": [{"kind": "asset", "id": item["asset_id"]} for item in evidence],
                "gaps": gaps,
            },
            "tool_trace": [], "model": "sentrix-thin-agent-v1",
        }
        if draft is not None:
            result["actions"] = [{"type": action.type, "target": action.target, "coverage": action.coverage} for action in draft.actions]
            result["facets"] = [{"dimension": facet.dimension, "surface_text": facet.surface_text} for facet in draft.facets]
            result["parser_mode"] = draft.mode
            result["parser_confidence"] = draft.confidence
        result.update(extra)
        return result
