"""The correctness-first Thin Agent runtime.

This module deliberately keeps model use at the language boundary.  Query
identity, evidence selection and the final evidence envelope remain code-owned.
"""

import os
import threading
import time
import uuid
from contextlib import nullcontext

from .answer_composer import compose_answer
from .claim_extractor import ClaimExtractor
from .complex_answer import ComplexAnswerBuilder
from .evidence_retrieval import EvidenceRetrievalKernel
from .model_routing import RequestDeadline
from .query_contracts import HARD, Constraint, QueryAction, build_query_spec
from .query_parser import QueryParser
from .router import ExplicitOperationDetector, Router
from .routing_rules import message_anchored


def _query_anchored(message, spec):
    """A query is anchored (strict-empty style) when it carries a concrete
    person / time / place signal — either parsed into the spec or present in
    the raw message when the parser hallucinated an empty draft."""
    if any(c.dimension in {"person", "time", "place"} for c in spec.constraints):
        return True
    return message_anchored(message)


class _StageTimer:
    def __init__(self, data, name):
        self.data = data
        self.name = name

    def __enter__(self):
        self._started = time.monotonic()
        return self

    def __exit__(self, *_exc):
        self.data.setdefault(self.name, 0.0)
        self.data[self.name] += time.monotonic() - self._started
        return False


class _Perf:
    """Thread-local stage trace collector (SENTRIX_AGENT_STAGE_TRACE).

    Benchmark-only: begin()/end() wrap one request and measure() records stage
    durations and call counters into the request's perf dict.  A request without
    the flag active gets a no-op context manager, so production overhead is a
    single attribute lookup.
    """

    _local = threading.local()

    @staticmethod
    def begin():
        _Perf._local.data = {}

    @staticmethod
    def end():
        data = getattr(_Perf._local, "data", None) or {}
        _Perf._local.data = None
        return data

    @staticmethod
    def measure(name):
        data = getattr(_Perf._local, "data", None)
        if data is None:
            return nullcontext()
        return _StageTimer(data, name)

    @staticmethod
    def count(name):
        data = getattr(_Perf._local, "data", None)
        if data is None:
            return
        data[name] = data.get(name, 0) + 1


class ThinAgentRuntime:
    def __init__(self, store, gamma=None, embedding_router=None, retrieval_config=None):
        self.store = store
        self.gamma = gamma
        self._detector = ExplicitOperationDetector()
        self._router = Router()
        self.embedding_router = embedding_router
        if embedding_router is not None:
            from .retrieval import RetrievalConfig, build_default_retrievers
            config = retrieval_config or RetrievalConfig()
            if config.multi_retriever:
                retrievers = build_default_retrievers(store, embedding_router=embedding_router, config=config)
                self.kernel = EvidenceRetrievalKernel(store, retrievers=retrievers,
                                                      embedding_router=embedding_router, config=config)
            else:
                self.kernel = EvidenceRetrievalKernel(store)
        else:
            self.kernel = EvidenceRetrievalKernel(store)
        self.router = None
        if gamma is not None:
            from .model_routing import ModelRouter
            self.router = ModelRouter(gamma=gamma)
        self.parser = QueryParser(gamma=gamma, router=self.router)
        self.complex_builder = ComplexAnswerBuilder(gamma=gamma)

    def answer_turn(self, message, conversation_id=None, feedback=None, scope_id=None, viewer_id=None, recent_turns="", selected_entity_id=None):
        # 12B-FC fix: RequestDeadline is per-REQUEST, not per-process.  The
        # shared ModelRouter's deadline is reset at the start of every turn so
        # model calls after the first ~20s of process life still have budget
        # (otherwise every call short-circuits to fallback and the model never
        # participates).  Concurrent resets are safe: they can only extend a
        # request's budget, never shrink it.
        if self.router is not None:
            self.router.deadline = RequestDeadline()
        from .validation import full_chain_profile as _prof
        from .validation import model_call_ledger as _ledger
        from .validation import assertions as _assert
        v_active = _prof.validation_active()
        if v_active:
            _ledger.begin_turn()
        trace_on = os.getenv("SENTRIX_AGENT_STAGE_TRACE", "0").lower() in {"1", "true", "on"}
        if trace_on:
            _Perf.begin()
            start_counts = dict(getattr(self.parser, "call_counts", {}) or {}) if self.parser else {}
        try:
            result = self._answer_turn_inner(message, conversation_id, feedback, scope_id,
                                             viewer_id, recent_turns, selected_entity_id)
        except Exception:
            if trace_on:
                _Perf.end()
            if v_active:
                _ledger.end_turn()
            raise
        if trace_on:
            perf = _Perf.end()
            if self.parser is not None:
                current = self.parser.call_counts or {}
                counts = {key: int(current.get(key, 0) - start_counts.get(key, 0))
                          for key in set(current) | set(start_counts)}
            else:
                counts = {}
            counts["answer"] = perf.get("answer_calls", 0)
            counts["claim"] = perf.get("claim_calls", 0)
            perf["model_calls"] = counts
            result["perf"] = perf
        if v_active:
            records = _ledger.end_turn()
            required = os.getenv("SENTRIX_PARSE_MODEL", "") or os.getenv("OLLAMA_MODEL", "gemma4:12b")
            parser_failed = (result.get("perf") or {}).get("degraded", 0) > 0
            result["validation"] = _assert.validate_turn(
                records, [], required_model=required, parser_failed=parser_failed)
            result["model_call_ledger"] = records
        return result

    def _answer_turn_inner(self, message, conversation_id=None, feedback=None, scope_id=None, viewer_id=None, recent_turns="", selected_entity_id=None):
        conversation_id = conversation_id or f"conversation_{uuid.uuid4().hex[:12]}"
        viewer_id = viewer_id or "owner"
        scope_id = scope_id if scope_id is not None else "home-default"
        api_signals = {"feedback": feedback, "selected_entity_id": selected_entity_id}
        # High-precision protocol fast path (writing / no-lookup) — no model call.
        with _Perf.measure("explicit_detector"):
            fast = self._detector.detect(message, api_signals=api_signals)
        if fast is not None and fast.mode == "none":
            empty_draft = self.parser._safe_fallback()  # deterministic empty draft, no LLM call
            return self._normal_chat(message, recent_turns, conversation_id, scope_id, viewer_id, fast, empty_draft)
        with _Perf.measure("parser"):
            draft = self.parser.parse(message, recent_turns=recent_turns)
        if getattr(draft, "parser_failed", False):
            # 12B-FC: a parser failure (timeout/degradation) is recorded so the
            # validation block can mark the case failed_due_to_degradation.
            _Perf.count("degraded")
        focus = self._load_focus(conversation_id, scope_id)
        with _Perf.measure("router"):
            decision = self._router.route(
                message, draft, api_signals=api_signals, conversation=recent_turns,
                focus=focus, entity_resolver=lambda name: self._resolve_person(name, scope_id),
                message_entity_resolver=lambda msg: self._message_entity_ids(msg, scope_id),
            )
        if decision.mode == "none":
            return self._normal_chat(message, recent_turns, conversation_id, scope_id, viewer_id,
                                     decision.as_gate_decision(), draft)
        if decision.mode == "contextual":
            return self._contextual(message, conversation_id, scope_id, viewer_id,
                                    decision.as_gate_decision(), draft)
        if decision.mode == "ambiguous":
            return self._ambiguous_path(message, recent_turns, conversation_id, scope_id, viewer_id,
                                        decision, draft)
        if decision.mode == "clarify":
            return self._clarify_envelope(message, conversation_id, scope_id, viewer_id,
                                          decision, None, draft)
        return self._evidence_path(message, recent_turns, conversation_id, scope_id, viewer_id,
                                   decision, draft)

    def _evidence_path(self, message, recent_turns, conversation_id, scope_id, viewer_id, decision, draft):
        with _Perf.measure("query_spec"):
            spec = build_query_spec(
                draft,
                scope_id=scope_id,
                viewer_id=viewer_id,
                conversation_id=conversation_id,
                entity_resolver=lambda name: self._resolve_person(name, scope_id),
                query_id=f"query_{uuid.uuid4().hex[:12]}",
            )
            if decision.focus_ids:
                spec = self._merge_focus(spec, decision.focus_ids, scope_id)
        with _Perf.measure("retrieval"):
            packet = self.kernel.retrieve(spec)
        if spec.answer_target == "person" and not spec.entity_ids:
            packet.assets = []
            packet.exact_results = []
            packet.strong_results = []
            packet.approximate_results = []
            packet.gaps = [{"condition": "confirmed_person", "reason": "没有找到当前 scope 内已确认的人物"}]
        result = self._evidence_answer(message, conversation_id, scope_id, viewer_id,
                                       decision.as_gate_decision(), spec, packet, draft)
        self._save_focus(conversation_id, scope_id, spec, packet)
        return result

    def _ambiguous_path(self, message, recent_turns, conversation_id, scope_id, viewer_id, decision, draft):
        """Ambiguous (weak parser signal / bare noun) — the NeutralProbe decides.

        The Router finalizes after the probe: upgrade -> evidence, clarify ->
        clarify envelope, no_household_match -> clear-general none / else clarify
        (never fabricated chat).
        """
        media_hint = None
        if getattr(draft, "media_expressions", None):
            media_hint = "image" if any("照片" in item or "图片" in item or "图" == item for item in draft.media_expressions) else "media"
        with _Perf.measure("probe"):
            probe = self._run_probe(message, scope_id, viewer_id,
                                    conversation_id=conversation_id, media_hint=media_hint)
        final = self._router.resolve_after_probe(probe, message, decision, draft)
        if final.mode == "evidence":
            if not draft.actions:
                draft.actions = [QueryAction(type="answer_question", target="general")]
            if probe.decision == "upgrade":
                draft.semantic_conditions.append(
                    {"dimension": "semantic", "value": message, "source_text": message}
                )
            return self._evidence_path(message, recent_turns, conversation_id, scope_id,
                                       viewer_id, final, draft)
        if final.mode == "clarify":
            return self._clarify_envelope(message, conversation_id, scope_id, viewer_id,
                                          decision, probe, draft)
        return self._normal_chat(message, recent_turns, conversation_id, scope_id, viewer_id,
                                 final.as_gate_decision(), draft)

    def _clarify_envelope(self, message, conversation_id, scope_id, viewer_id, decision, probe, draft):
        answer = "你是想让我在你存下的照片或记忆里找这个，还是想聊点别的？"
        counts = probe.channel_counts if probe is not None else {}
        reason = probe.reason if probe is not None else decision.reason
        trace = [{"stage": "gate", "status": "ambiguous",
                  "counts": {"query_parse": decision.query_parse_calls,
                             "probe": counts, "probe_decision": reason}}]
        return self._envelope(answer, conversation_id, scope_id, viewer_id,
                              decision.as_gate_decision(), [], [], "clarify", trace, draft=draft)

    def _run_probe(self, message, scope_id, viewer_id, conversation_id=None, media_hint=None):
        from .retrieval import NeutralProbe, RetrievalConfig
        channel_hits, index_health = self.kernel.probe(message, scope_id, viewer_id)
        focus = self._load_focus(conversation_id, scope_id) if conversation_id else {}
        return NeutralProbe(RetrievalConfig()).run(message, channel_hits,
                                                   scope_id=scope_id, viewer_id=viewer_id,
                                                   focus=focus, media_hint=media_hint,
                                                   index_health=index_health)

    def _load_focus(self, conversation_id, scope_id):
        try:
            return self.store.get_dialogue_state(conversation_id, scope_id) or {}
        except Exception:
            return {}

    def _save_focus(self, conversation_id, scope_id, spec, packet):
        try:
            entity_ids = list(dict.fromkeys(spec.entity_ids or []))
            state = {
                "scope_id": scope_id or "home-default",
                "active_entity_ids": entity_ids[:8],
                "active_event_ids": [],
                "evidence_ids": [item["asset_id"] for item in (packet.assets or [])][:40],
                "unresolved_ambiguity": not bool(packet.assets),
            }
            self.store.save_dialogue_state(conversation_id, scope_id or "home-default", state)
        except Exception:
            pass

    def _merge_focus(self, spec, focus_ids, scope_id):
        for entity_id in focus_ids:
            if entity_id in spec.entity_ids:
                continue
            name = self._entity_name(entity_id, scope_id)
            spec.entity_ids.append(entity_id)
            if name:
                spec.constraints.append(Constraint("person", name, HARD, "session_focus",
                                                   source_text=name))
        return spec

    def _entity_name(self, entity_id, scope_id):
        try:
            for entity in self.store.list_entities(status="confirmed", scope_id=scope_id):
                if entity.get("id") == entity_id:
                    return entity.get("canonical_name") or entity_id
        except Exception:
            pass
        return entity_id

    def _normal_chat(self, message, recent_turns, conversation_id, scope_id, viewer_id, decision, draft):
        answer = "我在听。"
        if self.gamma and hasattr(self.gamma, "chat"):
            prompt = (
                "你是 Sentrix，一个自然、克制的家庭数字助手。本轮不是家庭记忆查询，"
                "不要读取或猜测具体家庭事实。直接自然回答用户，不要提到数据库、检索或工具。\n"
                f"最近对话：{str(recent_turns or '')[-1200:]}\n用户：{message}"
            )
            try:
                with _Perf.measure("answer"):
                    if self.router is not None:
                        text = self.router.chat("answer", prompt, json_mode=False)
                    else:
                        text = self.gamma.chat(prompt, json_mode=False, role="answer")
                _Perf.count("answer_calls")
                answer = str(text or "").strip() or answer
            except Exception:
                pass
        trace = [{"stage": "gate", "status": "none", "counts": {"memory_tools": 0, "evidence": 0, "query_parse": decision.query_parse_calls}}]
        return self._envelope(answer, conversation_id, scope_id, viewer_id, decision, [], [], "not_applicable", trace, draft=draft)

    def _contextual(self, message, conversation_id, scope_id, viewer_id, decision, draft):
        cards = []
        # Prefer Core Memory Cards when SENTRIX_CORE_MEMORY_V1 is on; otherwise
        # fall back to the confirmed-entity placeholder from Phase 2R.
        import os
        core_flag = os.getenv("SENTRIX_CORE_MEMORY_V1", "0").lower() in {"1", "true", "on"}
        if core_flag:
            try:
                from .core_memory import CoreMemoryStore
                cms = CoreMemoryStore(self.store)
                cms_cards = cms.list_cards(scope_id=scope_id, limit=5)
                for card in cms_cards:
                    cms.record_access(card_id=card["card_id"], conversation_id=conversation_id, viewer_id=viewer_id)
                    for item in card.get("items", [])[:1]:
                        cards.append({"subject_id": card["subject_id"], "display_name": card["display_name"],
                                      "epistemic_type": item["epistemic_type"], "text": item["text"]})
            except Exception:
                cards = []
        if not cards:
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

    @staticmethod
    def _has_matched_condition(item):
        return any(cond.get("status") == "matched" for cond in (item.get("condition_results") or {}).values())

    @staticmethod
    def _recall_strength(item):
        """Normalized retrieval strength of an evidence item, from attributions.

        cosine_similarity is already [0,1]; token_hits count is capped;
        discrete/adjacency get a fixed low weight.  Returns None when the item
        carries no attributions (single-kernel path) — the gate then keeps it
        instead of guessing.
        """
        attributions = item.get("attributions") or []
        if not attributions:
            return None
        best = 0.0
        for attr in attributions:
            score = attr.get("score") or 0.0
            kind = attr.get("score_kind") or ""
            if kind == "cosine_similarity":
                best = max(best, float(score))
            elif kind == "token_hits":
                best = max(best, min(1.0, float(score) / 4.0))
            elif kind == "discrete":
                best = max(best, 1.0)
            elif kind == "adjacency":
                best = max(best, 0.1)
        return round(best, 4)

    @staticmethod
    def _gate_packet_approximate(packet, spec, anchored=False):
        """Drop weak approximate assets from the packet (R8-5).

        exact/strong always stay.  Approximate assets are kept only when their
        recall strength clears a config threshold; anchored queries (person /
        time / place) that find no directly-supported candidate must not surface
        weak approximate images (strict-empty style).
        """
        try:
            from .retrieval import RetrievalConfig
            config = RetrievalConfig()
            min_score = float(config.approximate("min_score", 0.15))
            max_count = int(config.approximate("max_count", 20))
            if anchored:
                min_score *= float(config.approximate("anchor_multiplier", 1.5))
        except Exception:
            return
        keep = []
        dropped = 0
        has_exact_strong = any(item["level"] in {"exact", "strong"} for item in packet.assets)
        for item in packet.assets:
            if item["level"] in {"exact", "strong"}:
                keep.append(item)
                continue
            if anchored and not has_exact_strong and not ThinAgentRuntime._has_matched_condition(item):
                # strict-empty: an anchored query whose candidates support NO
                # condition must not surface weak approximate images.  A
                # partially-supported candidate (e.g. time matched, activity
                # unknown) is still shown with disclosure.
                dropped += 1
                continue
            strength = ThinAgentRuntime._recall_strength(item)
            if strength is None or strength >= min_score:
                keep.append(item)
            else:
                dropped += 1
        if dropped:
            packet.excluded_count += dropped
        # Cap approximate count (best/top_k mode) — exact/strong stay.
        approximate_keep = [item for item in keep if item["level"] == "approximate"]
        if len(approximate_keep) > max_count:
            approximate_keep.sort(key=lambda item: ThinAgentRuntime._recall_strength(item) or 0.0, reverse=True)
            keep = [item for item in keep if item["level"] != "approximate"] + approximate_keep[:max_count]
        packet.assets = keep
        packet.exact_results = [item for item in packet.exact_results if item in keep]
        packet.strong_results = [item for item in packet.strong_results if item in keep]
        packet.approximate_results = [item for item in packet.approximate_results if item in keep]

    def _resolve_person(self, name, scope_id):
        try:
            for entity in self.store.list_entities(status="confirmed", scope_id=scope_id):
                if entity.get("canonical_name") == name:
                    return entity.get("id")
        except Exception:
            pass
        return None

    def _message_entity_ids(self, message, scope_id):
        """Confirmed entities whose canonical name appears in the raw message."""
        value = str(message or "")
        ids = []
        try:
            for entity in self.store.list_entities(status="confirmed", scope_id=scope_id):
                name = str(entity.get("canonical_name") or "").strip()
                if name and name in value:
                    ids.append(entity["id"])
        except Exception:
            pass
        return ids

    def _evidence_answer(self, message, conversation_id, scope_id, viewer_id, decision, spec, packet, draft):
        # R8-5: gate approximate evidence by recall strength before display —
        # weak approximate candidates must not surface as user-visible images.
        self._gate_packet_approximate(packet, spec, anchored=_query_anchored(message, spec))
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
                "recall_strength": self._recall_strength(item),
                "near_duplicate_group": item.get("near_duplicate_group"),
                "near_duplicate_size": item.get("near_duplicate_size", 1),
            })
        if packet.assets:
            try:
                from .retrieval.near_duplicate import NearDuplicateGrouper
                NearDuplicateGrouper(self.store).annotate(evidence)
            except Exception:
                pass
        person_summary = spec.answer_target == "person" and bool(spec.entity_ids)
        clothing_gap = spec.answer_target == "clothing" and bool(spec.entity_ids) and not self._has_subject_clothing(packet)
        v_evidence_answer = self._validation_active_evidence()
        with _Perf.measure("answer"):
            if spec.answer_target == "person" and not spec.entity_ids:
                answer, statements = ("目前没有找到当前范围内已确认的人物，不能把待确认人物簇直接当作人物介绍。", [])
            elif person_summary:
                answer, statements = self._person_summary_via_complex_or_fallback(message, spec, packet)
            elif clothing_gap:
                name = next((item.value for item in spec.constraints if item.dimension == "person"), "这个人")
                answer = f"现有记录没有把衣物字段可靠绑定到{name}，无法确认这件衣服属于他。"
                statements = []
            elif v_evidence_answer:
                # 12B-FC: under the validation profile the evidence answer is
                # generated by the 12B model (never a deterministic template),
                # covering matched / approximate / strict-empty equally.
                answer, statements = self._validation_evidence_answer(message, packet)
            elif not evidence:
                # Phase R R6: a household evidence query with no matched evidence
                # must refuse explicitly — never fall back to normal chat with a
                # fabricated general description.
                answer, statements = ("当前记忆中没有找到足够匹配的原始证据。", [])
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
        with _Perf.measure("claim"):
            extracted_claims = ClaimExtractor().scan(composed["answer"], composed.get("statements", []))
        _Perf.count("claim_calls")
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
        if getattr(packet, "channel_trace", None):
            trace.append({"stage": "channels", "status": "complete", "channels": packet.channel_trace})
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

    def _person_summary_via_complex_or_fallback(self, message, spec, packet):
        """Try Phase 4 Writer/Verifier chain; fall back to inline summary."""
        if os.getenv("SENTRIX_LLM_CLAIM_EXTRACTOR_V1", "0").lower() in {"1", "true", "on"}:
            with _Perf.measure("complex_chain"):
                result = self.complex_builder.build(message, spec, packet)
            _Perf.count("answer_calls")
            if not result.get("fallback"):
                return result["answer"], result["statements"]
        return self._person_summary(spec, packet)

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

    @staticmethod
    def _validation_active_evidence():
        # The 12B Evidence Answer is the production default (SENTRIX_EVIDENCE_
        # ANSWER_12B=1) and also active under the validation profile.  It turns
        # the deterministic evidence template into a natural 12B-generated
        # answer that distinguishes matched / possible / unknown.
        from .validation import full_chain_profile as _prof
        if _prof.validation_active():
            return True
        return os.getenv("SENTRIX_EVIDENCE_ANSWER_12B", "0").strip().lower() in {"1", "true", "on"}

    def _validation_evidence_answer(self, message, packet):
        """12B Evidence Answer (Phase 12B-FC V4).

        Under the validation profile the evidence answer is generated by the 12B
        answer model from the EvidencePacket — matched / possible / unknown all
        go through the model, never a deterministic template.
        """
        lines = []
        for item in packet.assets:
            conds = [f"{key.split(':', 1)[-1]}={cond.get('status')}"
                     for key, cond in (item.get("condition_results") or {}).items()]
            lines.append(f"- 证据(level={item.get('level')}): "
                         + ("; ".join(conds) if conds else "无条件") + f" [asset={item.get('asset_id')}]")
        if packet.gaps:
            lines.append("- 无法确认: " + "；".join(str(g.get("reason", "")) for g in packet.gaps))
        evidence_text = "\n".join(lines) if lines else "(无证据)"
        prompt = (
            "你是 Sentrix。基于以下家庭记忆证据回答用户，只使用这些证据，"
            "明确哪些是确定的（matched）、哪些只是可能（possible）、哪些无法确认（unknown）。"
            "不要编造证据外的事实，不要提到数据库或工具。\n\n证据：\n"
            + evidence_text + "\n\n用户：" + str(message)
        )
        try:
            with _Perf.measure("answer"):
                if self.router is not None:
                    text = self.router.chat("answer", prompt, json_mode=False)
                else:
                    text = self.gamma.chat(prompt, json_mode=False, role="answer")
            _Perf.count("answer_calls")
            answer = str(text or "").strip()
            if answer:
                return answer, []
        except Exception:
            pass
        return ("当前记忆中没有找到足够匹配的原始证据。", [])

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
    def _human_condition_text(key, status):
        """User-facing text for a condition — never exposes the internal key
        (e.g. ``clothing:浅黄``), the ANN score, the trace or table names."""
        value = key.split(":", 1)[1] if ":" in key else key
        if status == "matched":
            return f"记录中有「{value}」"
        if status == "possible":
            return f"记录中可能有「{value}」，但无法完全确认"
        return "目前无法确认其中的关键活动或视觉细节。"

    @staticmethod
    def _allowed_facts(packet):
        # Phase R R6: one fact per condition_key regardless of how many Assets
        # hit it; evidence_ids are the union across assets.  Prevents the
        # album1-01 "记录支持X" ten-times-repeat failure mode.
        facts, possibilities, unknowns = [], [], []
        facts_by_key: dict[str, dict] = {}
        possible_by_key: dict[str, dict] = {}
        for item in packet.assets:
            for key, condition in item.get("condition_results", {}).items():
                status = condition.get("status")
                bucket = facts_by_key if status == "matched" else possible_by_key if status == "possible" else None
                if bucket is None:
                    continue
                entry = bucket.get(key)
                if entry is None:
                    entry = {
                        "text": ThinAgentRuntime._human_condition_text(key, status),
                        "status": status, "condition_key": key, "evidence_ids": [],
                    }
                    bucket[key] = entry
                for evidence_id in item.get("evidence_ids", []):
                    if evidence_id not in entry["evidence_ids"]:
                        entry["evidence_ids"].append(evidence_id)
        facts.extend(facts_by_key.values())
        possibilities.extend(possible_by_key.values())
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
