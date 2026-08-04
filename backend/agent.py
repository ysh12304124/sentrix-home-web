import json
import re
import uuid

from .model_clients import ClipAdapter, GammaClient

VECTOR_EVIDENCE_MIN_SCORE = 0.35


def contains(value, query):
    value = str(value or "").lower()
    normalized_query = str(query or "").strip().lower()
    if not normalized_query:
        return True
    if normalized_query in value:
        return True
    terms = [term for term in re.findall(r"[\w\u4e00-\u9fff]+", normalized_query) if len(term) > 1]
    identifiers = [term for term in terms if "_" in term or any(character.isdigit() for character in term)]
    if identifiers and any(term in value for term in identifiers):
        return True
    if bool(terms) and all(term in value for term in terms):
        return True
    # Chinese questions often contain one continuous block (for example,
    # "餐桌旁发生了什么"). Recover meaningful 2-6 character clues without
    # falling back to unrelated records merely because the full sentence differs.
    semantic_query = re.sub(r"(发生了什么|在哪里|是什么|有哪些|相关证据|这张照片|这张图片|图片|照片|请问|吗|呢|的)", "", normalized_query)
    chinese_blocks = re.findall(r"[\u4e00-\u9fff]{2,}", semantic_query)
    clues = {
        block[index:index + size]
        for block in chinese_blocks
        for size in range(2, min(6, len(block)) + 1)
        for index in range(len(block) - size + 1)
        if block[index:index + size] not in {"什么", "哪里", "如何", "哪些", "发生", "图片"}
    }
    matched_clues = {clue for clue in clues if clue in value}
    required_clues = 2 if any(len(block) >= 6 for block in chinese_blocks) else 1
    return len(matched_clues) >= required_clues


class MemoryAgent:
    def __init__(self, store, gamma=None, clip=None):
        self.store = store
        self.gamma = gamma or GammaClient()
        self.clip = clip or ClipAdapter()
        self._conversations = {}
        self._dialogue_states = {}
        self._conversation_limit = 8

    @staticmethod
    def classify_intent(message, feedback=None):
        value = str(message or "").strip()
        if any(token in value for token in ("我说的是", "指的是", "不是", "而是", "澄清", "继续")):
            return "clarification"
        if feedback or any(token in value for token in ("纠正", "更正", "实际是", "应该是", "记错了")):
            return "feedback"
        return "query"

    @staticmethod
    def _looks_like_memory_question(message):
        value = str(message or "").strip()
        return any(token in value for token in (
            "照片", "图片", "相册", "回忆", "记得", "去年", "前年", "我们", "家里", "谁", "哪里",
            "什么时候", "发生", "介绍", "时间线", "比较", "推荐", "证据", "原图", "地点", "人物",
        ))

    def _household_memory_identity(self, scope_id=None):
        """Build a compact long-term memory primer for natural conversation."""
        people = [item for item in self.store.list_entities(scope_id=scope_id) if item.get("entity_type") == "person" and item.get("status") == "confirmed"]
        profiles = [self.store.get_semantic_profile(item["id"]) for item in people]
        groups = self.store.list_semantic_entity_groups(scope_id)
        events = self.store.list_events(12, scope_id=scope_id)
        return {
            "family_members": [{"id": item["id"], "name": item.get("canonical_name"), "role": item.get("family_role")} for item in people[:16]],
            "family_profiles": [item for item in profiles if item][:12],
            "places_and_things": [{"name": item["canonical_name"], "type": item["entity_type"], "labels": item["source_labels"][:5]} for item in groups[:40]],
            "recent_shared_memories": [{"id": item["id"], "title": item.get("title"), "summary": item.get("summary"), "time": item.get("time_start"), "place": item.get("place")} for item in events[:12]],
        }

    def _normal_chat_answer(self, message, conversation_context="", scope_id=None):
        value = str(message or "").strip()
        if not hasattr(self.gamma, "chat"):
            return "我在听。"
        prompt = """你是 Sentrix，一个中性、自然的家庭数字助手。你长期记得这个家庭已经整理出的成员、共同经历、地点和物件，但不模仿任何家庭成员。
请像一个熟悉家庭生活的助手一样自然交谈，不要把回答写成检索结果，不要主动讲工具、数据库、证据或置信度。可以共情、讨论和提出有帮助的下一步；只有在下方长期记忆中存在时，才把具体家庭往事当作事实说出。对不在记忆中的细节保持自然的不确定性，不要编造。
家庭长期记忆：""" + json.dumps(self._household_memory_identity(scope_id), ensure_ascii=False) + "\n最近对话：" + str(conversation_context or "")[-1200:] + "\n用户：" + value + "\n请直接自然回答，不要使用 Markdown 列表。"
        try:
            answer = str(self.gamma.chat(prompt) or "").strip()
            return answer or "我在听。"
        except Exception:
            return "我在听。"

    def _fallback_plan(self, message, feedback=None):
        intent = self.classify_intent(message, feedback)
        if intent == "feedback":
            return {"mode": "feedback", "tools": ["record_feedback"], "show_images": False, "reason": "用户正在纠正或确认记忆"}
        if intent == "clarification":
            return {"mode": "clarify", "tools": ["resolve_constraints", "request_clarification"], "show_images": False, "reason": "用户在补充或修正上文指代"}
        if not self._looks_like_memory_question(message):
            return {"mode": "chat", "tools": [], "show_images": False, "reason": "没有明确的家庭记忆请求"}
        tools = ["resolve_constraints", "find_events"]
        if self._is_recall_recommendation_query(message):
            tools.append("suggest_recall")
        elif self._is_timeline_query(message):
            tools.append("trace_timeline")
        elif self._is_evidence_request(message):
            tools.append("open_evidence")
        return {"mode": "memory", "tools": tools, "show_images": self._is_evidence_request(message), "reason": "问题要求家庭记忆或可验证事实"}

    def _plan_turn(self, message, conversation_context="", feedback=None):
        """Ask the model for a bounded plan, then validate every capability.

        The local deterministic planner remains available when the model is
        unavailable or returns malformed JSON.  The model receives no database
        evidence at this stage, so it cannot fabricate a result.
        """
        fallback = self._fallback_plan(message, feedback)
        if feedback or not hasattr(self.gamma, "chat"):
            return fallback | {"planner": "deterministic"}
        prompt = "你是 Sentrix 家庭数字助手的行动规划器。你的长期记忆来自家庭成员、共同经历、地点、物件与关系；只判断本轮是否需要读取更具体的家庭记忆，不回答事实，也不要调用工具。\n"
        prompt += "可选 mode: chat, memory, feedback, clarify。可选工具: resolve_constraints, describe_entity, find_events, trace_timeline, compare_memories, suggest_recall, open_evidence, request_clarification, record_feedback。\n"
        prompt += "普通聊天、情感支持、建议和延续对话可以是 chat；只有需要准确回忆具体家庭经历、人物、地点、时间、关系或用户要求依据时才 memory。图片只在问题确实需要视觉依据时展示。\n"
        prompt += f"用户消息：{message}\n最近对话：{conversation_context[-800:]}\n"
        prompt += '只返回 JSON：{"mode":"chat|memory|feedback|clarify","tools":["..."],"show_images":false,"reason":"..."}'
        try:
            raw = self.gamma.chat(prompt)
            parsed = json.loads(raw) if isinstance(raw, str) else raw
        except Exception:
            return fallback | {"planner": "deterministic"}
        if not isinstance(parsed, dict):
            return fallback | {"planner": "deterministic"}
        allowed_modes = {"chat", "memory", "feedback", "clarify"}
        allowed_tools = {"resolve_constraints", "describe_entity", "find_events", "trace_timeline", "compare_memories", "suggest_recall", "open_evidence", "request_clarification", "record_feedback"}
        mode = parsed.get("mode") if parsed.get("mode") in allowed_modes else fallback["mode"]
        # A model may choose a narrower memory tool, but it cannot discard an
        # explicit memory/feedback request as casual chat.
        if fallback["mode"] in {"memory", "feedback", "clarify"} and mode == "chat":
            mode = fallback["mode"]
        tools = [item for item in parsed.get("tools", []) if item in allowed_tools]
        if mode == "chat":
            tools = []
        elif mode == "memory":
            tools = list(dict.fromkeys(["resolve_constraints", *tools]))
            if len(tools) == 1:
                tools.append("find_events")
        elif mode == "feedback":
            tools = ["record_feedback"]
        else:
            tools = list(dict.fromkeys(["resolve_constraints", *tools, "request_clarification"]))
        return {
            "mode": mode, "tools": tools,
            "show_images": bool(parsed.get("show_images")) and mode == "memory",
            "reason": str(parsed.get("reason") or fallback["reason"])[:240], "planner": "model",
        }

    @staticmethod
    def _query_terms(query):
        value = str(query or "").lower()
        words = re.findall(r"[a-z0-9_]{2,}|[\u4e00-\u9fff]{2,}", value)
        terms = set(words)
        for word in words:
            if re.fullmatch(r"[\u4e00-\u9fff]+", word):
                terms.update(word[index:index + size] for size in range(2, min(5, len(word)) + 1) for index in range(len(word) - size + 1))
        return terms

    def _evidence_relevance(self, query, item, focused_entity_ids=()):
        terms = self._query_terms(query)
        text = json.dumps(item, ensure_ascii=False).lower()
        lexical = sum(term in text for term in terms) / max(1, len(terms))
        confidence = float(item.get("confidence", 0.62) or 0.62)
        entity_bonus = 0.18 if item.get("person_id") in set(focused_entity_ids or ()) else 0.0
        direct_bonus = 0.28 if item.get("kind") == "observation" and lexical >= 0.25 else 0.0
        return round(min(1.0, 0.15 + 0.52 * lexical + 0.18 * confidence + entity_bonus + direct_bonus), 4)

    def _rank_evidence_for_turn(self, query, result, plan, focused_entity_ids=()):
        ranked = []
        for item in result.get("evidence", []):
            value = dict(item)
            value["relevance"] = self._evidence_relevance(query, value, focused_entity_ids)
            ranked.append(value)
        ranked.sort(key=lambda item: (-item["relevance"], -float(item.get("confidence", 0) or 0), item.get("id", "")))
        result["evidence"] = ranked
        image_candidates = [item for item in ranked if item.get("kind") == "observation" and item.get("media_type") == "image" and item.get("asset_id") and item["relevance"] >= 0.42]
        result["image_results"] = self._image_results(image_candidates[:3]) if plan.get("show_images") else []
        result["evidence_presentation"] = {
            "image_limit": 3 if plan.get("show_images") else 0,
            "minimum_relevance": 0.42,
            "shown_image_count": len(result["image_results"]),
            "ranked_evidence_count": len(ranked),
        }
        result["evidence_layers"] = {
            "answers": result.get("evidence_layers", {}).get("answers", []),
            "people": [item for item in ranked if item["kind"] in {"person", "semantic_claim"}],
            "events": [item for item in ranked if item["kind"] == "event"],
            "claims": [item for item in ranked if item["kind"] in {"fact", "semantic_claim"}],
            "appearance": [item for item in ranked if item["kind"] == "person_appearance"],
            "observations": [item for item in ranked if item["kind"] == "observation"],
            "assets": [{"kind": "asset", "id": item["asset_id"]} for item in ranked if item.get("asset_id")],
            "gaps": result.get("evidence_layers", {}).get("gaps", []),
        }
        return result

    @staticmethod
    def _query_date(query):
        value = str(query or "")
        match = re.search(r"(20\d{2})[-/.年](\d{1,2})[-/.月](\d{1,2})", value)
        if not match:
            return None
        return f"{int(match.group(1)):04d}-{int(match.group(2)):02d}-{int(match.group(3)):02d}"

    @staticmethod
    def _is_timeline_query(query):
        return any(token in str(query or "") for token in ("时间线", "时间轴", "先后", "经历", "历程"))

    @staticmethod
    def _is_entity_introduction_query(query, focused_people):
        return bool(focused_people) and any(token in str(query or "") for token in ("介绍", "是谁", "了解", "档案", "画像"))

    @staticmethod
    def _is_comparison_query(query, focused_people):
        return len(focused_people) >= 2 and any(token in str(query or "") for token in ("比较", "区别", "不同", "共同", "对比"))

    @staticmethod
    def _is_recall_recommendation_query(query):
        return any(token in str(query or "") for token in ("推荐", "回顾", "回忆一下", "看看回忆"))

    @staticmethod
    def _is_evidence_request(query):
        return any(token in str(query or "") for token in ("证据", "原图", "照片", "图片", "依据", "为什么"))

    def _tool_trace(self, query, retrieved, evidence_count=0, insufficient=False):
        """Describe the bounded read-only memory tools used for this turn."""
        intent = retrieved.get("intent", {})
        focused_people = retrieved.get("focused_people", [])
        constraints = {
            "people": [item["id"] for item in focused_people],
            "date": self._query_date(query),
            "dimension": intent.get("dimension"),
            "event_filter": bool(intent.get("event_filter")),
        }
        trace = [{"tool": "resolve_constraints", "permission": "read", "status": "complete", "constraints": constraints}]
        if self._is_comparison_query(query, focused_people):
            trace.append({"tool": "compare_memories", "permission": "read", "status": "complete", "entity_ids": constraints["people"]})
        elif self._is_recall_recommendation_query(query):
            trace.append({"tool": "suggest_recall", "permission": "read", "status": "complete" if evidence_count else "requires_anchor", "entity_ids": constraints["people"]})
        elif self._is_entity_introduction_query(query, focused_people):
            trace.append({"tool": "describe_entity", "permission": "read", "status": "complete", "entity_ids": constraints["people"]})
        elif self._is_timeline_query(query):
            trace.append({"tool": "find_events", "permission": "read", "status": "complete", "event_count": len(retrieved.get("events", []))})
            trace.append({"tool": "trace_timeline", "permission": "read", "status": "complete", "event_count": len(retrieved.get("events", []))})
        else:
            trace.append({"tool": "find_events", "permission": "read", "status": "complete", "event_count": len(retrieved.get("events", []))})
        if self._is_evidence_request(query) or evidence_count:
            trace.append({"tool": "open_evidence", "permission": "read", "status": "complete" if evidence_count else "empty", "evidence_count": evidence_count})
        if insufficient:
            trace.append({"tool": "request_clarification", "permission": "read", "status": "required", "reason": "insufficient_evidence"})
        return trace

    @staticmethod
    def _is_contextual_follow_up(message):
        value = str(message or "").strip()
        return any(token in value for token in (
            "然后", "后来", "接着", "继续", "为什么", "具体", "详细", "还有呢", "那呢",
            "他呢", "她呢", "它呢", "这里呢", "那里呢", "这段呢", "那个呢",
        ))

    def _has_explicit_entity_reference(self, message, scope_id, active_entity_ids):
        """A named entity starts a new subject unless it is already in focus."""
        value = str(message or "")
        for entity in self.store.list_entities(scope_id=scope_id):
            name = str(entity.get("canonical_name") or "").strip()
            if name and name in value and entity["id"] not in set(active_entity_ids or []):
                return True
        return False

    @staticmethod
    def _dialogue_style(query, result):
        if result.get("insufficient_evidence"):
            return "clarifying"
        if any(token in str(query or "") for token in ("介绍", "时间线", "回顾", "比较", "经历", "然后", "后来", "继续")):
            return "narrative"
        return "concise"

    @staticmethod
    def _narrative_answer(query, result):
        if result.get("insufficient_evidence"):
            return result
        events = [item for item in result.get("evidence", []) if item.get("kind") == "event"]
        if not events:
            return result
        details = []
        for event in events[:3]:
            when = str(event.get("time_start") or "").replace("T", " ")[:16]
            place = event.get("place") or "地点未标注"
            summary = event.get("summary") or event.get("event_id")
            details.append(" · ".join(item for item in (when, place, summary) if item))
        prefix = "根据目前可回溯的记忆，"
        if any(token in str(query or "") for token in ("比较", "区别", "不同", "共同")):
            prefix = "从已确认的共同与独立事件来看，"
        elif any(token in str(query or "") for token in ("回顾", "推荐", "回忆")):
            prefix = "沿着有原始证据的回忆，可以这样回顾："
        result["answer"] = prefix + "；".join(details) + "。"
        result["model"] = "sentrix-dialogue-evidence"
        return result

    @staticmethod
    def _evidence_order(evidence):
        source_levels = {
            "fact": ("confirmed_fact", 0),
            "semantic_claim": ("semantic_claim", 1),
            "event": ("derived_event", 2),
            "observation": ("original_observation", 3),
            "person_appearance": ("original_person_appearance", 4),
        }
        ordered = []
        for item in evidence or []:
            source_level, source_rank = source_levels.get(item.get("kind"), ("other", 5))
            ordered.append({
                "id": item.get("id"), "kind": item.get("kind"), "source_level": source_level,
                "time": item.get("time_start") or item.get("captured_at"),
                "confidence": float(item.get("confidence", 0) or 0),
                "event_id": item.get("event_id"), "asset_id": item.get("asset_id"),
                "source_rank": source_rank,
            })
        return [
            {key: value for key, value in item.items() if key != "source_rank"}
            for item in sorted(ordered, key=lambda item: (item["source_rank"], item["time"] or "9999", -item["confidence"], item["id"] or ""))
        ]

    @staticmethod
    def _contextual_follow_up_answer(result):
        events = [item for item in result.get("evidence", []) if item.get("kind") == "event"]
        if not events:
            return result
        summaries = [item.get("summary") or item.get("event_id") for item in events[:3]]
        result["answer"] = "沿着刚才这段记忆，仍能确认的是：" + "；".join(summaries) + "。"
        result["model"] = "sentrix-dialogue-evidence"
        result["confidence"] = min(0.9, max(float(item.get("confidence", 0.62) or 0.62) for item in events))
        return result

    def _answer_from_verified_events(self, query, events):
        evidence = []
        for event in events[:8]:
            detail = self.store.get_event_detail(event["id"]) or {}
            observations = detail.get("observations", [])
            evidence.append({
                "kind": "event", "id": event["id"], "event_id": event["id"], "summary": event.get("summary", ""),
                "time_start": event.get("time_start"), "place": event.get("place"),
                "confidence": event.get("confidence", 0.62),
                "asset_ids": [item.get("asset_id") for item in observations if item.get("asset_id")],
            })
            for observation in observations[:8]:
                asset = observation.get("asset") or self.store.get_asset(observation["asset_id"]) or {}
                evidence.append({
                    "kind": "observation", "id": observation["id"], "observation_id": observation["id"], "event_id": event["id"],
                    "asset_id": observation.get("asset_id"), "file_name": asset.get("file_name"), "media_type": asset.get("media_type"),
                    "captured_at": observation.get("captured_at"), "caption": observation.get("caption"),
                    "transcript": observation.get("transcript"), "confidence": observation.get("confidence", 0),
                    "raw": observation.get("raw_json", {}),
                })
        result = {
            "answer": "", "confidence": 0.0, "insufficient_evidence": not bool(evidence), "model": "sentrix-dialogue-evidence",
            "modelEvidence": [], "evidence": evidence, "query": query,
        }
        result = self._contextual_follow_up_answer(result)
        result["retrieval_trace"] = [
            {"stage": "dialogue_state", "status": "complete", "counts": {"verified_events": len(events)}},
            {"stage": "evidence_validation", "status": "complete", "counts": {"evidence": len(evidence)}},
        ]
        result["evidence_layers"] = {
            "answers": [{"id": None, "text": result["answer"]}], "people": [],
            "events": [item for item in evidence if item["kind"] == "event"], "claims": [], "appearance": [],
            "observations": [item for item in evidence if item["kind"] == "observation"],
            "assets": [{"kind": "asset", "id": item["asset_id"]} for item in evidence if item.get("asset_id")], "gaps": [],
        }
        return result

    @staticmethod
    def _recall_recommendation_answer(events):
        summaries = [item.get("summary") or item.get("title") or item.get("id") for item in events[:3]]
        return "根据已锚定的本地事件证据，推荐回顾：" + "；".join(summaries) + "。"

    def _clarification_candidates(self, query, scope_id=None):
        value = str(query or "")
        candidates_by_type = {}
        for entity in self.store.list_entities(scope_id=scope_id):
            name = str(entity.get("canonical_name") or "")
            if not name:
                continue
            clues = {name[index:index + size] for size in range(2, min(4, len(name)) + 1) for index in range(len(name) - size + 1)}
            if not any(clue in value for clue in clues):
                continue
            candidates_by_type.setdefault(entity["entity_type"], []).append({
                "id": entity["id"], "name": name, "entity_type": entity["entity_type"],
                "evidence_count": entity.get("evidence_count", 0), "confidence": entity.get("confidence", 0),
            })
        eligible = [
            (entity_type, values)
            for entity_type, values in candidates_by_type.items()
            if len(values) >= 2
        ]
        if not eligible:
            return []
        entity_type, values = max(
            eligible,
            key=lambda item: (sum(value["evidence_count"] for value in item[1]), len(item[1]), item[0]),
        )
        return sorted(values, key=lambda item: (-item["evidence_count"], -item["confidence"], item["name"]))[:6]

    def _comparison_answer(self, people):
        event_sets = {person["id"]: set(self.store.entity_event_ids(person["id"])) for person in people[:2]}
        first, second = people[:2]
        shared = event_sets[first["id"]].intersection(event_sets[second["id"]])
        first_only = event_sets[first["id"]] - event_sets[second["id"]]
        second_only = event_sets[second["id"]] - event_sets[first["id"]]
        answer = (
            f"根据已确认的人物事件证据，{first['canonical_name']}与{second['canonical_name']}有 {len(shared)} 个共同事件；"
            f"{first['canonical_name']}另有 {len(first_only)} 个已关联事件，{second['canonical_name']}另有 {len(second_only)} 个已关联事件。"
        )
        return answer, shared | first_only | second_only

    @staticmethod
    def _object_values_for_query(query, objects):
        value = str(query or "")
        candidates = set()
        for block in re.findall(r"[\u4e00-\u9fff]{2,}", value):
            for size in range(2, min(5, len(block)) + 1):
                candidates.update(block[index:index + size] for index in range(len(block) - size + 1))
        matches = []
        for item in objects or []:
            text = str(item or "").strip()
            if text and (text in value or any(token in text for token in candidates)):
                matches.append(text)
        return list(dict.fromkeys(matches))

    def _private_place_replacements(self, scope_id=None):
        replacements = {}
        for entity in self.store.list_entities(scope_id=scope_id, public=False):
            if entity.get("entity_type") != "place":
                continue
            properties = {item["property_key"]: item for item in self.store.list_entity_properties(entity["id"])}
            private = properties.get("private_flag")
            if not private or private.get("value") is not True:
                continue
            alias = properties.get("alias", {}).get("value") or "私密地点"
            replacements[entity["canonical_name"]] = str(alias)
        return replacements

    @staticmethod
    def _redact_private_places(value, replacements):
        if not replacements:
            return value
        if isinstance(value, str):
            for private_name, alias in replacements.items():
                value = value.replace(private_name, alias)
            return value
        if isinstance(value, list):
            return [MemoryAgent._redact_private_places(item, replacements) for item in value]
        if isinstance(value, dict):
            return {key: MemoryAgent._redact_private_places(item, replacements) for key, item in value.items()}
        return value

    def _vector_hit_has_textual_anchor(self, hit, events):
        """A semantic similarity score is a ranking signal, not standalone evidence."""
        event_id = hit.get("source_id") if hit.get("source_type") == "event" else hit.get("metadata", {}).get("event_id")
        if event_id:
            event = next((item for item in events if item["id"] == event_id), None)
            if event and contains(json.dumps(event, ensure_ascii=False), hit.get("query", "")):
                return True
            detail = self.store.get_event_detail(event_id) or {}
            return any(
                contains(json.dumps(observation, ensure_ascii=False), hit.get("query", ""))
                for observation in detail.get("observations", [])
            )
        if hit.get("source_type") == "observation":
            observation = self.store.get_observation(hit.get("source_id"))
            return bool(observation and contains(json.dumps(observation, ensure_ascii=False), hit.get("query", "")))
        return False

    def retrieve(self, query, scope_id=None):
        events = self.store.list_events(100, scope_id=scope_id)
        observations = self.store.list_observations(1000, scope_id=scope_id)
        facts = self.store.list_facts(200, scope_id=scope_id)
        persons = self.store.list_persons()
        entities = self.store.list_entities(scope_id=scope_id)
        focused_people = [
            entity for entity in entities
            if entity.get("entity_type") == "person"
            and entity.get("status") != "rejected"
            and entity.get("canonical_name")
            and entity["canonical_name"] in str(query or "")
        ]
        focused_ids = {entity["id"] for entity in focused_people}
        focused_event_ids = {
            event_id for entity_id in focused_ids for event_id in self.store.entity_event_ids(entity_id)
        }
        dimension = self._query_dimension(query)
        date = self._query_date(query)
        place_event_ids = {
            event["id"] for event in events
            if event.get("place") and str(event["place"]) in str(query or "")
        }
        date_event_ids = {
            event["id"] for event in events
            if date and str(event.get("time_start") or "")[:10] == date
        }
        object_observations = [
            observation for observation in observations
            if dimension == "object" and self._object_values_for_query(query, observation.get("objects") or [])
        ]
        object_observation_ids = {item["id"] for item in object_observations}
        object_event_ids = {
            event["id"] for event in events
            if object_observation_ids.intersection(event.get("observation_ids", []))
        }
        constrained_event_ids = {event["id"] for event in events}
        has_event_constraint = False
        for event_ids in (focused_event_ids, place_event_ids, date_event_ids, object_event_ids):
            if event_ids:
                constrained_event_ids.intersection_update(event_ids)
                has_event_constraint = True
        local_events = [event for event in events if event["id"] in constrained_event_ids] if has_event_constraint else [
            event for event in events if contains(json.dumps(event, ensure_ascii=False), query)
        ]
        local_observations = []
        for item in observations:
            asset = self.store.get_asset(item["asset_id"]) or {}
            searchable = {**item, "asset_file_name": asset.get("file_name", "")}
            if contains(json.dumps(searchable, ensure_ascii=False), query):
                searchable_text = json.dumps(searchable, ensure_ascii=False).lower()
                normalized_query = str(query or "").strip().lower()
                exact = 2 if normalized_query and normalized_query in searchable_text else 0
                terms = [term for term in re.findall(r"[\w\u4e00-\u9fff]+", normalized_query) if len(term) > 1]
                coverage = sum(term in searchable_text for term in terms)
                local_observations.append((exact, coverage, item))
        local_facts = [item for item in facts if contains(json.dumps(item, ensure_ascii=False), query)]
        # Structured semantic/event constraints are authoritative and cheaper
        # than a full vector scan. Only fall back to vector recall when they do
        # not produce usable local evidence.
        structured_hit = bool(has_event_constraint or local_observations or local_facts or focused_ids)
        query_embedding = []
        vector_hits = []
        vector_candidates = []
        vector_available = bool(getattr(self.clip, "evidence_ready", True))
        if not structured_hit and vector_available:
            query_embedding = self.clip.embed_text(query)
            vector_candidates = self.store.search_vectors("episodic", query_embedding, 12, scope_id=scope_id) + self.store.search_vectors("semantic", query_embedding, 12, scope_id=scope_id)
            scored_hits = [item for item in vector_candidates if float(item.get("score", 0) or 0) >= VECTOR_EVIDENCE_MIN_SCORE]
            for item in scored_hits:
                item["query"] = query
            vector_hits = [item for item in scored_hits if self._vector_hit_has_textual_anchor(item, events)]
        vector_event_ids = [item["source_id"] for item in vector_hits if item["source_type"] == "event"]
        vector_event_ids.extend(item.get("metadata", {}).get("event_id") for item in vector_hits if item.get("metadata", {}).get("event_id"))
        vector_events = [event for event in events if event["id"] in vector_event_ids]
        observation_event_ids = {item[2]["id"] for item in local_observations}
        observation_events = [event for event in events if observation_event_ids.intersection(event.get("observation_ids", []))]
        relationships = self.store.list_relationships(scope_id=scope_id)
        semantic_claims = []
        profiles = []
        if focused_ids:
            for entity_id in focused_ids:
                semantic_claims.extend(self.store.list_semantic_claims(entity_id, 500))
                profile = self.store.get_semantic_profile(entity_id)
                if profile:
                    profiles.append(profile)
        else:
            semantic_claims = [
                claim for claim in self.store.list_semantic_claims(None, 500)
                if (not scope_id or claim.get("scope_id") == scope_id) and contains(json.dumps(claim, ensure_ascii=False), query)
            ]
        if dimension == "clothing" and focused_ids:
            semantic_claims = [claim for claim in semantic_claims if claim.get("dimension") == "clothing"]
        appearance_evidence = []
        if dimension == "clothing" and focused_ids:
            for entity_id in focused_ids:
                appearance_evidence.extend(self.store.list_person_appearance_evidence(entity_id))
        ranked_observations = [item for _, _, item in sorted(local_observations, key=lambda value: (-value[0], -value[1], value[2]["id"]))]
        if object_observations:
            ranked_observations = object_observations
        elif has_event_constraint:
            relevant_observation_ids = {
                observation_id for event in local_events for observation_id in event.get("observation_ids", [])
            }
            ranked_observations = [item for item in observations if item["id"] in relevant_observation_ids]
        return {
            "events": local_events if has_event_constraint else (local_events or observation_events or vector_events),
            "observations": ranked_observations,
            "focus_observation_ids": observation_event_ids,
            "facts": local_facts,
            "semantic_claims": semantic_claims,
            "appearance_evidence": appearance_evidence,
            "profiles": profiles,
            "focused_people": focused_people,
            "persons": persons[:50],
            "entities": entities[:100],
            "relationships": relationships[:100],
            "vectors": vector_hits,
            "vector_candidate_count": len(vector_candidates),
            "vector_skipped": structured_hit,
            "vector_available": vector_available,
            "intent": {
                "activity": self._is_activity_query(query),
                "dimension": dimension,
                "event_filter": has_event_constraint,
            },
            "scope_id": scope_id,
        }

    @staticmethod
    def _is_pending_identity_query(query):
        value = str(query or "")
        return any(token in value for token in ("待命名", "未命名人物", "未命名成员", "候选人物", "谁还没命名"))

    def _pending_identity_answer(self, query, scope_id=None):
        """Return an evidence-only review request without exposing candidate labels."""
        pending_people = [
            entity for entity in self.store.list_entities(scope_id=scope_id)
            if entity.get("entity_type") == "person" and entity.get("status") == "pending"
        ]
        clusters_by_entity = {}
        for cluster in self.store.list_face_clusters():
            if scope_id and cluster.get("scope_id") != scope_id:
                continue
            if cluster.get("entity_id"):
                clusters_by_entity.setdefault(cluster["entity_id"], []).append(cluster)

        evidence = []
        candidate_asset_ids = []
        for index, entity in enumerate(pending_people, 1):
            clusters = clusters_by_entity.get(entity["id"], [])
            sample_asset_ids = list(dict.fromkeys(
                sample.get("asset_id") for cluster in clusters for sample in cluster.get("samples", []) if sample.get("asset_id")
            ))
            candidate_asset_ids.extend(sample_asset_ids)
            evidence.append({
                "kind": "person", "id": entity["id"], "person_id": entity["id"],
                "name": f"待命名成员 {index}", "status": "pending",
                "confidence": entity.get("confidence", 0),
                "cluster_ids": [cluster["id"] for cluster in clusters],
                "asset_ids": sample_asset_ids,
            })
        gap = self.store.create_query_gap(
            query, "identity", list(dict.fromkeys(candidate_asset_ids)), [item["id"] for item in evidence],
        )
        count = len(evidence)
        answer = (
            f"当前有 {count} 位待命名成员，系统尚未确认其姓名。"
            "请在人脸与关系页面查看每个簇的原始样本后确认或驳回；在确认前不会把候选身份写入长期人物记忆。"
            if count else "当前范围内没有待命名成员。"
        )
        return {
            "answer": answer,
            "confidence": 0.0 if count else 1.0,
            "insufficient_evidence": bool(count),
            "model": "sentrix-identity-review",
            "modelEvidence": [],
            "evidence": evidence,
            "retrieval_trace": [
                {"stage": "identity_review", "status": "complete", "counts": {"pending_people": count, "query_gaps": 1 if count else 0}},
                {"stage": "vector", "status": "skipped", "counts": {"hits": 0}},
                {"stage": "evidence_validation", "status": "complete", "counts": {"evidence": len(evidence)}},
            ],
            "evidence_layers": {
                "answers": [{"id": None, "text": answer}], "people": evidence, "events": [], "claims": [],
                "appearance": [], "observations": [], "assets": [
                    {"kind": "asset", "id": asset_id} for asset_id in dict.fromkeys(candidate_asset_ids)
                ], "gaps": [gap] if count else [],
            },
            "query": query,
            "query_gap_id": gap["id"] if count else None,
        }

    def context(self, retrieved):
        lines = [
            "Sentrix evidence only. Evidence不足时明确说明，不得编造。",
            "每条 JSON 都是独立证据；引用时只能使用已有的 id、observation_id 或 asset_id。",
            "[PERSON_PROFILES]",
        ]
        for profile in retrieved.get("profiles", []):
            lines.append(json.dumps(profile, ensure_ascii=False))
        lines.extend([
            "[EVENTS]",
        ])
        for event in retrieved["events"]:
            asset_ids = []
            for observation_id in event.get("observation_ids", []):
                observation = self.store.get_observation(observation_id)
                if observation:
                    asset_ids.append(observation["asset_id"])
            lines.append(json.dumps({"id": event["id"], "title": event["title"], "time": event.get("time_start"), "place": event.get("place"), "summary": event.get("summary"), "observation_ids": event.get("observation_ids", []), "asset_ids": list(dict.fromkeys(asset_ids)), "participant_roles": event.get("participant_roles", [])}, ensure_ascii=False))
        lines.append("[OBSERVATIONS]")
        for observation in retrieved["observations"]:
            lines.append(json.dumps({"id": observation["id"], "asset_id": observation["asset_id"], "caption": observation.get("caption"), "ocr_text": observation.get("ocr_text"), "transcript": observation.get("transcript"), "event_type": observation.get("event_type"), "clothing": observation.get("clothing", []), "spatial_relations": observation.get("spatial_relations", []), "source_owner_id": observation.get("source_owner_id")}, ensure_ascii=False))
        lines.append("[FACTS]")
        for fact in retrieved["facts"]:
            lines.append(json.dumps({"id": fact["id"], "subject": fact["subject"], "predicate": fact["predicate"], "object": fact["object"], "status": fact["status"], "evidence_ids_json": fact.get("evidence_ids_json", [])}, ensure_ascii=False))
        lines.append("[ENTITIES]")
        lines.extend(json.dumps(entity, ensure_ascii=False) for entity in retrieved.get("entities", []))
        lines.append("[RELATIONSHIPS]")
        lines.extend(json.dumps(relationship, ensure_ascii=False) for relationship in retrieved.get("relationships", []))
        lines.append("[VECTOR_HITS]")
        lines.extend(json.dumps(hit, ensure_ascii=False) for hit in retrieved.get("vectors", []))
        lines.append("[SEMANTIC_CLAIMS]")
        lines.extend(json.dumps(claim, ensure_ascii=False) for claim in retrieved.get("semantic_claims", []))
        lines.append("[PERSON_APPEARANCE_EVIDENCE]")
        lines.extend(json.dumps(item, ensure_ascii=False) for item in retrieved.get("appearance_evidence", []))
        return "\n".join(lines)

    @staticmethod
    def _query_dimension(query):
        value = str(query or "").lower()
        if any(token in value for token in ("衣服", "穿着", "外套", "裤子", "裙子", "鞋", "帽子", "衣物")):
            return "clothing"
        if any(token in value for token in ("在哪里", "位置", "旁边", "左边", "右边", "前面", "后面")):
            return "spatial_relation"
        if any(token in value for token in ("拿着", "物品", "东西", "蛋糕", "礼物", "包", "麦克风", "眼镜", "相关证据")):
            return "object"
        return None

    def _refine_visual_memory(self, query, retrieved):
        dimension = self._query_dimension(query)
        if not dimension:
            return retrieved, None
        if dimension == "object":
            existing = [
                observation for observation in retrieved.get("observations", [])
                if self._object_values_for_query(query, observation.get("objects") or [])
            ]
            if existing:
                return retrieved, None
        candidates = []
        seen = set()
        for event in retrieved["events"][:8]:
            detail = self.store.get_event_detail(event["id"])
            for observation in (detail or {}).get("observations", []):
                if observation["id"] in seen or observation.get("asset", {}).get("media_type") != "image":
                    continue
                if dimension == "clothing" and observation.get("clothing"):
                    continue
                if dimension == "spatial_relation" and observation.get("spatial_relations"):
                    continue
                if dimension == "object" and observation.get("objects"):
                    continue
                seen.add(observation["id"])
                candidates.append(observation)
        candidate_asset_ids = [item["asset_id"] for item in candidates[:4]]
        if not candidates:
            return retrieved, None
        refined_ids = []
        for observation in candidates[:4]:
            asset = observation.get("asset") or self.store.get_asset(observation["asset_id"])
            if not asset or not asset.get("path") or not hasattr(self.gamma, "analyze_image_focus"):
                continue
            try:
                details = self.gamma.analyze_image_focus(asset["path"], dimension, {
                    "file_name": asset.get("file_name"),
                    "captured_at": asset.get("captured_at"),
                    "source_owner_id": asset.get("source_owner_id"),
                })
                if any(details.get(key) for key in ("clothing", "objects", "spatial_relations")):
                    updated = self.store.enrich_observation(observation["id"], details)
                    refined_ids.append(updated["id"])
                    text = " ".join(str(item) for key in ("clothing", "objects", "spatial_relations") for item in (updated.get(key) or []))
                    vector = self.clip.embed_text(text)
                    event_id = next((row["event_id"] for row in self.store._rows("SELECT event_id FROM event_observations WHERE observation_id = ?", (observation["id"],))), None)
                    self.store.upsert_vector("semantic", "observation", observation["id"], vector, self.clip.model_name, {"asset_id": observation["asset_id"], "event_id": event_id, "refined_dimension": dimension})
            except Exception:
                continue
        gap = self.store.create_query_gap(query, dimension, candidate_asset_ids, refined_ids)
        return self.retrieve(query, retrieved.get("scope_id")), gap

    @staticmethod
    def _is_activity_query(query):
        value = str(query or "")
        return any(token in value for token in ("活动", "参与", "参加", "出席", "经历", "做过"))

    @classmethod
    def _fallback_answer(cls, query, evidence):
        claims = [item for item in evidence if item["kind"] == "semantic_claim" and item.get("dimension") == "activity"]
        if claims and cls._is_activity_query(query):
            values = list(dict.fromkeys(item.get("value_text") for item in claims if item.get("value_text")))
            event_ids = list(dict.fromkeys(event_id for item in claims for event_id in item.get("supporting_event_ids", [])))
            return {
                "answer": "根据人物语义记忆，参与过：" + "；".join(values[:12]) + "。支撑事件：" + "、".join(event_ids[:8]) + "。",
                "confidence": max(float(item.get("confidence", 0.5) or 0.5) for item in claims),
                "insufficient_evidence": False,
            }
        clothing_claims = [item for item in evidence if item["kind"] == "semantic_claim" and item.get("dimension") == "clothing"]
        if clothing_claims and cls._query_dimension(query) == "clothing":
            values = list(dict.fromkeys(item.get("value_text") for item in clothing_claims if item.get("value_text")))
            event_ids = list(dict.fromkeys(event_id for item in clothing_claims for event_id in item.get("supporting_event_ids", [])))
            return {
                "answer": "根据人物语义记忆，曾穿着：" + "；".join(values[:20]) + "。支撑事件：" + "、".join(event_ids[:8]) + "。",
                "confidence": max(float(item.get("confidence", 0.5) or 0.5) for item in clothing_claims),
                "insufficient_evidence": False,
            }
        if cls._query_dimension(query) == "clothing":
            scene_observations = [item for item in evidence if item.get("kind") == "observation" and item.get("clothing")]
            references = list(dict.fromkeys(item.get("file_name") or item["id"] for item in scene_observations))
            return {
                "answer": "当前没有可归属到该人物的衣物事实；关联照片保留了场景级衣物观察，需人物级视觉确认后才能写入画像。场景证据：" + "、".join(references[:12]) + "。",
                "confidence": 0.0,
                "insufficient_evidence": True,
            }
        object_observations = []
        for item in evidence:
            if item.get("kind") != "observation":
                continue
            matches = cls._object_values_for_query(query, item.get("objects") or [])
            if matches:
                object_observations.append((item, matches))
        if object_observations and cls._query_dimension(query) == "object":
            values = list(dict.fromkeys(value for _, matches in object_observations for value in matches))
            references = list(dict.fromkeys(item.get("file_name") or item["id"] for item, _ in object_observations))
            return {
                "answer": "根据原始图片观察，发现：" + "；".join(values[:12]) + "。原始证据：" + "、".join(references[:12]) + "。",
                "confidence": 0.72,
                "insufficient_evidence": False,
            }
        observations = [item for item in evidence if item["kind"] == "observation"]
        events = [item for item in evidence if item["kind"] == "event"]
        if events:
            summaries = [item.get("summary") or item.get("id") for item in events[:8]]
            return {
                "answer": f"根据本地事件记忆，检索到 {len(events)} 个相关事件：" + "；".join(summaries) + "。",
                "confidence": 0.62,
                "insufficient_evidence": False,
            }
        if observations:
            summaries = [item.get("caption") or item.get("transcript") or item.get("file_name") or item["id"] for item in observations[:3]]
            references = [item.get("file_name") or item["id"] for item in observations[:3]]
            return {
                "answer": f"根据本地证据，检索到 {len(observations)} 条相关观察：" + "；".join(summaries) + "。原始证据：" + "、".join(references) + "。",
                "confidence": 0.62,
                "insufficient_evidence": False,
            }
        return {"answer": f"当前本地记忆没有找到能回答“{query}”的证据。", "confidence": 0.0, "insufficient_evidence": True}

    def answer(self, query, conversation_context=None, scope_id=None):
        if self._is_pending_identity_query(query):
            return self._pending_identity_answer(query, scope_id)
        retrieved = self.retrieve(query, scope_id)
        retrieved, query_gap = self._refine_visual_memory(query, retrieved)
        private_places = self._private_place_replacements(scope_id)
        public_retrieved = self._redact_private_places(retrieved, private_places)
        evidence = []
        seen = set()
        activity_query = self._is_activity_query(query)
        intent = public_retrieved.get("intent", {})
        recall_recommendation = self._is_recall_recommendation_query(query)
        for event in public_retrieved["events"][:8]:
            detail = self.store.get_event_detail(event["id"])
            if not detail:
                continue
            asset_ids = [item.get("asset_id") for item in detail["observations"] if item.get("asset_id")]
            item = {"kind": "event", "id": event["id"], "event_id": event["id"], "asset_ids": asset_ids, "summary": event.get("summary", ""), "time_start": event.get("time_start"), "place": event.get("place")}
            evidence.append(item)
            seen.add(event["id"])
            observations = sorted(detail["observations"], key=lambda item: item["id"] not in public_retrieved.get("focus_observation_ids", set()))
            if intent.get("dimension") == "object":
                observations = [
                    observation for observation in observations
                    if self._object_values_for_query(query, observation.get("objects") or [])
                ]
            for observation in observations[:8]:
                observation = self._redact_private_places(observation, private_places)
                asset = observation.get("asset") or {}
                evidence.append({"kind": "observation", "id": observation["id"], "observation_id": observation["id"], "event_id": event["id"], "asset_id": observation.get("asset_id"), "file_name": asset.get("file_name"), "media_type": asset.get("media_type"), "captured_at": observation.get("captured_at"), "caption": observation.get("caption"), "transcript": observation.get("transcript"), "clothing": observation.get("clothing", []), "objects": observation.get("objects", []), "spatial_relations": observation.get("spatial_relations", []), "source_owner_id": asset.get("source_owner_id"), "raw": observation.get("raw_json", {})})
        for fact in public_retrieved["facts"][:12]:
            evidence.append({"kind": "fact", "id": fact["id"], "fact_id": fact["id"], "subject": fact["subject"], "predicate": fact["predicate"], "object": fact["object"], "status": fact["status"], "evidence_ids": fact.get("evidence_ids_json", [])})
        claims = public_retrieved.get("semantic_claims", [])
        if activity_query:
            activity_claims = [claim for claim in claims if claim.get("dimension") == "activity"]
            other_claims = [claim for claim in claims if claim.get("dimension") != "activity"]
            claims = activity_claims + other_claims
        claim_limit = len(claims) if activity_query else 20
        for claim in claims[:claim_limit]:
            evidence.append({"kind": "semantic_claim", "id": claim["id"], "claim_id": claim["id"], "person_id": claim["person_id"], "dimension": claim["dimension"], "predicate": claim["predicate"], "value_text": claim["value_text"], "status": claim["status"], "evidence_ids": claim.get("evidence_ids_json", []), "supporting_event_ids": claim.get("supporting_event_ids_json", [])})
        if intent.get("dimension") == "clothing":
            for appearance in public_retrieved.get("appearance_evidence", []):
                evidence.append({
                    "kind": "person_appearance", "id": appearance["id"], "person_id": appearance["person_id"],
                    "face_instance_id": appearance["face_instance_id"], "observation_id": appearance["observation_id"],
                    "asset_id": appearance["asset_id"], "file_name": appearance.get("file_name"),
                    "crop_bbox": appearance.get("crop_bbox_json", []), "clothing": appearance.get("clothing_json", []),
                    "confidence": appearance.get("confidence", 0), "model": appearance.get("model_name"),
                })
        if not evidence:
            if recall_recommendation:
                result = {
                    "answer": "请先告诉我想回顾的人物、地点或日期，我会只从有原始证据的事件中推荐。",
                    "confidence": 0.0,
                    "insufficient_evidence": True,
                    "model": "sentrix-evidence-fallback",
                    "modelEvidence": [], "evidence": [], "clarification_candidates": [],
                }
                result["retrieval_trace"] = [
                    {"stage": "lexical", "status": "complete", "counts": {"events": 0, "observations": 0, "facts": 0}},
                    {"stage": "semantic", "status": "complete", "counts": {"claims": 0, "entities": len(public_retrieved.get("entities", [])), "relationships": 0}},
                    {"stage": "vector", "status": "skipped", "counts": {"hits": 0, "accepted": 0, "candidates": 0}},
                    {"stage": "evidence_validation", "status": "requires_anchor", "counts": {"evidence": 0, "query_gaps": 0}},
                ]
                result["evidence_layers"] = {"answers": [{"id": None, "text": result["answer"]}], "people": [], "events": [], "claims": [], "appearance": [], "observations": [], "assets": [], "gaps": []}
                result["query"] = query
                result["tool_trace"] = self._tool_trace(query, public_retrieved, insufficient=True)
                return result
            candidates = self._clarification_candidates(query, scope_id)
            candidate_asset_ids = list(dict.fromkeys(
                item.get("metadata", {}).get("asset_id") for item in public_retrieved.get("vectors", [])
                if item.get("metadata", {}).get("asset_id")
            ))
            gap = self.store.create_query_gap(query, intent.get("dimension") or "semantic", candidate_asset_ids, [])
            answer = (
                "当前有多个可能的实体，请确认你指的是：" + "、".join(item["name"] for item in candidates) + "。"
                if len(candidates) >= 2 else f"当前本地记忆没有找到能回答“{query}”的证据。"
            )
            result = {
                "answer": answer,
                "confidence": 0.0,
                "insufficient_evidence": True,
                "model": "sentrix-evidence-fallback",
                "modelEvidence": [],
                "evidence": [],
                "query_gap_id": gap["id"],
                "clarification_candidates": candidates if len(candidates) >= 2 else [],
            }
            result["retrieval_trace"] = [
                {"stage": "lexical", "status": "complete", "counts": {"events": 0, "observations": 0, "facts": 0}},
                {"stage": "semantic", "status": "complete", "counts": {"claims": 0, "entities": len(public_retrieved.get("entities", [])), "relationships": 0}},
                {"stage": "vector", "status": "unavailable" if not public_retrieved.get("vector_available", True) else "complete", "counts": {"hits": len(public_retrieved.get("vectors", [])), "accepted": len(public_retrieved.get("vectors", [])), "candidates": public_retrieved.get("vector_candidate_count", 0)}},
                {"stage": "evidence_validation", "status": "insufficient", "counts": {"evidence": 0, "query_gaps": 1}},
            ]
            result["evidence_layers"] = {"answers": [{"id": None, "text": result["answer"]}], "people": [], "events": [], "claims": [], "appearance": [], "observations": [], "assets": [], "gaps": [gap]}
            result["query"] = query
            result["tool_trace"] = self._tool_trace(query, public_retrieved, insufficient=True)
            return result
        deterministic_query = (
            (activity_query and public_retrieved.get("focused_people"))
            or (intent.get("dimension") == "clothing" and public_retrieved.get("focused_people"))
            or intent.get("dimension") == "object"
            or intent.get("event_filter")
        )
        comparison_query = self._is_comparison_query(query, public_retrieved.get("focused_people", []))
        if comparison_query:
            answer, compared_event_ids = self._comparison_answer(public_retrieved["focused_people"])
            result = {
                "answer": answer,
                "confidence": 0.9,
                "insufficient_evidence": False,
                "model": "sentrix-evidence",
                "evidence": [],
                "compared_event_ids": sorted(compared_event_ids),
            }
        elif recall_recommendation:
            result = {
                "answer": self._recall_recommendation_answer([item for item in evidence if item["kind"] == "event"]),
                "confidence": 0.75, "insufficient_evidence": False, "model": "sentrix-evidence", "evidence": [],
            }
        elif deterministic_query:
            result = self._fallback_answer(query, evidence)
            result["model"] = "sentrix-evidence"
            result["evidence"] = []
        else:
            try:
                context = self.context(public_retrieved)
                if conversation_context:
                    context += "\n[CONVERSATION_CONTEXT]\n" + conversation_context
                result = self.gamma.answer(query, context)
            except Exception:
                result = {"answer": "证据不足，模型暂时不可用。", "confidence": 0.2, "evidence": [], "insufficient_evidence": True, "model": self.gamma.model}
        result["modelEvidence"] = result.get("evidence", [])
        known_ids = {item["id"] for item in evidence}
        model_evidence = result.get("modelEvidence") or []
        valid_model_evidence = [item for item in model_evidence if isinstance(item, dict) and item.get("id") in known_ids]
        if evidence and not (comparison_query or recall_recommendation) and (result.get("insufficient_evidence") or not valid_model_evidence):
            result.update(self._fallback_answer(query, evidence))
        if public_retrieved.get("focused_people") and public_retrieved.get("semantic_claims") and activity_query:
            semantic_answer = self._fallback_answer(query, evidence)
            activity_values = [
                item.get("value_text") for item in evidence
                if item["kind"] == "semantic_claim" and item.get("dimension") == "activity"
            ]
            if semantic_answer.get("answer") and not all(value and value in str(result.get("answer") or "") for value in activity_values):
                result.update(semantic_answer)
        result["answer"] = self._redact_private_places(result.get("answer", ""), private_places)
        result["evidence"] = evidence
        result["retrieval_trace"] = [
            {"stage": "lexical", "status": "complete", "counts": {"events": len(public_retrieved.get("events", [])), "observations": len(public_retrieved.get("observations", [])), "facts": len(public_retrieved.get("facts", []))}},
            {"stage": "semantic", "status": "complete", "counts": {"claims": len(public_retrieved.get("semantic_claims", [])), "entities": len(public_retrieved.get("entities", [])), "relationships": len(public_retrieved.get("relationships", []))}},
            {"stage": "vector", "status": "skipped" if public_retrieved.get("vector_skipped") else "unavailable" if not public_retrieved.get("vector_available", True) else "complete", "counts": {"hits": len(public_retrieved.get("vectors", [])), "accepted": len(public_retrieved.get("vectors", [])), "candidates": public_retrieved.get("vector_candidate_count", 0)}},
            {"stage": "evidence_validation", "status": "complete", "counts": {"evidence": len(evidence)}},
        ]
        result["evidence_layers"] = {
            "answers": [{"id": result.get("query"), "text": result.get("answer", "")}],
            "people": [item for item in evidence if item["kind"] in {"person", "semantic_claim"}],
            "events": [item for item in evidence if item["kind"] == "event"],
            "claims": [item for item in evidence if item["kind"] in {"fact", "semantic_claim"}],
            "appearance": [item for item in evidence if item["kind"] == "person_appearance"],
            "observations": [item for item in evidence if item["kind"] == "observation"],
            "assets": [{"kind": "asset", "id": item["asset_id"]} for item in evidence if item.get("asset_id")],
            "gaps": [query_gap] if query_gap else [],
        }
        result["query"] = query
        result["tool_trace"] = self._tool_trace(query, public_retrieved, len(evidence), result.get("insufficient_evidence", False))
        if query_gap:
            result["query_gap_id"] = query_gap["id"]
        return result

    def _conversation_text(self, conversation_id):
        turns = self._conversations.get(conversation_id, [])
        return "\n".join(f"{turn['role']}: {turn['text']}" for turn in turns[-self._conversation_limit:])

    def _remember_turn(self, conversation_id, role, text):
        turns = self._conversations.setdefault(conversation_id, [])
        turns.append({"role": role, "text": str(text or "")[:2000]})
        del turns[:-self._conversation_limit]

    def answer_turn(self, message, conversation_id=None, feedback=None, scope_id=None, selected_entity_id=None):
        conversation_id = conversation_id or f"conversation_{uuid.uuid4().hex[:12]}"
        intent = self.classify_intent(message, feedback)
        previous = self._conversation_text(conversation_id)
        turn_plan = self._plan_turn(message, previous, feedback)
        persisted_state = self._dialogue_states.get(conversation_id) or self.store.get_dialogue_state(conversation_id, scope_id) or {}
        if intent == "feedback":
            feedback = feedback or {}
            gap_id = feedback.get("query_gap_id")
            correction = feedback.get("correction") or str(message or "").strip()
            persisted = None
            target_entity_id = feedback.get("target_entity_id")
            target_event_id = feedback.get("target_event_id")
            target_claim_id = feedback.get("target_claim_id")
            target_property_key = feedback.get("target_property_key")
            if (gap_id and self.store.get_query_gap(gap_id)) or any((target_entity_id, target_event_id, target_claim_id)):
                persisted = self.store.add_memory_feedback(
                    gap_id, feedback.get("user_id"), feedback.get("accepted_answer"), correction, target_claim_id,
                    target_entity_id, target_event_id, target_property_key,
                )
            result = {
                "intent": "feedback", "conversation_id": conversation_id,
                "answer": "已记录你的修正，相关记忆会保留原始证据并进入更新链。",
                "confidence": 1.0 if persisted else 0.0, "insufficient_evidence": not bool(persisted),
                "evidence": [], "image_results": [], "retrieval_trace": [{"stage": "feedback", "status": "complete", "counts": {"persisted": 1 if persisted else 0}}],
                "model": "sentrix-feedback", "feedback": persisted,
            }
            result["agent_plan"] = turn_plan
            result["tool_trace"] = [{"tool": "plan_turn", "permission": "read", "status": "complete", "mode": turn_plan["mode"], "reason": turn_plan["reason"]}, {"tool": "record_feedback", "permission": "explicit_user_action", "status": "complete" if persisted else "requires_target"}]
            self._remember_turn(conversation_id, "user", message)
            self._remember_turn(conversation_id, "assistant", result["answer"])
            return result
        if turn_plan["mode"] == "chat" and not selected_entity_id and not ((previous or persisted_state.get("active_event_ids")) and self._is_contextual_follow_up(message)):
            result = {
                "intent": "chat", "conversation_id": conversation_id,
                "answer": self._normal_chat_answer(message, previous, scope_id), "confidence": 1.0,
                "insufficient_evidence": False, "evidence": [], "image_results": [],
                "retrieval_trace": [{"stage": "agent_plan", "status": "chat", "counts": {"memory_tools": 0, "evidence": 0}}],
                "evidence_layers": {"answers": [{"id": None, "text": "自然对话未引用家庭记忆"}], "people": [], "events": [], "claims": [], "appearance": [], "observations": [], "assets": [], "gaps": []},
                "tool_trace": [{"tool": "plan_turn", "permission": "read", "status": "complete", "mode": "chat", "reason": turn_plan["reason"]}],
                "agent_plan": turn_plan,
            }
            self._remember_turn(conversation_id, "user", message)
            self._remember_turn(conversation_id, "assistant", result["answer"])
            return result
        prior_state = persisted_state
        query = str(message or "").strip()
        selected_entity = self.store.get_entity(selected_entity_id) if selected_entity_id else None
        if selected_entity and selected_entity.get("scope_id") != (scope_id or "home-default"):
            selected_entity = None
        explicit_new_subject = self._has_explicit_entity_reference(query, scope_id, prior_state.get("active_entity_ids"))
        contextual_follow_up = (
            self._is_contextual_follow_up(query)
            and prior_state.get("scope_id") == scope_id
            and prior_state.get("active_event_ids")
            and not explicit_new_subject
        )
        if selected_entity:
            result = self.answer(query, previous, scope_id)
            result.setdefault("tool_trace", []).insert(0, {
                "tool": "resolve_constraints", "permission": "read", "status": "complete",
                "constraints": {"selected_entity_id": selected_entity["id"]},
            })
            result["clarification_candidates"] = []
            dialogue_mode = "clarification_selection"
        elif contextual_follow_up:
            event_ids = prior_state["active_event_ids"]
            events = [self.store.get_event(event_id) for event_id in event_ids]
            events = [event for event in events if event and (not scope_id or event.get("scope_id") == scope_id)]
            result = self._answer_from_verified_events(query, events)
            result["tool_trace"] = [
                {"tool": "resolve_constraints", "permission": "read", "status": "complete", "constraints": {"reused_events": event_ids}},
                {"tool": "trace_timeline", "permission": "read", "status": "complete", "event_count": len(events)},
                {"tool": "open_evidence", "permission": "read", "status": "complete", "evidence_count": len(result["evidence"])},
            ]
            dialogue_mode = "contextual_follow_up"
        elif intent == "clarification" and previous:
            query = previous + "\n当前澄清：" + query
            result = self.answer(query, previous, scope_id)
            dialogue_mode = "clarification"
        else:
            result = self.answer(query, previous, scope_id)
            dialogue_mode = "planned_query"
        result["intent"] = intent
        result["conversation_id"] = conversation_id
        focused_entity_ids = [item.get("person_id") for item in result.get("evidence", []) if item.get("person_id")]
        result = self._rank_evidence_for_turn(query, result, turn_plan, focused_entity_ids)
        result["evidence_order"] = self._evidence_order(result.get("evidence", []))
        active_event_ids = list(dict.fromkeys(
            item.get("event_id") or item.get("id")
            for item in result.get("evidence", []) if item.get("kind") == "event"
        ))
        active_entity_ids = list(dict.fromkeys(
            item.get("person_id") for item in result.get("evidence", []) if item.get("person_id")
        ))
        if selected_entity:
            active_entity_ids.insert(0, selected_entity["id"])
        dialogue_state = {
            "scope_id": scope_id, "active_event_ids": active_event_ids[:8],
            "active_entity_ids": active_entity_ids[:8],
            "evidence_ids": [item.get("id") for item in result.get("evidence", []) if item.get("id")][:40],
            "unresolved_ambiguity": bool(result.get("clarification_candidates") or result.get("insufficient_evidence")),
        }
        self._dialogue_states[conversation_id] = dialogue_state
        self.store.save_dialogue_state(conversation_id, scope_id or "home-default", dialogue_state)
        result["agent_plan"] = turn_plan
        result["dialogue_plan"] = {
            "mode": dialogue_mode, "style": self._dialogue_style(message, result),
            "layers": ["semantic", "episodic", "original_evidence"],
        }
        execution = list(result.get("tool_trace", []))
        existing = {item.get("tool"): item for item in execution}
        for tool in turn_plan["tools"]:
            if tool in existing:
                continue
            item = dict(existing.get(tool) or {"tool": tool, "permission": "read", "status": "complete"})
            if tool == "open_evidence":
                item["status"] = "complete" if result.get("image_results") or result.get("evidence") else "empty"
                item["evidence_count"] = len(result.get("evidence", []))
            execution.append(item)
        result["tool_trace"] = [*execution, {"tool": "plan_turn", "permission": "read", "status": "complete", "mode": turn_plan["mode"], "reason": turn_plan["reason"], "planner": turn_plan["planner"]}]
        if result["dialogue_plan"]["style"] == "narrative":
            result = self._narrative_answer(message, result)
            result["evidence_layers"]["answers"] = [{"id": result.get("query"), "text": result.get("answer", "")}]
        result["dialogue_state"] = dialogue_state
        self._remember_turn(conversation_id, "user", message)
        self._remember_turn(conversation_id, "assistant", result.get("answer", ""))
        return result

    @staticmethod
    def _image_results(evidence):
        results = []
        seen = set()
        for item in evidence or []:
            if item.get("kind") != "observation" or item.get("media_type") != "image":
                continue
            asset_id = item.get("asset_id")
            if not asset_id or asset_id in seen:
                continue
            seen.add(asset_id)
            results.append({
                "asset_id": asset_id, "observation_id": item.get("observation_id"),
                "file_name": item.get("file_name"), "caption": item.get("caption"),
                "captured_at": item.get("captured_at"), "relevance": item.get("relevance", 0),
                "media_url": f"/api/assets/{asset_id}/file",
            })
        return results[:3]
