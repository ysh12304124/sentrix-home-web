import json
import re

from .model_clients import ClipAdapter, GammaClient


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
    return bool(terms) and all(term in value for term in terms)


class MemoryAgent:
    def __init__(self, store, gamma=None, clip=None):
        self.store = store
        self.gamma = gamma or GammaClient()
        self.clip = clip or ClipAdapter()

    def retrieve(self, query):
        events = self.store.list_events(100)
        observations = self.store.list_observations(1000)
        facts = self.store.list_facts(200)
        persons = self.store.list_persons()
        local_events = [event for event in events if contains(json.dumps(event, ensure_ascii=False), query)]
        local_observations = []
        for item in observations:
            asset = self.store.get_asset(item["asset_id"]) or {}
            searchable = {**item, "asset_file_name": asset.get("file_name", "")}
            if contains(json.dumps(searchable, ensure_ascii=False), query):
                local_observations.append(item)
        local_facts = [item for item in facts if contains(json.dumps(item, ensure_ascii=False), query)]
        query_embedding = self.clip.embed_text(query)
        vector_hits = self.store.search_vectors("episodic", query_embedding, 12) + self.store.search_vectors("semantic", query_embedding, 12)
        vector_event_ids = [item["source_id"] for item in vector_hits if item["source_type"] == "event"]
        vector_events = [event for event in events if event["id"] in vector_event_ids]
        observation_event_ids = {observation_id for item in local_observations for observation_id in [item["id"]]}
        observation_events = [event for event in events if observation_event_ids.intersection(event.get("observation_ids", []))]
        entities = self.store.list_entities()
        relationships = self.store.list_relationships()
        semantic_claims = self.store.list_semantic_claims(None, 500)
        return {"events": local_events or observation_events or vector_events or events[:8], "observations": local_observations or observations[:16], "focus_observation_ids": observation_event_ids, "facts": local_facts or facts[:16], "semantic_claims": semantic_claims, "persons": persons[:50], "entities": entities[:100], "relationships": relationships[:100], "vectors": vector_hits}

    def context(self, retrieved):
        lines = [
            "Sentrix evidence only. Evidence不足时明确说明，不得编造。",
            "每条 JSON 都是独立证据；引用时只能使用已有的 id、observation_id 或 asset_id。",
            "[EVENTS]",
        ]
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
        return "\n".join(lines)

    @staticmethod
    def _query_dimension(query):
        value = str(query or "").lower()
        if any(token in value for token in ("衣服", "穿着", "外套", "裤子", "裙子", "鞋", "帽子", "衣物")):
            return "clothing"
        if any(token in value for token in ("在哪里", "位置", "旁边", "左边", "右边", "前面", "后面")):
            return "spatial_relation"
        if any(token in value for token in ("拿着", "物品", "东西", "蛋糕", "礼物", "包")):
            return "object"
        return None

    def _refine_visual_memory(self, query, retrieved):
        dimension = self._query_dimension(query)
        if not dimension:
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
        return self.retrieve(query), gap

    @staticmethod
    def _fallback_answer(query, evidence):
        observations = [item for item in evidence if item["kind"] == "observation"]
        events = [item for item in evidence if item["kind"] == "event"]
        if observations:
            summaries = [item.get("caption") or item.get("transcript") or item.get("file_name") or item["id"] for item in observations[:3]]
            references = [item.get("file_name") or item["id"] for item in observations[:3]]
            return {
                "answer": f"根据本地证据，检索到 {len(observations)} 条相关观察：" + "；".join(summaries) + "。原始证据：" + "、".join(references) + "。",
                "confidence": 0.62,
                "insufficient_evidence": False,
            }
        if events:
            summaries = [item.get("summary") or item.get("id") for item in events[:3]]
            return {
                "answer": f"根据本地事件记忆，检索到 {len(events)} 个相关事件：" + "；".join(summaries) + "。",
                "confidence": 0.52,
                "insufficient_evidence": False,
            }
        return {"answer": f"当前本地记忆没有找到能回答“{query}”的证据。", "confidence": 0.0, "insufficient_evidence": True}

    def answer(self, query):
        retrieved = self.retrieve(query)
        retrieved, query_gap = self._refine_visual_memory(query, retrieved)
        evidence = []
        seen = set()
        for event in retrieved["events"][:8]:
            detail = self.store.get_event_detail(event["id"])
            if not detail:
                continue
            asset_ids = [item.get("asset_id") for item in detail["observations"] if item.get("asset_id")]
            item = {"kind": "event", "id": event["id"], "event_id": event["id"], "asset_ids": asset_ids, "summary": event.get("summary", ""), "time_start": event.get("time_start"), "place": event.get("place")}
            evidence.append(item)
            seen.add(event["id"])
            observations = sorted(detail["observations"], key=lambda item: item["id"] not in retrieved.get("focus_observation_ids", set()))
            for observation in observations[:8]:
                asset = observation.get("asset") or {}
                evidence.append({"kind": "observation", "id": observation["id"], "observation_id": observation["id"], "event_id": event["id"], "asset_id": observation.get("asset_id"), "file_name": asset.get("file_name"), "media_type": asset.get("media_type"), "captured_at": observation.get("captured_at"), "caption": observation.get("caption"), "transcript": observation.get("transcript"), "clothing": observation.get("clothing", []), "objects": observation.get("objects", []), "spatial_relations": observation.get("spatial_relations", []), "source_owner_id": asset.get("source_owner_id"), "raw": observation.get("raw_json", {})})
        for fact in retrieved["facts"][:12]:
            evidence.append({"kind": "fact", "id": fact["id"], "fact_id": fact["id"], "subject": fact["subject"], "predicate": fact["predicate"], "object": fact["object"], "status": fact["status"], "evidence_ids": fact.get("evidence_ids_json", [])})
        for claim in retrieved.get("semantic_claims", [])[:20]:
            evidence.append({"kind": "semantic_claim", "id": claim["id"], "claim_id": claim["id"], "person_id": claim["person_id"], "dimension": claim["dimension"], "predicate": claim["predicate"], "value_text": claim["value_text"], "status": claim["status"], "evidence_ids": claim.get("evidence_ids_json", [])})
        try:
            result = self.gamma.answer(query, self.context(retrieved))
        except Exception:
            result = {"answer": "证据不足，模型暂时不可用。", "confidence": 0.2, "evidence": [], "insufficient_evidence": True, "model": self.gamma.model}
        result["modelEvidence"] = result.get("evidence", [])
        known_ids = {item["id"] for item in evidence}
        model_evidence = result.get("modelEvidence") or []
        valid_model_evidence = [item for item in model_evidence if isinstance(item, dict) and item.get("id") in known_ids]
        if evidence and (result.get("insufficient_evidence") or not valid_model_evidence):
            result.update(self._fallback_answer(query, evidence))
        result["evidence"] = evidence
        result["retrieval_trace"] = ["event_memory", "semantic_memory", "native_vector_memory"]
        result["query"] = query
        if query_gap:
            result["query_gap_id"] = query_gap["id"]
        return result
