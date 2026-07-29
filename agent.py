import json
import re

from .model_clients import ClipAdapter, GammaClient


def contains(value, query):
    value = str(value or "").lower()
    terms = [term for term in re.findall(r"[\w\u4e00-\u9fff]+", query.lower()) if len(term) > 1]
    return not terms or any(term in value for term in terms)


class MemoryAgent:
    def __init__(self, store, gamma=None, clip=None):
        self.store = store
        self.gamma = gamma or GammaClient()
        self.clip = clip or ClipAdapter()

    def retrieve(self, query):
        events = self.store.list_events(100)
        observations = self.store.list_observations(200)
        facts = self.store.list_facts(200)
        persons = self.store.list_persons()
        local_events = [event for event in events if contains(json.dumps(event, ensure_ascii=False), query)]
        local_observations = [item for item in observations if contains(json.dumps(item, ensure_ascii=False), query)]
        local_facts = [item for item in facts if contains(json.dumps(item, ensure_ascii=False), query)]
        vector_hits = self.store.search_vectors("episodic", self.clip.embed_text(query), 12)
        vector_event_ids = [item["source_id"] for item in vector_hits if item["source_type"] == "event"]
        vector_events = [event for event in events if event["id"] in vector_event_ids]
        entities = self.store.list_entities()
        relationships = self.store.list_relationships()
        return {"events": local_events or vector_events or events[:8], "observations": local_observations or observations[:16], "facts": local_facts or facts[:16], "persons": persons[:50], "entities": entities[:100], "relationships": relationships[:100], "vectors": vector_hits}

    def context(self, retrieved):
        lines = ["Sentrix evidence only. Evidence不足时明确说明，不得编造。", "[EVENTS]"]
        for event in retrieved["events"]:
            lines.append(json.dumps({"id": event["id"], "title": event["title"], "time": event.get("time_start"), "place": event.get("place"), "summary": event.get("summary"), "observation_ids": event.get("observation_ids", [])}, ensure_ascii=False))
        lines.append("[OBSERVATIONS]")
        for observation in retrieved["observations"]:
            lines.append(json.dumps({"id": observation["id"], "asset_id": observation["asset_id"], "caption": observation.get("caption"), "ocr_text": observation.get("ocr_text"), "transcript": observation.get("transcript"), "event_type": observation.get("event_type")}, ensure_ascii=False))
        lines.append("[FACTS]")
        for fact in retrieved["facts"]:
            lines.append(json.dumps({"id": fact["id"], "subject": fact["subject"], "predicate": fact["predicate"], "object": fact["object"], "status": fact["status"], "evidence_ids_json": fact.get("evidence_ids_json", [])}, ensure_ascii=False))
        lines.append("[ENTITIES]")
        lines.extend(json.dumps(entity, ensure_ascii=False) for entity in retrieved.get("entities", []))
        lines.append("[RELATIONSHIPS]")
        lines.extend(json.dumps(relationship, ensure_ascii=False) for relationship in retrieved.get("relationships", []))
        lines.append("[VECTOR_HITS]")
        lines.extend(json.dumps(hit, ensure_ascii=False) for hit in retrieved.get("vectors", []))
        return "\n".join(lines)

    def answer(self, query):
        retrieved = self.retrieve(query)
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
            for observation in detail["observations"][:8]:
                asset = observation.get("asset") or {}
                evidence.append({"kind": "observation", "id": observation["id"], "observation_id": observation["id"], "event_id": event["id"], "asset_id": observation.get("asset_id"), "file_name": asset.get("file_name"), "media_type": asset.get("media_type"), "captured_at": observation.get("captured_at"), "caption": observation.get("caption"), "transcript": observation.get("transcript"), "raw": observation.get("raw_json", {})})
        for fact in retrieved["facts"][:12]:
            evidence.append({"kind": "fact", "id": fact["id"], "fact_id": fact["id"], "subject": fact["subject"], "predicate": fact["predicate"], "object": fact["object"], "status": fact["status"], "evidence_ids": fact.get("evidence_ids_json", [])})
        try:
            result = self.gamma.answer(query, self.context(retrieved))
        except Exception:
            result = {"answer": "证据不足，模型暂时不可用。", "confidence": 0.2, "evidence": [], "insufficient_evidence": True, "model": self.gamma.model}
        result["modelEvidence"] = result.get("evidence", [])
        result["evidence"] = evidence
        result["retrieval_trace"] = ["event_memory", "semantic_memory", "native_vector_memory"]
        result["query"] = query
        return result
