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

    @staticmethod
    def _query_date(query):
        value = str(query or "")
        match = re.search(r"(20\d{2})[-/.年](\d{1,2})[-/.月](\d{1,2})", value)
        if not match:
            return None
        return f"{int(match.group(1)):04d}-{int(match.group(2)):02d}-{int(match.group(3)):02d}"

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

    def retrieve(self, query):
        events = self.store.list_events(100)
        observations = self.store.list_observations(1000)
        facts = self.store.list_facts(200)
        persons = self.store.list_persons()
        entities = self.store.list_entities()
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
        query_embedding = self.clip.embed_text(query)
        vector_hits = self.store.search_vectors("episodic", query_embedding, 12) + self.store.search_vectors("semantic", query_embedding, 12)
        vector_event_ids = [item["source_id"] for item in vector_hits if item["source_type"] == "event"]
        vector_events = [event for event in events if event["id"] in vector_event_ids]
        observation_event_ids = {item[2]["id"] for item in local_observations}
        observation_events = [event for event in events if observation_event_ids.intersection(event.get("observation_ids", []))]
        relationships = self.store.list_relationships()
        semantic_claims = []
        profiles = []
        if focused_ids:
            for entity_id in focused_ids:
                semantic_claims.extend(self.store.list_semantic_claims(entity_id, 500))
                profile = self.store.get_semantic_profile(entity_id)
                if profile:
                    profiles.append(profile)
        else:
            semantic_claims = self.store.list_semantic_claims(None, 500)
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
            "events": local_events if has_event_constraint else (local_events or observation_events or vector_events or events[:8]),
            "observations": ranked_observations or observations[:16],
            "focus_observation_ids": observation_event_ids,
            "facts": local_facts or facts[:16],
            "semantic_claims": semantic_claims,
            "appearance_evidence": appearance_evidence,
            "profiles": profiles,
            "focused_people": focused_people,
            "persons": persons[:50],
            "entities": entities[:100],
            "relationships": relationships[:100],
            "vectors": vector_hits,
            "intent": {
                "activity": self._is_activity_query(query),
                "dimension": dimension,
                "event_filter": has_event_constraint,
            },
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
        return self.retrieve(query), gap

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

    def answer(self, query):
        retrieved = self.retrieve(query)
        retrieved, query_gap = self._refine_visual_memory(query, retrieved)
        evidence = []
        seen = set()
        activity_query = self._is_activity_query(query)
        intent = retrieved.get("intent", {})
        for event in retrieved["events"][:8]:
            detail = self.store.get_event_detail(event["id"])
            if not detail:
                continue
            asset_ids = [item.get("asset_id") for item in detail["observations"] if item.get("asset_id")]
            item = {"kind": "event", "id": event["id"], "event_id": event["id"], "asset_ids": asset_ids, "summary": event.get("summary", ""), "time_start": event.get("time_start"), "place": event.get("place")}
            evidence.append(item)
            seen.add(event["id"])
            observations = sorted(detail["observations"], key=lambda item: item["id"] not in retrieved.get("focus_observation_ids", set()))
            if intent.get("dimension") == "object":
                observations = [
                    observation for observation in observations
                    if self._object_values_for_query(query, observation.get("objects") or [])
                ]
            for observation in observations[:8]:
                asset = observation.get("asset") or {}
                evidence.append({"kind": "observation", "id": observation["id"], "observation_id": observation["id"], "event_id": event["id"], "asset_id": observation.get("asset_id"), "file_name": asset.get("file_name"), "media_type": asset.get("media_type"), "captured_at": observation.get("captured_at"), "caption": observation.get("caption"), "transcript": observation.get("transcript"), "clothing": observation.get("clothing", []), "objects": observation.get("objects", []), "spatial_relations": observation.get("spatial_relations", []), "source_owner_id": asset.get("source_owner_id"), "raw": observation.get("raw_json", {})})
        for fact in retrieved["facts"][:12]:
            evidence.append({"kind": "fact", "id": fact["id"], "fact_id": fact["id"], "subject": fact["subject"], "predicate": fact["predicate"], "object": fact["object"], "status": fact["status"], "evidence_ids": fact.get("evidence_ids_json", [])})
        claims = retrieved.get("semantic_claims", [])
        if activity_query:
            activity_claims = [claim for claim in claims if claim.get("dimension") == "activity"]
            other_claims = [claim for claim in claims if claim.get("dimension") != "activity"]
            claims = activity_claims + other_claims
        claim_limit = len(claims) if activity_query else 20
        for claim in claims[:claim_limit]:
            evidence.append({"kind": "semantic_claim", "id": claim["id"], "claim_id": claim["id"], "person_id": claim["person_id"], "dimension": claim["dimension"], "predicate": claim["predicate"], "value_text": claim["value_text"], "status": claim["status"], "evidence_ids": claim.get("evidence_ids_json", []), "supporting_event_ids": claim.get("supporting_event_ids_json", [])})
        if intent.get("dimension") == "clothing":
            for appearance in retrieved.get("appearance_evidence", []):
                evidence.append({
                    "kind": "person_appearance", "id": appearance["id"], "person_id": appearance["person_id"],
                    "face_instance_id": appearance["face_instance_id"], "observation_id": appearance["observation_id"],
                    "asset_id": appearance["asset_id"], "file_name": appearance.get("file_name"),
                    "crop_bbox": appearance.get("crop_bbox_json", []), "clothing": appearance.get("clothing_json", []),
                    "confidence": appearance.get("confidence", 0), "model": appearance.get("model_name"),
                })
        deterministic_query = (
            (activity_query and retrieved.get("focused_people"))
            or (intent.get("dimension") == "clothing" and retrieved.get("focused_people"))
            or intent.get("dimension") == "object"
            or intent.get("event_filter")
        )
        if deterministic_query:
            result = self._fallback_answer(query, evidence)
            result["model"] = "sentrix-evidence"
            result["evidence"] = []
        else:
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
        if retrieved.get("focused_people") and retrieved.get("semantic_claims") and activity_query:
            semantic_answer = self._fallback_answer(query, evidence)
            activity_values = [
                item.get("value_text") for item in evidence
                if item["kind"] == "semantic_claim" and item.get("dimension") == "activity"
            ]
            if semantic_answer.get("answer") and not all(value and value in str(result.get("answer") or "") for value in activity_values):
                result.update(semantic_answer)
        result["evidence"] = evidence
        result["retrieval_trace"] = [
            {"stage": "lexical", "status": "complete", "counts": {"events": len(retrieved.get("events", [])), "observations": len(retrieved.get("observations", [])), "facts": len(retrieved.get("facts", []))}},
            {"stage": "semantic", "status": "complete", "counts": {"claims": len(retrieved.get("semantic_claims", [])), "entities": len(retrieved.get("entities", [])), "relationships": len(retrieved.get("relationships", []))}},
            {"stage": "vector", "status": "complete", "counts": {"hits": len(retrieved.get("vectors", []))}},
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
        if query_gap:
            result["query_gap_id"] = query_gap["id"]
        return result
