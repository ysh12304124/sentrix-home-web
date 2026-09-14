import json


class FamilyGraphService:
    """Build and persist one scope's family graph from existing semantic evidence."""

    def __init__(self, store, gamma=None):
        self.store = store
        self.gamma = gamma

    def _build_text_evidence(self, scope_id):
        rows = self.store.connection.execute(
            """SELECT em.entity_id, em.face_instance_id, em.confidence,
                      o.id AS observation_id, o.caption, o.activity,
                      o.canonical_json, o.detail_json
               FROM entity_mentions em
               JOIN observations o ON o.id = em.observation_id
               JOIN entities e ON e.id = em.entity_id
               WHERE o.scope_id = ? AND e.scope_id = ?
               ORDER BY o.created_at ASC""",
            (scope_id, scope_id),
        ).fetchall()
        people = {}
        for row in rows:
            value = dict(row)
            person = people.setdefault(value["entity_id"], {
                "person_id": value["entity_id"],
                "face_instance_ids": [],
                "observation_ids": [],
                "descriptions": [],
                "confidence": 0.0,
            })
            if value.get("face_instance_id"):
                person["face_instance_ids"].append(value["face_instance_id"])
            person["observation_ids"].append(value["observation_id"])
            person["confidence"] = max(person["confidence"], float(value.get("confidence") or 0))
            description = self._semantic_description(value)
            if description:
                person["descriptions"].append(description)
        values = []
        for person in people.values():
            person["face_instance_ids"] = list(dict.fromkeys(person["face_instance_ids"]))
            person["observation_ids"] = list(dict.fromkeys(person["observation_ids"]))
            person["descriptions"] = list(dict.fromkeys(person["descriptions"]))[:20]
            values.append(person)
        return {"scope_id": scope_id, "input_mode": "semantic_text_only", "people": values}

    @staticmethod
    def _semantic_description(row):
        parts = [str(row.get(key) or "").strip() for key in ("caption", "activity")]
        for key in ("canonical_json", "detail_json"):
            try:
                payload = json.loads(row.get(key) or "{}")
            except (TypeError, ValueError):
                payload = {}
            if payload:
                parts.append(json.dumps(payload, ensure_ascii=False, sort_keys=True))
        return "；".join(part for part in parts if part)
