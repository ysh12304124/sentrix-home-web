import json

from .family_graph import validate_relationship_pair


FAMILY_GRAPH_PROMPT = """你是家庭相册关系推断器。只能使用输入中已经落库的文字描述、事件与人物绑定 ID；
绝不查看、索取或假设原始图片。请判断人物是否为家庭成员，并在有足够文本证据时输出直接中文关系。
成员 membership 只能是 family、friend、unknown；低频或证据不足必须为 unknown。
关系只能在证据充分时输出，必须填写合法的正反向标签。只返回 JSON：
{"memberships":[{"person_id":"","membership":"family|friend|unknown","confidence":0.0}],
 "relationships":[{"subject_entity_id":"","predicate":"父亲","object_entity_id":"","inverse_predicate":"女儿","confidence":0.0}]}。
不要编造姓名、关系或不存在的人物 ID。输入："""


class FamilyGraphService:
    """Build and persist one scope's family graph from existing semantic evidence."""

    def __init__(self, store, gamma=None):
        self.store = store
        self.gamma = gamma

    def run(self, run_id, scope_id):
        run = self.store.get_family_analysis_run(run_id)
        if not run:
            raise KeyError(run_id)
        if run.get("status") == "running":
            raise RuntimeError("run already running")
        self.store.update_family_analysis_run(run_id, status="running", stage="build_text_evidence")
        try:
            evidence = self._build_text_evidence(scope_id)
            self.store.update_family_analysis_run(run_id, status="running", stage="infer_graph")
            output = self._infer(evidence)
            self.store.update_family_analysis_run(run_id, status="running", stage="write_graph")
            counts = self._persist(scope_id, run_id, evidence, output)
            counts["event_watermark"] = self._event_count(scope_id)
            return self.store.update_family_analysis_run(
                run_id, status="completed", stage="done", stats={**counts, "input_mode": "semantic_text_only"}
            )
        except Exception as error:
            return self.store.update_family_analysis_run(run_id, status="failed", error=str(error))

    def _event_count(self, scope_id):
        return self.store.connection.execute(
            "SELECT COUNT(*) FROM events WHERE scope_id = ?", (scope_id,)
        ).fetchone()[0]

    def _infer(self, evidence):
        if not self.gamma:
            return {"memberships": [], "relationships": []}
        response = self.gamma.chat(
            FAMILY_GRAPH_PROMPT + json.dumps(evidence, ensure_ascii=False),
            images=None, json_mode=True, role="verify",
        )
        if isinstance(response, dict):
            return response
        try:
            return json.loads(response or "{}")
        except (TypeError, ValueError):
            return {"memberships": [], "relationships": []}

    def _persist(self, scope_id, run_id, evidence, output):
        people = {item["person_id"]: item for item in evidence.get("people") or []}
        memberships = 0
        relationships = 0
        by_person = {
            str(item.get("person_id") or ""): item
            for item in (output.get("memberships") or []) if isinstance(item, dict)
        }
        for person_id, person in people.items():
            item = by_person.get(person_id) or {}
            membership = item.get("membership") if item.get("membership") in {"family", "friend", "unknown"} else "unknown"
            self.store.set_family_membership(
                scope_id, person_id, membership, source="model",
                confidence=float(item.get("confidence") or 0),
                evidence_refs=person.get("observation_ids") or [], inference_run_id=run_id,
            )
            memberships += 1
        for item in output.get("relationships") or []:
            if not isinstance(item, dict):
                continue
            subject = str(item.get("subject_entity_id") or "")
            object_id = str(item.get("object_entity_id") or "")
            predicate = str(item.get("predicate") or "")
            inverse = str(item.get("inverse_predicate") or "")
            if subject not in people or object_id not in people or subject == object_id:
                continue
            try:
                validate_relationship_pair(predicate, inverse)
            except ValueError:
                continue
            self.store.set_family_relationship(
                scope_id, subject, predicate, object_id, inverse, source="model",
                confidence=float(item.get("confidence") or 0),
                evidence_refs=list(dict.fromkeys(
                    (people[subject].get("observation_ids") or []) + (people[object_id].get("observation_ids") or [])
                )), inference_run_id=run_id,
            )
            relationships += 1
        return {"people": len(people), "memberships": memberships, "relationships": relationships}

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
