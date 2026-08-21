"""Deterministic core-person ranking and representative photo selection.

Ranking and selection are pure, deterministic functions so the pipeline can be
tuned against the same `album3-max` evidence without touching the import path.
"""

import math

STAGES = (
    "rank_people",
    "select_representatives",
    "extract_moments",
    "infer_graph",
    "compile_portraits",
    "write_portraits",
)


def cosine(left, right):
    if not left or not right:
        return 0.0
    numerator = sum(a * b for a, b in zip(left, right))
    denominator = math.sqrt(sum(a * a for a in left)) * math.sqrt(sum(b * b for b in right))
    if denominator == 0:
        return 0.0
    return numerator / denominator


def core_person_score(row):
    date_score = min(int(row.get("date_count") or 0) / 8.0, 1.0)
    event_score = min(int(row.get("event_count") or 0) / 6.0, 1.0)
    coverage_score = min(int(row.get("member_count") or 0) / 20.0, 1.0)
    co_person_score = min(int(row.get("co_person_count") or 0) / 4.0, 1.0)
    scene_score = min(int(row.get("scene_count") or 0) / 4.0, 1.0)
    quality_score = max(0.0, min(float(row.get("quality") or 0), 1.0))
    confirmed_boost = 0.20 if row.get("confirmed") else 0.0
    single_day_penalty = 0.35 if int(row.get("date_count") or 0) <= 1 else 0.0
    total = (
        0.30 * date_score + 0.20 * event_score + 0.15 * coverage_score
        + 0.10 * co_person_score + 0.10 * scene_score + 0.15 * quality_score
        + confirmed_boost - single_day_penalty
    )
    row["score_breakdown"] = {
        "date_score": round(date_score, 6),
        "event_score": round(event_score, 6),
        "coverage_score": round(coverage_score, 6),
        "co_person_score": round(co_person_score, 6),
        "scene_score": round(scene_score, 6),
        "quality_score": round(quality_score, 6),
        "confirmed_boost": round(confirmed_boost, 6),
        "single_day_penalty": round(single_day_penalty, 6),
    }
    return round(max(0.0, min(total, 1.0)), 6)


def rank_core_people(rows, limit=10):
    values = [{**row, "core_score": core_person_score(row)} for row in rows]
    values.sort(key=lambda item: (
        not bool(item.get("confirmed")),
        -item["core_score"],
        -int(item.get("date_count") or 0),
        str(item["person_id"]),
    ))
    core_ids = {
        item["person_id"] for item in values[:max(1, min(int(limit), 10))]
        if item.get("confirmed") or int(item.get("date_count") or 0) >= 2
    }
    return [{
        **item,
        "tier": "core" if item["person_id"] in core_ids else (
            "common" if int(item.get("date_count") or 0) >= 2 else "incidental"
        ),
    } for item in values]


def select_representatives(candidates, vector_by_asset, limit=12,
                           per_event=2, duplicate_threshold=0.94):
    selected = []
    event_counts = {}
    ordered = sorted(candidates, key=lambda item: (
        -float(item.get("body_visibility") or 0),
        -float(item.get("quality") or 0),
        str(item.get("captured_at") or ""),
        str(item["face_instance_id"]),
    ))
    for item in ordered:
        event_id = item.get("event_id")
        if event_counts.get(event_id, 0) >= per_event:
            continue
        vector = vector_by_asset.get(item.get("asset_id")) or []
        if vector and any(
            cosine(vector, vector_by_asset.get(chosen.get("asset_id")) or [])
            >= duplicate_threshold for chosen in selected
        ):
            continue
        selected.append(item)
        event_counts[event_id] = event_counts.get(event_id, 0) + 1
        if len(selected) >= limit:
            break
    for item in selected:
        item["selection_json"] = {
            "version": "representative-v1",
            "quality": round(float(item.get("quality") or 0), 6),
            "body_visibility": round(float(item.get("body_visibility") or 0), 6),
            "duplicate_threshold": duplicate_threshold,
            "event_coverage": event_counts.get(item.get("event_id"), 0),
        }
    return selected


class PersonInsightService:
    """Run the fixed person-insight pipeline stages for one memory space."""

    def __init__(self, store, gamma=None):
        self.store = store
        self.gamma = gamma

    def run(self, run_id, scope_id, config):
        run = self.store.get_person_insight_run(run_id)
        if not run:
            raise KeyError(run_id)
        if run.get("status") == "running":
            raise RuntimeError("run already running")
        from .db import MemoryStore

        store = MemoryStore(self.store.path)
        state = {}
        try:
            self.store.update_person_insight_run(run_id, status="running")
            start_stage = run.get("current_stage") or "queued"
            resumed = start_stage not in STAGES
            for stage in STAGES:
                if not resumed:
                    if start_stage == stage:
                        resumed = True
                    else:
                        continue
                store.update_person_insight_run(run_id, status="running", stage=stage)
                state = self._run_stage(store, stage, run_id, scope_id, config, state)
            stats = {
                "event_watermark": self._event_count(store, scope_id),
                "stages": list(STAGES),
            }
            store.update_person_insight_run(run_id, status="completed", stage="done", stats=stats)
        except Exception as error:
            self.store.update_person_insight_run(run_id, status="failed", error=str(error))
        finally:
            store.close()
        return self.store.get_person_insight_run(run_id)

    @staticmethod
    def _event_count(store, scope_id):
        return store.connection.execute(
            "SELECT COUNT(*) FROM events WHERE scope_id = ?", (scope_id,)
        ).fetchone()[0]

    def _run_stage(self, store, stage, run_id, scope_id, config, state):
        if stage == "rank_people":
            features = store.person_candidate_features(scope_id)
            ranked = rank_core_people(features, limit=int(config.get("max_core_people") or 10))
            state["ranked"] = ranked
            state["core"] = [item for item in ranked if item["tier"] == "core"]
        elif stage == "select_representatives":
            state["selections"] = self._select_representatives(
                store, scope_id, state.get("core") or []
            )
        elif stage == "extract_moments":
            from .person_moments import PersonMomentExtractor

            result = PersonMomentExtractor(store, self.gamma).extract(
                scope_id, run_id, state.get("selections") or []
            )
            state["moment_result"] = result
        elif stage == "infer_graph":
            state["graph"] = self._infer_graph(store, run_id, scope_id, state)
        elif stage == "compile_portraits":
            packs = {}
            for item in state.get("core") or []:
                from .person_portraits import compile_portrait_evidence

                packs[item["person_id"]] = compile_portrait_evidence(store, item["person_id"])
            state["packs"] = packs
        elif stage == "write_portraits":
            from .person_portraits import deterministic_portrait, validate_portrait

            for item in state.get("core") or []:
                person_id = item["person_id"]
                pack = (state.get("packs") or {}).get(person_id)
                if not pack:
                    continue
                portrait = self.gamma.write_person_portrait(pack, role="writer")
                ok, errors = validate_portrait(pack, portrait)
                if not ok:
                    portrait = self.gamma.write_person_portrait(pack, role="repair")
                    ok, errors = validate_portrait(pack, portrait)
                if not ok:
                    portrait = deterministic_portrait(pack)
                    ok, errors = validate_portrait(pack, portrait)
                if not ok:
                    continue
                refs = [
                    ref for theme in (portrait.get("themes") or [])
                    for ref in (theme.get("evidence_refs") or [])
                ]
                store.create_portrait_revision(person_id, {
                    "portrait_text": portrait["portrait_text"],
                    "themes": portrait.get("themes") or [],
                    "evidence_refs": refs,
                    "trigger_type": "pipeline",
                    "model_name": getattr(self.gamma, "model", "unknown"),
                    "prompt_version": "person-portrait-v1",
                })
        return state

    def _select_representatives(self, store, scope_id, core_people):
        selections = []
        for item in core_people:
            person_id = item["person_id"]
            candidates = store._rows(
                """SELECT fi.id AS face_instance_id, fi.asset_id, fi.quality,
                    COALESCE(fi.area_ratio, 0) AS body_visibility,
                    o.captured_at, eo.event_id, fi.cluster_id, c.entity_id AS person_id
                FROM face_instances fi
                JOIN face_clusters c ON c.id = fi.cluster_id
                JOIN observations o ON o.id = fi.observation_id
                LEFT JOIN event_observations eo ON eo.observation_id = o.id
                WHERE c.entity_id = ? AND c.status != 'rejected'""",
                (person_id,),
            )
            picked = select_representatives(candidates, {}, limit=12, per_event=2)
            for selection in picked:
                face = store.get_face_instance(selection["face_instance_id"])
                selections.append({
                    "asset_id": selection["asset_id"],
                    "face_instance_id": selection["face_instance_id"],
                    "person_id": person_id,
                    "cluster_id": selection["cluster_id"],
                    "observation_id": (face or {}).get("observation_id"),
                    "event_id": selection.get("event_id"),
                })
        return selections

    def _infer_graph(self, store, run_id, scope_id, state):
        core = state.get("core") or []
        if not core:
            return {"people": [], "roles": [], "relationships": []}
        ref_map = {item["person_id"]: f"P{index + 1:02d}" for index, item in enumerate(core)}
        rev_ref = {ref: person_id for person_id, ref in ref_map.items()}
        events = store._rows(
            "SELECT id, place, activity, time_start FROM events WHERE scope_id = ?", (scope_id,)
        )
        paths = []
        for selection in state.get("selections") or []:
            asset = store.get_asset(selection["asset_id"])
            if asset and asset.get("path"):
                paths.append(asset["path"])
        graph_payload = {
            "people": [ref_map[item["person_id"]] for item in core],
            "events": [
                {"id": e["id"], "place": e["place"], "activity": e["activity"],
                 "date": e["time_start"]} for e in events
            ],
            "cooccurrence": {},
            "moments": [],
            "devices": {},
        }
        result = self.gamma.infer_person_graph(paths, graph_payload)
        role_rows = []
        for role in result.get("roles") or []:
            person_id = rev_ref.get(str(role.get("person_ref") or ""))
            if not person_id:
                continue
            relative_to = rev_ref.get(str(role.get("relative_to") or "")) if role.get("relative_to") else None
            for rank, candidate in enumerate((role.get("candidates") or [])[:3], start=1):
                role_rows.append({
                    "person_id": person_id,
                    "relative_to_person_id": relative_to,
                    "role": candidate.get("role"),
                    "rank": rank,
                    "confidence": candidate.get("confidence") or 0,
                    "reason_summary": candidate.get("reason") or "",
                    "model_name": getattr(self.gamma, "model", "unknown"),
                    "prompt_version": "person-graph-v1",
                })
        if role_rows:
            store.replace_role_hypotheses(scope_id, run_id, role_rows)
        rel_rows = []
        for rel in result.get("relationships") or []:
            subject = rev_ref.get(str(rel.get("subject_ref") or ""))
            obj = rev_ref.get(str(rel.get("object_ref") or ""))
            if not subject or not obj:
                continue
            rel_rows.append({
                "subject_person_id": subject,
                "predicate": rel.get("predicate"),
                "object_person_id": obj,
                "inverse_predicate": rel.get("inverse_predicate") or "",
                "confidence": rel.get("confidence") or 0,
                "evidence_event_ids": rel.get("evidence_event_ids") or [],
                "evidence_moment_ids": rel.get("evidence_moment_ids") or [],
                "reason_summary": rel.get("reason") or "",
                "model_name": getattr(self.gamma, "model", "unknown"),
                "prompt_version": "person-graph-v1",
            })
        if rel_rows:
            store.replace_relationship_hypotheses(scope_id, run_id, rel_rows)
        return {"people": [item["person_id"] for item in core],
                "roles": role_rows, "relationships": rel_rows}
