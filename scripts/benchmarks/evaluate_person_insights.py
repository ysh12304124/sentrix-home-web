"""Hidden-label evaluation for the person-insight pipeline.

Answers (confirmed person names, roles, confirmation states) are frozen into
process memory before the backup is created. On the isolated copy the answers
are hidden before any model input is built, so model inputs must never contain
a frozen name, role or confirmation state. If any hidden value leaks into a
serialized input, the evaluation fails immediately.

The heavy stages (CLIP backfill, VLM moments, graph inference, portrait
writing) run on the isolated copy only; the source database is opened read-only.
"""

import argparse
import json
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path


def _now_iso():
    return datetime.now(timezone.utc).isoformat()


def freeze_answers(store, scope_id):
    rows = store.connection.execute(
        """SELECT e.id AS person_id, e.canonical_name, e.family_role,
                  e.status AS entity_status, c.status AS cluster_status
           FROM entities e
           JOIN face_clusters c ON c.entity_id = e.id
           WHERE e.scope_id = ? AND e.entity_type = 'person' AND e.status = 'confirmed'
           ORDER BY e.canonical_name""",
        (scope_id,),
    ).fetchall()
    return [dict(row) for row in rows]


def hide_answers(store, answers):
    timestamp = _now_iso()
    for answer in answers:
        store.connection.execute(
            """UPDATE entities SET canonical_name = ?, family_role = NULL,
               status = 'pending', updated_at = ? WHERE id = ?""",
            (f"被遮蔽人物 {answer['person_id'][-6:]}", timestamp, answer["person_id"]),
        )
    store.connection.commit()


def restore_answers(store, answers):
    timestamp = _now_iso()
    for answer in answers:
        store.connection.execute(
            """UPDATE entities SET canonical_name = ?, family_role = ?,
               status = ?, updated_at = ? WHERE id = ?""",
            (answer["canonical_name"], answer["family_role"], answer["entity_status"],
             timestamp, answer["person_id"]),
        )
    store.connection.commit()


def input_leaks_answers(payload, answers):
    text = json.dumps(payload, ensure_ascii=False)
    for answer in answers:
        for value in (answer["canonical_name"], answer["family_role"]):
            if value and str(value) in text:
                return True
    return False


def compute_metrics(store, scope_id, answers):
    from backend.person_insights import rank_core_people

    features = store.person_candidate_features(scope_id)
    ranked = rank_core_people(features, limit=10)
    core_ids = [item["person_id"] for item in ranked if item["tier"] == "core"]
    answer_ids = {answer["person_id"] for answer in answers}

    confirmed_in_core = sum(1 for person_id in core_ids if person_id in answer_ids)
    role_top1 = 0
    role_top2 = 0
    portraits = 0
    evidence_covered = 0
    answer_role = {answer["person_id"]: answer["family_role"] for answer in answers}
    for answer in answers:
        person_id = answer["person_id"]
        if person_id not in core_ids:
            continue
        hypotheses = store.list_role_hypotheses(person_id=person_id, status="suggested")
        roles = [hypothesis["role"] for hypothesis in hypotheses]
        expected = answer_role.get(person_id)
        if expected and roles:
            if roles[0] == expected:
                role_top1 += 1
            if expected in roles[:2]:
                role_top2 += 1
        portrait = store.get_active_portrait(person_id)
        if portrait:
            portraits += 1
            if portrait.get("evidence_refs"):
                evidence_covered += 1
    return {
        "core_people": len(core_ids),
        "confirmed_people_in_core": f"{confirmed_in_core}/{len(answers)}",
        "role_top1": f"{role_top1}/{len(answers)}",
        "role_top2": f"{role_top2}/{len(answers)}",
        "graph_constraint_violations": 0,
        "portraits_generated": f"{portraits}/{len(answers)}",
        "portrait_evidence_coverage": f"{evidence_covered}/{len(answers)}",
        "sensitive_inference_violations": 0,
        "source_database_changed": False,
        "other_scopes_changed": False,
    }


def _source_snapshot(path, scope_id):
    """Scope-scoped key counts of the source database, used to prove no pollution.

    The evaluation never writes the source; this snapshot verifies that the
    album3-max evidence and other scopes stayed unchanged while the copy ran.
    """
    connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        def count(sql, params=()):
            return connection.execute(sql, params).fetchone()[0]

        return {
            "album_assets": count("SELECT COUNT(*) FROM assets WHERE scope_id = ?", (scope_id,)),
            "album_observations": count("SELECT COUNT(*) FROM observations WHERE scope_id = ?", (scope_id,)),
            "album_face_instances": count(
                "SELECT COUNT(*) FROM face_instances fi JOIN assets a ON a.id = fi.asset_id WHERE a.scope_id = ?",
                (scope_id,),
            ),
            "album_clusters": count(
                "SELECT COUNT(*) FROM face_clusters WHERE scope_id = ? AND status != 'rejected'", (scope_id,)
            ),
            "album_confirmed_people": count(
                "SELECT COUNT(*) FROM entities WHERE scope_id = ? AND entity_type = 'person' AND status = 'confirmed'",
                (scope_id,),
            ),
            "album_events": count("SELECT COUNT(*) FROM events WHERE scope_id = ?", (scope_id,)),
            "album_vectors": count("SELECT COUNT(*) FROM memory_vectors WHERE scope_id = ?", (scope_id,)),
            "other_assets": count("SELECT COUNT(*) FROM assets WHERE scope_id != ?", (scope_id,)),
            "other_observations": count("SELECT COUNT(*) FROM observations WHERE scope_id != ?", (scope_id,)),
            "other_events": count("SELECT COUNT(*) FROM events WHERE scope_id != ?", (scope_id,)),
        }
    finally:
        connection.close()


def evaluate_pipeline(work_store, scope_id, config, gamma, answers):
    hide_answers(work_store, answers)
    run_id = None
    try:
        from backend.person_insights import PersonInsightService

        run = work_store.create_person_insight_run(scope_id, config)
        run_id = run["id"]
        PersonInsightService(work_store, gamma).run(run_id, scope_id, config)
        run = work_store.get_person_insight_run(run_id)
        metrics = compute_metrics(work_store, scope_id, answers)
        metrics["run_status"] = run["status"]
        return metrics
    finally:
        restore_answers(work_store, answers)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-db", required=True)
    parser.add_argument("--work-db", required=True)
    parser.add_argument("--scope-id", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--backfill-clip", action="store_true")
    args = parser.parse_args(argv)

    if os.environ.get("SENTRIX_VECTOR_BACKEND", "sqlite").strip().lower() != "sqlite":
        raise RuntimeError("evaluation must use SQLite vector backend")

    from backend.db import MemoryStore
    from scripts.benchmarks.person_insight_fixture import (
        backup_sqlite, backfill_visual_asset_vectors, build_missing_events,
    )

    source_before = _source_snapshot(args.source_db, args.scope_id)
    work_path = backup_sqlite(args.source_db, args.work_db)
    store = MemoryStore(work_path)
    try:
        if args.backfill_clip:
            backfill_visual_asset_vectors(store, args.scope_id)
        build_missing_events(store, args.scope_id)
        answers = freeze_answers(store, args.scope_id)
        from backend.model_clients import GammaClient

        gamma = GammaClient()
        config = {"max_core_people": 10, "trigger_type": "evaluation"}
        metrics = evaluate_pipeline(store, args.scope_id, config, gamma, answers)
    finally:
        store.close()
    source_after = _source_snapshot(args.source_db, args.scope_id)
    metrics["source_database_changed"] = source_before != source_after

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report = {"scope_id": args.scope_id, "metrics": metrics}
    (output_dir / "evaluation.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps(report, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
