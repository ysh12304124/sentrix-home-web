"""Inspect a Sentrix database for Agent replay readiness without writing to it."""

import argparse
import json
import sqlite3
from pathlib import Path
from urllib.parse import quote


REQUIRED_SCHEMA = {
    "entities": ("id", "scope_id", "entity_type", "canonical_name", "status", "family_role"),
    "relationships": ("id", "scope_id", "subject_entity_id", "predicate", "object_entity_id", "status", "evidence_ids_json"),
    "semantic_claims": ("id", "scope_id", "person_id", "dimension", "value_text", "supporting_event_ids_json", "evidence_ids_json", "status"),
    "person_patterns": ("id", "scope_id", "person_id", "pattern_type", "value_text", "supporting_event_ids_json", "evidence_ids_json", "status"),
    "semantic_profiles": ("id", "scope_id", "person_id", "summary_zh", "evidence_ids_json"),
    "events": ("id", "scope_id", "time_start", "time_end", "place", "summary"),
    "observations": ("id", "scope_id", "asset_id", "captured_at", "caption", "transcript", "people_json"),
    "assets": ("id", "scope_id", "media_type", "file_name", "captured_at"),
    "person_event_memory": ("id", "scope_id", "person_id", "event_id", "evidence_ids_json"),
    "entity_observations": ("entity_id", "observation_id"),
    "event_entities": ("event_id", "entity_id", "evidence_ids_json"),
    "memory_vectors": ("id", "scope_id", "source_type", "source_id"),
}

COUNT_TABLES = (
    "entities",
    "relationships",
    "semantic_claims",
    "person_patterns",
    "events",
    "observations",
    "assets",
    "person_event_memory",
    "memory_vectors",
)

OPTIONAL_SCHEMA = {
    "person_appearance_evidence": (
        "id", "person_id", "observation_id", "asset_id", "clothing_json", "status"
    ),
}


def _connect_readonly(database):
    path = Path(database).expanduser().resolve()
    uri = f"file:{quote(str(path), safe='/')}?mode=ro"
    connection = sqlite3.connect(uri, uri=True)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA query_only = ON")
    return connection


def _table_names(connection):
    return {
        row["name"]
        for row in connection.execute("SELECT name FROM sqlite_master WHERE type = 'table'")
    }


def _columns(connection, table):
    return {row["name"] for row in connection.execute(f"PRAGMA table_info({table})")}


def _count(connection, table):
    return int(connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])


def _entity_dependency_counts(connection, entity_id, tables):
    result = {
        "claims": int(connection.execute("SELECT COUNT(*) FROM semantic_claims WHERE person_id = ?", (entity_id,)).fetchone()[0]) if "semantic_claims" in tables else 0,
        "patterns": int(connection.execute("SELECT COUNT(*) FROM person_patterns WHERE person_id = ?", (entity_id,)).fetchone()[0]) if "person_patterns" in tables else 0,
        "relationships": int(connection.execute("SELECT COUNT(*) FROM relationships WHERE subject_entity_id = ? OR object_entity_id = ?", (entity_id, entity_id)).fetchone()[0]) if "relationships" in tables else 0,
        "person_events": int(connection.execute("SELECT COUNT(*) FROM person_event_memory WHERE person_id = ?", (entity_id,)).fetchone()[0]) if "person_event_memory" in tables else 0,
        "observations": 0,
        "assets": 0,
        "evidence_link_paths": [],
        "profiles": int(connection.execute("SELECT COUNT(*) FROM semantic_profiles WHERE person_id = ?", (entity_id,)).fetchone()[0]) if "semantic_profiles" in tables else 0,
    }
    observation_ids = {row[0] for row in connection.execute("SELECT id FROM observations")} if "observations" in tables else set()
    asset_ids = {row[0] for row in connection.execute("SELECT id FROM assets")} if "assets" in tables else set()
    linked_observation_ids = set()
    linked_asset_ids = set()
    if {"entity_observations", "observations"}.issubset(tables):
        linked_observation_ids.update(row[0] for row in connection.execute(
            "SELECT observation_id FROM entity_observations WHERE entity_id = ?", (entity_id,)
        ))
        result["evidence_link_paths"].append("entity_observations")
    if "person_appearance_evidence" in tables:
        linked_observation_ids.update(row[0] for row in connection.execute(
            "SELECT observation_id FROM person_appearance_evidence WHERE person_id = ?", (entity_id,)
        ))
        linked_asset_ids.update(row[0] for row in connection.execute(
            "SELECT asset_id FROM person_appearance_evidence WHERE person_id = ?", (entity_id,)
        ))
        result["evidence_link_paths"].append("person_appearance_evidence")
    if "person_event_memory" in tables:
        for row in connection.execute(
            "SELECT evidence_ids_json FROM person_event_memory WHERE person_id = ?", (entity_id,)
        ):
            try:
                evidence_ids = json.loads(row[0] or "[]")
            except (TypeError, ValueError):
                evidence_ids = []
            linked_observation_ids.update(item for item in evidence_ids if item in observation_ids)
            linked_asset_ids.update(item for item in evidence_ids if item in asset_ids)
        result["evidence_link_paths"].append("person_event_memory.evidence_ids_json")
    linked_observation_ids.intersection_update(observation_ids)
    if "observations" in tables:
        linked_asset_ids.update(
            row[0] for row in connection.execute(
                "SELECT asset_id FROM observations WHERE id IN ({})".format(",".join("?" for _ in linked_observation_ids)),
                tuple(linked_observation_ids),
            )
        ) if linked_observation_ids else None
    result["observations"] = len(linked_observation_ids)
    result["assets"] = len(linked_asset_ids.intersection(asset_ids))
    return result


def inspect_database(database, target_terms=("明哥",), sample_limit=30):
    """Return a JSON-safe readiness report from a read-only database connection."""
    connection = _connect_readonly(database)
    try:
        tables = _table_names(connection)
        schema = {}
        missing_tables = []
        missing_columns = {}
        for table, required_columns in REQUIRED_SCHEMA.items():
            if table not in tables:
                missing_tables.append(table)
                schema[table] = {"present": False, "missing_columns": list(required_columns), "row_count": None}
                continue
            actual_columns = _columns(connection, table)
            missing = sorted(set(required_columns) - actual_columns)
            if missing:
                missing_columns[table] = missing
            schema[table] = {"present": True, "missing_columns": missing, "row_count": _count(connection, table)}

        optional_schema = {}
        for table, expected_columns in OPTIONAL_SCHEMA.items():
            if table not in tables:
                optional_schema[table] = {"present": False, "missing_columns": [], "row_count": None}
                continue
            actual_columns = _columns(connection, table)
            optional_schema[table] = {
                "present": True,
                "missing_columns": sorted(set(expected_columns) - actual_columns),
                "row_count": _count(connection, table),
            }

        integrity_check = str(connection.execute("PRAGMA integrity_check").fetchone()[0])
        counts = {table: _count(connection, table) for table in COUNT_TABLES if table in tables}
        counts.update({
            "confirmed_people": int(connection.execute(
                "SELECT COUNT(*) FROM entities WHERE entity_type = 'person' AND status = 'confirmed'"
            ).fetchone()[0]) if "entities" in tables else 0,
            "pending_people": int(connection.execute(
                "SELECT COUNT(*) FROM entities WHERE entity_type = 'person' AND status <> 'confirmed'"
            ).fetchone()[0]) if "entities" in tables else 0,
        })

        target_entities = []
        if "entities" in tables:
            entity_columns = _columns(connection, "entities")
            confidence_column = "confidence" if "confidence" in entity_columns else "NULL"
            for term in target_terms:
                rows = connection.execute(
                    f"""SELECT id, canonical_name, status, family_role, {confidence_column} AS confidence, scope_id
                    FROM entities
                    WHERE entity_type = 'person' AND canonical_name LIKE ?
                    ORDER BY canonical_name""",
                    (f"%{term}%",),
                ).fetchall()
            for row in rows:
                if any(item["id"] == row["id"] for item in target_entities):
                    continue
                dependency = _entity_dependency_counts(connection, row["id"], tables)
                target_entities.append({
                        "id": row["id"],
                        "name": row["canonical_name"],
                        "status": row["status"],
                        "family_role": row["family_role"],
                        "confidence": row["confidence"] if "confidence" in row.keys() else None,
                        "scope_id": row["scope_id"],
                        **dependency,
                    })

        warnings = []
        if integrity_check != "ok":
            warnings.append(f"integrity_check={integrity_check}")
        if missing_tables:
            warnings.append(f"missing_tables={','.join(sorted(missing_tables))}")
        if missing_columns:
            warnings.append(f"missing_columns={json.dumps(missing_columns, ensure_ascii=False, sort_keys=True)}")
        if not counts.get("confirmed_people", 0):
            warnings.append("no_confirmed_people")
        if target_terms and not target_entities:
            warnings.append(f"target_not_found={','.join(target_terms)}")
        if not counts.get("semantic_claims", 0) or not counts.get("person_patterns", 0):
            warnings.append("semantic_support_is_empty")

        ready = bool(
            integrity_check == "ok"
            and not missing_tables
            and not missing_columns
            and counts.get("confirmed_people", 0) > 0
            and (not target_terms or bool(target_entities))
        )
        return {
            "database": str(Path(database).expanduser().resolve()),
            "readonly": True,
            "integrity_check": integrity_check,
            "sample_limit": max(1, int(sample_limit)),
            "schema": schema,
            "optional_schema": optional_schema,
            "missing_tables": sorted(missing_tables),
            "missing_columns": missing_columns,
            "counts": counts,
            "target_terms": list(target_terms),
            "target_entities": target_entities,
            "warnings": warnings,
            "ready": ready,
        }
    finally:
        connection.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", required=True, type=Path)
    parser.add_argument("--limit", type=int, default=30)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--target-term", action="append", dest="target_terms")
    args = parser.parse_args()
    report = inspect_database(args.db, tuple(args.target_terms or ("明哥",)), args.limit)
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    raise SystemExit(0 if report["ready"] else 2)


if __name__ == "__main__":
    main()
