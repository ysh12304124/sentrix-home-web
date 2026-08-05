#!/usr/bin/env python3
"""Asset-level retrieval diagnosis for one query / asset (Phase R R0).

Explains *why* a Ground-Truth Asset did or did not enter the EvidencePacket
for a query.  Supports three modes:

  query-only          -- run the kernel for a query, summarize the packet.
  query + expected    -- as above, plus a full diagnosis block per expected
                         Asset ID (metadata, observations, event, vectors,
                         kernel rank, condition matrix, exclusion reason).
  asset-only          -- dump the full DB state for an Asset without running
                         retrieval (useful for Hidden-set / real-user assets).

A QuerySpec is built three ways, first match wins:
  1. --spec-json <file>       cached QuerySpec (JSON) for pure retrieval replay
  2. --parser                 real QueryParser (needs GammaClient / Ollama)
  3. --semantic/--hard/--time/--media/--exclude CLI flags (deterministic,
     no model) so the script works offline on 153 and local fixtures.

This is a diagnostic tool.  It never imports benchmark case data and never
modifies the database.
"""

import argparse
import json
import os
import sys
from pathlib import Path


def _load_store(db_path):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from backend.db import MemoryStore
    return MemoryStore(db_path)


def _build_spec(args):
    """Return (spec, source_label).  Deterministic when no parser is requested."""
    from backend.query_contracts import QueryParseDraft, build_query_spec
    from backend.query_parser import QueryParser

    if args.spec_json:
        spec = _spec_from_file(args.spec_json, args.scope, args.viewer)
        return spec, f"spec_json:{args.spec_json}"

    if args.parser:
        gamma = _gamma_client()
        draft = QueryParser(gamma=gamma).parse(args.query or "")
        spec = _build_from_draft(draft, args)
        return spec, "parser"

    draft = QueryParseDraft(intent="answer", answer_target=args.target or "general")
    for item in args.semantic or []:
        dimension, _, value = item.partition("=")
        draft.semantic_conditions.append({"dimension": dimension.strip(), "value": value.strip()})
    for item in args.hard or []:
        dimension, _, value = item.partition("=")
        draft.semantic_conditions.append({"dimension": dimension.strip(), "value": value.strip(), "strictness": "deterministic_hard"})
    if args.time:
        draft.time_expression = args.time
    for media in args.media or []:
        draft.media_expressions.append(media)
    for item in args.exclude or []:
        dimension, _, value = item.partition("=")
        draft.negative_conditions.append({"dimension": dimension.strip(), "value": value.strip()})
    return _build_from_draft(draft, args), "deterministic"


def _build_from_draft(draft, args):
    from backend.query_contracts import build_query_spec
    return build_query_spec(
        draft,
        scope_id=args.scope,
        viewer_id=args.viewer,
        conversation_id=f"inspect_{os.getpid()}",
        query_id=f"inspect_{os.getpid()}",
    )


def _spec_from_file(path, scope, viewer):
    from backend.query_contracts import Constraint, QueryAction, QueryFacet, QuerySpec
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    scope_ids = data.get("scope_ids") or ([scope] if scope else [])
    return QuerySpec(
        query_id=data.get("query_id", "inspect"),
        scope_mode=data.get("scope_mode", "single"),
        scope_ids=scope_ids,
        viewer_id=viewer or data.get("viewer_id") or "owner",
        conversation_id=data.get("conversation_id") or "inspect",
        intent=data.get("intent", "answer"),
        answer_target=data.get("answer_target", "general"),
        constraints=[Constraint(**item) for item in data.get("constraints", [])],
        entity_ids=data.get("entity_ids", []),
        result_requirement=data.get("result_requirement", {}),
        actions=[QueryAction(**item) for item in data.get("actions", [])],
        facets=[QueryFacet(**item) for item in data.get("facets", [])],
        ambiguities=data.get("ambiguities", []),
    )


def _gamma_client():
    from backend.model_clients import GammaClient
    return GammaClient()


def _dump_asset_state(store, asset_id):
    """Full DB-side state for one Asset, independent of retrieval."""
    asset = store.get_asset(asset_id) or {}
    observations = [dict(row) for row in store.list_observations(limit=10_000) if row.get("asset_id") == asset_id]
    events = []
    for obs in observations:
        event_id = obs.get("event_id")
        if event_id:
            event = store.get_event(event_id)
            if event:
                events.append({k: event.get(k) for k in ("id", "title", "time_start", "time_end", "place", "activity", "summary")})
    vectors = []
    try:
        ids = [asset_id] + [obs.get("id") for obs in observations[:1] if obs.get("id")]
        placeholders = ",".join("?" for _ in ids)
        rows = store.connection.execute(
            f"SELECT space, source_type, source_id, model_name, length(vector_json) AS vec_len, updated_at "
            f"FROM memory_vectors WHERE source_id IN ({placeholders})",
            ids,
        ).fetchall()
        vectors = [dict(row) for row in rows]
    except Exception as error:
        vectors = [{"error": str(error)}]
    return {
        "asset_id": asset_id,
        "asset_metadata": {k: asset.get(k) for k in ("file_name", "media_type", "captured_at", "status", "source_device_id", "source_album_id", "captured_location") if asset.get(k) is not None},
        "observation_count": len(observations),
        "observations": [{
            "id": obs.get("id"), "caption": obs.get("caption"), "activity": obs.get("activity"),
            "place": obs.get("place"), "people": obs.get("people_json"), "objects": obs.get("objects_json"),
            "ocr_text": (obs.get("ocr_text") or "")[:200], "clothing": obs.get("clothing_json"),
            "event_type": obs.get("event_type"), "confidence": obs.get("confidence"), "revision": obs.get("revision"),
        } for obs in observations],
        "events": events,
        "vectors": vectors,
    }


def _packet_result(packet, asset_id):
    """Rank/condition/exclusion of an Asset inside the packet."""
    for index, item in enumerate(packet.assets):
        if item["asset_id"] == asset_id:
            return {"returned": True, "rank": index + 1, "level": item["level"], "score": item["score"],
                    "condition_results": item.get("condition_results", {})}
    return {"returned": False, "rank": None, "excluded_count": packet.excluded_count, "gaps": packet.gaps}


def _run(args):
    store = _load_store(args.db)
    from backend.evidence_retrieval import EvidenceRetrievalKernel
    kernel = EvidenceRetrievalKernel(store)

    report = {"db": args.db, "scope": args.scope or "all_authorized"}
    spec = None
    if args.query or args.spec_json or args.parser:
        spec, spec_source = _build_spec(args)
        report["spec_source"] = spec_source
        report["query"] = args.query
        report["spec"] = {
            "scope_ids": spec.scope_ids, "answer_target": spec.answer_target,
            "constraints": [{"dimension": c.dimension, "value": c.value, "strictness": c.strictness, "negated": c.negated} for c in spec.constraints],
            "entity_ids": spec.entity_ids, "result_requirement": spec.result_requirement,
        }
        packet = kernel.retrieve(spec)
        report["packet"] = {
            "assets": [{"rank": i + 1, "asset_id": item["asset_id"], "level": item["level"], "score": item["score"]} for i, item in enumerate(packet.assets)],
            "excluded_count": packet.excluded_count,
            "exact_count": len(packet.exact_results),
            "strong_count": len(packet.strong_results),
            "approximate_count": len(packet.approximate_results),
            "gaps": packet.gaps,
        }
    else:
        packet = None

    target_ids = []
    if args.asset:
        target_ids.append(args.asset)
    for item in (args.expected or "").split(","):
        item = item.strip()
        if item:
            target_ids.append(item)
    target_ids = list(dict.fromkeys(target_ids))

    report["target_assets"] = []
    for asset_id in target_ids:
        block = _dump_asset_state(store, asset_id)
        if packet is not None:
            block["kernel"] = _packet_result(packet, asset_id)
        report["target_assets"].append(block)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", default=os.getenv("SENTRIX_DB_PATH", "data/sentrix.db"))
    parser.add_argument("--scope", default=None)
    parser.add_argument("--viewer", default="owner")
    parser.add_argument("--query", default=None, help="user query text (query-only mode)")
    parser.add_argument("--expected", default=None, help="comma-separated Ground-Truth Asset IDs")
    parser.add_argument("--asset", default=None, help="diagnose a single Asset without retrieval")
    parser.add_argument("--spec-json", default=None, help="cached QuerySpec JSON for pure retrieval replay")
    parser.add_argument("--parser", action="store_true", help="use the real QueryParser (needs GammaClient)")
    parser.add_argument("--semantic", action="append", default=[], metavar="dim=value", help="semantic condition, repeatable")
    parser.add_argument("--hard", action="append", default=[], metavar="dim=value", help="deterministic-hard condition, repeatable")
    parser.add_argument("--time", default=None, help="time expression e.g. '2024 年 5 月'")
    parser.add_argument("--media", action="append", default=[], help="media expression, repeatable")
    parser.add_argument("--exclude", action="append", default=[], metavar="dim=value", help="negated condition, repeatable")
    parser.add_argument("--target", default="general", help="answer_target when no parser")
    parser.add_argument("--report", default=None, help="write JSON report to this path")
    args = parser.parse_args()

    if not (args.query or args.spec_json or args.parser or args.asset):
        parser.error("one of --query / --spec-json / --parser / --asset is required")

    report = _run(args)
    text = json.dumps(report, ensure_ascii=False, indent=2)
    if args.report:
        Path(args.report).write_text(text, encoding="utf-8")
        print(f"wrote {args.report}")
    else:
        print(text)


if __name__ == "__main__":
    main()
