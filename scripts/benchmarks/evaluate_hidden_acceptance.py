#!/usr/bin/env python3
"""Phase R9-5 — Hidden Acceptance blind runner (read-only).

Runs the 16 frozen hidden cases through the PRODUCTION route + retrieval path
(Router + QueryParser + EvidenceRetrievalKernel) and writes per-case
PREDICTIONS ONLY — no GT.  The full hidden GT is held by the user; they grade
offline with ``score_hidden.py``.

Run on 153 (needs the 12B parser + DB + ANN):
  PYTHONPATH=. .venv-mac/bin/python scripts/benchmarks/evaluate_hidden_acceptance.py
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def _resolve_person(store, scope_id):
    def resolve(name):
        try:
            for entity in store.list_entities(status="confirmed", scope_id=scope_id):
                if entity.get("canonical_name") == name:
                    return entity.get("id")
        except Exception:
            pass
        return None
    return resolve


def _message_entity_ids(store, scope_id):
    def mentions(message):
        value = str(message or "")
        ids = []
        try:
            for entity in store.list_entities(status="confirmed", scope_id=scope_id):
                name = str(entity.get("canonical_name") or "").strip()
                if name and name in value:
                    ids.append(entity["id"])
        except Exception:
            pass
        return ids
    return mentions


def main():
    # Match the production start script so the blind run uses the same retrieval
    # stack (Chinese-CLIP visual, CLIP text, multi-retriever kernel).  The parser
    # budget mirrors the production ModelRouter phase budget (4s) so a GPU-blocked
    # 12B parser falls back exactly as it does in the live API.
    os.environ.setdefault("SENTRIX_IMAGE_EMBEDDER", "chinese_clip")
    os.environ.setdefault("SENTRIX_TEXT_EMBEDDER", "clip")
    os.environ.setdefault("SENTRIX_EVIDENCE_MULTI_RETRIEVER_V1", "1")
    os.environ.setdefault("CLIP_DEVICE", "cpu")
    os.environ.setdefault("OLLAMA_TIMEOUT_SECONDS", "4")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", default=os.getenv("SENTRIX_DB_PATH", "data/sentrix.db"))
    parser.add_argument("--manifest", default="docs/baseline/hidden_set_manifest.json")
    parser.add_argument("--report", default="docs/baseline/hidden_predictions.json")
    parser.add_argument("--ranking", default="visual_backbone")
    parser.add_argument("--channels", default="full_hybrid")
    args = parser.parse_args()

    sys.path.insert(0, str(REPO_ROOT))
    from backend.db import MemoryStore
    from backend.model_clients import GammaClient
    from backend.query_contracts import QueryAction, build_query_spec
    from backend.query_parser import QueryParser
    from backend.router import Router
    from backend.retrieval import NeutralProbe, RetrievalConfig
    from evaluate_retrieval_kernel import _build_kernel, _make_embedding_router

    manifest_path = Path(args.manifest)
    if not manifest_path.is_file():
        print(f"missing manifest: {args.manifest}")
        return 1
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    store = MemoryStore(args.db)
    kernel = _build_kernel(store, args.channels, _make_embedding_router(), ranking=args.ranking)
    qp = QueryParser(gamma=GammaClient())
    router = Router()

    rows = []
    for entry in manifest.get("hidden_keys", []):
        key = entry.get("key")
        query = entry.get("query_cn") or ""
        scope_id = str(key).split("-")[0] if key else None
        resolver = _resolve_person(store, scope_id)
        mentions = _message_entity_ids(store, scope_id)

        started = time.perf_counter()
        draft = qp.parse(query)
        decision = router.route(query, draft, focus={}, entity_resolver=resolver,
                                message_entity_resolver=mentions)
        route = decision.mode
        reason = decision.reason
        final_decision = decision
        if route == "ambiguous":
            channel_hits, health = kernel.probe(query, scope_id, "owner")
            outcome = NeutralProbe(RetrievalConfig()).run(query, channel_hits, scope_id=scope_id,
                                                          viewer_id="owner", index_health=health)
            final_decision = router.resolve_after_probe(outcome, query, decision, draft)
            route = final_decision.mode

        # Mirror thin_agent._ambiguous_path: on a probe upgrade the raw query
        # becomes a semantic condition so the formal retrieval has something to
        # evaluate (an empty parser-failed draft otherwise retrieves nothing).
        if route == "evidence" and getattr(final_decision, "reason", "").startswith("probe_upgrade"):
            if not draft.actions:
                draft.actions = [QueryAction(type="answer_question", target="general")]
            draft.semantic_conditions.append(
                {"dimension": "semantic", "value": query, "source_text": query}
            )
        spec = build_query_spec(
            draft, scope_id=scope_id, viewer_id="owner", conversation_id="hidden",
            entity_resolver=resolver, query_id=f"hidden_{key}",
        )
        if final_decision.focus_ids:
            spec.entity_ids = list(dict.fromkeys(list(spec.entity_ids) + list(final_decision.focus_ids)))
        if route == "evidence":
            packet = kernel.retrieve(spec)
            retrieved = [item["asset_id"] for item in packet.assets]
            levels = [item["level"] for item in packet.assets]
            strengths = [item.get("recall_strength") for item in packet.assets]
            gaps = list(packet.gaps)
            excluded = packet.excluded_count
        else:
            retrieved, levels, strengths, gaps, excluded = [], [], [], [], 0

        rows.append({
            "key": key, "query": query, "scope_id": scope_id, "category": entry.get("category"),
            "route": route, "reason": reason, "final_reason": final_decision.reason,
            "parser": {"mode": draft.proposed_mode,
                       "actions": [a.type for a in draft.actions],
                       "facets": [f.dimension for f in draft.facets],
                       "time": bool(draft.time_expression),
                       "media": bool(draft.media_expressions),
                       "negative": bool(draft.negative_conditions)},
            "retrieved_asset_ids": retrieved, "evidence_levels": levels,
            "recall_strengths": strengths, "gaps": gaps, "excluded_count": excluded,
            "elapsed_s": round(time.perf_counter() - started, 4),
        })
        print(f"{key} [{entry.get('category')}] route={route} reason={final_decision.reason} "
              f"retrieved={len(retrieved)}", flush=True)

    report = {
        "generated_at": "2026-08-06",
        "note": "predictions only; full GT held by the user. Grade with score_hidden.py --gt <gt.json>",
        "count": len(rows),
        "cases": rows,
    }
    out = Path(args.report)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
