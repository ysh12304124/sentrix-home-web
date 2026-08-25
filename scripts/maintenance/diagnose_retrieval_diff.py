#!/usr/bin/env python3
"""Run fixed search_memories queries against one retrieval backend.

This is a read-only differential harness for the 153 retrieval diagnosis. It
does not call the chat model and reports exact candidate IDs, preview size,
channel counts, and retrieval timing.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


DEFAULT_QUERIES = [
    "2017年10月4日保定亲友婚礼的照片",
    "宜昌滨水纪念广场截流石雕塑",
    "馆陶县婚礼伴娘穿什么",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=("qdrant", "sqlite"), required=True)
    parser.add_argument("--qdrant-path", default="")
    parser.add_argument("--scope", default="album_cba01be9502b")
    parser.add_argument("--db", default="data/sentrix.db")
    parser.add_argument("--out", required=True)
    parser.add_argument("queries", nargs="*")
    args = parser.parse_args()

    os.environ["SENTRIX_VECTOR_BACKEND"] = args.backend
    if args.qdrant_path:
        os.environ["SENTRIX_QDRANT_PATH"] = args.qdrant_path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

    from backend.agent_runtime import tools as runtime_tools
    from backend.db import MemoryStore
    from backend.embeddings import EmbeddingRouter
    from backend.model_clients import ClipAdapter
    from backend.retrieval import RetrievalConfig
    from backend.agent_runtime.canonical_intent import extract_constraints

    store = MemoryStore(args.db)
    router = EmbeddingRouter.from_clip(ClipAdapter())
    runtime_tools.bind_runtime(
        store, embedding_router=router, retrieval_config=RetrievalConfig()
    )
    runtime_tools.register_tools()

    rows = []
    for query in args.queries or DEFAULT_QUERIES:
        result = runtime_tools._search_memories(
            {"query": query, "mode": "best"},
            context={
                "scope_id": args.scope,
                "viewer_id": "owner",
                "task_state": {"user_goal": query},
            },
        )
        result_set = runtime_tools._RUNTIME["result_sets"].get(result.get("result_set_id"))
        asset_ids = list(result_set.asset_ids) if result_set else list(result.get("asset_ids") or [])
        if not asset_ids and str(result.get("result_set_id") or "").startswith("event_"):
            constraints = extract_constraints(query, store, args.scope)
            event = runtime_tools._event_resolution_geo(
                query,
                store,
                args.scope,
                time_expr=constraints.get("time"),
                place=constraints.get("place"),
            )
            asset_ids = list((event or {}).get("asset_ids") or [])
        files = []
        for asset_id in asset_ids:
            asset = store.get_asset(asset_id) or {}
            files.append(Path(str(asset.get("file_name") or "")).name)
        timing = result.get("retrieval_timing") or {}
        channels = {
            name: (channel or {}).get("candidate_count")
            for name, channel in (timing.get("channels") or {}).items()
        }
        rows.append(
            {
                "query": query,
                "total": result.get("total"),
                "preview_len": len(result.get("preview") or []),
                "asset_ids": asset_ids,
                "files": files,
                "channels": channels,
                "retrieval_timing": timing,
            }
        )

    payload = {"backend": args.backend, "qdrant_path": args.qdrant_path, "rows": rows}
    Path(args.out).write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
