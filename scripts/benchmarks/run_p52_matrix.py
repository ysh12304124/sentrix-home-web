"""P5.2 retrieval-validation pool/batch matrix on the authoritative 153 DB."""
from __future__ import annotations

import json
import os
import statistics
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from backend import app
from backend.agent_runtime import tools
from backend.agent_runtime.result_set import ResultSet
from backend.embeddings import EmbeddingRouter
from backend.retrieval import RetrievalConfig


def main() -> None:
    router = EmbeddingRouter.from_clip(app.pipeline.clip)
    tools.bind_runtime(app.store, gamma=app.gamma, embedding_router=router,
                       retrieval_config=RetrievalConfig())

    class EphemeralResultSets:
        def __init__(self):
            self.items = {}

        def new(self, *, scope_id, query, asset_ids, unresolved=None, owner="", revision=""):
            import uuid
            rs = ResultSet(result_set_id=f"matrix_{uuid.uuid4().hex[:10]}", scope_id=scope_id,
                           query=query, asset_ids=list(asset_ids), total=len(asset_ids),
                           unresolved=unresolved or [], owner=owner, revision=revision,
                           shown=min(6, len(asset_ids)))
            self.items[rs.result_set_id] = rs
            return rs

        def get(self, result_set_id):
            return self.items.get(result_set_id)

        def resolve_handle(self, result_set_id, handle):
            rs = self.get(result_set_id)
            return rs.handles().get(handle) if rs else None

    # Search results are not the subject of this matrix; do not write 192
    # transient result sets into the production SQLite database.
    tools._RUNTIME["result_sets"] = EphemeralResultSets()
    qa_path = "services/photobench/data/album3-max/qa/album3-max-100qa.jsonl"
    wanted = {"006", "007", "025", "048", "052", "058", "063", "071",
              "078", "088", "095", "097"}
    rows = []
    for line in open(qa_path, encoding="utf-8"):
        row = json.loads(line)
        if str(row.get("qa_id", "")).rsplit("-", 1)[-1] in wanted:
            rows.append(row)
    rows.sort(key=lambda row: row["qa_id"])

    def one(row):
        started = time.perf_counter()
        try:
            result = tools._search_memories(
                {"query": row["question"], "filters": {}},
                context={"scope_id": "album_ca0cc0ddda3a",
                         "task_state": {"user_goal": row["question"]}},
            )
            gt_assets = set()
            for image_id in row.get("retrieval_image_ids") or []:
                file_name = str(image_id).rsplit("/", 1)[-1]
                hit = app.store.connection.execute(
                    "SELECT id FROM assets WHERE scope_id=? AND file_name=?",
                    ("album_ca0cc0ddda3a", file_name),
                ).fetchone()
                if hit:
                    gt_assets.add(hit["id"])
            retrieved = set(result.get("_retrieved_asset_ids") or [])
            evidence = set(result.get("evidence_asset_ids") or [])
            metrics = result.get("_model_call_metrics") or []
            return {
                "qa_id": row["qa_id"], "total": result.get("total", 0),
                "retrieval_hit": bool(retrieved & gt_assets),
                "evidence_hit": bool(evidence & gt_assets),
                "evidence_count": len(evidence),
                "validation_batches": result.get("validation_batches", 0),
                "validator_calls": len(metrics),
                "prompt_tokens": sum(int(item.get("prompt_tokens") or 0) for item in metrics),
                "completion_tokens": sum(int(item.get("completion_tokens") or 0) for item in metrics),
                "latency_s": round(time.perf_counter() - started, 2),
            }
        except Exception as exc:  # preserve a per-question diagnostic result
            return {"qa_id": row["qa_id"], "error": f"{type(exc).__name__}: {exc}",
                    "latency_s": round(time.perf_counter() - started, 2)}

    matrix = []
    for pool in (12, 18, 24, 30):
        for batch in (4, 6, 8, 12):
            os.environ["SENTRIX_SEARCH_VALIDATION_MAX_CANDIDATES"] = str(pool)
            os.environ["SENTRIX_SEARCH_VALIDATION_BATCH_SIZE"] = str(batch)
            started = time.perf_counter()
            with ThreadPoolExecutor(max_workers=12) as executor:
                values = [future.result() for future in
                          as_completed([executor.submit(one, row) for row in rows])]
            valid = [value for value in values if "error" not in value]
            values_sorted = sorted(value["latency_s"] for value in valid)
            p95 = values_sorted[max(0, int(len(values_sorted) * .95) - 1)] if values_sorted else 0
            summary = {
                "pool": pool, "batch": batch, "n": len(valid),
                "retrieval_hit_rate": round(sum(v["retrieval_hit"] for v in valid) / len(valid), 3) if valid else 0,
                "evidence_hit_rate": round(sum(v["evidence_hit"] for v in valid) / len(valid), 3) if valid else 0,
                "mean_evidence_count": round(statistics.mean(v["evidence_count"] for v in valid), 2) if valid else 0,
                "mean_validation_batches": round(statistics.mean(v["validation_batches"] for v in valid), 2) if valid else 0,
                "mean_validator_calls": round(statistics.mean(v["validator_calls"] for v in valid), 2) if valid else 0,
                "mean_prompt_tokens": round(statistics.mean(v["prompt_tokens"] for v in valid), 1) if valid else 0,
                "p95_latency_s": round(p95, 2),
                "wall_s": round(time.perf_counter() - started, 2),
                "errors": [v for v in values if "error" in v],
            }
            matrix.append(summary)
            print(json.dumps(summary, ensure_ascii=False), flush=True)
    print("MATRIX_JSON=" + json.dumps(matrix, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
