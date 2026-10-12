#!/usr/bin/env python3
"""Evaluate Qwen3-VL embedding recall and Qwen3-VL reranking without an Agent.

The benchmark is deliberately two-phase so the two 2B models are never
resident at the same time:

1. Encode every question and query the visual Qdrant space for Top-N assets.
2. Unload the embedder, temporally cluster the saved candidates, and rerank
   them with the multimodal reranker.

Ground-truth media are used only after both rankings have been produced.
"""

from __future__ import annotations

import argparse
import atexit
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

import httpx
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def load_env(path: Path) -> None:
    if not path.is_file():
        return
    for raw in path.read_text(encoding="utf-8-sig").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(
        encoding="utf-8-sig",
    ).splitlines() if line.strip()]


def media_refs(row: dict) -> list[dict]:
    refs = row.get("retrieval_media_refs") or row.get("answer_evidence_media_refs") or []
    if refs:
        return [item for item in refs if isinstance(item, dict)]
    return [
        *({"media_type": "image", "media_id": value}
          for value in (row.get("retrieval_image_ids") or [])),
        *({"media_type": "video", "media_id": value}
          for value in (row.get("retrieval_video_ids") or [])),
    ]


def retrieval_query(row: dict) -> str:
    messages = [str(turn.get("message") or "").strip()
                for turn in (row.get("conversation") or [])]
    messages = [message for message in messages if message]
    return " ".join(messages[-3:]) or str(row.get("question") or "").strip()


def select_cases(qa_dir: Path, selection: str) -> list[dict]:
    if selection == "all487":
        return read_jsonl(qa_dir / "mixed-image-video-487q.jsonl")
    videos = []
    for row in read_jsonl(qa_dir / "single-video-48q.jsonl"):
        refs = media_refs(row)
        if refs and all(item.get("media_type") == "video" for item in refs):
            videos.append(row)
    if selection == "video48":
        return videos[:48]
    images = []
    for row in read_jsonl(qa_dir / "image-related-439q.jsonl"):
        refs = media_refs(row)
        if refs and all(item.get("media_type") == "image" for item in refs):
            images.append(row)
        if len(images) >= 50:
            break
    return videos[:48] + images


def group_name(row: dict) -> str:
    types = {str(item.get("media_type") or "") for item in media_refs(row)}
    if types == {"image"}:
        return "image"
    if types == {"video"}:
        return "video"
    return "mixed"


def asset_maps(store, scope_id: str) -> tuple[dict[str, str], dict[str, str]]:
    by_name, name_by_id = {}, {}
    for asset in store.list_assets(scope_id=scope_id, limit=100000):
        asset_id = str(asset.get("id") or "")
        name = str(asset.get("file_name") or "")
        if asset_id and name:
            by_name.setdefault(name.casefold(), asset_id)
            name_by_id[asset_id] = name
    return by_name, name_by_id


def temporary_video_vectors(store, scope_id: str, embedder_url: str) -> dict[str, np.ndarray]:
    """Encode scope videos in memory without modifying SQLite or Qdrant."""
    result = {}
    videos = [asset for asset in store.list_assets(scope_id=scope_id, limit=100000)
              if str(asset.get("media_type") or "").lower() == "video"]
    for index, asset in enumerate(videos, 1):
        response = httpx.post(
            embedder_url.rstrip("/") + "/embed",
            json={"video": str(asset.get("path") or "")}, timeout=600,
        )
        response.raise_for_status()
        vector = np.asarray(response.json().get("vector") or [], dtype=np.float32)
        if vector.size:
            norm = float(np.linalg.norm(vector))
            result[str(asset["id"])] = vector / norm if norm else vector
        print(f"temporary video vectors [{index}/{len(videos)}]", flush=True)
    return result


def gt_ids(row: dict, by_name: dict[str, str]) -> list[str]:
    result = []
    for ref in media_refs(row):
        name = str(ref.get("media_id") or "").replace("\\", "/").rsplit("/", 1)[-1]
        if ref.get("media_type") == "video" and not Path(name).suffix:
            name += ".mp4"
        asset_id = by_name.get(name.casefold())
        if asset_id and asset_id not in result:
            result.append(asset_id)
    return result


def query_spec(text: str, scope_id: str):
    from backend.query_contracts import QueryParseDraft, build_query_spec

    draft = QueryParseDraft(intent="search", answer_target="asset_set")
    draft.semantic_conditions.append({
        "dimension": "semantic", "value": text, "source_text": text,
        "strictness": "semantic_required",
    })
    draft.result_requirement = {"mode": "best", "top_k": 50}
    return build_query_spec(
        draft, scope_id=scope_id, viewer_id="owner",
        conversation_id="retrieval-only", query_id="retrieval-only",
    )


def rank_metrics(ranked: list[str], expected: list[str], cutoff: int = 8) -> dict:
    expected_set = set(expected)
    first = next((index for index, value in enumerate(ranked, 1)
                  if value in expected_set), None)
    top = ranked[:cutoff]
    hits = len(set(top) & expected_set)
    precision = hits / len(top) if top else 0.0
    recall = hits / len(expected_set) if expected_set else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if precision + recall else 0.0
    return {
        "r1": int(first is not None and first <= 1),
        "r3": int(first is not None and first <= 3),
        "r5": int(first is not None and first <= 5),
        "r8": int(first is not None and first <= 8),
        "mrr": 1.0 / first if first else 0.0,
        "first_rank": first,
        "precision_at_8": precision,
        "recall_at_8": recall,
        "f1_at_8": f1,
    }


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int((len(ordered) - 1) * q)))
    return ordered[index]


def summarize(rows: list[dict], key: str) -> dict:
    selected = [row for row in rows if row.get(key)]
    count = len(selected)
    if not count:
        return {"questions": 0}
    metrics = [row[key] for row in selected]
    latency_key = "coarse_ms" if key == "coarse" else "rerank_ms"
    latencies = [float(row[latency_key]) for row in selected]
    return {
        "questions": count,
        "r_at_1": round(sum(item["r1"] for item in metrics) / count, 4),
        "r_at_3": round(sum(item["r3"] for item in metrics) / count, 4),
        "r_at_5": round(sum(item["r5"] for item in metrics) / count, 4),
        "r_at_8": round(sum(item["r8"] for item in metrics) / count, 4),
        "mrr": round(sum(item["mrr"] for item in metrics) / count, 4),
        "precision_at_8_macro": round(sum(item["precision_at_8"] for item in metrics) / count, 4),
        "recall_at_8_macro": round(sum(item["recall_at_8"] for item in metrics) / count, 4),
        "f1_at_8_macro": round(sum(item["f1_at_8"] for item in metrics) / count, 4),
        "latency_median_ms": round(statistics.median(latencies), 1),
        "latency_p95_ms": round(percentile(latencies, 0.95), 1),
        "latency_max_ms": round(max(latencies), 1),
    }


def unload(url: str) -> None:
    try:
        httpx.post(url.rstrip("/") + "/unload", timeout=10)
    except Exception:
        pass


def start_isolated_sidecars() -> list[tuple[subprocess.Popen, object, object]]:
    """Start no-idle-unload sidecars on private benchmark ports."""
    python = os.getenv(
        "SENTRIX_QWEN3_VL_PYTHON_PATH",
        str(ROOT / ".venv-qwen3-vl" / "Scripts" / "python.exe"),
    )
    code_path = os.getenv("SENTRIX_QWEN3_VL_CODE_PATH", "")
    logs = ROOT / "data" / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    processes = []
    specs = [
        ("embedding", 8112, os.getenv("SENTRIX_QWEN3_VL_EMBED_MODEL_PATH", ""),
         os.getenv("SENTRIX_QWEN3_VL_EMBED_MODEL_ID", "Qwen/Qwen3-VL-Embedding-2B")),
        ("reranker", 8113, os.getenv("SENTRIX_RERANKER_MODEL_PATH", ""),
         "Qwen/Qwen3-VL-Reranker-2B"),
    ]
    creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    for mode, port, model_path, model_id in specs:
        env = os.environ.copy()
        env.update({
            "SENTRIX_QWEN3_VL_MODE": mode,
            "SENTRIX_QWEN3_VL_MODEL_PATH": model_path,
            "SENTRIX_QWEN3_VL_MODEL_ID": model_id,
            "SENTRIX_QWEN3_VL_CODE_PATH": code_path,
            "SENTRIX_QWEN3_VL_DEVICE": "cuda",
            "SENTRIX_QWEN3_VL_IDLE_UNLOAD_SECONDS": "0",
            "SENTRIX_QWEN3_VL_MAX_LENGTH": os.getenv("SENTRIX_RERANKER_MAX_LENGTH", "4096"),
            "SENTRIX_QWEN3_VL_MAX_PIXELS": os.getenv("SENTRIX_QWEN3_VL_MAX_PIXELS", "602112"),
            "SENTRIX_QWEN3_VL_MAX_FRAMES": os.getenv("SENTRIX_QWEN3_VL_MAX_FRAMES", "8"),
        })
        stdout = (logs / f"retrieval-only-{mode}.log").open("w", encoding="utf-8")
        stderr = (logs / f"retrieval-only-{mode}.error.log").open("w", encoding="utf-8")
        process = subprocess.Popen(
            [python, str(ROOT / "scripts" / "maintenance" / "qwen3_vl_retrieval_sidecar.py"),
             "--host", "127.0.0.1", "--port", str(port)],
            cwd=ROOT, env=env, stdout=stdout, stderr=stderr,
            creationflags=creationflags,
        )
        processes.append((process, stdout, stderr))

    def cleanup() -> None:
        for process, stdout, stderr in processes:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
            stdout.close()
            stderr.close()

    atexit.register(cleanup)
    for url in ("http://127.0.0.1:8112/health", "http://127.0.0.1:8113/health"):
        deadline = time.monotonic() + 60
        while time.monotonic() < deadline:
            try:
                if httpx.get(url, timeout=2).status_code == 200:
                    break
            except Exception:
                time.sleep(0.5)
        else:
            raise RuntimeError(f"temporary sidecar did not start: {url}")
    os.environ["SENTRIX_QWEN3_VL_EMBEDDER_URL"] = "http://127.0.0.1:8112"
    os.environ["SENTRIX_RERANKER_URL"] = "http://127.0.0.1:8113"
    os.environ["SENTRIX_RERANKER_TIMEOUT"] = "600"
    print("isolated sidecars ready on 8112/8113; idle unload disabled", flush=True)
    return processes


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default=str(ROOT / "data" / "sentrix.db"))
    parser.add_argument("--qa-dir", default=str(
        ROOT / "services" / "photobench" / "data" / "album3-max-video10" / "qa"
    ))
    parser.add_argument("--scope", default="album3-max-video10-migrated")
    parser.add_argument("--selection", choices=("balanced98", "video48", "all487"),
                        default="balanced98")
    parser.add_argument("--coarse-top-k", type=int, default=50)
    parser.add_argument("--execution", choices=("interleaved", "two_phase"),
                        default="interleaved")
    parser.add_argument("--isolated-sidecars", action="store_true")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    load_env(ROOT / ".env.local")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ["SENTRIX_VECTOR_BACKEND"] = "qdrant"
    if args.isolated_sidecars:
        start_isolated_sidecars()

    from backend.db import MemoryStore
    from backend.embeddings import EmbeddingRouter
    from backend.embeddings.qwen3_vl_remote import create_visual_media_embedder
    from backend.retrieval.base import HardFilterContext, RetrievalQuery
    from backend.retrieval.reranker import get_local_reranker, rerank_candidates
    from backend.retrieval.visual_ann import VisualAnnRetriever

    store = MemoryStore(args.db)
    visual_adapter = create_visual_media_embedder()
    router = EmbeddingRouter.from_clip(visual_adapter)
    visual = VisualAnnRetriever(store, embedding_router=router)
    cases = select_cases(Path(args.qa_dir), args.selection)
    by_name, name_by_id = asset_maps(store, args.scope)
    embedder_url = os.getenv("SENTRIX_QWEN3_VL_EMBEDDER_URL", "http://127.0.0.1:8102")
    video_vectors = (temporary_video_vectors(store, args.scope, embedder_url)
                     if args.selection == "video48" else {})

    rows, missing_gt, errors = [], [], []
    scorer = get_local_reranker()

    def apply_reranker(row: dict) -> None:
        started = time.perf_counter()
        try:
            result = rerank_candidates(
                row["question"], row["coarse_ids"], store, scorer=scorer,
                window_seconds=float(os.getenv("SENTRIX_RERANKER_CLUSTER_WINDOW_SECONDS", "1")),
            )
            ranked = result.asset_ids
            row["temporal_clusters"] = len(result.clusters)
            row["rerank_telemetry"] = result.telemetry
            row["rerank_status"] = result.telemetry.get("status")
        except Exception as exc:
            ranked = list(row["coarse_ids"])
            row["temporal_clusters"] = len(ranked)
            row["rerank_status"] = "error"
            errors.append({"qa_id": row["qa_id"], "error": repr(exc)})
        row["rerank_ms"] = round((time.perf_counter() - started) * 1000, 1)
        row["reranked_ids"] = ranked
        row["reranked_top8"] = [name_by_id.get(value, value) for value in ranked[:8]]
        row["reranked"] = rank_metrics(ranked, row["gt_asset_ids"])

    print(f"coarse phase: {len(cases)} questions", flush=True)
    for index, case in enumerate(cases, 1):
        query = retrieval_query(case)
        expected = gt_ids(case, by_name)
        if not expected:
            missing_gt.append(case.get("qa_id"))
            continue
        started = time.perf_counter()
        if video_vectors:
            vector = np.asarray(router.embed_visual(query), dtype=np.float32)
            norm = float(np.linalg.norm(vector))
            vector = vector / norm if norm else vector
            coarse_ids = [asset_id for asset_id, _score in sorted(
                ((asset_id, float(np.dot(vector, media_vector)))
                 for asset_id, media_vector in video_vectors.items()),
                key=lambda item: (-item[1], item[0]),
            )[:args.coarse_top_k]]
        else:
            spec = query_spec(query, args.scope)
            hits = visual.retrieve(
                RetrievalQuery.from_spec(spec, embedding_router=router),
                HardFilterContext.from_spec(spec), args.coarse_top_k,
            )
            coarse_ids = [hit.asset_id for hit in hits]
        elapsed_ms = (time.perf_counter() - started) * 1000
        row = {
            "qa_id": case.get("qa_id"), "group": group_name(case),
            "question": query, "gt_asset_ids": expected,
            "gt_file_names": [name_by_id.get(value, value) for value in expected],
            "coarse_ids": coarse_ids,
            "coarse_top8": [name_by_id.get(value, value) for value in coarse_ids[:8]],
            "coarse": rank_metrics(coarse_ids, expected),
            "coarse_ms": round(elapsed_ms, 1),
        }
        rows.append(row)
        if args.execution == "interleaved":
            apply_reranker(row)
        if index == 1 or index % 10 == 0 or index == len(cases):
            print(f"coarse [{index}/{len(cases)}]", flush=True)

    if args.execution == "two_phase":
        unload(embedder_url)
        print("embedding model unloaded; rerank phase", flush=True)
        for index, row in enumerate(rows, 1):
            apply_reranker(row)
            if index == 1 or index % 10 == 0 or index == len(rows):
                print(f"rerank [{index}/{len(rows)}]", flush=True)

    groups = {}
    for group in ("image", "video", "mixed", "all"):
        selected = rows if group == "all" else [row for row in rows if row["group"] == group]
        if not selected:
            continue
        coarse = summarize(selected, "coarse")
        reranked = summarize(selected, "reranked")
        groups[group] = {
            "coarse_embedding": coarse,
            "after_reranker": reranked,
            "quality_delta": {
                key: round(reranked[key] - coarse[key], 4)
                for key in ("r_at_1", "r_at_3", "r_at_5", "r_at_8", "mrr",
                            "precision_at_8_macro", "recall_at_8_macro", "f1_at_8_macro")
            },
            "average_candidates": round(statistics.mean(
                len(row["coarse_ids"]) for row in selected
            ), 1),
            "average_temporal_clusters": round(statistics.mean(
                row["temporal_clusters"] for row in selected
            ), 1),
        }

    result = {
        "schema_version": "qwen3_vl_retrieval_only_v1",
        "selection": args.selection,
        "selected": len(cases), "evaluated": len(rows),
        "scope_id": args.scope, "coarse_top_k": args.coarse_top_k,
        "execution": args.execution,
        "embedding_model": os.getenv("SENTRIX_QWEN3_VL_EMBED_MODEL_ID"),
        "reranker_model": Path(os.getenv("SENTRIX_RERANKER_MODEL_PATH", "")).name,
        "ground_truth_policy": "metrics_only_after_ranking",
        "precision_recall_cutoff": 8,
        "missing_gt": missing_gt, "errors": errors,
        "groups": groups, "rows": rows,
    }
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(groups, ensure_ascii=False, indent=2), flush=True)
    print(f"saved: {output.resolve()}", flush=True)
    scorer.unload()
    return 0 if not missing_gt and not errors else 2


if __name__ == "__main__":
    raise SystemExit(main())
