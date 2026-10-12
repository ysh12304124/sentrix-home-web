#!/usr/bin/env python3
"""Isolated GPU-memory and latency benchmark for Sentrix retrieval models.

Each invocation loads exactly one model.  Run models in separate processes so
CUDA allocations from an earlier model cannot contaminate the next result.
The workloads mirror the project runtime:

* CLIP / Qwen3-VL: one text query and one real album image;
* BGE / Qwen3 text embedders: one query plus a 12-document indexing batch;
* BGE / Qwen3 rerankers: one query against 50 passages, batch size eight.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from pathlib import Path
from typing import Callable


QUERY = "镜头从大坝主体继续移动后展示了什么？"
PASSAGE_BASE = (
    "Time: 2018-10-15 12:49:54; Place: 湖北宜昌。"
    "Caption: 镜头展示三峡工程模型、大坝主体以及附近的船闸和水域。"
    "Object: 大坝; 河流; 船只; 建筑模型。"
    "Relation: 船只位于水面，大坝横跨河道。Action/Event: 镜头缓慢向右移动。"
)


def _sync(torch) -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _device_used_mib(torch) -> float:
    free, total = torch.cuda.mem_get_info()
    return (total - free) / (1024 ** 2)


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = (len(ordered) - 1) * percentile
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] * (upper - index) + ordered[upper] * (index - lower)


def _time_once(torch, operation: Callable[[], object]) -> tuple[float, object]:
    _sync(torch)
    started = time.perf_counter()
    output = operation()
    _sync(torch)
    return (time.perf_counter() - started) * 1000.0, output


def _measure_operation(torch, operation: Callable[[], object], repeats: int) -> dict:
    torch.cuda.reset_peak_memory_stats()
    first_ms, output = _time_once(torch, operation)
    del output
    steady_ms = []
    for _ in range(max(1, repeats)):
        elapsed, output = _time_once(torch, operation)
        steady_ms.append(elapsed)
        del output
    return {
        "first_ms": round(first_ms, 2),
        "steady_median_ms": round(statistics.median(steady_ms), 2),
        "steady_p95_ms": round(_percentile(steady_ms, 0.95), 2),
        "steady_runs_ms": [round(value, 2) for value in steady_ms],
        "pytorch_inference_peak_allocated_mib": round(
            torch.cuda.max_memory_allocated() / (1024 ** 2), 1
        ),
        "pytorch_inference_peak_reserved_mib": round(
            torch.cuda.max_memory_reserved() / (1024 ** 2), 1
        ),
    }


def _load_clip(args, torch):
    import clip
    from PIL import Image

    model, preprocess = clip.load("ViT-B/32", device="cuda", jit=False)
    model.eval()
    image = Image.open(args.image).convert("RGB")
    image_tensor = preprocess(image).unsqueeze(0).to("cuda")
    text_tokens = clip.tokenize([QUERY], truncate=True).to("cuda")

    def query_operation():
        with torch.inference_mode():
            return torch.nn.functional.normalize(model.encode_text(text_tokens), dim=-1)

    def image_operation():
        with torch.inference_mode():
            return torch.nn.functional.normalize(model.encode_image(image_tensor), dim=-1)

    return model, {
        "query_batch1": lambda: _measure_operation(torch, query_operation, args.repeats),
        "image_batch1": lambda: _measure_operation(torch, image_operation, args.repeats),
    }, {"precision": str(next(model.parameters()).dtype), "dimension": 512}


def _text_pooling(model, tokenizer, values: list[str], *, qwen: bool, torch):
    if qwen:
        instruction = (
            "Given a personal memory question, retrieve media descriptions that "
            "contain evidence relevant to answering the question."
        )
        values = [f"Instruct: {instruction}\nQuery:{value}" for value in values]
    tokens = tokenizer(
        values, padding=True, truncation=True, max_length=512, return_tensors="pt"
    ).to("cuda")
    with torch.inference_mode():
        hidden = model(**tokens).last_hidden_state
        embeddings = hidden[:, -1] if qwen else hidden[:, 0]
        return torch.nn.functional.normalize(embeddings, p=2, dim=1)


def _load_text_embedder(args, torch, *, qwen: bool):
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)
    if qwen:
        tokenizer.padding_side = "left"
    model = AutoModel.from_pretrained(
        args.model_path, local_files_only=True, dtype=torch.float16,
        low_cpu_mem_usage=True,
    ).to("cuda").eval()
    documents = [f"{PASSAGE_BASE} Candidate {index}." for index in range(12)]

    def query_operation():
        return _text_pooling(model, tokenizer, [QUERY], qwen=qwen, torch=torch)

    def document_operation():
        return _text_pooling(model, tokenizer, documents, qwen=False, torch=torch)

    return model, {
        "query_batch1": lambda: _measure_operation(torch, query_operation, args.repeats),
        "documents_batch12": lambda: _measure_operation(
            torch, document_operation, args.repeats
        ),
    }, {
        "precision": str(next(model.parameters()).dtype),
        "dimension": int(model.config.hidden_size),
        "pooling": "last_token" if qwen else "cls",
    }


def _load_reranker(args, torch):
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from backend.retrieval.reranker import LocalReranker

    scorer = LocalReranker(
        args.model_path, device="cuda", batch_size=8, max_length=512,
        idle_unload_seconds=0,
    )
    scorer._load()
    passages = [f"{PASSAGE_BASE} Candidate {index}." for index in range(50)]

    def operation():
        return scorer.score(QUERY, passages)

    return scorer, {
        "rerank_50_candidates": lambda: _measure_operation(
            torch, operation, max(3, min(args.repeats, 5))
        ),
    }, {
        "precision": str(next(scorer._model.parameters()).dtype),
        "backend": scorer.backend,
        "batch_size": 8,
        "max_length": 512,
        "candidate_count": 50,
    }


def _load_qwen_vl(args, torch):
    code_path = Path(args.code_path).resolve()
    if str(code_path) not in sys.path:
        sys.path.insert(0, str(code_path))
    from PIL import Image
    from src.models.qwen3_vl_embedding import Qwen3VLEmbedder

    model = Qwen3VLEmbedder(
        args.model_path,
        max_length=4096,
        min_pixels=4096,
        max_pixels=786432,
        max_frames=16,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    )
    image = Image.open(args.image).convert("RGB")

    def query_operation():
        return model.process([{
            "text": QUERY,
            "instruction": "Represent the question for retrieving relevant media.",
        }])

    def image_operation():
        return model.process([{
            "image": image,
            "instruction": "Represent the image for personal memory retrieval.",
        }])

    return model, {
        "query_batch1": lambda: _measure_operation(torch, query_operation, args.repeats),
        "image_batch1": lambda: _measure_operation(torch, image_operation, args.repeats),
    }, {
        "precision": str(next(model.model.parameters()).dtype),
        "dimension": int(getattr(
            model.model.config,
            "hidden_size",
            getattr(model.model.config.text_config, "hidden_size"),
        )),
        "max_pixels": 786432,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kind", required=True, choices=(
        "clip", "bge_embedding", "qwen_embedding", "bge_reranker",
        "qwen_reranker", "qwen_vl_embedding",
    ))
    parser.add_argument("--model-path", default="")
    parser.add_argument("--code-path", default="")
    parser.add_argument("--image", required=True)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark")
    torch.cuda.empty_cache()
    _sync(torch)
    card_baseline_mib = _device_used_mib(torch)
    torch.cuda.reset_peak_memory_stats()
    load_started = time.perf_counter()

    if args.kind == "clip":
        model, operations, details = _load_clip(args, torch)
    elif args.kind == "bge_embedding":
        model, operations, details = _load_text_embedder(args, torch, qwen=False)
    elif args.kind == "qwen_embedding":
        model, operations, details = _load_text_embedder(args, torch, qwen=True)
    elif args.kind in {"bge_reranker", "qwen_reranker"}:
        model, operations, details = _load_reranker(args, torch)
    else:
        model, operations, details = _load_qwen_vl(args, torch)

    _sync(torch)
    load_ms = (time.perf_counter() - load_started) * 1000.0
    card_loaded_mib = _device_used_mib(torch)
    result = {
        "kind": args.kind,
        "model_path": args.model_path or "openai/clip-vit-base-patch32",
        "device": torch.cuda.get_device_name(0),
        "workload": {
            "query": QUERY,
            "image": str(Path(args.image).resolve()),
            "repeats": args.repeats,
        },
        "load_ms": round(load_ms, 2),
        "pytorch_loaded_allocated_mib": round(
            torch.cuda.memory_allocated() / (1024 ** 2), 1
        ),
        "pytorch_loaded_reserved_mib": round(
            torch.cuda.memory_reserved() / (1024 ** 2), 1
        ),
        "pytorch_load_peak_allocated_mib": round(
            torch.cuda.max_memory_allocated() / (1024 ** 2), 1
        ),
        "whole_card_baseline_mib": round(card_baseline_mib, 1),
        "whole_card_loaded_mib": round(card_loaded_mib, 1),
        "whole_card_loaded_delta_mib": round(card_loaded_mib - card_baseline_mib, 1),
        "details": details,
        "operations": {},
    }
    for name, measure in operations.items():
        result["operations"][name] = measure()

    output = Path(args.out).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    del model
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
