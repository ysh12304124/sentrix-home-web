#!/usr/bin/env python3
"""Run unchanged WorldMM retrieval/QA for two videos with a read-only sidecar trace.

The original benchmark service, model files, captions, and memories are only read.
Every checkpoint and HippoRAG index created by this process stays in --output.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

SERVICE = Path("/home/sscy/evaluations/sentrix-worldmm-hippovlog-20260928/services/worldmm_hippovlog")
SOURCE_RUN = SERVICE / "results/20260928-163006/svd_esvd"
DATASET = Path("/ssd/sscy/datasets/HippoVlog-svd-worldmm-20260928")
VIDEOS = ("6Z_qEtbmK34", "Ei7hTKr8Ins")
QUERY_TIME = int("123595999")
GRANULARITIES = ("10sec", "30sec", "3min", "10min")


def stamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def atomic_json(path: Path, value) -> None:
    pending = path.with_suffix(path.suffix + ".tmp")
    pending.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    pending.replace(path)


def append_jsonl(path: Path, value) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, default=str) + "\n")
        handle.flush()


class Sampler:
    def __init__(self, output: Path, torch_module):
        self.path = output / "resources.jsonl"
        self.torch = torch_module
        self.stage = "model_load"
        self.video_id = None
        self.question_id = None
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self.run, daemon=True)

    def start(self) -> None:
        self.thread.start()

    def run(self) -> None:
        import psutil
        proc = psutil.Process()
        while not self.stop.is_set():
            gpu_alloc = None
            gpu_reserved = None
            try:
                gpu_alloc = round(self.torch.cuda.memory_allocated(0) / 1048576, 1)
                gpu_reserved = round(self.torch.cuda.memory_reserved(0) / 1048576, 1)
            except Exception:
                pass
            append_jsonl(self.path, {
                "utc": stamp(), "stage": self.stage, "video_id": self.video_id,
                "question_id": self.question_id,
                "cpu_percent": proc.cpu_percent(interval=None),
                "rss_mib": round(proc.memory_info().rss / 1048576, 1),
                "system_memory_percent": psutil.virtual_memory().percent,
                "gpu_allocated_mib": gpu_alloc, "gpu_reserved_mib": gpu_reserved,
            })
            self.stop.wait(3)

    def close(self) -> None:
        self.stop.set()
        self.thread.join(timeout=5)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    answers_path = output / "answers.partial.jsonl"
    retrieval_path = output / "retrieval_trace.jsonl"
    stages_path = output / "stage_metrics.jsonl"
    status_path = output / "status.json"
    done = {}
    if answers_path.exists():
        for line in answers_path.read_text(encoding="utf-8").splitlines():
            try:
                row = json.loads(line)
                done[row["ID"]] = row
            except (ValueError, KeyError):
                continue

    # The same embedding weights/queries on CPU free GPU memory for Qwen3-VL
    # visual QA. This is a placement change, not a retrieval algorithm change.
    os.environ["WORLDMM_TEXT_MODEL_DEVICE"] = "cpu"
    os.environ["WORLDMM_TEXT_BATCH_SIZE"] = "8"
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    os.environ["WORLDMM_TEXT_MODEL_PATH"] = str(SERVICE / "models/Qwen3-Embedding-4B")
    os.environ["WORLDMM_VIS_MODEL_PATH"] = str(SERVICE / "models/VLM2Vec-V2.0")
    os.environ["WORLDMM_VIS_BASE_MODEL_PATH"] = str(SERVICE / "models/Qwen2-VL-2B-Instruct")
    os.environ["WORLDMM_SOURCE_ROOT"] = str(DATASET / "source/WorldMM")
    os.environ["HIPPOVLOG_DATASET_ROOT"] = str(DATASET)
    sys.path.insert(0, str(SERVICE))
    import qa as original_qa
    from runtime import configure_visual_sdpa, configure_worldmm, limit_worldmm_generation_threads

    started = time.perf_counter()
    configure_worldmm()
    limit_worldmm_generation_threads()
    configure_visual_sdpa()
    logging.getLogger("worldmm").setLevel(logging.CRITICAL)
    import torch
    from worldmm.embedding import EmbeddingModel
    from worldmm.llm import LLMModel, PromptTemplateManager
    from worldmm.memory import WorldMemory
    sampler = Sampler(output, torch)
    sampler.start()
    try:
        build_choices, evaluate_prediction = original_qa.official_eval_functions()
        questions = []
        for video_id in VIDEOS:
            qa_file = DATASET / "worldmm_adapter/questions_by_video" / f"{video_id}.jsonl"
            for line in qa_file.read_text(encoding="utf-8").splitlines():
                row = json.loads(line)
                row["ID"] = f"{video_id}-{row['question_id']}"
                questions.append(row)
        if len(questions) != 80 or len({r["ID"] for r in questions}) != 80:
            raise RuntimeError(f"expected 80 unique questions, found {len(questions)}")
        atomic_json(output / "questions.json", questions)
        atomic_json(status_path, {"status": "running", "done": len(done), "total": 80, "utc": stamp()})
        embedding_model = EmbeddingModel()
        retriever_llm = LLMModel(model_name="qwen3vl-4b")
        responder_llm = LLMModel(model_name="qwen3vl-4b", fps=1)
        memory = WorldMemory(
            embedding_model=embedding_model,
            retriever_llm_model=retriever_llm,
            respond_llm_model=responder_llm,
            prompt_template_manager=PromptTemplateManager(),
            episodic_granularities=list(GRANULARITIES),
            qa_template_name="qa", max_rounds=5, max_errors=5,
        )
        memory.set_retrieval_top_k(episodic=3, semantic=10, visual=3)
        append_jsonl(stages_path, {"stage": "model_load", "seconds": round(time.perf_counter() - started, 3), "utc": stamp()})

        current = {"video_id": None, "question_id": None}
        original_generate = responder_llm.generate

        def traced_generate(*pos, **kw):
            try:
                return original_generate(*pos, **kw)
            except Exception as error:
                append_jsonl(output / "qa_errors.jsonl", {
                    "utc": stamp(), **current, "error_type": type(error).__name__,
                    "error": str(error)[:1200],
                })
                raise

        responder_llm.generate = traced_generate
        for method_name, memory_type in (
            ("retrieve_from_episodic", "episodic"),
            ("retrieve_from_semantic", "semantic"),
            ("retrieve_from_visual", "visual"),
        ):
            original = getattr(memory, method_name)

            def traced(query, *pos, _original=original, _type=memory_type, **kw):
                t0 = time.perf_counter()
                value = _original(query, *pos, **kw)
                content = value[0]
                if _type == "visual":
                    records = [{"clip_key": str(key), "image_count": len(images)} for key, images in (content or {}).items()]
                else:
                    records = str(content or "")
                append_jsonl(retrieval_path, {
                    "utc": stamp(), **current, "memory_type": _type, "query": query,
                    "top_k_config": {"episodic": 3, "semantic": 10, "visual": 3}[_type],
                    "result": records, "result_count": len(records) if isinstance(records, list) else int(bool(records)),
                    "seconds": round(time.perf_counter() - t0, 3),
                })
                return value

            setattr(memory, method_name, traced)

        by_video = {v: [r for r in questions if r["video_id"] == v] for v in VIDEOS}
        for video_id, video_questions in by_video.items():
            if all(row["ID"] in done for row in video_questions):
                continue
            memory.reset()
            sampler.video_id = video_id
            sampler.stage = "memory_index"
            cache = output / "episodic_cache" / video_id / "episodic_memory"
            memory.episodic_memory.save_dir_root = str(cache)
            captions = SOURCE_RUN / "captions" / video_id
            caption_files = {name: str(captions / f"{name}.json") for name in GRANULARITIES}
            semantic = SOURCE_RUN / "metadata/semantic_memory" / video_id / "semantic_consolidation_results_qwen3vl-4b.json"
            visual = SOURCE_RUN / "metadata/visual_memory" / video_id / "visual_embeddings.pkl"
            for path in [*map(Path, caption_files.values()), semantic, visual]:
                if not path.is_file():
                    raise FileNotFoundError(path)
            t0 = time.perf_counter()
            memory.load_episodic_captions(caption_files=caption_files)
            memory.load_semantic_triples(file_path=str(semantic))
            memory.load_visual_clips(embeddings_path=str(visual), clips_data=json.loads((captions / "10sec.json").read_text(encoding="utf-8")))
            memory.index(QUERY_TIME)
            append_jsonl(stages_path, {"stage": "memory_index", "video_id": video_id, "seconds": round(time.perf_counter() - t0, 3), "utc": stamp()})
            for row in video_questions:
                if row["ID"] in done:
                    continue
                current.update(video_id=video_id, question_id=row["ID"])
                sampler.question_id = row["ID"]
                sampler.stage = "retrieval_qa"
                t0 = time.perf_counter()
                choices = build_choices({
                    "choice_a": row["options"]["A"], "choice_b": row["options"]["B"],
                    "choice_c": row["options"]["C"], "choice_d": row["options"]["D"],
                })
                qa_result = memory.answer(query=row["question_text"], choices=choices, until_time=QUERY_TIME)
                response = qa_result.answer
                result = {
                    "ID": row["ID"], "video_id": video_id, "type": row["category"],
                    "question": row["question_text"], "explanation": row.get("explanation"),
                    "choices": choices, "answer": row["correct_answer"], "response": response,
                    "round_history": qa_result.round_history, "num_rounds": qa_result.num_rounds,
                    "evaluate": evaluate_prediction(response, row["correct_answer"], choices),
                    "seconds": round(time.perf_counter() - t0, 3), "utc": stamp(),
                }
                append_jsonl(answers_path, result)
                done[row["ID"]] = result
                torch.cuda.empty_cache()
                atomic_json(status_path, {"status": "running", "done": len(done), "total": 80,
                                          "video_id": video_id, "question_id": row["ID"], "utc": stamp()})
                print(f"[qa] {len(done)}/80 {row['ID']} correct={result['evaluate']} seconds={result['seconds']}", flush=True)
        if len(done) != 80:
            raise RuntimeError(f"only {len(done)}/80 answers completed")
        atomic_json(output / "answers.json", [done[row["ID"]] for row in questions])
        atomic_json(status_path, {"status": "complete", "done": 80, "total": 80, "utc": stamp()})
        memory.cleanup()
    except BaseException as error:
        atomic_json(status_path, {"status": "failed", "done": len(done), "total": 80,
                                  "error": f"{type(error).__name__}: {error}", "utc": stamp()})
        raise
    finally:
        sampler.close()


if __name__ == "__main__":
    main()
