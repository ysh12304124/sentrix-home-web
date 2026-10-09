#!/usr/bin/env python3
"""WorldMM adaptive retrieval with per-question HippoVlog checkpoints."""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from runtime import configure_visual_sdpa, configure_worldmm, limit_worldmm_generation_threads, worldmm_root

GRANULARITIES = ("10sec", "30sec", "3min", "10min")
QUERY_TIME = int("1" + "23595999")


def atomic_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_suffix(path.suffix + ".tmp")
    pending.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    pending.replace(path)


def official_eval_functions():
    target = worldmm_root() / "eval" / "eval.py"
    spec = importlib.util.spec_from_file_location("worldmm_official_eval", target)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import official WorldMM evaluation functions from {target}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.build_choices, module.evaluate_prediction


def run_qa(run_dir: Path, algorithm: str) -> None:
    # Keep retrieval on the RTX 3090; a one-item batch avoids the old
    # 256-item embedding peak while preserving the same embedding model.
    os.environ["WORLDMM_TEXT_MODEL_DEVICE"] = "cuda"
    os.environ["WORLDMM_TEXT_BATCH_SIZE"] = "1"
    configure_worldmm()
    limit_worldmm_generation_threads()
    configure_visual_sdpa()
    from worldmm.embedding import EmbeddingModel
    from worldmm.llm import LLMModel, PromptTemplateManager
    from worldmm.memory import WorldMemory

    build_choices, evaluate_prediction = official_eval_functions()
    questions = json.loads((run_dir / "eval.json").read_text(encoding="utf-8"))
    by_video = defaultdict(list)
    for row in questions:
        by_video[row["video_id"]].append(row)
    output = run_dir / algorithm / "eval" / "qwen3vl_4b_qwen3vl_4b"
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = output / "answers.partial.jsonl"
    final = output / "videomme_eval.json"
    done = {}
    if checkpoint.is_file():
        for line in checkpoint.read_text(encoding="utf-8").splitlines():
            try:
                row = json.loads(line)
                done[row["ID"]] = row
            except (ValueError, KeyError):
                continue
    if len(done) == len(questions) and final.is_file():
        print(f"[resume] {algorithm}: 1000 QA already complete", flush=True)
        return

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
    status_path = run_dir / "status.json"

    for video_id, video_questions in sorted(by_video.items()):
        if all(row["ID"] in done for row in video_questions):
            continue
        memory.reset()
        memory.episodic_memory.save_dir_root = str(run_dir / algorithm / "episodic_cache" / video_id / "episodic_memory")
        caption_root = run_dir / algorithm / "captions" / video_id
        caption_files = {name: str(caption_root / f"{name}.json") for name in GRANULARITIES}
        missing = [name for name, path in caption_files.items() if not Path(path).is_file()]
        if missing:
            raise FileNotFoundError(f"{video_id} missing WorldMM captions: {missing}")
        memory.load_episodic_captions(caption_files=caption_files)
        semantic_path = run_dir / algorithm / "metadata" / "semantic_memory" / video_id / "semantic_consolidation_results_qwen3vl-4b.json"
        visual_path = run_dir / algorithm / "metadata" / "visual_memory" / video_id / "visual_embeddings.pkl"
        if not semantic_path.is_file() or not visual_path.is_file():
            raise FileNotFoundError(f"{video_id} missing semantic/visual memory: {semantic_path}, {visual_path}")
        memory.load_semantic_triples(file_path=str(semantic_path))
        memory.load_visual_clips(embeddings_path=str(visual_path), clips_data=json.loads((caption_root / "10sec.json").read_text(encoding="utf-8")))
        memory.index(QUERY_TIME)
        for row in video_questions:
            if row["ID"] in done:
                continue
            choices = build_choices(row)
            qa_result = memory.answer(query=row["question"], choices=choices, until_time=QUERY_TIME)
            answer = qa_result.answer
            result = {
                "ID": row["ID"], "video_id": video_id, "type": row["type"],
                "duration": row["duration"], "question": row["question"],
                "choices": choices, "answer": row["answer"], "response": answer,
                "round_history": qa_result.round_history, "num_rounds": qa_result.num_rounds,
                "evaluate": evaluate_prediction(answer, row["answer"], choices),
            }
            with checkpoint.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(result, ensure_ascii=False) + "\n")
            done[row["ID"]] = result
            state = json.loads(status_path.read_text(encoding="utf-8"))
            state.update(
                question_done=len(done) + (1000 if algorithm == "svd_esvd" else 0),
                question_total=2000,
                message=f"{algorithm} QA {len(done)}/1000 · {video_id}",
                updated_at=datetime.now().isoformat(timespec="seconds"),
            )
            atomic_json(status_path, state)
            print(f"[qa] {algorithm} {len(done)}/1000 {row['ID']} correct={result['evaluate']}", flush=True)
    if len(done) != 1000:
        raise RuntimeError(f"{algorithm}: only {len(done)}/1000 completed answers")
    atomic_json(final, [done[row["ID"]] for row in questions])
    memory.cleanup()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--algorithm", required=True, choices=["worldmm_uniform", "svd_esvd"])
    args = parser.parse_args()
    run_qa(args.run_dir, args.algorithm)


if __name__ == "__main__":
    main()
