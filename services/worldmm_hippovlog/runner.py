#!/usr/bin/env python3
"""Checkpointed two-arm HippoVlog comparison using WorldMM's own pipeline."""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

from runtime import dataset_root, worldmm_root

SERVICE = Path(__file__).resolve().parent
PROJECT = SERVICE.parents[1]
METHODS = ("worldmm_uniform", "svd_esvd")
LABELS = {"worldmm_uniform": "WorldMM 均匀抽帧", "svd_esvd": "Sentrix SVD/eSVD"}
EXPECTED_CATEGORIES = {"audio", "audio_visual", "visual", "summarization"}


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def read_json(path: Path, default):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default


class Run:
    def __init__(self, root: Path):
        self.root = root.resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.state_file = self.root / "status.json"
        self.state = read_json(self.state_file, {})
        self.state.update(status="running", updated_at=datetime.now().isoformat(timespec="seconds"))
        self.save()
        self.env = os.environ.copy()
        self.env.update({
            "HIPPOVLOG_DATASET_ROOT": str(dataset_root()),
            "WORLDMM_SOURCE_ROOT": str(worldmm_root()),
            "WORLDMM_BENCH_RUN_DIR": str(self.root),
            "SENTRIX_VIDEO_DEVICE": "cuda",
            "SENTRIX_VIDEO_PYTHON": sys.executable,
            "HF_ENDPOINT": "https://hf-mirror.com",
            "HF_HUB_DISABLE_XET": "1",
            "HF_HUB_DOWNLOAD_TIMEOUT": "600",
            "HF_HUB_ETAG_TIMEOUT": "30",
            "WORLDMM_TEXT_MODEL_PATH": str(SERVICE / "models" / "Qwen3-Embedding-4B"),
            "WORLDMM_VIS_MODEL_PATH": str(SERVICE / "models" / "VLM2Vec-V2.0"),
            "WORLDMM_VIS_BASE_MODEL_PATH": str(SERVICE / "models" / "Qwen2-VL-2B-Instruct"),
            "PYTHONPATH": os.pathsep.join([str(worldmm_root() / "src"), str(PROJECT), self.env.get("PYTHONPATH", "")]),
        })
        yolo = SERVICE / "models" / "yolo11n.pt"
        if yolo.is_file():
            self.env["SENTRIX_VIDEO_YOLO_MODEL"] = str(yolo)

    def save(self) -> None:
        self.state["updated_at"] = datetime.now().isoformat(timespec="seconds")
        atomic_json(self.state_file, self.state)

    def stage(self, name: str, progress: float, message: str, **values) -> None:
        self.state.update(stage=name, progress=round(progress, 2), message=message, **values)
        self.save()
        print(f"[stage] {name}: {message}", flush=True)

    def command(self, *parts: str) -> None:
        print("[run] " + " ".join(parts), flush=True)
        subprocess.run([sys.executable, *parts], cwd=PROJECT, env=self.env, check=True)

    def start_command(self, *parts: str) -> subprocess.Popen:
        print("[run] " + " ".join(parts), flush=True)
        return subprocess.Popen([sys.executable, *parts], cwd=PROJECT, env=self.env)


def prepare(run: Run) -> list[str]:
    root = dataset_root()
    qa_dir = root / "worldmm_adapter" / "questions_by_video"
    manifest_path = root / "worldmm_adapter" / "hippovlog_manifest.jsonl"
    if not qa_dir.is_dir() or not manifest_path.is_file() or not worldmm_root().is_dir():
        raise FileNotFoundError("HippoVlog questions, manifest, or WorldMM source is missing")
    videos = sorted(path.stem for path in (root / "videos").glob("*.mp4"))
    manifests = {row["video_id"]: row for row in map(json.loads, manifest_path.read_text(encoding="utf-8").splitlines())}
    if len(videos) != 25 or set(videos) != set(manifests):
        raise ValueError(f"expected 25 matching videos and manifests, got {len(videos)} videos")
    questions = []
    for video_id in videos:
        qa_file = qa_dir / f"{video_id}.jsonl"
        if not qa_file.is_file():
            raise FileNotFoundError(qa_file)
        for line in qa_file.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            choices = row["options"]
            if set(choices) != {"A", "B", "C", "D"} or row["category"] not in EXPECTED_CATEGORIES:
                raise ValueError(f"invalid choices/category for {video_id}: {row.get('question_id')}")
            questions.append({
                "ID": f"{video_id}-{row['question_id']}", "video_id": video_id,
                "question": row["question_text"], "answer": row["correct_answer"],
                "type": row["category"], "duration": "long",
                **{f"choice_{letter.lower()}": choices[letter] for letter in "ABCD"},
            })
    if len(questions) != 1000 or len({row["ID"] for row in questions}) != 1000:
        raise ValueError(f"expected 1000 unique questions, got {len(questions)}")
    if any(count != 250 for count in Counter(row["type"] for row in questions).values()):
        raise ValueError("category counts differ from the HippoVlog protocol")
    atomic_json(run.root / "eval.json", questions)
    atomic_json(run.root / "experiment.json", {
        "source_project_commit": subprocess.check_output(["git", "-C", str(PROJECT), "rev-parse", "HEAD"], text=True).strip(),
        "worldmm_commit": subprocess.check_output(["git", "-C", str(worldmm_root()), "rev-parse", "HEAD"], text=True).strip(),
        "dataset": str(root), "videos": len(videos), "questions": len(questions),
        "methods": list(METHODS), "model": "qwen3vl-4b",
        "baseline_sampling": "WorldMM 1 fps in each 10-second window",
        "candidate_sampling": "Sentrix SVD/eSVD with required CUDA YOLO semantic events",
        "memory_and_qa": "WorldMM official source, shared visual encoding protocol",
    })
    return videos


def extract_keyframes(run: Run, videos: list[str]) -> None:
    caption_process = None
    caption_name = ""

    def schedule_caption() -> None:
        nonlocal caption_process, caption_name
        if caption_process is not None:
            result = caption_process.poll()
            if result is None:
                return
            if result != 0:
                raise RuntimeError(f"parallel caption task {caption_name} exited with code {result}")
            caption_process = None
        for video_id in videos:
            for algorithm in ("svd_esvd", "worldmm_uniform"):
                if algorithm == "svd_esvd" and not (
                    run.root / algorithm / "keyframes" / video_id / "manifest.json"
                ).is_file():
                    continue
                caption_dir = run.root / algorithm / "captions" / video_id
                if (caption_dir / "metrics.json").is_file():
                    continue
                caption_name = f"{algorithm}/{video_id}"
                caption_process = run.start_command(
                    str(SERVICE / "worker.py"), "caption", "--algorithm", algorithm,
                    "--video-id", video_id, "--run-dir", str(run.root),
                )
                print(f"[parallel caption] {caption_name} while SVD extraction continues", flush=True)
                return

    for index, video_id in enumerate(videos):
        run.stage("svd_esvd_keyframes", 2 + 18 * index / len(videos),
                  f"SVD/eSVD CUDA 抽帧 {index + 1}/{len(videos)}：{video_id}",
                  video_done=index, video_total=len(videos))
        manifest = run.root / "svd_esvd" / "keyframes" / video_id / "manifest.json"
        if manifest.is_file():
            schedule_caption()
            continue
        extraction = run.start_command(
            str(SERVICE / "worker.py"), "extract", "--video-id", video_id, "--run-dir", str(run.root)
        )
        while extraction.poll() is None:
            schedule_caption()
            time.sleep(5)
        if extraction.returncode != 0:
            raise RuntimeError(f"SVD/eSVD extraction failed for {video_id}: exit={extraction.returncode}")
        schedule_caption()
    if caption_process is not None:
        result = caption_process.wait()
        if result != 0:
            raise RuntimeError(f"parallel caption task {caption_name} exited with code {result}")
    run.stage("svd_esvd_keyframes", 20, "25 个视频的 SVD/eSVD 抽帧完成", video_done=len(videos))


def make_captions(run: Run, videos: list[str], algorithm: str, begin: int, span: int) -> None:
    for index, video_id in enumerate(videos):
        run.stage(f"{algorithm}_captions", begin + span * index / len(videos),
                  f"{LABELS[algorithm]}：片段描述与多尺度摘要 {index + 1}/{len(videos)}：{video_id}",
                  video_done=index, video_total=len(videos))
        run.command(str(SERVICE / "worker.py"), "caption", "--algorithm", algorithm,
                    "--video-id", video_id, "--run-dir", str(run.root))


def build_memories(run: Run, algorithm: str, begin: int) -> None:
    captions = run.root / algorithm / "captions"
    metadata = run.root / algorithm / "metadata"
    for index, step in enumerate(("episodic", "semantic", "visual")):
        if step == "visual" and algorithm == "svd_esvd":
            baseline_visual = run.root / "worldmm_uniform" / "metadata" / "visual_memory"
            target = metadata / "visual_memory"
            if not baseline_visual.is_dir():
                raise FileNotFoundError(baseline_visual)
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists():
                target.symlink_to(baseline_visual, target_is_directory=True)
            continue
        run.stage(f"{algorithm}_{step}_memory", begin + index * 3,
                  f"{LABELS[algorithm]}：WorldMM {step} 记忆构建")
        run.command(str(SERVICE / "bootstrap.py"), "build_memory",
                    "--caption-dir", str(captions), "--output-dir", str(metadata),
                    "--model", "qwen3vl-4b", "--step", step, "--gpu", "0", "--num-frames", "16")


def summarize(run: Run, algorithm: str, results: list[dict]) -> None:
    if len(results) != 1000:
        raise RuntimeError(f"{algorithm} has {len(results)}/1000 QA rows")
    if any(str(row.get("response", "")).startswith(("Error", "Index error")) for row in results):
        raise RuntimeError(f"{algorithm} contains failed QA responses; refusing to report accuracy")
    by_category = {category: [row for row in results if row.get("type") == category] for category in EXPECTED_CATEGORIES}
    if any(len(rows) != 250 for rows in by_category.values()):
        raise RuntimeError("category coverage is incomplete")
    captions = run.root / algorithm / "captions"
    metrics = [read_json(path, {}) for path in captions.glob("*/metrics.json")]
    keyframe_seconds = None
    keyframe_count = sum(int(row.get("input_frame_count", 0)) for row in metrics)
    if algorithm == "svd_esvd":
        manifests = [read_json(path, {}) for path in (run.root / algorithm / "keyframes").glob("*/manifest.json")]
        keyframe_seconds = sum(float(row.get("runtime_seconds", 0)) for row in manifests)
        keyframe_count = sum(int(row.get("selected_keyframe_count", 0)) for row in manifests)
    else:
        keyframe_seconds = sum(float(row.get("selection_seconds", 0)) for row in metrics)
    summary = read_json(run.root / "summary.json", {"items": []})
    item = {
        "algorithm": algorithm, "algorithm_label": LABELS[algorithm],
        "qa_done": len(results), "qa_total": 1000,
        "accuracy": sum(bool(row["evaluate"]) for row in results) / len(results),
        "keyframe_count": keyframe_count,
        "caption_input_frame_count": sum(int(row.get("input_frame_count", 0)) for row in metrics),
        "fallback_window_count": sum(int(row.get("fallback_window_count", 0)) for row in metrics),
        "keyframe_seconds": keyframe_seconds,
        **{f"{category}_accuracy": sum(bool(row["evaluate"]) for row in rows) / len(rows)
           for category, rows in by_category.items()},
    }
    item["summary_accuracy"] = item["summarization_accuracy"]
    summary["items"] = [row for row in summary["items"] if row.get("algorithm") != algorithm] + [item]
    atomic_json(run.root / "summary.json", summary)


def evaluate(run: Run, algorithm: str, begin: int) -> None:
    run.stage(f"{algorithm}_qa", begin, f"{LABELS[algorithm]}：WorldMM 自适应检索与 1,000 题 QA")
    output = run.root / algorithm / "eval"
    result_file = output / "qwen3vl_4b_qwen3vl_4b" / "videomme_eval.json"
    if not result_file.is_file():
        run.command(str(SERVICE / "qa.py"), "--run-dir", str(run.root), "--algorithm", algorithm)
    summarize(run, algorithm, json.loads(result_file.read_text(encoding="utf-8")))
    done = (1 if algorithm == "worldmm_uniform" else 2) * 1000
    run.stage(f"{algorithm}_qa", begin + 5, f"{LABELS[algorithm]} 1,000 题完成", question_done=done)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    run = Run(args.run_dir)
    try:
        run.stage("prepare", 0, "验证 25 个视频与 1,000 题")
        videos = prepare(run)
        extract_keyframes(run, videos)
        make_captions(run, videos, "worldmm_uniform", 20, 22)
        make_captions(run, videos, "svd_esvd", 42, 22)
        build_memories(run, "worldmm_uniform", 64)
        build_memories(run, "svd_esvd", 73)
        evaluate(run, "worldmm_uniform", 83)
        evaluate(run, "svd_esvd", 92)
        run.stage("complete", 100, "两种方法的 2,000 条 QA 结果均已保存", status="complete")
    except KeyboardInterrupt:
        run.stage("stopped", run.state.get("progress", 0), "任务已停止，可用原运行目录续跑", status="cancelled")
        raise
    except Exception as error:
        run.stage("failed", run.state.get("progress", 0), f"{type(error).__name__}: {error}", status="failed")
        raise


if __name__ == "__main__":
    main()
