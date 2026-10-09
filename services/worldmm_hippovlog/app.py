#!/usr/bin/env python3
"""Small, isolated HTTP UI for the Sentrix/WorldMM HippoVlog evaluation."""
from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import threading
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parent
STATIC = ROOT / "static"
RUNS = ROOT / "results"
DATASET = Path(os.environ.get(
    "HIPPOVLOG_DATASET_ROOT",
    "/ssd/sscy/datasets/HippoVlog-svd-worldmm-20260928",
)).resolve()
WORLDMM = DATASET / "source" / "WorldMM"
LOCK = threading.Lock()
PROCESS: subprocess.Popen | None = None
ACTIVE_RUN: Path | None = None


def read_json(path: Path, fallback):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return fallback


def dataset_info() -> dict:
    manifest = DATASET / "worldmm_adapter" / "hippovlog_manifest.jsonl"
    questions_dir = DATASET / "worldmm_adapter" / "questions_by_video"
    videos = sorted((DATASET / "videos").glob("*.mp4"))
    question_files = sorted(questions_dir.glob("*.jsonl")) if questions_dir.is_dir() else []
    question_count = sum(
        1 for question_file in question_files
        for line in question_file.open(encoding="utf-8") if line.strip()
    )
    manifest_count = sum(1 for line in manifest.open(encoding="utf-8") if line.strip()) if manifest.is_file() else 0
    return {
        "dataset_root": str(DATASET),
        "dataset_ready": DATASET.is_dir() and bool(videos) and question_count > 0,
        "video_count": len(videos),
        "question_count": question_count,
        "manifest_count": manifest_count,
        "question_files": len(question_files),
        "question_root": str(questions_dir),
        "worldmm_root": str(WORLDMM),
        "worldmm_source_present": WORLDMM.is_dir(),
        "models": {
            "caption_retriever_responder": "Qwen3-VL-4B-Instruct (local, WorldMM adapter config)",
            "text_embedding": "Qwen3-Embedding-4B",
            "visual_embedding": "VLM2Vec-V2.0 / Qwen2-VL-2B",
        },
    }


def latest_run() -> Path | None:
    candidates = sorted((path for path in RUNS.iterdir() if path.is_dir()), reverse=True) if RUNS.exists() else []
    return candidates[0] if candidates else None


def current_status() -> dict:
    with LOCK:
        run_dir = ACTIVE_RUN
        process = PROCESS
    if run_dir is None:
        run_dir = latest_run()
    if run_dir is None:
        return {"status": "idle", "stage": "ready", "progress": 0, "run_id": None}
    state = read_json(run_dir / "status.json", {})
    if process is not None and process.poll() is None:
        state.setdefault("status", "running")
    elif state.get("status") == "running":
        state["status"] = "failed"
        state["message"] = state.get("message") or "评测进程已退出；请查看运行日志。"
    state["run_id"] = run_dir.name
    state["result_dir"] = str(run_dir)
    return state


class Handler(BaseHTTPRequestHandler):
    server_version = "SentrixWorldMMBench/0.1"

    def log_message(self, fmt, *args):
        print("[http] " + fmt % args, flush=True)

    def send_json(self, payload, status=200):
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = urlparse(self.path).path
        if path == "/api/config":
            self.send_json(dataset_info())
            return
        if path == "/api/status":
            self.send_json(current_status())
            return
        if path == "/api/results":
            run_dir = latest_run()
            self.send_json(read_json(run_dir / "summary.json", {"items": []}) if run_dir else {"items": []})
            return
        if path == "/api/log":
            run_dir = latest_run()
            text = ""
            if run_dir:
                try:
                    text = (run_dir / "run.log").read_text(encoding="utf-8", errors="replace")[-24000:]
                except OSError:
                    pass
            self.send_json({"text": text})
            return
        if path in {"/", "/index.html"}:
            target = STATIC / "index.html"
        else:
            target = (STATIC / path.lstrip("/")).resolve()
            if STATIC.resolve() not in target.parents:
                self.send_error(404)
                return
        if not target.is_file():
            self.send_error(404)
            return
        body = target.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        global PROCESS, ACTIVE_RUN
        path = urlparse(self.path).path
        if path == "/api/stop":
            with LOCK:
                process = PROCESS
            if process and process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                run_dir = ACTIVE_RUN
                if run_dir:
                    state = read_json(run_dir / "status.json", {})
                    state.update(status="cancelled", stage="stopped", message="已停止，可从该运行目录续跑")
                    (run_dir / "status.json").write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
                self.send_json({"ok": True, "message": "已请求停止当前运行"})
            else:
                self.send_json({"ok": False, "message": "当前没有运行中的任务"}, 409)
            return
        if path not in {"/api/run", "/api/resume"}:
            self.send_error(404)
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            request = json.loads(self.rfile.read(length) or b"{}")
        except (ValueError, json.JSONDecodeError):
            self.send_json({"error": "请求不是有效 JSON"}, 400)
            return
        info = dataset_info()
        if not info["dataset_ready"] or not info["worldmm_source_present"]:
            self.send_json({"error": "数据集或 WorldMM 源码未就绪", "config": info}, 409)
            return
        with LOCK:
            if PROCESS is not None and PROCESS.poll() is None:
                self.send_json({"error": "已有评测正在运行"}, 409)
                return
            if path == "/api/resume":
                run_dir = latest_run()
                if run_dir is None or read_json(run_dir / "status.json", {}).get("status") == "complete":
                    self.send_json({"error": "没有可续跑的任务"}, 409)
                    return
                run_id = run_dir.name
            else:
                run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
                run_dir = RUNS / run_id
                run_dir.mkdir(parents=True, exist_ok=False)
                state = {
                    "status": "queued", "stage": "starting", "progress": 0,
                    "video_done": 0, "video_total": info["video_count"],
                    "question_done": 0, "question_total": 2 * info["question_count"],
                    "algorithms": ["worldmm_uniform", "svd_esvd"],
                    "message": "任务排队中", "updated_at": datetime.now().isoformat(timespec="seconds"),
                }
                (run_dir / "status.json").write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
            worker_python = os.environ.get("WORLDMM_BENCH_PYTHON", str(ROOT / ".venv" / "bin" / "python"))
            if not Path(worker_python).is_file():
                worker_python = sys.executable
            env = os.environ.copy()
            env["HIPPOVLOG_DATASET_ROOT"] = str(DATASET)
            env["WORLDMM_SOURCE_ROOT"] = str(WORLDMM)
            env["WORLDMM_BENCH_RUN_DIR"] = str(run_dir)
            PROCESS = subprocess.Popen(
                [worker_python, str(ROOT / "runner.py"), "--run-dir", str(run_dir)],
                cwd=str(ROOT), env=env, stdout=(run_dir / "run.log").open("ab"),
                stderr=subprocess.STDOUT, start_new_session=True,
            )
            ACTIVE_RUN = run_dir
        self.send_json({"ok": True, "run_id": run_id, "status": "queued"}, 202)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8776)
    args = parser.parse_args()
    print(json.dumps({"listening": f"{args.host}:{args.port}", **dataset_info()}, ensure_ascii=False), flush=True)
    ThreadingHTTPServer((args.host, args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
