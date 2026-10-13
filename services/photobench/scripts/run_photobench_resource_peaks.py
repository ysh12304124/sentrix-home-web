#!/usr/bin/env python3
"""Run PhotoBench album3-14/compact-10q and record strict resource peaks."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
from statistics import mean
from urllib.error import HTTPError
from urllib.request import Request, urlopen

TERMINAL = {"completed", "completed_with_errors", "failed", "cancelled", "interrupted"}


def http_json(url: str, *, method: str = "GET", payload: dict | None = None, timeout: float = 30) -> dict:
    data = None if payload is None else json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = Request(url, data=data, method=method, headers={"Content-Type": "application/json"})
    try:
        with urlopen(req, timeout=timeout) as response:
            value = json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        body = exc.read().decode("utf-8", "replace")
        raise RuntimeError(f"HTTP {exc.code} from {url}: {body[:2000]}") from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object from {url}")
    return value


def model_ids(base_url: str) -> list[str]:
    value = http_json(base_url.rstrip("/") + "/models", timeout=15)
    ids = []
    for item in (value.get("data") or []) + (value.get("models") or []):
        if isinstance(item, dict):
            ident = item.get("id") or item.get("model") or item.get("name")
            if ident:
                ids.append(str(ident))
    return sorted(set(ids))


def switch_model(script: str, model: str) -> None:
    result = subprocess.run([script, model], text=True, capture_output=True, timeout=900)
    if result.returncode:
        raise RuntimeError(f"switch {model} failed:\n{result.stdout[-4000:]}\n{result.stderr[-4000:]}")
    print(result.stdout[-4000:], flush=True)


def wait_model(base_url: str, expected: str, timeout: float = 600) -> list[str]:
    deadline = time.time() + timeout
    last = []
    while time.time() < deadline:
        try:
            last = model_ids(base_url)
            if last == [expected]:
                return last
        except Exception:
            pass
        time.sleep(2)
    raise RuntimeError(f"endpoint model mismatch: expected {[expected]!r}, last={last!r}")


def wait_idle(api: str, timeout: float = 120) -> None:
    deadline = time.time() + timeout
    active = []
    while time.time() < deadline:
        value = http_json(api.rstrip("/") + "/api/runs", timeout=30)
        runs = value.get("runs") if isinstance(value.get("runs"), list) else []
        active = [r for r in runs if isinstance(r, dict) and r.get("status") in {"running", "pending", "cancelling"}]
        if not active:
            return
        time.sleep(2)
    raise RuntimeError(f"active runs remain: {active!r}")


def start_run(api: str, payload: dict) -> dict:
    last: Exception | None = None
    for attempt in range(1, 4):
        try:
            return http_json(api.rstrip("/") + "/api/runs", method="POST", payload=payload, timeout=120)
        except Exception as exc:
            last = exc
            print(f"start attempt {attempt}/3 failed: {exc}", file=sys.stderr, flush=True)
            time.sleep(8 * attempt)
            wait_idle(api)
    raise RuntimeError(f"start failed after retries: {last}")


def fetch_run(api: str, run_id: str) -> dict:
    value = http_json(api.rstrip("/") + "/api/runs/" + run_id, timeout=60)
    return value.get("run") if isinstance(value.get("run"), dict) else value


def wait_run(api: str, run_id: str, timeout: float) -> dict:
    deadline = time.time() + timeout
    last = {}
    while time.time() < deadline:
        last = fetch_run(api, run_id)
        status = str(last.get("status") or "")
        progress = ((last.get("phases") or {}).get("qa_eval") or {}).get("progress") or {}
        print(f"[{run_id}] {status} {progress.get('completed')}/{progress.get('total')}", flush=True)
        if status in TERMINAL:
            return last
        time.sleep(10)
    raise RuntimeError(f"run timeout: {run_id}; last={last!r}")


def values(rows: list[dict], key: str) -> list[float]:
    return [float(row[key]) for row in rows if isinstance(row.get(key), (int, float))]


def summarize(project: Path, run: dict, expected_model: str, platform: str) -> dict:
    run_id = str(run.get("run_id") or "")
    result_dir = project / "services" / "photobench" / "results" / run_id
    run_path = result_dir / "run.json"
    telemetry_path = result_dir / "gpu_samples.jsonl"
    results_path = result_dir / "results.jsonl"
    if not all(path.is_file() for path in (run_path, telemetry_path, results_path)):
        raise RuntimeError(f"missing result artifacts: {result_dir}")
    saved = json.loads(run_path.read_text(encoding="utf-8"))
    samples = [json.loads(line) for line in telemetry_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    items = [json.loads(line) for line in results_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if saved.get("status") != "completed":
        raise RuntimeError(f"not clean completed: {run_id} status={saved.get('status')!r}")
    if saved.get("album_id") != "album3-14" or saved.get("qa_set") != "compact-10q":
        raise RuntimeError(f"wrong subset: {saved.get('album_id')}/{saved.get('qa_set')}")
    if len(items) != 10 or not samples:
        raise RuntimeError(f"unexpected artifact counts: results={len(items)} samples={len(samples)}")
    snapshot = saved.get("current_model_snapshot") or {}
    actual = str(snapshot.get("served_model_name") or saved.get("model_name") or saved.get("model_profile") or "")
    if actual != expected_model:
        raise RuntimeError(f"model mismatch: expected={expected_model!r} actual={actual!r}")
    required = ("model_process_memory_used_mib", "model_process_system_memory_used_mib",
                "product_stack_memory_mib", "benchmark_process_memory_used_mib")
    missing = {key: sum(row.get(key) is None for row in samples) for key in required}
    if any(missing.values()):
        raise RuntimeError(f"required telemetry missing: {missing}")
    product_scopes = sorted({str(row.get("benchmark_process_memory_scope")) for row in samples})
    if "None" in product_scopes:
        raise RuntimeError(f"product RAM scope missing: {product_scopes}")
    gpu_process = values(samples, "benchmark_process_gpu_memory_mib")
    if platform == "153":
        if len(gpu_process) != len(samples) or any(row.get("benchmark_process_gpu_memory_scope") in (None, "None") for row in samples):
            raise RuntimeError("153 GPU process attribution/scope incomplete")
    elif any(row.get("benchmark_process_gpu_memory_mib") is not None for row in samples):
        raise RuntimeError("118 reported independent GPU VRAM")

    def peak(key: str) -> float | None:
        data = values(samples, key)
        return round(max(data), 2) if data else None

    recalls = values(items, "retrieval_recall")
    precisions = values(items, "retrieval_precision")
    wall = values(items, "wall_clock_ms")
    calls = [len(item.get("model_call_metrics") or []) for item in items]
    prompt_total = sum(float((item.get("llm_summary") or {}).get("prompt_tokens_total") or 0) for item in items)
    completion_total = sum(float((item.get("llm_summary") or {}).get("completion_tokens_total") or 0) for item in items)
    parse_total = sum(int((item.get("agent_stability") or {}).get("json_parse_total") or 0) for item in items)
    parse_success = sum(int((item.get("agent_stability") or {}).get("json_parse_success") or 0) for item in items)
    within = [1 if (item.get("agent_stability") or {}).get("completed_within_steps") else 0 for item in items if isinstance((item.get("agent_stability") or {}).get("completed_within_steps"), bool)]
    scores = [float((item.get("judge") or {}).get("score")) for item in items if isinstance((item.get("judge") or {}).get("score"), (int, float))]
    return {
        "run_id": run_id, "platform": platform, "album_id": saved.get("album_id"), "qa_set": saved.get("qa_set"),
        "status": saved.get("status"), "model_id": actual, "model_profile": saved.get("model_profile"),
        "runtime_framework": saved.get("runtime_framework"), "telemetry_source": saved.get("telemetry_source"),
        "sample_count": len(samples), "result_count": len(items),
        "model_gpu_or_pss_peak_mib": peak("model_process_memory_used_mib"),
        "model_process_vmrss_peak_mib": peak("model_process_system_memory_used_mib"),
        "product_stack_memory_peak_mib": peak("product_stack_memory_mib"),
        "benchmark_gpu_process_peak_mib": peak("benchmark_process_gpu_memory_mib"),
        "device_gpu_total_peak_mib": peak("memory_used_mib"), "host_memory_used_peak_mib": peak("system_memory_used_mib"),
        "model_process_memory_scope": sorted({str(row.get("process_memory_scope")) for row in samples}),
        "model_process_system_memory_scope": sorted({str(row.get("model_process_system_memory_scope")) for row in samples}),
        "product_stack_memory_scope": product_scopes,
        "benchmark_gpu_scope": sorted({str(row.get("benchmark_process_gpu_memory_scope")) for row in samples}),
        "retrieval_recall_mean": round(mean(recalls), 6) if recalls else None,
        "retrieval_precision_mean": round(mean(precisions), 6) if precisions else None,
        "answer_score_mean": round(mean(scores), 6) if scores else None,
        "within_steps_rate": round(mean(within), 6) if within else None,
        "json_parse_success_rate": round(parse_success / parse_total, 6) if parse_total else None,
        "average_wall_clock_s": round(mean(wall) / 1000, 6) if wall else None,
        "average_model_calls": round(mean(calls), 6) if calls else None,
        "prompt_tokens_total": int(prompt_total), "completion_tokens_total": int(completion_total),
        "artifacts": {"run": str(run_path), "telemetry": str(telemetry_path), "results": str(results_path)},
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", required=True)
    parser.add_argument("--api", default="http://127.0.0.1:8771")
    parser.add_argument("--model-base-url", default="http://127.0.0.1:8100/v1")
    parser.add_argument("--sentrix-url", required=True)
    parser.add_argument("--switch-script", required=True)
    parser.add_argument("--platform", choices=("118", "153"), required=True)
    parser.add_argument("--models", required=True, help="comma-separated explicit endpoint model ids")
    parser.add_argument("--output", required=True)
    parser.add_argument("--run-timeout", type=float, default=4 * 3600)
    args = parser.parse_args()
    project = Path(args.project).resolve()
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    summaries: list[dict] = []
    failures: list[dict] = []
    for model in [item.strip() for item in args.models.split(",") if item.strip()]:
        try:
            print(f"=== switch {model} ===", flush=True)
            switch_model(args.switch_script, model)
            print(f"endpoint ready: {wait_model(args.model_base_url, model)}", flush=True)
            # The local Sentrix process may still be applying the previous
            # external-runtime binding after llama-server is reachable.
            time.sleep(10)
            wait_idle(args.api)
            payload = {"album_id": "album3-14", "qa_set": "compact-10q", "mode": "full", "models": ["__current__"],
                       "sentrix_url": args.sentrix_url, "model_base_url": args.model_base_url, "endpoint_model": model,
                       "runtime_framework": "llama.cpp", "vllm_manager_url": "", "delete_scope_after_run": False}
            response = start_run(args.api, payload)
            run_ids = response.get("run_ids") or []
            if len(run_ids) != 1:
                raise RuntimeError(f"unexpected start response: {response!r}")
            run_id = str(run_ids[0])
            print(f"started {run_id} for {model}", flush=True)
            run = wait_run(args.api, run_id, args.run_timeout)
            summary = summarize(project, run, model, args.platform)
            summaries.append(summary)
            print(json.dumps(summary, ensure_ascii=False, sort_keys=True), flush=True)
        except Exception as exc:
            failure = {"model_id": model, "error": f"{type(exc).__name__}: {exc}"}
            failures.append(failure)
            print(json.dumps(failure, ensure_ascii=False), file=sys.stderr, flush=True)
    output.write_text(json.dumps({"platform": args.platform, "summaries": summaries, "failures": failures}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return 0 if not failures else 2


if __name__ == "__main__":
    raise SystemExit(main())
