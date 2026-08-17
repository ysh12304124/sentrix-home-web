#!/usr/bin/env python3
"""Thin HTTP wrapper around sentrix_vllm_manager.py.

Exposes start/stop/switch/status/list as REST endpoints so remote hosts
(e.g. 192.168.0.100) can manage vLLM profiles without SSH command splicing.

Run from the repo root:
  python3 services/vllm_manager/app.py --host 0.0.0.0 --port 8500

The manager CLI and registry default to repo-local files
(configs/sentrix_vllm_registry_192_168_0_153.json), overridable via
SENTRIX_VLLM_MANAGER / SENTRIX_VLLM_REGISTRY.
"""
from __future__ import annotations

import os
import argparse
import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parent.parent.parent
MANAGER = Path(os.environ.get("SENTRIX_VLLM_MANAGER", str(ROOT / "services/vllm_manager/manager.py")))
REGISTRY = Path(os.environ.get("SENTRIX_VLLM_REGISTRY", str(ROOT / "configs/sentrix_vllm_registry_192_168_0_153.json")))
STATE_FILE_DEFAULT = "/home/asus/sentrix-vllm/state/current.json"

app = FastAPI(title="Sentrix vLLM Manager API", version="1.0.0")


def _read_json(path: Path, fallback):
    try:
        return json.loads(path.read_text())
    except Exception:
        return fallback


def _state_file_path() -> Path:
    registry = _read_json(REGISTRY, {})
    return Path(registry.get("state_file") or STATE_FILE_DEFAULT)


def _process_tree(root_pid: int) -> set[int]:
    """Return the tracked vLLM process and all of its current descendants."""
    seen: set[int] = set()
    pending = [root_pid]
    while pending:
        pid = pending.pop()
        if pid in seen:
            continue
        seen.add(pid)
        children_path = Path(f"/proc/{pid}/task/{pid}/children")
        try:
            pending.extend(int(value) for value in children_path.read_text().split())
        except (FileNotFoundError, PermissionError, ValueError):
            continue
    return seen


def _compute_process_memory() -> list[dict]:
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,process_name,used_memory",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        timeout=5,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "nvidia-smi failed")
    processes = []
    for line in result.stdout.strip().splitlines():
        parts = [part.strip() for part in line.split(",", 2)]
        if len(parts) != 3:
            continue
        try:
            processes.append({
                "pid": int(parts[0]),
                "process_name": parts[1],
                "memory_used_mib": float(parts[2]),
            })
        except ValueError:
            continue
    return processes


def _vllm_runtime_metrics(port: int) -> dict:
    wanted = {
        "vllm:kv_cache_usage_perc": "kv_cache_usage_pct",
        "vllm:num_requests_running": "requests_running",
        "vllm:num_requests_waiting": "requests_waiting",
    }
    metrics = {}
    try:
        with urlopen(f"http://127.0.0.1:{port}/metrics", timeout=3) as response:
            text = response.read().decode("utf-8", errors="replace")
        for line in text.splitlines():
            if not line or line.startswith("#"):
                continue
            metric_name = line.split("{", 1)[0].split(" ", 1)[0]
            output_name = wanted.get(metric_name)
            if not output_name:
                continue
            try:
                value = float(line.rsplit(" ", 1)[-1])
            except ValueError:
                continue
            metrics[output_name] = value * 100 if output_name == "kv_cache_usage_pct" else value
    except Exception as exc:
        metrics["error"] = str(exc)
    return metrics


def _latest_memory_profile(log_path: str | None) -> dict:
    """Parse absolute vLLM memory components from the latest successful startup."""
    if not log_path:
        return {}
    path = Path(log_path)
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError as exc:
        return {"error": str(exc), "log_path": str(path)}
    kv_pattern = re.compile(r"Available KV cache memory:\s*(-?[0-9.]+)\s*GiB")
    profile_pattern = re.compile(
        r"Actual usage is\s*([0-9.]+)\s*GiB for weight,\s*"
        r"([0-9.]+)\s*GiB for peak activation,\s*"
        r"(-?[0-9.]+)\s*GiB for non-torch memory, and\s*"
        r"([0-9.]+)\s*GiB for CUDAGraph memory.*?"
        r"Current kv cache memory in use is\s*([0-9.]+)\s*GiB"
    )
    token_pattern = re.compile(r"GPU KV cache size:\s*([0-9,]+)\s*tokens")
    result = {"log_path": str(path)}
    for line in reversed(lines):
        if "GPU KV cache size:" in line and "kv_cache_capacity_tokens" not in result:
            match = token_pattern.search(line)
            if match:
                result["kv_cache_capacity_tokens"] = int(match.group(1).replace(",", ""))
        if "Actual usage is" in line and "weight_gib" not in result:
            match = profile_pattern.search(line)
            if match:
                weight, activation, non_torch, cuda_graph, kv = map(float, match.groups())
                result.update({
                    "weight_gib": weight,
                    "peak_activation_gib": activation,
                    "non_torch_gib": non_torch,
                    "cuda_graph_gib": cuda_graph,
                    "kv_cache_capacity_gib": kv,
                })
        if "Available KV cache memory:" in line and "available_kv_cache_gib" not in result:
            match = kv_pattern.search(line)
            if match:
                result["available_kv_cache_gib"] = float(match.group(1))
        if "weight_gib" in result and "kv_cache_capacity_tokens" in result:
            break
    return result


@app.get("/state")
def get_state():
    return _read_json(_state_file_path(), None)


@app.get("/registry")
def get_registry():
    return _read_json(REGISTRY, {"profiles": {}})


@app.get("/profiles")
def list_profiles():
    """Return profiles with local availability."""
    registry = _read_json(REGISTRY, {"profiles": {}})
    result = []
    for name, profile in registry.get("profiles", {}).items():
        model_path = Path(profile.get("model", ""))
        model_ok = not model_path.is_absolute() or model_path.exists()
        missing = []
        if not model_ok:
            missing.append(str(model_path))
        for module in profile.get("lora_modules") or []:
            p = module.get("path")
            if p and not Path(p).exists():
                missing.append(p)
        result.append({
            "id": name,
            "model": profile.get("model"),
            "served_model_name": profile.get("served_model_name") or name,
            "available": not missing,
            "missing_paths": missing,
            "dtype": profile.get("dtype"),
            "quantization": profile.get("quantization"),
            "load_format": profile.get("load_format"),
            "max_model_len": profile.get("max_model_len"),
            "max_num_seqs": profile.get("max_num_seqs"),
            "gpu_memory_utilization": profile.get("gpu_memory_utilization"),
            "default_max_tokens": profile.get("default_max_tokens"),
            "enable_lora": bool(profile.get("enable_lora")),
            "lora_modules": profile.get("lora_modules") or [],
            "limit_mm_per_prompt": profile.get("limit_mm_per_prompt") or {},
            "notes": profile.get("notes", ""),
        })
    return result


class SwitchRequest(BaseModel):
    profile: str
    max_model_len: int | None = None
    max_num_seqs: int | None = None
    max_num_batched_tokens: int | None = None
    gpu_memory_utilization: float | None = None
    quantization: str | None = None
    load_format: str | None = None
    dtype: str | None = None
    default_max_tokens: int | None = None
    cuda_visible_devices: str | None = None
    wait_ready: bool = True
    ready_timeout: int = 600
    dry_run: bool = False


def _build_command(action: str, req: SwitchRequest) -> list[str]:
    cmd = [sys.executable, str(MANAGER), "--registry", str(REGISTRY), action, req.profile]
    option_map = {
        "max_model_len": "--max-model-len",
        "max_num_seqs": "--max-num-seqs",
        "max_num_batched_tokens": "--max-num-batched-tokens",
        "gpu_memory_utilization": "--gpu-memory-utilization",
        "quantization": "--quantization",
        "load_format": "--load-format",
        "dtype": "--dtype",
        "default_max_tokens": "--default-max-tokens",
        "cuda_visible_devices": "--cuda-visible-devices",
    }
    values = req.model_dump()
    for field, flag in option_map.items():
        val = values.get(field)
        if val is not None and val != "":
            cmd.extend([flag, str(val)])
    if req.wait_ready and action in ("start", "switch"):
        cmd.extend(["--wait-ready", "--ready-timeout", str(max(30, req.ready_timeout))])
    if req.dry_run:
        cmd.append("--dry-run")
    return cmd


@app.post("/switch")
def switch_profile(req: SwitchRequest):
    cmd = _build_command("switch", req)
    timeout = max(60, req.ready_timeout + 90) if req.wait_ready else 60
    completed = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if completed.returncode != 0:
        raise HTTPException(status_code=502, detail={
            "message": "vLLM switch failed",
            "command": cmd,
            "stdout": completed.stdout[-4000:],
            "stderr": completed.stderr[-4000:],
        })
    return {
        "accepted": True,
        "profile": req.profile,
        "state": _read_json(_state_file_path(), None),
        "stdout": completed.stdout[-4000:],
        "stderr": completed.stderr[-4000:],
    }


@app.post("/start")
def start_profile(req: SwitchRequest):
    cmd = _build_command("start", req)
    timeout = max(60, req.ready_timeout + 90) if req.wait_ready else 60
    completed = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if completed.returncode != 0:
        raise HTTPException(status_code=502, detail={
            "message": "vLLM start failed",
            "command": cmd,
            "stdout": completed.stdout[-4000:],
            "stderr": completed.stderr[-4000:],
        })
    return {
        "accepted": True,
        "profile": req.profile,
        "state": _read_json(_state_file_path(), None),
        "stdout": completed.stdout[-4000:],
        "stderr": completed.stderr[-4000:],
    }


class StopRequest(BaseModel):
    profile: str | None = None
    timeout: int = 60
    force: bool = False


class TokenizeRequest(BaseModel):
    messages: list[dict]
    tools: list[dict] | None = None
    add_generation_prompt: bool = True


@app.post("/tokenize-current")
def tokenize_current(req: TokenizeRequest):
    """Tokenize a chat request with the model currently managed by this instance."""
    state = _read_json(_state_file_path(), None)
    if not state or not state.get("pid") or not state.get("port"):
        raise HTTPException(status_code=503, detail="no active managed vLLM model")
    if not req.messages:
        raise HTTPException(status_code=422, detail="messages must not be empty")

    payload = {
        "messages": req.messages,
        "add_generation_prompt": req.add_generation_prompt,
    }
    if req.tools is not None:
        payload["tools"] = req.tools
    request = Request(
        f"http://127.0.0.1:{int(state['port'])}/tokenize",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=15) as response:
            tokenized = json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[-4000:]
        raise HTTPException(status_code=502, detail={
            "message": "current vLLM tokenizer rejected the request",
            "status": exc.code,
            "response": detail,
        }) from exc
    except (URLError, TimeoutError, ValueError) as exc:
        raise HTTPException(status_code=502, detail={
            "message": "current vLLM tokenizer unavailable",
            "error": str(exc),
        }) from exc

    return {
        "profile": state.get("profile"),
        "served_model_name": state.get("served_model_name"),
        "port": int(state["port"]),
        "prompt_tokens": int(tokenized.get("count") or 0),
        "max_model_len": int(
            tokenized.get("max_model_len")
            or state.get("max_model_len")
            or 0
        ),
    }


@app.post("/stop")
def stop_profile(req: StopRequest):
    cmd = [sys.executable, str(MANAGER), "--registry", str(REGISTRY), "stop"]
    if req.profile:
        cmd.append(req.profile)
    cmd.extend(["--timeout", str(req.timeout)])
    if req.force:
        cmd.append("--force")
    completed = subprocess.run(cmd, capture_output=True, text=True, timeout=max(60, req.timeout + 30))
    if completed.returncode != 0:
        raise HTTPException(status_code=502, detail={
            "message": "vLLM stop failed",
            "stdout": completed.stdout[-4000:],
            "stderr": completed.stderr[-4000:],
        })
    return {
        "accepted": True,
        "stdout": completed.stdout[-4000:],
        "stderr": completed.stderr[-4000:],
    }




@app.get("/gpu-stats")
def gpu_stats():
    """Return current GPU stats as JSON via nvidia-smi."""
    try:
        result = subprocess.run(
            ["nvidia-smi",
             "--query-gpu=index,timestamp,temperature.gpu,utilization.gpu,memory.used,memory.total,power.draw,clocks.sm,clocks.mem",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5)
        if result.returncode != 0:
            raise HTTPException(status_code=502, detail={"error": result.stderr.strip()})
        gpus = []
        for line in result.stdout.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 9:
                continue
            gpus.append({
                "index": int(parts[0]),
                "timestamp": parts[1],
                "temperature_c": float(parts[2]),
                "gpu_utilization_pct": float(parts[3]),
                "memory_used_mib": float(parts[4]),
                "memory_total_mib": float(parts[5]),
                "power_draw_w": float(parts[6]),
                "sm_clock_mhz": float(parts[7]),
                "mem_clock_mhz": float(parts[8]),
            })
        return {"gpus": gpus}
    except subprocess.TimeoutExpired:
        raise HTTPException(status_code=504, detail="nvidia-smi timeout")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/process-memory")
def process_memory():
    """Return GPU memory physically occupied by the managed vLLM process tree."""
    state = _read_json(_state_file_path(), None)
    if not state or not state.get("pid"):
        return {
            "sampled_at_utc": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
            "active": False,
            "process_memory_used_mib": 0.0,
            "processes": [],
            "vllm_metrics": {},
        }
    try:
        root_pid = int(state["pid"])
        tracked_pids = _process_tree(root_pid)
        all_processes = _compute_process_memory()
        model_processes = [item for item in all_processes if item["pid"] in tracked_pids]
        return {
            "sampled_at_utc": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
            "active": Path(f"/proc/{root_pid}").exists(),
            "profile": state.get("profile"),
            "served_model_name": state.get("served_model_name"),
            "root_pid": root_pid,
            "tracked_pids": sorted(tracked_pids),
            "process_memory_used_mib": round(sum(item["memory_used_mib"] for item in model_processes), 1),
            "process_memory_limit_mib": 10240.0,
            "process_memory_over_limit": sum(item["memory_used_mib"] for item in model_processes) > 10240.0,
            "processes": model_processes,
            "configured_gpu_memory_utilization": state.get("gpu_memory_utilization"),
            "configured_max_model_len": state.get("max_model_len"),
            "configured_max_num_seqs": state.get("max_num_seqs"),
            "configured_default_max_tokens": state.get("default_max_tokens"),
            "memory_profile": _latest_memory_profile(state.get("log_path")),
            "vllm_metrics": _vllm_runtime_metrics(int(state.get("port") or 8100)),
        }
    except subprocess.TimeoutExpired:
        raise HTTPException(status_code=504, detail="nvidia-smi timeout")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


if __name__ == "__main__":
    import uvicorn
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8500)
    args = parser.parse_args()
    uvicorn.run(app, host=args.host, port=args.port)
