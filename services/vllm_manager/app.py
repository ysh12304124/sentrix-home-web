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
import subprocess
import sys
from pathlib import Path

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


if __name__ == "__main__":
    import uvicorn
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8500)
    args = parser.parse_args()
    uvicorn.run(app, host=args.host, port=args.port)
