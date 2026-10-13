"""Hardware snapshot — QA run 硬件参数采集（GPU / CPU / 内存）。

只读采集运行服务器（153）的 GPU 与系统资源状态，供
run_qa_benchmark 写入 run meta、QA Dashboard 展示。
全部容错：任何一项不可用返回空值，不影响主流程。
"""

from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess
from datetime import datetime


def _run(cmd: list[str], timeout_s: float = 8.0) -> str:
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s)
        return out.stdout if out.returncode == 0 else ""
    except Exception:
        return ""


def collect_gpu() -> list[dict]:
    """nvidia-smi 查询：index / name / VRAM 用量 / 利用率 / 温度。"""
    if not shutil.which("nvidia-smi"):
        return []
    out = _run([
        "nvidia-smi",
        "--query-gpu=index,name,memory.used,memory.total,utilization.gpu,temperature.gpu",
        "--format=csv,noheader,nounits",
    ])
    gpus = []
    for line in out.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 6:
            continue
        try:
            gpus.append({
                "index": int(parts[0]),
                "name": parts[1],
                "memory_used_mib": int(parts[2]),
                "memory_total_mib": int(parts[3]),
                "utilization_percent": int(parts[4]),
                "temperature_c": int(parts[5]),
            })
        except Exception:
            continue
    return gpus


_CPU_HWMON_NAMES = ("k10temp", "coretemp", "cpu_thermal", "cpu-thermal")
_CPU_TEMP_LABELS = ("tctl", "tdie", "package id 0", "cpu")


def read_cpu_temperature_c(root: Path | None = None) -> float | None:
    """CPU package temperature in Celsius, or None when the sensor is absent.

    153 is AMD k10temp Tctl. Intel boards expose coretemp. Orin may expose
    cpu-thermal; tegrastats cpu@ remains the preferred source there.
    NVMe, NIC and amdgpu sensors are ignored.
    """
    base = root or Path("/sys/class/hwmon")
    if not base.is_dir():
        return None
    for hw in sorted(base.iterdir()):
        try:
            name = (hw / "name").read_text(encoding="ascii").strip().lower()
        except OSError:
            continue
        if name not in _CPU_HWMON_NAMES:
            continue
        chosen = None
        chosen_rank = 99
        for sensor in sorted(hw.glob("temp*_input")):
            label = ""
            label_path = hw / sensor.name.replace("_input", "_label")
            try:
                label = label_path.read_text(encoding="ascii").strip().lower()
            except OSError:
                pass
            try:
                celsius = round(float(sensor.read_text(encoding="ascii").strip()) / 1000.0, 3)
            except (OSError, ValueError):
                continue
            rank = _CPU_TEMP_LABELS.index(label) if label in _CPU_TEMP_LABELS else 50
            if chosen is None or rank < chosen_rank:
                chosen = celsius
                chosen_rank = rank
        if chosen is not None:
            return chosen
    return None


def collect_cpu() -> dict:
    cores = None
    try:
        cores = max(1, os.cpu_count() or 1)
    except Exception:
        pass
    load_avg = None
    try:
        with open("/proc/loadavg", encoding="utf-8") as fh:
            parts = fh.read().split()
            if len(parts) >= 3:
                load_avg = [round(float(x), 2) for x in parts[:3]]
    except Exception:
        pass
    return {"cores": cores, "load_avg": load_avg, "temperature_c": read_cpu_temperature_c()}


def collect_memory() -> dict:
    try:
        data = {}
        with open("/proc/meminfo", encoding="utf-8") as fh:
            for line in fh:
                key, _, value = line.partition(":")
                data[key.strip()] = int(value.strip().split()[0])  # kB
        total_kb = data.get("MemTotal")
        avail_kb = data.get("MemAvailable")
        return {
            "total_mib": round(total_kb / 1024) if total_kb else None,
            "available_mib": round(avail_kb / 1024) if avail_kb else None,
        }
    except Exception:
        return {}


def collect_hardware() -> dict:
    return {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "gpu": collect_gpu(),
        "cpu": collect_cpu(),
        "memory": collect_memory(),
    }
