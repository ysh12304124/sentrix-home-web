"""Apple Silicon unified-memory telemetry for PhotoBench.

Mac mini M4 has no discrete VRAM. ``ps`` RSS does not include Metal/GPU
pages, so per-process occupancy is ``phys_footprint`` from
``proc_pid_rusage``. That value already charges GPU allocations to the
process. The product total is the sum of related footprints. Adding an
Orin-style VmRSS-minus-Rss UMA term would count those pages twice.

Host used memory follows Activity Monitor's used set: active + wired +
pages occupied by the compressor. Inactive and speculative file cache
are not included.

GPU utilization prefers macmon's gpu_active_ratio, then the AGXAccelerator
"Device Utilization %" counter. CPU/GPU temperature and system power come
from ``macmon pipe -i 1000``, which reads Apple Silicon sensors without
root. If macmon is not installed those fields stay absent and are not
replaced with 0. ``macmon pipe -s 10`` is only a finite probe; the service
keeps one pipe open for the process lifetime.
"""
from __future__ import annotations

import atexit
import ctypes
import json
import platform
import re
import shutil
import subprocess
import threading
import time
from pathlib import Path
from urllib.parse import urlparse


SYSTEM_PROCESS_PORTS = {"8771", "8091", "8500", "8501", "8100", "8101", "6333"}
SYSTEM_PROCESS_KEYWORDS = (
    "photobench", "benchmark_orchestrator.py", "sentrix", "vllm",
    "llama-server", "ollama", "qdrant", "mlx",
)


def is_apple_silicon() -> bool:
    return platform.system() == "Darwin" and platform.machine() in {"arm64", "aarch64"}


def parse_vm_stat(text: str, page_size: int, total_bytes: int | None = None) -> dict | None:
    pages: dict[str, int] = {}
    for line in text.splitlines():
        name, sep, rest = line.partition(":")
        if not sep:
            continue
        token = rest.strip().rstrip(".").split()
        if not token:
            continue
        try:
            pages[name.strip().lower()] = int(token[0])
        except ValueError:
            continue
    needed = ("pages active", "pages wired down", "pages occupied by compressor")
    if any(key not in pages for key in needed) or page_size <= 0:
        return None
    used_bytes = (
        pages["pages active"] + pages["pages wired down"] + pages["pages occupied by compressor"]
    ) * page_size
    data = {
        "system_memory_used_mib": round(used_bytes / 1048576, 2),
        "system_memory_scope": "apple_app_wired_compressed",
    }
    if total_bytes and total_bytes > 0:
        data["system_memory_total_mib"] = round(total_bytes / 1048576, 2)
        data["system_memory_available_mib"] = round(max(0, total_bytes - used_bytes) / 1048576, 2)
    return data


def parse_macmon_sample(payload: dict) -> dict:
    """Map one ``macmon pipe`` JSON object onto PhotoBench sample fields."""
    if not isinstance(payload, dict):
        return {}
    data: dict = {}
    temp = payload.get("temp") if isinstance(payload.get("temp"), dict) else {}
    cpu_temp = temp.get("cpu_temp_avg")
    gpu_temp = temp.get("gpu_temp_avg")
    if isinstance(cpu_temp, (int, float)):
        data["cpu_temperature_c"] = round(float(cpu_temp), 3)
    if isinstance(gpu_temp, (int, float)):
        data["temperature_c"] = round(float(gpu_temp), 3)
    sys_power = payload.get("sys_power")
    if isinstance(sys_power, (int, float)):
        data["power_draw_w"] = round(float(sys_power), 3)
        data["power_scope"] = "macmon_sys_power"
    for source_key, target_key in (("cpu_power", "cpu_power_w"), ("gpu_power", "gpu_power_w")):
        value = payload.get(source_key)
        if isinstance(value, (int, float)):
            data[target_key] = round(float(value), 3)
    ratio = payload.get("gpu_active_ratio")
    if isinstance(ratio, (int, float)):
        data["gpu_utilization_pct"] = round(float(ratio) * 100, 3)
    return data


class _MacmonStream:
    """One long-lived ``macmon pipe -i 1000``. A finite ``-s`` run would stop."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._latest: dict = {}
        self._proc: subprocess.Popen | None = None
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        binary = shutil.which("macmon") or "/opt/homebrew/bin/macmon"
        if not Path(binary).is_file() or self._proc is not None:
            return
        self._proc = subprocess.Popen(
            [binary, "pipe", "-i", "1000"],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        self._thread = threading.Thread(target=self._read, name="macmon-pipe", daemon=True)
        self._thread.start()
        atexit.register(self.stop)

    def stop(self) -> None:
        proc = self._proc
        if proc is None or proc.poll() is not None:
            return
        proc.terminate()
        try:
            proc.wait(timeout=2)
        except subprocess.TimeoutExpired:
            proc.kill()

    def latest(self) -> dict:
        if self._proc is None:
            self.start()
        with self._lock:
            return dict(self._latest)

    def _read(self) -> None:
        proc = self._proc
        if proc is None or proc.stdout is None:
            return
        for line in proc.stdout:
            line = line.strip()
            if not line.startswith("{"):
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            parsed = parse_macmon_sample(payload)
            if not parsed:
                continue
            with self._lock:
                self._latest = parsed


_MACMON = _MacmonStream()


def parse_agx_performance(text: str) -> dict:
    """Read integrated-GPU counters. Missing counters stay absent."""
    data: dict = {}
    util = re.search(r'Device Utilization %"=(\d+(?:\.\d+)?)', text)
    if util:
        data["gpu_utilization_pct"] = round(float(util.group(1)), 3)
    allocated = re.search(r'"Alloc system memory"=(\d+)', text)
    if allocated:
        data["gpu_allocated_unified_memory_mib"] = round(int(allocated.group(1)) / 1048576, 2)
    in_use = re.search(r'"In use system memory"=(\d+)', text)
    if in_use:
        data["gpu_in_use_unified_memory_mib"] = round(int(in_use.group(1)) / 1048576, 2)
    return data


class _RUsageInfoV0(ctypes.Structure):
    _fields_ = [
        ("ri_uuid", ctypes.c_uint8 * 16),
        ("ri_user_time", ctypes.c_uint64),
        ("ri_system_time", ctypes.c_uint64),
        ("ri_pkg_idle_wkups", ctypes.c_uint64),
        ("ri_interrupt_wkups", ctypes.c_uint64),
        ("ri_pageins", ctypes.c_uint64),
        ("ri_wired_size", ctypes.c_uint64),
        ("ri_resident_size", ctypes.c_uint64),
        ("ri_phys_footprint", ctypes.c_uint64),
        ("ri_proc_start_abstime", ctypes.c_uint64),
        ("ri_proc_exit_abstime", ctypes.c_uint64),
    ]


_LIBPROC = None
_LIBPROC_LOCK = threading.Lock()


def _libproc():
    global _LIBPROC
    if _LIBPROC is None:
        library = ctypes.CDLL("/usr/lib/libproc.dylib")
        library.proc_pid_rusage.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.POINTER(_RUsageInfoV0)]
        library.proc_pid_rusage.restype = ctypes.c_int
        _LIBPROC = library
    return _LIBPROC


def _phys_footprint_bytes(pid: int) -> int | None:
    """Activity Monitor footprint, including GPU pages charged to this process.

    Flavor 0 is rusage_info_v0, which already contains phys_footprint. A newer
    flavor writes a larger struct and would overflow this buffer. The struct
    and library handle stay at module scope: redefining them per call crashes
    the second invocation on this macOS.
    """
    try:
        info = _RUsageInfoV0()
        with _LIBPROC_LOCK:
            rc = _libproc().proc_pid_rusage(int(pid), 0, ctypes.byref(info))
            footprint = int(info.ri_phys_footprint)
        if rc != 0 or footprint <= 0:
            return None
        return footprint
    except (OSError, AttributeError, ValueError):
        return None


def _run_text(command: list[str], timeout: float = 3) -> str:
    try:
        completed = subprocess.run(command, capture_output=True, text=True, timeout=timeout, check=False)
    except (OSError, subprocess.SubprocessError):
        return ""
    return completed.stdout or ""


def _page_size() -> int:
    text = _run_text(["sysctl", "-n", "hw.pagesize"], timeout=2).strip()
    try:
        return int(text)
    except ValueError:
        return 16384


def _total_bytes() -> int | None:
    text = _run_text(["sysctl", "-n", "hw.memsize"], timeout=2).strip()
    try:
        return int(text)
    except ValueError:
        return None


def _process_rows() -> list[dict]:
    text = _run_text(["ps", "-ax", "-o", "pid=", "-o", "command="], timeout=3)
    rows = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        pid_text, _, command = stripped.partition(" ")
        if not pid_text.isdigit() or not command:
            continue
        rows.append({
            "pid": int(pid_text),
            "command": command,
            "identity": command.lower()[:1000],
            "process_name": Path(command.split(" ", 1)[0]).name,
            "listening_ports": [],
        })
    return rows


def _listening_ports(ports: set[str]) -> dict[int, set[str]]:
    command = ["lsof", "-nP", "-sTCP:LISTEN"]
    for port in sorted(ports):
        command.extend(["-iTCP:" + port])
    text = _run_text(command, timeout=4)
    found: dict[int, set[str]] = {}
    for line in text.splitlines()[1:]:
        parts = line.split()
        if len(parts) < 9 or not parts[1].isdigit():
            continue
        match = re.search(r":(\d+)$", parts[-1])
        if not match:
            continue
        found.setdefault(int(parts[1]), set()).add(match.group(1))
    return found


def _is_related(row: dict) -> bool:
    ports = {str(port) for port in row.get("listening_ports") or []}
    if ports & SYSTEM_PROCESS_PORTS:
        return True
    identity = str(row.get("identity") or "")
    return any(keyword in identity for keyword in SYSTEM_PROCESS_KEYWORDS)


def _is_model_process(row: dict, endpoint_port: str) -> bool:
    identity = str(row.get("identity") or "")
    ports = {str(port) for port in row.get("listening_ports") or []}
    if endpoint_port and endpoint_port in ports and ("llama-server" in identity or "mlx" in identity or "vllm" in identity):
        return True
    if endpoint_port and f"--port {endpoint_port}" in identity and "llama-server" in identity:
        return True
    return False


class LocalAppleUnifiedTelemetryProvider:
    source = "apple_unified"

    def __init__(self, endpoint_url: str = ""):
        parsed = urlparse(endpoint_url or "")
        self.port = str(parsed.port or "")
        self._last_memory: dict = {"status": "unavailable", "reason": "not_sampled"}
        self._device_sample: dict = {}
        self._last_device_probe = 0.0
        self._ports_by_pid: dict[int, set[str]] = {}
        self._last_port_probe = 0.0
        self._system_baseline_mib: float | None = None

    def _device(self) -> dict:
        if self._device_sample and time.monotonic() - self._last_device_probe < 1.0:
            return self._device_sample
        self._last_device_probe = time.monotonic()
        sample = parse_agx_performance(_run_text(["ioreg", "-r", "-c", "AGXAccelerator", "-l"], timeout=3))
        sample.update(_MACMON.latest())
        memory = parse_vm_stat(_run_text(["vm_stat"], timeout=3), _page_size(), _total_bytes())
        if memory:
            sample.update(memory)
        self._device_sample = sample
        return sample

    def _ports(self) -> dict[int, set[str]]:
        if time.monotonic() - self._last_port_probe < 2.0:
            return self._ports_by_pid
        self._last_port_probe = time.monotonic()
        self._ports_by_pid = _listening_ports(SYSTEM_PROCESS_PORTS)
        return self._ports_by_pid

    def _augment(self) -> dict:
        device = self._device()
        ports = self._ports()
        rows = _process_rows()
        for row in rows:
            row["listening_ports"] = sorted(ports.get(row["pid"], set()), key=int)
        model_rows = [row for row in rows if _is_model_process(row, self.port)]
        related_rows = [row for row in rows if _is_related(row)]
        model_pids = {row["pid"] for row in model_rows}
        sentrix_rows = [row for row in related_rows if row["pid"] not in model_pids]
        for row in sentrix_rows + model_rows:
            footprint = _phys_footprint_bytes(row["pid"])
            row["footprint_mib"] = None if footprint is None else round(footprint / 1048576, 2)
        data: dict = {}
        if len(model_rows) == 1 and model_rows[0].get("footprint_mib") is not None:
            model = model_rows[0]
            data["root_pid"] = model["pid"]
            data["model_process_system_memory_used_mib"] = model["footprint_mib"]
            data["model_process_system_memory_scope"] = "apple_phys_footprint"
            data["process_memory_used_mib"] = model["footprint_mib"]
            data["process_memory_scope"] = "apple_phys_footprint"
            data["model_system_processes"] = [{
                "pid": model["pid"],
                "process_name": model["process_name"],
                "footprint_mib": model["footprint_mib"],
                "listening_ports": model["listening_ports"],
            }]
        sentrix_complete = bool(sentrix_rows) and all(row.get("footprint_mib") is not None for row in sentrix_rows)
        model_complete = bool(model_rows) and all(row.get("footprint_mib") is not None for row in model_rows)
        if sentrix_complete:
            sentrix = round(sum(row["footprint_mib"] for row in sentrix_rows), 2)
            data["sentrix_stack_pss_mib"] = sentrix
            data["sentrix_stack_pss_scope"] = "apple_phys_footprint_excluding_model"
        if sentrix_complete and model_complete and len(model_rows) == 1:
            product = round(sum(row["footprint_mib"] for row in sentrix_rows + model_rows), 2)
            data["product_stack_memory_mib"] = product
            data["benchmark_process_memory_used_mib"] = product
            data["benchmark_process_memory_scope"] = "apple_phys_footprint_related"
            data["benchmark_processes"] = [{
                "pid": row["pid"],
                "process_name": row["process_name"],
                "footprint_mib": row["footprint_mib"],
                "listening_ports": row["listening_ports"],
            } for row in (sentrix_rows + model_rows)[:50]]
        used = device.get("system_memory_used_mib")
        if self._system_baseline_mib is None and isinstance(used, (int, float)):
            self._system_baseline_mib = float(used)
        if self._system_baseline_mib is not None and isinstance(used, (int, float)):
            data["system_memory_baseline_mib"] = round(self._system_baseline_mib, 2)
            data["system_memory_delta_mib"] = round(float(used) - self._system_baseline_mib, 2)
        data["memory_unit"] = "apple_phys_footprint_mib"
        data["memory_profile"] = {"method": "apple_phys_footprint_v1"}
        return data

    def gpu_stats(self) -> dict:
        device = self._device()
        data = self._augment()
        data.update({key: value for key, value in device.items() if key not in data})
        self._last_memory = {"status": "available", "source": self.source, "data": data}
        gpu = {"index": 0, "memory_unit": "apple_phys_footprint_mib"}
        for key in ("gpu_utilization_pct", "gpu_allocated_unified_memory_mib", "gpu_in_use_unified_memory_mib",
                    "temperature_c", "cpu_temperature_c", "power_draw_w", "power_scope",
                    "cpu_power_w", "gpu_power_w"):
            if key in device:
                gpu[key] = device[key]
        return {"status": "available", "source": self.source, "data": {"gpus": [gpu]}}

    def process_memory(self) -> dict:
        if self._last_memory.get("status") != "available":
            self.gpu_stats()
        return self._last_memory

    def system_memory(self) -> dict:
        data = self._device()
        if "system_memory_used_mib" not in data:
            return {"status": "unavailable", "source": "vm_stat", "reason": "vm_stat_unavailable"}
        keys = ("system_memory_used_mib", "system_memory_total_mib", "system_memory_available_mib", "system_memory_scope")
        return {"status": "available", "source": "vm_stat", "data": {key: data[key] for key in keys if key in data}}

    def kv_cache(self) -> dict:
        return {"status": "not_applicable", "source": "apple_unified", "reason": "unified_memory_has_no_discrete_kv_pool"}
