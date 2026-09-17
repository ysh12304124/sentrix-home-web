"""Local Jetson llama.cpp telemetry; UMA physical memory is not GPU VRAM."""
from __future__ import annotations

import os
from pathlib import Path
import re
import subprocess
import threading
import time
from urllib.parse import urlparse

import httpx


def _configured_set(env_name: str, defaults: tuple[str, ...]) -> set[str]:
    raw = os.environ.get(env_name)
    if not raw:
        return set(defaults)
    return {item.strip().lower() for item in raw.split(",") if item.strip()}


SYSTEM_PROCESS_PORTS = _configured_set(
    "PHOTOBENCH_SYSTEM_PROCESS_PORTS",
    ("8771", "8091", "8500", "8501", "8100", "8101", "6333"),
)
SYSTEM_PROCESS_KEYWORDS = _configured_set(
    "PHOTOBENCH_SYSTEM_PROCESS_KEYWORDS",
    (
        "photobench", "benchmark_orchestrator.py", "sentrix", "vllm",
        "llama-server", "ollama", "qdrant",
    ),
)


def _process_cmdline(pid: int) -> str:
    try:
        raw = Path(f"/proc/{pid}/cmdline").read_bytes()
        text = raw.replace(bytes([0]), bytes([32])).decode("utf-8", "replace").strip()
        if text:
            return text
    except (OSError, ValueError):
        pass
    try:
        return Path(f"/proc/{pid}/comm").read_text(encoding="utf-8", errors="replace").strip()
    except OSError:
        return ""


def _process_rss_mib(pid: int) -> float | None:
    try:
        for line in Path(f"/proc/{pid}/status").read_text(encoding="ascii").splitlines():
            if line.startswith("VmRSS:"):
                return round(float(line.split()[1]) / 1024.0, 2)
    except (OSError, ValueError, IndexError):
        return None
    return None


def _listening_ports_by_pid(target_ports: set[str] | None = None) -> dict[int, set[str]]:
    target_ports = {str(port) for port in (target_ports or set())}
    socket_ports: dict[str, str] = {}
    for table in ("/proc/net/tcp", "/proc/net/tcp6"):
        try:
            lines = Path(table).read_text(encoding="ascii").splitlines()[1:]
        except OSError:
            continue
        for line in lines:
            parts = line.split()
            if len(parts) < 10 or parts[3] != "0A":
                continue
            try:
                port = str(int(parts[1].rsplit(":", 1)[1], 16))
            except (ValueError, IndexError):
                continue
            if target_ports and port not in target_ports:
                continue
            socket_ports[parts[9]] = port
    if not socket_ports:
        return {}
    result: dict[int, set[str]] = {}
    for proc in Path("/proc").iterdir():
        if not proc.name.isdecimal():
            continue
        fd_dir = proc / "fd"
        try:
            fds = list(fd_dir.iterdir())
        except OSError:
            continue
        for fd in fds:
            try:
                target = os.readlink(fd)
            except OSError:
                continue
            match = re.fullmatch(r"socket:\[(\d+)\]", target)
            if match and match.group(1) in socket_ports:
                result.setdefault(int(proc.name), set()).add(socket_ports[match.group(1)])
    return result


def _host_process_rows() -> list[dict]:
    ports_by_pid = _listening_ports_by_pid(SYSTEM_PROCESS_PORTS)
    rows = []
    for proc in Path("/proc").iterdir():
        if not proc.name.isdecimal():
            continue
        pid = int(proc.name)
        rss_mib = _process_rss_mib(pid)
        cmdline = _process_cmdline(pid)
        if rss_mib is None and not cmdline:
            continue
        rows.append({
            "pid": pid,
            "process_name": Path(cmdline.split(" ", 1)[0]).name if cmdline else "",
            "rss_mib": rss_mib,
            "listening_ports": sorted(ports_by_pid.get(pid, set()), key=lambda value: int(value)),
            "identity": cmdline.lower()[:1000],
        })
    return rows


def _is_system_related_process(row: dict) -> bool:
    identity = str(row.get("identity") or "").lower()
    ports = {str(port) for port in row.get("listening_ports") or []}
    if ports & SYSTEM_PROCESS_PORTS:
        return True
    return any(keyword and keyword in identity for keyword in SYSTEM_PROCESS_KEYWORDS)


def _sum_mib(rows: list[dict], key: str) -> float | None:
    values = [float(row[key]) for row in rows if isinstance(row.get(key), (int, float))]
    return round(sum(values), 2) if values else None


def _augment_local_process_memory(data: dict | None, *, model_pid: int | None, endpoint_port: str) -> dict:
    data = dict(data or {})
    try:
        rows = _host_process_rows()
    except Exception:
        rows = []
    model_rows = [
        row for row in rows
        if row["pid"] == model_pid or "llama-server" in str(row.get("identity") or "")
        or str(endpoint_port) in str(row.get("identity") or "")
    ]
    system_rows = [row for row in rows if _is_system_related_process(row)]
    model_rss = _sum_mib(model_rows, "rss_mib")
    system_rss = _sum_mib(system_rows, "rss_mib")
    if model_rss is not None:
        data["model_process_system_memory_used_mib"] = model_rss
        data["model_process_system_memory_scope"] = "host_process_rss"
        data["model_system_processes"] = [
            {key: row.get(key) for key in ("pid", "process_name", "rss_mib", "listening_ports")}
            for row in model_rows[:20]
        ]
    if system_rss is not None:
        data["benchmark_process_memory_used_mib"] = system_rss
        data["benchmark_process_memory_scope"] = "photobench_related_host_process_rss"
        data["benchmark_processes"] = [
            {key: row.get(key) for key in ("pid", "process_name", "rss_mib", "listening_ports")}
            for row in system_rows[:50]
        ]
    return data


def is_jetson_host() -> bool:
    try:
        return "jetson" in Path("/proc/device-tree/model").read_bytes().replace(b"\0", b"").decode("utf-8", "replace").lower()
    except OSError:
        return False


JETSON_SAMPLE_INTERVAL_SECONDS = max(
    0.1, float(os.environ.get("PHOTOBENCH_JETSON_SAMPLE_INTERVAL_SECONDS", "0.5"))
)


def _meminfo_mib() -> dict:
    values: dict[str, float] = {}
    try:
        for line in Path("/proc/meminfo").read_text(encoding="ascii").splitlines():
            name, _, rest = line.partition(":")
            if name in {"MemTotal", "MemAvailable", "MemFree"}:
                values[name] = float(rest.split()[0]) / 1024.0
    except (OSError, ValueError, IndexError):
        return {}
    total = values.get("MemTotal")
    available = values.get("MemAvailable", values.get("MemFree"))
    if total is None or available is None:
        return {}
    return {
        "system_memory_used_mib": round(max(0.0, total - available), 3),
        "system_memory_total_mib": round(total, 3),
        "system_memory_available_mib": round(max(0.0, available), 3),
        "system_memory_scope": "host_all_processes",
    }


def _parse_tegrastats_line(line: str) -> dict:
    sample = {}
    for key, pattern, scale in (
        ("system_memory_used_mib", r"RAM\s+(\d+)/(\d+)MB", 1),
        ("gpu_utilization_pct", r"GR3D_FREQ\s+(\d+)%", 1),
        ("power_draw_w", r"VDD_GPU_SOC\s+(\d+)mW", 0.001),
        ("temperature_c", r"(?:gpu|tj)@([\d.]+)C", 1),
    ):
        match = re.search(pattern, line)
        if not match:
            continue
        sample[key] = round(float(match.group(1)) * scale, 3)
        if key == "system_memory_used_mib":
            sample["system_memory_total_mib"] = round(float(match.group(2)) * scale, 3)
            sample["system_memory_available_mib"] = round(
                max(0.0, sample["system_memory_total_mib"] - sample[key]), 3
            )
            sample["system_memory_scope"] = "host_all_processes"
    return sample


class _TegrastatsStream:
    """Keep one tegrastats process so 0.5s sampling does not spawn a child every tick."""

    def __init__(self, interval_ms: int):
        self.interval_ms = max(100, int(interval_ms))
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._proc: subprocess.Popen | None = None
        self._thread: threading.Thread | None = None
        self._latest = ""

    def latest_line(self) -> str:
        self._ensure()
        with self._lock:
            return self._latest

    def _ensure(self) -> None:
        with self._lock:
            if self._proc is not None and self._proc.poll() is None:
                return
            self._stop.clear()
            try:
                self._proc = subprocess.Popen(
                    ["tegrastats", "--interval", str(self.interval_ms)],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.DEVNULL,
                    text=True,
                    bufsize=1,
                )
            except OSError:
                self._proc = None
                return
            self._thread = threading.Thread(target=self._read, daemon=True, name="tegrastats-reader")
            self._thread.start()

    def _read(self) -> None:
        proc = self._proc
        if proc is None or proc.stdout is None:
            return
        for line in proc.stdout:
            if self._stop.is_set():
                break
            text = line.strip()
            if text:
                with self._lock:
                    self._latest = text


_TEGRASTATS = _TegrastatsStream(int(JETSON_SAMPLE_INTERVAL_SECONDS * 1000))


class LocalJetsonLlamaCppTelemetryProvider:
    """Sample a verified local llama-server process, never an arbitrary remote PID.

    Device stats and PSS follow the same 0.5s GpuSampler cadence as 153 NVML.
    tegrastats is kept as a single long-lived process. Missing PSS/KV values stay
    null; no whole-system RAM is mislabeled as GPU VRAM.
    """

    def __init__(self, endpoint_url: str, *, pid_file: str | None = None, pss_interval: float | None = None):
        parsed = urlparse(endpoint_url)
        if parsed.scheme not in {"http", "https"} or not parsed.port:
            raise ValueError("a local llama.cpp endpoint with an explicit port is required")
        self.endpoint = f"{parsed.scheme}://{parsed.netloc}"
        self.port = parsed.port
        self.pid_file = Path(pid_file or os.environ.get(
            "PHOTOBENCH_LLAMA_PID_FILE",
            "/home/orin/VLM/gemma4/results/orin/server_photobench_8100.pid",
        ))
        self.pss_interval = JETSON_SAMPLE_INTERVAL_SECONDS if pss_interval is None else max(0.0, float(pss_interval))
        self.device_interval = self.pss_interval
        self._last_probe = 0.0
        self._last_device_probe = 0.0
        self._device_sample: dict = {}
        self._last_pid = None
        self._last_memory = {"status": "unavailable", "reason": "not_sampled"}

    def _pid_from_proc(self, pid: int) -> int:
        args = Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0")
        if not args or Path(os.fsdecode(args[0])).name != "llama-server":
            raise ValueError("PID does not belong to llama-server")
        command = [os.fsdecode(arg) for arg in args if arg]
        if "--port" not in command or command[command.index("--port") + 1] != str(self.port):
            raise ValueError("PID is not serving the selected endpoint port")
        return pid

    def _pid_from_port(self) -> int:
        for entry in Path("/proc").iterdir():
            if not entry.name.isdecimal():
                continue
            try:
                return self._pid_from_proc(int(entry.name))
            except (ValueError, FileNotFoundError, PermissionError, OSError):
                continue
        raise ValueError(f"no llama-server is listening on port {self.port}")

    def _pid(self) -> int:
        try:
            raw = self.pid_file.read_text(encoding="ascii").strip()
            if raw.isdecimal():
                return self._pid_from_proc(int(raw))
        except (OSError, ValueError):
            pass
        pid = self._pid_from_port()
        try:
            self.pid_file.parent.mkdir(parents=True, exist_ok=True)
            self.pid_file.write_text(f"{pid}\n", encoding="ascii")
        except OSError:
            pass
        return pid

    @staticmethod
    def _pss_mib(pid: int) -> float:
        with open(f"/proc/{pid}/smaps_rollup", encoding="ascii") as file:
            for line in file:
                if line.startswith("Pss:"):
                    return round(int(line.split()[1]) / 1024, 2)
        raise ValueError("Pss not found in smaps_rollup")

    def _metrics(self) -> dict:
        data = {}
        try:
            response = httpx.get(f"{self.endpoint}/metrics", timeout=2)
            response.raise_for_status()
            values = {}
            for line in response.text.splitlines():
                if line.startswith("llamacpp:kv_cache_") and "{" not in line:
                    name, _, value = line.partition(" ")
                    try:
                        values[name] = float(value.strip().split()[0])
                    except (ValueError, IndexError):
                        continue
            ratio = values.get("llamacpp:kv_cache_usage_ratio")
            if ratio is not None and 0 <= ratio <= 1:
                data["vllm_metrics"] = {"kv_cache_usage_pct": round(ratio * 100, 2)}
            tokens = values.get("llamacpp:kv_cache_tokens")
            if tokens is not None and tokens >= 0:
                data["kv_cache_used_tokens"] = int(tokens)
        except (httpx.HTTPError, ValueError):
            pass
        return data

    def _device(self) -> dict:
        if time.monotonic() - self._last_device_probe < self.device_interval and self._device_sample:
            return self._device_sample
        self._last_device_probe = time.monotonic()
        sample = _meminfo_mib()
        line = _TEGRASTATS.latest_line()
        if line:
            parsed = _parse_tegrastats_line(line)
            sample.update(parsed)
        self._device_sample = sample
        return self._device_sample

    def gpu_stats(self) -> dict:
        data = dict(self._last_memory.get("data") or {})
        data.update(self._metrics())
        data["memory_unit"] = "process_pss_uma_mib"
        data["memory_scope"] = "model_process_pss"
        data["memory_profile"] = {"method": "orin_process_pss_uma_v1"}
        try:
            pid = self._pid()
            if pid != self._last_pid:
                self._last_pid = pid
                self._last_probe = 0.0
                data.pop("process_memory_used_mib", None)
            data["root_pid"] = pid
            if time.monotonic() - self._last_probe >= self.pss_interval:
                self._last_probe = time.monotonic()
                data["process_memory_used_mib"] = self._pss_mib(pid)
                data["process_memory_scope"] = "llama_server_pss"
                data["pss_sampled_at_monotonic"] = time.monotonic()
                data.pop("pss_error", None)
        except (OSError, ValueError, IndexError) as exc:
            data.pop("process_memory_used_mib", None)
            data["pss_error"] = str(exc)
        self._last_memory = {"status": "available", "source": "jetson_local_pss", "data": data}
        return {"status": "available", "source": "jetson_local_pss", "data": {
            "gpus": [{"index": 0, "memory_unit": "process_pss_uma_mib", "memory_scope": "model_process_pss", **self._device()}],
        }}

    def process_memory(self) -> dict:
        if self._last_memory.get("status") == "available":
            data = self._last_memory.get("data") or {}
            pid = int(data["root_pid"]) if str(data.get("root_pid") or "").isdigit() else None
            return {
                **self._last_memory,
                "data": _augment_local_process_memory(data, model_pid=pid, endpoint_port=str(self.port)),
            }
        return self._last_memory

    def system_memory(self) -> dict:
        data = self._device_sample.copy()
        keys = {"system_memory_used_mib", "system_memory_total_mib", "system_memory_available_mib", "system_memory_scope"}
        data = {key: value for key, value in data.items() if key in keys}
        if "system_memory_used_mib" not in data:
            return {"status": "unavailable", "source": "tegrastats", "reason": "ram_sample_unavailable"}
        return {"status": "available", "source": "tegrastats", "data": data}

    def kv_cache(self) -> dict:
        data = self._last_memory.get("data") or {}
        if "kv_cache_used_tokens" not in data and "vllm_metrics" not in data:
            return {"status": "unavailable", "source": "llamacpp_metrics", "reason": "kv_metrics_not_exposed"}
        return {"status": "available", "source": "llamacpp_metrics", "data": {
            "used_tokens": data.get("kv_cache_used_tokens"),
            "usage_pct": (data.get("vllm_metrics") or {}).get("kv_cache_usage_pct"),
            "used_bytes": None,
        }}
