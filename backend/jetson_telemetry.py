"""Local Jetson llama.cpp telemetry; UMA physical memory is not GPU VRAM."""
from __future__ import annotations

import os
from pathlib import Path
import re
import subprocess
import time
from urllib.parse import urlparse

import httpx


def is_jetson_host() -> bool:
    try:
        return "jetson" in Path("/proc/device-tree/model").read_bytes().replace(b"\0", b"").decode("utf-8", "replace").lower()
    except OSError:
        return False


class LocalJetsonLlamaCppTelemetryProvider:
    """Sample a verified local llama-server process, never an arbitrary remote PID.

    PSS is sampled less frequently than HTTP metrics. Missing PSS/KV values stay
    null; no whole-system RAM or per-token formula is mislabeled as GPU VRAM.
    """

    def __init__(self, endpoint_url: str, *, pid_file: str | None = None, pss_interval: float = 5):
        parsed = urlparse(endpoint_url)
        if parsed.scheme not in {"http", "https"} or not parsed.port:
            raise ValueError("a local llama.cpp endpoint with an explicit port is required")
        self.endpoint = f"{parsed.scheme}://{parsed.netloc}"
        self.port = parsed.port
        self.pid_file = Path(pid_file or os.environ.get(
            "PHOTOBENCH_LLAMA_PID_FILE",
            "/home/orin/VLM/gemma4/results/orin/server_photobench_8100.pid",
        ))
        self.pss_interval = pss_interval
        self._last_probe = 0.0
        self._last_device_probe = 0.0
        self._device_sample: dict = {}
        self._last_pid = None
        self._last_memory = {"status": "unavailable", "reason": "not_sampled"}

    def _pid(self) -> int:
        raw = self.pid_file.read_text(encoding="ascii").strip()
        if not raw.isdecimal():
            raise ValueError("invalid llama-server PID file")
        pid = int(raw)
        args = Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0")
        if not args or Path(os.fsdecode(args[0])).name != "llama-server":
            raise ValueError("PID does not belong to llama-server")
        command = [os.fsdecode(arg) for arg in args if arg]
        if "--port" not in command or command[command.index("--port") + 1] != str(self.port):
            raise ValueError("PID is not serving the selected endpoint port")
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
        if time.monotonic() - self._last_device_probe < 5:
            return self._device_sample
        self._last_device_probe = time.monotonic()
        proc = None
        try:
            proc = subprocess.Popen(["tegrastats", "--interval", "100"],
                                    stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
            try:
                output, _ = proc.communicate(timeout=0.35)
            except subprocess.TimeoutExpired:
                proc.terminate()
                output, _ = proc.communicate(timeout=1)
            line = output.splitlines()[0] if output.splitlines() else ""
            sample = {}
            for key, pattern, scale in (
                ("system_memory_used_mib", r"RAM\s+(\d+)/(\d+)MB", 1),
                ("gpu_utilization_pct", r"GR3D_FREQ\s+(\d+)%", 1),
                ("power_draw_w", r"VDD_GPU_SOC\s+(\d+)mW", 0.001),
                ("temperature_c", r"(?:gpu|tj)@([\d.]+)C", 1),
            ):
                match = re.search(pattern, line)
                if match:
                    sample[key] = round(float(match.group(1)) * scale, 3)
                    if key == "system_memory_used_mib":
                        sample["system_memory_total_mib"] = round(float(match.group(2)) * scale, 3)
                        sample["system_memory_available_mib"] = round(
                            max(0.0, sample["system_memory_total_mib"] - sample[key]), 3
                        )
                        sample["system_memory_scope"] = "host_all_processes"
            self._device_sample = sample
        except (OSError, subprocess.SubprocessError, ValueError):
            self._device_sample = {}
        finally:
            if proc is not None and proc.poll() is None:
                proc.kill()
                proc.communicate()
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
