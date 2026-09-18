"""Pluggable model-runtime providers shared by Sentrix and PhotoBench.

Inference is required. Lifecycle and telemetry are optional so a plain
OpenAI-compatible llama.cpp, Ollama, or vLLM endpoint can be evaluated without
installing the Sentrix vLLM Manager.
"""
from __future__ import annotations

from dataclasses import dataclass
import os
import re
import csv
import io
import subprocess
import time
from pathlib import Path
from urllib.parse import urlparse

import httpx
from backend.jetson_telemetry import LocalJetsonLlamaCppTelemetryProvider, is_jetson_host


def normalize_service_url(value: str | None) -> str:
    value = str(value or "").strip().rstrip("/")
    if value and not value.startswith(("http://", "https://")):
        value = f"http://{value}"
    return value


def normalize_openai_base_url(value: str | None) -> str:
    value = normalize_service_url(value)
    if not value:
        return ""
    return value if re.search(r"/v\d+$", value, flags=re.IGNORECASE) else f"{value}/v1"


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


def _read_linux_system_memory() -> dict:
    values = {}
    try:
        for line in Path("/proc/meminfo").read_text(encoding="ascii").splitlines():
            key, _, raw = line.partition(":")
            if key in {"MemTotal", "MemAvailable"}:
                # Linux /proc/meminfo reports kB. Convert once here so every
                # API field whose suffix is _mib is genuinely MiB.
                values[key] = float(raw.strip().split()[0]) / 1024.0
        total = values.get("MemTotal")
        available = values.get("MemAvailable")
        if total is None or available is None:
            raise ValueError("MemTotal/MemAvailable missing")
        return {"status": "available", "source": "proc_meminfo", "data": {
            "system_memory_used_mib": round(max(0.0, total - available), 2),
            "system_memory_total_mib": round(total, 2),
            "system_memory_available_mib": round(available, 2),
            "system_memory_scope": "host_all_processes",
        }}
    except (OSError, ValueError, IndexError) as exc:
        return {"status": "unavailable", "source": "proc_meminfo", "reason": str(exc)}


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


def _query_compute_apps() -> list[dict]:
    result = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory",
         "--format=csv,noheader,nounits"],
        capture_output=True, text=True, timeout=5, check=True,
    )
    rows = []
    for row in csv.reader(io.StringIO(result.stdout), skipinitialspace=True):
        if len(row) < 3:
            continue
        try:
            rows.append({
                "pid": int(float(row[0])),
                "process_name": row[1],
                "used_memory_mib": float(row[2]),
            })
        except (ValueError, TypeError):
            continue
    return rows



def _process_pss_mib(pid: int) -> float | None:
    try:
        with open(f"/proc/{pid}/smaps_rollup", encoding="ascii") as file:
            for line in file:
                if line.startswith("Pss:"):
                    return round(int(line.split()[1]) / 1024, 2)
    except (OSError, ValueError, IndexError):
        return None
    return None


def _sum_mib(rows: list[dict], key: str) -> float | None:
    values = [float(row[key]) for row in rows if isinstance(row.get(key), (int, float))]
    return round(sum(values), 2) if values else None


def _augment_host_process_memory(
    data: dict | None, *, process_hint: str = "", endpoint_port: str = "",
    model_pids: set[int] | None = None, compute_apps: list[dict] | None = None,
) -> dict:
    """Add host RAM and PhotoBench-process GPU attribution to telemetry data.

    The extra fields are best-effort and remain absent when the host cannot
    expose them. Missing values must not be interpreted as 0.
    """
    data = dict(data or {})
    try:
        process_rows = _host_process_rows()
    except Exception:
        process_rows = []
    try:
        compute_apps = list(compute_apps) if compute_apps is not None else _query_compute_apps()
    except Exception:
        compute_apps = []

    known_model_pids = {int(pid) for pid in (model_pids or set()) if str(pid).isdigit()}
    for key in ("root_pid",):
        if str(data.get(key) or "").isdigit():
            known_model_pids.add(int(data[key]))
    for pid in data.get("tracked_pids") or []:
        if str(pid).isdigit():
            known_model_pids.add(int(pid))
    for process in data.get("processes") or data.get("model_processes") or []:
        if str((process or {}).get("pid") or "").isdigit():
            known_model_pids.add(int(process["pid"]))

    hint = str(process_hint or "").lower()
    port = str(endpoint_port or "")
    model_rows = []
    for row in process_rows:
        identity = str(row.get("identity") or "")
        if row["pid"] in known_model_pids or (hint and hint in identity) or (port and port in identity):
            model_rows.append(row)

    system_rows = [row for row in process_rows if _is_system_related_process(row)]
    system_pids = {row["pid"] for row in system_rows}
    system_gpu_rows = [row for row in compute_apps if row.get("pid") in system_pids]

    model_pids = {row["pid"] for row in model_rows}
    sentrix_rows = [row for row in system_rows if row["pid"] not in model_pids]
    for row in system_rows:
        row["pss_mib"] = _process_pss_mib(int(row["pid"]))
    model_rss = _sum_mib(model_rows, "rss_mib")
    sentrix_pss = _sum_mib(sentrix_rows, "pss_mib")
    product_pss = _sum_mib(system_rows, "pss_mib")
    system_gpu = _sum_mib(system_gpu_rows, "used_memory_mib")
    if model_rss is not None:
        data["model_process_system_memory_used_mib"] = model_rss
        data["model_process_system_memory_scope"] = "host_process_rss"
        data["model_system_processes"] = [
            {key: row.get(key) for key in ("pid", "process_name", "rss_mib", "pss_mib", "listening_ports")}
            for row in model_rows[:20]
        ]
    if sentrix_pss is not None:
        data["sentrix_stack_pss_mib"] = sentrix_pss
        data["sentrix_stack_pss_scope"] = "sentrix_related_pss_excluding_model"
    if product_pss is not None:
        data["product_stack_memory_mib"] = product_pss
        data["benchmark_process_memory_used_mib"] = product_pss
        data["benchmark_process_memory_scope"] = "photobench_related_host_process_pss"
        data["benchmark_processes"] = [
            {key: row.get(key) for key in ("pid", "process_name", "rss_mib", "pss_mib", "listening_ports")}
            for row in system_rows[:50]
        ]
    if system_gpu is not None:
        data["benchmark_process_gpu_memory_mib"] = system_gpu
        data["benchmark_process_gpu_memory_scope"] = "photobench_related_gpu_compute_processes"
        data["benchmark_gpu_processes"] = system_gpu_rows[:50]
    if compute_apps and data.get("all_processes_memory_mib") is None:
        data["all_processes_memory_mib"] = round(sum(row["used_memory_mib"] for row in compute_apps), 2)
        data["all_processes"] = compute_apps
        data["all_processes_scope"] = "gpu_compute_processes"
    return data


class InferenceProvider:
    def health(self) -> dict:  # pragma: no cover - interface contract
        raise NotImplementedError

    def list_models(self) -> dict:  # pragma: no cover - interface contract
        raise NotImplementedError

    def chat(self, payload: dict, *, timeout: float | None = None):  # pragma: no cover
        raise NotImplementedError

    def chat_stream(self, payload: dict, *, timeout: float | None = None):  # pragma: no cover
        raise NotImplementedError

    def token_count(self, messages: list[dict], *, timeout: float = 15) -> dict | None:
        return None

    def capabilities(self) -> dict:  # pragma: no cover - interface contract
        raise NotImplementedError


class LifecycleProvider:
    def profiles(self) -> dict:  # pragma: no cover - interface contract
        raise NotImplementedError

    def start(self, payload: dict, *, timeout: float = 120) -> dict:  # pragma: no cover
        raise NotImplementedError

    def stop(self, payload: dict | None = None, *, timeout: float = 90) -> dict:  # pragma: no cover
        raise NotImplementedError

    def state(self) -> dict:  # pragma: no cover - interface contract
        raise NotImplementedError


class TelemetryProvider:
    def gpu_stats(self) -> dict:  # pragma: no cover - interface contract
        raise NotImplementedError

    def process_memory(self) -> dict:  # pragma: no cover - interface contract
        raise NotImplementedError

    def kv_cache(self) -> dict:  # pragma: no cover - interface contract
        raise NotImplementedError

    def system_memory(self) -> dict:
        return {"status": "unavailable", "reason": "system_memory_not_supported"}


class OpenAICompatibleInferenceProvider(InferenceProvider):
    def __init__(self, base_url: str, *, api_key: str = "", api_mode: str = "generic",
                 manager_url: str = "", timeout: float = 180):
        self.base_url = normalize_openai_base_url(base_url)
        if not self.base_url:
            raise ValueError("model base URL is required")
        self.api_key = str(api_key or "")
        self.api_mode = str(api_mode or "generic").strip().lower()
        self.manager_url = normalize_service_url(manager_url)
        self.timeout = float(timeout)

    @property
    def headers(self) -> dict:
        return {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}

    def health(self) -> dict:
        try:
            models = self.list_models()
            return {"status": "available", "source": "openai_models", **models}
        except Exception as models_error:
            root = self.base_url.removesuffix("/v1")
            for suffix in ("/health", "/api/health"):
                try:
                    response = httpx.get(f"{root}{suffix}", headers=self.headers, timeout=min(10, self.timeout))
                    response.raise_for_status()
                    body = response.json() if response.content else {}
                    return {"status": "available", "source": suffix, "detail": body}
                except Exception:
                    continue
            return {"status": "unavailable", "error": str(models_error)}

    def list_models(self) -> dict:
        response = httpx.get(f"{self.base_url}/models", headers=self.headers, timeout=min(15, self.timeout))
        response.raise_for_status()
        body = response.json()
        models = [
            str(item.get("id")) for item in body.get("data") or []
            if isinstance(item, dict) and item.get("id")
        ]
        return {"models": models, "raw": body}

    def chat(self, payload: dict, *, timeout: float | None = None):
        body = dict(payload or {})
        if self.api_mode == "generic":
            body.pop("chat_template_kwargs", None)
            body = {key: value for key, value in body.items() if value is not None}
        response = httpx.post(
            f"{self.base_url}/chat/completions", json=body, headers=self.headers,
            timeout=timeout or self.timeout,
        )
        response.raise_for_status()
        return response

    def chat_stream(self, payload: dict, *, timeout: float | None = None):
        body = dict(payload or {})
        if self.api_mode == "generic":
            body.pop("chat_template_kwargs", None)
            body = {key: value for key, value in body.items() if value is not None}
        return httpx.stream(
            "POST", f"{self.base_url}/chat/completions", json=body,
            headers=self.headers, timeout=timeout or self.timeout,
        )

    def token_count(self, messages: list[dict], *, timeout: float = 15) -> dict | None:
        if not self.manager_url:
            return None
        response = httpx.post(
            f"{self.manager_url}/tokenize-current",
            json={"messages": messages, "add_generation_prompt": True},
            timeout=min(timeout, self.timeout),
        )
        response.raise_for_status()
        value = response.json()
        if int(value.get("prompt_tokens") or 0) < 1 or int(value.get("max_model_len") or 0) < 1:
            raise ValueError("invalid tokenizer budget response")
        return value

    def capabilities(self) -> dict:
        managed = bool(self.manager_url)
        return {
            "provider": "openai_compatible",
            "api_mode": self.api_mode,
            "chat": True,
            "chat_stream": True,
            "list_models": True,
            "token_count": managed,
            "vision": "unknown",
            "json_object": "optional",
            "stream_usage": "optional",
            "vllm_extensions": self.api_mode == "vllm",
        }


class ManagerLifecycleProvider(LifecycleProvider):
    def __init__(self, manager_url: str):
        self.base_url = normalize_service_url(manager_url)
        if not self.base_url:
            raise ValueError("manager URL is required")

    def _request(self, path: str, *, payload=None, method="GET", timeout=30):
        response = httpx.request(method, f"{self.base_url}{path}", json=payload, timeout=timeout)
        response.raise_for_status()
        return response.json() if response.content else {}

    def profiles(self) -> dict:
        value = self._request("/profiles", timeout=15)
        profiles = value if isinstance(value, list) else value.get("profiles") or []
        return {"status": "available", "profiles": profiles}

    def start(self, payload: dict, *, timeout: float = 120) -> dict:
        return self._request("/start", payload=payload, method="POST", timeout=timeout)

    def stop(self, payload: dict | None = None, *, timeout: float = 90) -> dict:
        return self._request("/stop", payload=payload or {"timeout": 60}, method="POST", timeout=timeout)

    def state(self) -> dict:
        return self._request("/state", timeout=15)


class ManagerTelemetryProvider(TelemetryProvider):
    def __init__(self, manager_url: str):
        self.base_url = normalize_service_url(manager_url)
        if not self.base_url:
            raise ValueError("manager URL is required")

    def _optional(self, path: str) -> dict:
        try:
            response = httpx.get(f"{self.base_url}{path}", timeout=10)
            response.raise_for_status()
            value = response.json() if response.content else {}
            return {"status": "available", "data": value}
        except Exception as exc:
            return {"status": "unavailable", "error": str(exc)}

    def gpu_stats(self) -> dict:
        return self._optional("/gpu-stats")

    def process_memory(self) -> dict:
        value = self._optional("/process-memory")
        if value.get("status") == "available":
            data = value.get("data") or {}
            model_pids = set()
            for pid in data.get("tracked_pids") or []:
                try:
                    model_pids.add(int(pid))
                except (TypeError, ValueError):
                    pass
            if data.get("root_pid") is not None:
                try:
                    model_pids.add(int(data["root_pid"]))
                except (TypeError, ValueError):
                    pass
            value["data"] = _augment_host_process_memory(
                data, process_hint="vllm", model_pids=model_pids,
            )
        return value

    def system_memory(self) -> dict:
        return _read_linux_system_memory()

    def kv_cache(self) -> dict:
        memory = self.process_memory()
        if memory.get("status") != "available":
            return memory
        metrics = (memory.get("data") or {}).get("vllm_metrics") or {}
        return {"status": "available", "data": metrics}


class UnavailableLifecycleProvider(LifecycleProvider):
    _result = {"status": "not_applicable", "reason": "model_manager_not_configured"}

    def profiles(self) -> dict:
        return {**self._result, "profiles": []}

    def start(self, payload: dict, *, timeout: float = 120) -> dict:
        return dict(self._result)

    def stop(self, payload: dict | None = None, *, timeout: float = 90) -> dict:
        return dict(self._result)

    def state(self) -> dict:
        return dict(self._result)


class HostNvidiaTelemetryProvider(TelemetryProvider):
    """Framework-neutral local NVIDIA telemetry for external runtimes."""
    framework = "generic"
    def __init__(self, process_hint: str = "", endpoint_url: str = ""):
        self.process_hint = str(process_hint or "").strip().lower()
        self.endpoint_url = normalize_service_url(endpoint_url)

    @property
    def endpoint_port(self) -> str:
        parsed = urlparse(self.endpoint_url)
        return str(parsed.port or "")

    @staticmethod
    def _query(query: str):
        result = subprocess.run(["nvidia-smi", f"--query-{query}", "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=5, check=True)
        return list(csv.reader(io.StringIO(result.stdout), skipinitialspace=True))
    def gpu_stats(self) -> dict:
        try:
            rows = self._query("gpu=index,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw,clocks.sm")
            gpus = []
            for row in rows:
                if len(row) < 7: continue
                v = [float(x) for x in row[1:]]
                gpus.append({"index": int(float(row[0])), "gpu_utilization_pct": v[0], "memory_used_mib": v[1], "memory_total_mib": v[2], "temperature_c": v[3], "power_draw_w": v[4], "sm_clock_mhz": v[5]})
            return {"status": "available", "source": "host_nvidia_smi", "data": {
                "gpus": gpus,
                "memory_scope": "gpu_device",
            }}
        except Exception as exc:
            return {"status": "unavailable", "source": "host_nvidia_smi", "error": str(exc)}

    @staticmethod
    def _read_system_memory() -> dict:
        return _read_linux_system_memory()

    def system_memory(self) -> dict:
        return self._read_system_memory()
    def process_memory(self) -> dict:
        try:
            candidates = _query_compute_apps()
            hints = [self.process_hint]
            matches = []
            for item in candidates:
                process_name = item["process_name"].lower()
                identity = process_name
                try:
                    raw_cmdline = Path(f"/proc/{item['pid']}/cmdline").read_bytes()
                    identity = f"{identity} {raw_cmdline.replace(bytes([0]), bytes([32])).decode('utf-8', 'replace').lower()}"
                except (OSError, ValueError):
                    pass
                item["identity"] = identity[:500]
                if any(hint and hint in identity for hint in hints):
                    matches.append(item)
                elif self.endpoint_port and self.endpoint_port in identity:
                    # A generic Python executable may hide the runtime name;
                    # a command line bound to this endpoint is still a useful
                    # best-effort identity signal for unmanaged runtimes.
                    matches.append(item)
            if not matches:
                data = _augment_host_process_memory(
                    {
                            "processes": [], "model_processes": [],
                            "all_processes": candidates,
                            "all_processes_memory_mib": sum(x["used_memory_mib"] for x in candidates),
                            "process_attribution": "unavailable",
                            "process_attribution_reason": "matching_model_process_not_found",
                    },
                    process_hint=self.process_hint, endpoint_port=self.endpoint_port,
                    compute_apps=candidates,
                )
                return {"status": "unavailable", "source": "host_nvidia_smi",
                        "reason": "matching_model_process_not_found", "data": data}
            others = [item for item in candidates if item not in matches]
            data = _augment_host_process_memory({
                "process_memory_used_mib": sum(x["used_memory_mib"] for x in matches),
                "processes": matches, "model_processes": matches,
                "model_process_scope": "gpu_compute_processes",
                "other_processes_memory_mib": sum(x["used_memory_mib"] for x in others),
                "other_processes": others, "all_processes_memory_mib": sum(x["used_memory_mib"] for x in candidates),
                "all_processes": candidates, "all_processes_scope": "gpu_compute_processes",
            }, process_hint=self.process_hint, endpoint_port=self.endpoint_port,
                model_pids={item["pid"] for item in matches}, compute_apps=candidates)
            return {"status": "available", "source": "host_nvidia_smi", "data": data}
        except Exception as exc:
            return {"status": "unavailable", "source": "host_nvidia_smi", "error": str(exc)}
    def kv_cache(self) -> dict:
        return {"status": "not_applicable", "source": "host_nvidia_smi", "runtime_framework": self.framework, "reason": "framework_does_not_expose_kv_cache"}

class LlamaCppTelemetryProvider(HostNvidiaTelemetryProvider):
    framework = "llama.cpp"
    def __init__(self, endpoint_url: str = ""):
        super().__init__(process_hint="llama-server", endpoint_url=endpoint_url)

class VllmTelemetryProvider(HostNvidiaTelemetryProvider):
    """Best-effort process attribution for a local, unmanaged vLLM endpoint."""
    framework = "vllm"
    def __init__(self, endpoint_url: str = ""):
        super().__init__(process_hint="vllm", endpoint_url=endpoint_url)


class OrinLlamaCppTelemetryProvider(TelemetryProvider):
    """Read the *specific* 118 llama-server process over SSH; Orin has UMA, not VRAM.

    The remote identity, PID file and command are fixed, not derived from a
    user-supplied URL. A missing/changed process must never be reported as 0 MB.
    """
    endpoint = "http://192.168.0.118:8100"
    remote = "orin@192.168.0.118"
    pid_file = "/home/orin/VLM/gemma4/results/orin/server_photobench_8100.pid"

    def __init__(self, endpoint_url: str = ""):
        parsed = urlparse(normalize_service_url(endpoint_url))
        if parsed.hostname != "192.168.0.118" or parsed.port != 8100:
            raise ValueError("Orin telemetry is only configured for the 118:8100 endpoint")
        self._last_memory: dict = {"status": "unavailable", "reason": "not_sampled"}
        self._last_pss_probe = 0.0

    def gpu_stats(self) -> dict:
        data = dict(self._last_memory.get("data") or {})
        data.setdefault("memory_unit", "process_pss_uma_mib")
        data.setdefault("memory_profile", {"method": "orin_process_pss_uma_v1"})
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
        if time.monotonic() - self._last_pss_probe < 5:
            self._last_memory = {"status": "available", "source": "orin_ssh_pss", "data": data}
            return {"status": "available", "source": "orin_ssh_pss", "data": {"gpus": [{"index": 0, "memory_unit": "process_pss_uma_mib"}]}}
        self._last_pss_probe = time.monotonic()
        command = (
            f'pid=$(cat {self.pid_file}) || exit 1; '
            'case "$pid" in *[!0-9]*|"") exit 2;; esac; '
            'ps -p "$pid" -o args= | grep -Fq "/llama-server --model " || exit 3; '
            'ps -p "$pid" -o args= | grep -Fq -- "--port 8100" || exit 3; '
            'awk \'/^Pss:/{print $2}\' "/proc/$pid/smaps_rollup"; '
            'printf "pid=%s\\n" "$pid"'
        )
        try:
            completed = subprocess.run(
                ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=3",
                 "-o", "ConnectionAttempts=1", self.remote, command],
                capture_output=True, text=True, timeout=5, check=True,
            )
            lines = completed.stdout.strip().splitlines()
            pss_mib = int(lines[0]) / 1024
            pid = int(lines[1].removeprefix("pid="))
            data["root_pid"] = pid
            data["process_memory_used_mib"] = round(pss_mib, 2)
            data["pss_sampled_at_monotonic"] = time.monotonic()
            self._last_memory = {"status": "available", "source": "orin_ssh_pss", "data": data}
            return {"status": "available", "source": "orin_ssh_pss", "data": {"gpus": [{"index": 0, "memory_unit": "process_pss_uma_mib"}]}}
        except (OSError, subprocess.SubprocessError, ValueError, IndexError) as exc:
            data.pop("process_memory_used_mib", None)
            data["pss_error"] = str(exc)
            self._last_memory = {"status": "available", "source": "orin_ssh_pss", "data": data}
            return {"status": "available", "source": "orin_ssh_pss", "data": {"gpus": [{"index": 0, "memory_unit": "process_pss_uma_mib"}]}}

    def process_memory(self) -> dict:
        if self._last_memory.get("status") == "available":
            data = _augment_host_process_memory(
                self._last_memory.get("data") or {},
                process_hint="llama-server", endpoint_port="8100",
                model_pids={
                    int((self._last_memory.get("data") or {}).get("root_pid"))
                } if str((self._last_memory.get("data") or {}).get("root_pid") or "").isdigit() else set(),
            )
            return {**self._last_memory, "data": data}
        return self._last_memory

    def kv_cache(self) -> dict:
        data = self._last_memory.get("data") or {}
        if "kv_cache_used_tokens" not in data and "vllm_metrics" not in data:
            return {"status": "unavailable", "source": "llamacpp_metrics", "reason": "metrics_missing_or_disabled"}
        return {"status": "available", "source": "llamacpp_metrics", "data": {
            "used_tokens": data.get("kv_cache_used_tokens"),
            "usage_pct": (data.get("vllm_metrics") or {}).get("kv_cache_usage_pct"),
            "used_bytes": None, "bytes_status": "unavailable_without_model_kv_allocation_bytes",
        }}


class OllamaTelemetryProvider(HostNvidiaTelemetryProvider):
    framework = "ollama"
    def __init__(self, endpoint_url: str = ""):
        super().__init__(process_hint="ollama", endpoint_url=endpoint_url)


class UnavailableTelemetryProvider(TelemetryProvider):
    _result = {"status": "not_applicable", "reason": "telemetry_provider_not_configured"}

    def gpu_stats(self) -> dict:
        return dict(self._result)

    def process_memory(self) -> dict:
        return dict(self._result)

    def kv_cache(self) -> dict:
        return dict(self._result)


@dataclass(frozen=True)
class RuntimeProviders:
    inference: InferenceProvider
    lifecycle: LifecycleProvider
    telemetry: TelemetryProvider


def create_runtime_providers(model_base_url: str, *, manager_url: str = "", api_key: str = "",
                             api_mode: str = "generic", timeout: float = 180) -> RuntimeProviders:
    inference = OpenAICompatibleInferenceProvider(
        model_base_url, api_key=api_key, api_mode=api_mode,
        manager_url=manager_url, timeout=timeout,
    )
    manager_url = normalize_service_url(manager_url)
    if manager_url:
        return RuntimeProviders(
            inference=inference,
            lifecycle=ManagerLifecycleProvider(manager_url),
            telemetry=ManagerTelemetryProvider(manager_url),
        )
    return RuntimeProviders(
        inference=inference,
        lifecycle=UnavailableLifecycleProvider(),
        telemetry=UnavailableTelemetryProvider(),
    )
