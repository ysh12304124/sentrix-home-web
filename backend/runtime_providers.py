"""Pluggable model-runtime providers shared by Sentrix and PhotoBench.

Inference is required. Lifecycle and telemetry are optional so a plain
OpenAI-compatible llama.cpp, Ollama, or vLLM endpoint can be evaluated without
installing the Sentrix vLLM Manager.
"""
from __future__ import annotations

from dataclasses import dataclass
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
        return self._optional("/process-memory")

    def system_memory(self) -> dict:
        return self._optional("/system-memory")

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
        values = {}
        try:
            for line in Path("/proc/meminfo").read_text(encoding="ascii").splitlines():
                key, _, raw = line.partition(":")
                if key in {"MemTotal", "MemAvailable"}:
                    # Linux /proc/meminfo reports kB (despite the historical
                    # field name). Convert once here so every API field whose
                    # suffix is _mib is genuinely MiB.
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

    def system_memory(self) -> dict:
        return self._read_system_memory()
    def process_memory(self) -> dict:
        if not self.process_hint:
            return {"status": "unavailable", "source": "host_nvidia_smi",
                    "reason": "model_process_identity_not_configured"}
        try:
            rows = self._query("compute-apps=pid,process_name,used_memory")
            candidates = [{"pid": int(float(r[0])), "process_name": r[1], "used_memory_mib": float(r[2])}
                          for r in rows if len(r) >= 3]
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
                return {"status": "unavailable", "source": "host_nvidia_smi",
                        "reason": "matching_model_process_not_found", "data": {
                            "processes": [], "model_processes": [],
                            "all_processes": candidates,
                            "all_processes_memory_mib": sum(x["used_memory_mib"] for x in candidates),
                            "process_attribution": "unavailable",
                            "process_attribution_reason": "matching_model_process_not_found",
                        }}
            others = [item for item in candidates if item not in matches]
            return {"status": "available", "source": "host_nvidia_smi", "data": {
                "process_memory_used_mib": sum(x["used_memory_mib"] for x in matches),
                "processes": matches, "model_processes": matches,
                "model_process_scope": "gpu_compute_processes",
                "other_processes_memory_mib": sum(x["used_memory_mib"] for x in others),
                "other_processes": others, "all_processes_memory_mib": sum(x["used_memory_mib"] for x in candidates),
                "all_processes": candidates, "all_processes_scope": "gpu_compute_processes",
            }}
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
