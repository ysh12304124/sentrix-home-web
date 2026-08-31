"""Pluggable model-runtime providers shared by Sentrix and PhotoBench.

Inference is required. Lifecycle and telemetry are optional so a plain
OpenAI-compatible llama.cpp, Ollama, or vLLM endpoint can be evaluated without
installing the Sentrix vLLM Manager.
"""
from __future__ import annotations

from dataclasses import dataclass
import re

import httpx


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
