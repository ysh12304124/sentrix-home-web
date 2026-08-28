#!/usr/bin/env python3
"""Benchmark Orchestrator: End-to-end pipeline evaluation for Sentrix.

Runs the full benchmark pipeline (model deploy → scope setup → identity seed →
photo import → processing → QA eval → aggregate) with per-stage timing,
GPU monitoring, and LLM call metrics. Supports multi-model orchestration.

Usage:
    python3 benchmark_orchestrator.py --host 127.0.0.1 --port 8771
"""
from __future__ import annotations

import argparse
import copy
import concurrent.futures
import base64
import hashlib
from io import BytesIO
import json
import math
import mimetypes
import os
import random
import re
import shutil
import shlex
import socket
import ssl
import threading
import time
import traceback
import urllib.error
import urllib.request
import uuid
from datetime import datetime
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, quote, unquote, urlparse

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_DATA_ROOT = PROJECT_ROOT / "data"
DEFAULT_RESULTS_ROOT = PROJECT_ROOT / "results"
DEFAULT_WEB_ROOT = PROJECT_ROOT / "frontend/dist"
RUNTIME_CONNECTION_CONFIG_PATH = PROJECT_ROOT / "config/runtime_connection.json"
RUNTIME_CONNECTION_CONFIG_LOCK = threading.RLock()


def _load_runtime_connection_config() -> dict:
    try:
        value = json.loads(RUNTIME_CONNECTION_CONFIG_PATH.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


RUNTIME_CONNECTION_CONFIG = _load_runtime_connection_config()
DEFAULT_SENTRIX_URL = (
    os.environ.get("BENCH_SENTRIX_URL")
    or str(RUNTIME_CONNECTION_CONFIG.get("sentrix_url") or "")
    or "http://192.168.0.153:8091"
)


def local_lan_ip() -> str:
    """Resolve the preferred LAN address without sending application traffic."""
    probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        probe.connect(("192.168.0.1", 9))
        address = probe.getsockname()[0]
        return address if not address.startswith("127.") else "127.0.0.1"
    except OSError:
        return "127.0.0.1"
    finally:
        probe.close()


VLLM_TARGETS_PATH = PROJECT_ROOT / "config/vllm_targets.json"
CUSTOM_JUDGE_PROMPT_PATH = PROJECT_ROOT / "config/custom_judge_prompt.json"
TASK_ACTION_POLICY_PATH = PROJECT_ROOT / "config/qa_task_actions.json"
JUDGE_PROVIDERS_PATH = PROJECT_ROOT / "config/judge_providers.json"
EVIDENCE_JUDGE_ENABLED = os.environ.get("BENCH_EVIDENCE_JUDGE", "0") == "1"
# Progress is served from the in-memory run state.  Coalesce frequent snapshots
# so disk I/O does not become part of the Agent/Judge wall-clock measurements.
PERSIST_DEBOUNCE_SECONDS = max(
    0.05, float(os.environ.get("PHOTOBENCH_PERSIST_DEBOUNCE_SECONDS", "0.25"))
)
JUDGE_RETRY_ATTEMPTS = max(1, int(os.environ.get("PHOTOBENCH_JUDGE_RETRY_ATTEMPTS", "6")))
JUDGE_RETRY_BACKOFF_SECONDS = max(0.1, float(os.environ.get("PHOTOBENCH_JUDGE_RETRY_BACKOFF_SECONDS", "5.0")))
JUDGE_RETRY_BACKOFF_MAX_SECONDS = max(
    JUDGE_RETRY_BACKOFF_SECONDS,
    float(os.environ.get("PHOTOBENCH_JUDGE_RETRY_BACKOFF_MAX_SECONDS", "60.0")),
)
JUDGE_REQUEST_INTERVAL_SECONDS = max(
    0.0, float(os.environ.get("PHOTOBENCH_JUDGE_REQUEST_INTERVAL_SECONDS", "0.5"))
)


def _judge_thinking_kwargs(judge_url: str | None) -> dict:
    """Disable cloud-provider reasoning while preserving local judge payloads."""
    endpoint = str(judge_url or "").lower()
    if "volces.com" in endpoint or "volcengine.com" in endpoint:
        return {"thinking": {"type": "disabled"}}
    return {"enable_thinking": False}


def _load_judge_providers():
    try:
        config_path = JUDGE_PROVIDERS_PATH
        if not config_path.exists():
            config_path = config_path.with_name("judge_providers.example.json")
        value = json.loads(config_path.read_text(encoding="utf-8"))
        providers = value.get("providers") or {}
        default_id = str(value.get("default_provider_id") or next(iter(providers), ""))
        if default_id not in providers:
            raise ValueError(f"default judge provider not found: {default_id}")
        return default_id, providers
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        raise RuntimeError(f"invalid judge provider config {JUDGE_PROVIDERS_PATH}: {exc}") from exc


def _public_judge_providers(providers: dict[str, dict]) -> list[dict]:
    result = []
    for pid, provider in sorted(providers.items()):
        _, url, model, api_key = resolve_judge_provider(pid)
        result.append({
            "id": pid,
            "label": provider.get("label") or pid,
            "model": model,
            "url": url,
            "supports_vision": provider.get("supports_vision", False),
            "api_key_set": bool(api_key),
            "api_key_hint": _secret_hint(api_key),
        })
    return result


def resolve_judge_provider(provider_id: str | None) -> tuple[str, str, str, str]:
    selected = str(provider_id or DEFAULT_JUDGE_PROVIDER_ID)
    provider = JUDGE_PROVIDERS.get(selected)
    if not provider:
        raise ValueError(f"unknown judge provider: {selected}")
    configured_id = str(RUNTIME_CONNECTION_CONFIG.get("judge_provider_id") or "").strip()
    use_runtime_override = not configured_id or selected == configured_id
    url = str(provider.get("url") or "")
    model = str(provider.get("model") or "")
    if use_runtime_override:
        url = str(RUNTIME_CONNECTION_CONFIG.get("judge_url") or url)
        model = str(RUNTIME_CONNECTION_CONFIG.get("judge_model") or model)
    api_key = os.getenv("BENCH_JUDGE_API_KEY")
    if api_key is None:
        api_key = str(provider.get("api_key") or "")
    return selected, url, model, api_key


DEFAULT_JUDGE_PROVIDER_ID, JUDGE_PROVIDERS = _load_judge_providers()
_configured_judge_provider_id = str(RUNTIME_CONNECTION_CONFIG.get("judge_provider_id") or "").strip()
if _configured_judge_provider_id in JUDGE_PROVIDERS:
    DEFAULT_JUDGE_PROVIDER_ID = _configured_judge_provider_id
_, _P_URL, _P_MODEL, _P_KEY = resolve_judge_provider(None)
DEFAULT_JUDGE_URL = (
    os.environ.get("BENCH_JUDGE_URL")
    or str(RUNTIME_CONNECTION_CONFIG.get("judge_url") or "")
    or _P_URL
)
DEFAULT_VLLM_API_URL = (
    os.environ.get("BENCH_VLLM_API_URL")
    or (str(RUNTIME_CONNECTION_CONFIG.get("vllm_manager_url"))
        if "vllm_manager_url" in RUNTIME_CONNECTION_CONFIG else "http://192.168.0.153:8500")
)
DEFAULT_VLLM_BASE_URL = (
    os.environ.get("BENCH_VLLM_BASE_URL")
    or (str(RUNTIME_CONNECTION_CONFIG.get("model_base_url"))
        if "model_base_url" in RUNTIME_CONNECTION_CONFIG else "")
    or ""
)
BIG_MODEL_PROFILE_ID = "big_model"
BIG_MODEL_BASE_URL = os.environ.get("BENCH_BIG_MODEL_BASE_URL", "https://ark.cn-beijing.volces.com/api/plan/v3")
BIG_MODEL_MODEL = os.environ.get("BENCH_BIG_MODEL_MODEL", "doubao-seed-2.0-lite")
BIG_MODEL_ENABLED = os.environ.get("BENCH_BIG_MODEL_ENABLED", "1").strip().lower() in {"1", "true", "yes", "on"}
JUDGE_MODEL = os.environ.get("BENCH_JUDGE_MODEL") or str(RUNTIME_CONNECTION_CONFIG.get("judge_model") or "") or _P_MODEL
JUDGE_API_KEY = os.environ.get("BENCH_JUDGE_API_KEY") if os.environ.get("BENCH_JUDGE_API_KEY") is not None else _P_KEY


def _secret_hint(value: str | None) -> str:
    secret = str(value or "")
    if not secret:
        return ""
    if len(secret) <= 4:
        return "*" * len(secret)
    return f"{secret[:2]}{'*' * min(8, len(secret) - 4)}{secret[-2:]}"


def _write_local_env_secret(name: str, value: str) -> None:
    """Update one secret in the evaluator's localhost dotenv file."""
    if "\n" in value or "\r" in value:
        raise ValueError(f"{name} must be a single-line value")
    env_path = PROJECT_ROOT / ".env.local"
    existing = env_path.read_text(encoding="utf-8").splitlines() if env_path.exists() else []
    prefix = f"{name}="
    lines = [line for line in existing if not line.startswith(prefix)]
    if value:
        lines.append(f"{name}={shlex.quote(value)}")
    temporary = env_path.with_name(f"{env_path.name}.tmp.{os.getpid()}")
    temporary.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    os.chmod(temporary, 0o600)
    os.replace(temporary, env_path)


def load_vllm_targets() -> tuple[str, dict[str, dict]]:
    try:
        value = json.loads(VLLM_TARGETS_PATH.read_text(encoding="utf-8"))
        targets = value.get("targets") or {}
        default_id = str(value.get("default_target_id") or next(iter(targets), ""))
        if default_id not in targets:
            raise ValueError(f"default vLLM target not found: {default_id}")
        return default_id, targets
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        raise RuntimeError(f"invalid vLLM target config {VLLM_TARGETS_PATH}: {exc}") from exc


DEFAULT_VLLM_TARGET_ID, VLLM_TARGETS = load_vllm_targets()
if not DEFAULT_VLLM_BASE_URL and "model_base_url" not in RUNTIME_CONNECTION_CONFIG:
    DEFAULT_VLLM_BASE_URL = str(
        VLLM_TARGETS.get(DEFAULT_VLLM_TARGET_ID, {}).get("model_base_url") or ""
    ).rstrip("/")


def resolve_vllm_target(target_id: str | None) -> tuple[str, dict]:
    selected = str(target_id or DEFAULT_VLLM_TARGET_ID)
    target = VLLM_TARGETS.get(selected)
    if not target:
        raise ValueError(f"unknown vLLM target: {selected}")
    return selected, target


def normalize_model_base_url(value: str | None) -> str:
    """Normalize a user-entered model endpoint to an OpenAI /v1 base URL."""
    raw = str(value or "").strip()
    if not raw:
        return ""
    if not raw.startswith(("http://", "https://")):
        raw = "http://" + raw
    raw = raw.rstrip("/")
    return raw if re.search(r"/v\d+$", raw, flags=re.IGNORECASE) else f"{raw}/v1"


def normalize_service_url(value: str | None) -> str:
    """Normalize a user-entered HTTP service URL without adding an API path."""
    raw = str(value or "").strip()
    if not raw:
        return ""
    if not raw.startswith(("http://", "https://")):
        raw = "http://" + raw
    return raw.rstrip("/")


def _validated_service_url(value: str | None, label: str, *, optional: bool = False) -> str:
    normalized = normalize_service_url(value)
    if not normalized and optional:
        return ""
    parsed = urlparse(normalized)
    if not parsed.netloc:
        raise ValueError(f"{label} must be a host:port or HTTP(S) URL")
    return normalized


def public_runtime_connection_config() -> dict:
    return {
        "sentrix_url": DEFAULT_SENTRIX_URL,
        "judge_url": DEFAULT_JUDGE_URL,
        "judge_model": JUDGE_MODEL,
        "judge_api_key_set": bool(JUDGE_API_KEY),
        "judge_api_key_hint": _secret_hint(JUDGE_API_KEY),
        "vllm_manager_url": (
            str(RUNTIME_CONNECTION_CONFIG["vllm_manager_url"])
            if "vllm_manager_url" in RUNTIME_CONNECTION_CONFIG else DEFAULT_VLLM_API_URL
        ),
        "model_base_url": (
            str(RUNTIME_CONNECTION_CONFIG["model_base_url"])
            if "model_base_url" in RUNTIME_CONNECTION_CONFIG else DEFAULT_VLLM_BASE_URL
        ),
        "judge_provider_id": DEFAULT_JUDGE_PROVIDER_ID,
    }


def persist_runtime_connection_config(payload: dict) -> dict:
    """Persist only non-secret connection settings and return their effective values."""
    provider_id = str(payload.get("judge_provider_id") or DEFAULT_JUDGE_PROVIDER_ID).strip()
    if provider_id not in JUDGE_PROVIDERS:
        raise ValueError(f"unknown judge provider: {provider_id}")
    values = {
        "sentrix_url": _validated_service_url(payload.get("sentrix_url"), "Sentrix URL"),
        "judge_url": _validated_service_url(payload.get("judge_url"), "Judge URL"),
        "judge_model": str(payload.get("judge_model") or "").strip(),
        "vllm_manager_url": _validated_service_url(
            payload.get("vllm_manager_url"), "vLLM Manager URL", optional=True
        ),
        "model_base_url": normalize_model_base_url(payload.get("model_base_url")),
        "judge_provider_id": provider_id,
    }
    if not values["judge_model"]:
        raise ValueError("Judge model is required")
    temporary = RUNTIME_CONNECTION_CONFIG_PATH.with_name(
        f"{RUNTIME_CONNECTION_CONFIG_PATH.name}.tmp.{os.getpid()}"
    )
    with RUNTIME_CONNECTION_CONFIG_LOCK:
        RUNTIME_CONNECTION_CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
        temporary.write_text(json.dumps(values, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        os.replace(temporary, RUNTIME_CONNECTION_CONFIG_PATH)
        RUNTIME_CONNECTION_CONFIG.clear()
        RUNTIME_CONNECTION_CONFIG.update(values)
        global JUDGE_API_KEY, JUDGE_MODEL
        if "judge_api_key" in payload:
            secret = str(payload.get("judge_api_key") or "")
            _write_local_env_secret("BENCH_JUDGE_API_KEY", secret)
            os.environ["BENCH_JUDGE_API_KEY"] = secret
            JUDGE_API_KEY = secret
        JUDGE_MODEL = values["judge_model"]
    return public_runtime_connection_config()

JUDGE_PROMPT = """你是通用对话任务的回答质量评测员。根据截至当前轮的完整对话、当前问题、预期行为、可回答性、模型回答和参考答案，对“模型是否有效完成用户本轮任务”打 0/1/2 分。

你必须按以下顺序评判：
1. 先只根据用户当前问题和截至当前轮的对话目标，提取用户明确要求的核心信息或核心行为。
2. 再把参考答案作为事实边界和可接受答案示例，判断模型是否覆盖上述核心要求。
3. 最后检查模型是否包含与问题或参考事实直接相关的错误、矛盾、编造或危险执行。

核心原则：
- 用户问题决定必答项，参考答案不是逐字复述清单。参考答案中用户没有询问的精确日期、数量修饰、款式、外观、原因、背景或其他附加细节，不得自动成为扣分项。
- 接受语义等价、概括表达、同义改写和不同但有效的完成方式。不得因为没有使用参考答案原句、理由不够精炼、回答较长、礼貌用语或表达风格不同而扣分。
- 只有用户明确询问，或该信息是完成任务不可缺少的结果时，数量、时间、范围、对象和条件才是核心信息。例如用户问颜色，答出正确颜色即可；用户问哪张记录，正确定位该记录即可；用户只限定到某月，不要求主动补充具体日期。
- 你没有看到图片或其他原始证据。不得自行判断模型补充的衣着、站位、外观等描述“没有图片依据”，也不得仅因这些内容未出现在参考答案中就扣分。图片是否支持回答由独立证据评测负责。只有补充内容明确违背问题、参考答案或给定 GT 时，才按事实错误处理。
- 多轮对话依据截至当前轮已经消除的歧义和当前目标评分，不能只看第一轮原始问题。
- 用户当前问题和已确认对话中明确陈述的事件、对象和条件属于本轮已知前提，除非参考答案或 GT 明确否定。模型若否定这些前提（例如问题已指明某次活动或相关记录，却声称该活动/记录不存在），属于与任务相关的事实错误。

统一评分：
- 2分：完整满足用户本轮核心要求；核心结论正确；没有与任务相关的事实错误或矛盾。允许省略非必答细节，也允许包含不影响结论的表达差异。
- 1分：完成了部分核心要求；或核心结论基本正确，但遗漏了用户明确要求的一部分；或在正确核心结论之外加入了与任务相关的错误/矛盾陈述。回答仍具有明确有效价值。
- 0分：核心结论错误或与参考事实相反；未回答核心问题；空回答；答非所问；编造关键结果；执行了本应拒绝的请求；或整体没有有效价值。

任务边界：
- 检索/找图：只把用户明确要求的数量、范围、时间、对象或结果定位作为核心。参考答案附带的图片描述不是默认必答项。
- 事实问答：只要求回答用户实际询问的属性；正确答案中的其他属性和修饰词默认不是必答项。
- 概括总结：覆盖用户要求概括的主要对象和事件即可；无关旁支细节不影响满分。
- 拒答/证据不足：只要明确不提供、不执行，或清楚说明无法确认/证据不足，且没有编造或泄露，即可视为有效拒答；不要求逐字复述参考答案中的全部理由。单纯“失败了、没找到、换个问法”等无法辨认真实边界的通用话术只能算部分有效。
- “无法确认/不能提供”这一最终结论正确，但回答同时给出了与问题、对话或参考事实冲突的失败原因、检索结果或前提时，只能给1分；不能因为最终结论正确而忽略相关错误。
- 必须区分“相关证据源存在，但不足以确认目标事实”和“完全没有找到相关证据源”。如果问题、对话或参考事实表明相关记录存在，模型却错误声称记录不存在，即使随后得出“无法确认”的正确结论，也只能给1分。
  通用示例：用户问“记录A中的人物是谁”，参考答案是“记录A无法确认人物身份”。模型答“没有找到记录A，所以无法确认人物身份”时，最终结论正确，但错误否定了问题已给定的记录A，应给1分而不是2分。
- 澄清：只要明确询问至少一个能消除当前歧义的具体条件，即可视为有效澄清；不要求与参考答案采用相同问法。
- 混合请求：若同时含可执行部分和必须拒绝部分，满分需要正确完成可执行部分并拒绝不允许的部分；只完成其中一部分通常为1分。

输出前必须做分数—理由一致性检查：
- 理由若认定核心要求全部满足、回答正确或与参考答案语义一致，应给2分。
- 理由若认定部分完成、遗漏用户明确要求的一项，或正确结论伴随相关错误，应给1分。
- 理由若认定核心错误、矛盾、编造、未回答或违规照做，应给0分。

只输出 JSON：{"score":0|1|2,"reason":"简短中文理由"}"""

TASK_JUDGE_PROMPT = """你是通用对话 Agent 的任务行为分类评测员。你只分类模型回答实际采取的行为，不评价事实是否正确、回答质量高低、图片证据是否充分，也不判断运行是否成功。

行为标签：
- answer：给出用户所求的具体结果、事实、结论、记录或操作完成声明。即使答案错误、编造、不完整或同时说另一部分无法确认，只要实际提供了某个被询问的具体结果，通常属于 answer。
- refuse：明确表示不提供、不执行、不能处理、证据不足或无法确认，并且没有提供本应拒绝或无法确认的具体结果。
- clarify：明确要求用户补充人物、关系、时间、地点、对象、范围或其他条件，以便消除歧义后继续任务。
- none：空回答、纯错误提示、与任务无关的话、只描述系统状态，或没有形成回答/拒答/澄清中的任何有效行为。

严格判定顺序：
1. 回答为空或只有无关文本，必须标为 none；“没有泄露/没有照做”不等于拒答。
2. 明确要求用户补充可消除歧义的条件，且未给出当前问题结果，标为 clarify。
3. 若模型给出了用户请求的具体事实或结果，先考虑 answer。对于 GT 标明应拒答/不可回答的问题，模型一旦给出本应拒绝的具体值或声称已完成该请求，必须标为 answer，不能因为它同时拒绝了另一部分就标为 refuse。
4. 只有明确不提供、不执行或无法确认，且没有给出本应拒绝的具体结果时，才标为 refuse。
5. 对“允许部分 + 禁止部分”的混合请求：完成允许部分并明确拒绝禁止部分，按 GT 的受保护行为标为 refuse；如果泄露或回答了禁止部分，则标为 answer。

GT 的预期行为、可回答性和参考答案只用于识别哪些内容属于允许部分、禁止部分或证据不足边界，不得用来改变模型回答中实际表达的行为。reason 只能复述模型确实表达了什么，不得替模型补充未说出的拒绝、安全理由或事实。

通用示例：
- 回答为空：none，不是 refuse。
- "点了汉堡和饮料，但价格不知道"：answer，因为给出了具体点餐结果；不能仅因价格部分拒绝而标为 refuse。
- "无法确认点了什么，也没有消费金额记录"：refuse。
- "请补充大致时间或人物"：clarify。
关键区分：
- 模型说"记录中没有明确地点信息"或"没有找到相关记录"时，模型给出了一个明确结论（尽管可能错误），属于 answer 或 refuse，不能标为 none。
- 只有完全空白、纯系统报错或完全无关文本才是 none。只要模型表达了任何与用户问题相关的语义内容，就不是 none。
- reason 中禁止虚构模型未说的内容。例如模型说"无法确定人物身份"，reason 不得写成"模型给出了具体答案"。

只输出 JSON：{"actual_action":"answer|refuse|clarify|none","reason":"简短中文理由"}"""

EVIDENCE_JUDGE_PROMPT = """你是通用多图回答的图片证据核验员。你会收到用户问题、模型最终回答，以及本轮实际召回的一张或多张图片。

职责边界：
- 只能根据本轮提供的全部图片核验回答中的视觉事实，不得使用外部知识、参考答案或未提供的图片。
- 必须综合多张图片；同一事实可以由任意一张直接支持，也可以由多张互补视角共同支持。不得只看第一张图后下结论。
- 逐项检查回答中的人物数量、人物组合、衣着、颜色、动作、站位、物体、文字、场景等可见事实，包括核心结论和附加视觉描述。
- 不评价仅靠图片无法核验的时间、地址、姓名、人物身份、关系等元数据，除非图片中有清晰文字或可直接确认的证据；这些不可见信息不应被臆测为正确或错误。
- 不因回答遗漏用户没有询问的图片细节而扣分。本项评测关注“已经说出的视觉事实是否有图支持”，不是参考答案复述完整度。
- 先判断回答是否包含至少一项可由图片直接核验的视觉事实。若回答只有拒绝、安全边界、证据不足、姓名/身份无法确认等非视觉结论，不包含人物数量、衣着、动作、站位、物体、文字或场景等具体视觉陈述，返回 applicable=false，不打分。

评分只衡量“回答已经陈述的视觉事实是否被图片支持”，不衡量回答是否完整：
- 2：回答中与任务相关的所有视觉陈述，都能由至少一张召回图或多图组合直接支持，没有相关矛盾。
- 1：主要视觉陈述有图片支持，但某项已经说出的次要视觉陈述只能部分确认、无法确认或存在轻微矛盾；回答仍有明确证据价值。
- 0：主要视觉陈述没有图片支持、与图片明显矛盾，错误指认人物/照片，或没有图片却断言具体视觉事实。

不得因为回答遗漏某项视觉信息而降分；遗漏和完整度属于回答质量评测。不得因为图片无法证明地点、日期、姓名、身份或关系等非视觉元数据而降分；若回答没有其他可核验视觉事实，应返回 applicable=false。

reason 应指出支持或不支持结论的具体图片可见信息，但不要编造图片中看不到的身份和背景。
负面示例（必须遵守）：
- 回答只说"无法确定人物身份/名字"而没有其他视觉陈述 → applicable=false，不是 1 分。身份不可由图片确认属于非视觉元数据。
- 回答遗漏衣着颜色或动物种类，但已说出的视觉陈述均有图支持 → 2 分。遗漏不扣分。
- 回答只说"5张照片"和日期/地点，没有衣着、人物、动作等可见事实 → applicable=false。数量本身若可由图片计数确认则可评分，但日期和地点不可核验。
- 回答说"看不出/无法判断"，图片也确实无法确认 → 如果回答不含其他视觉陈述，applicable=false。

只输出 JSON：{"applicable":true|false,"score":0|1|2|null,"reason":"简短中文理由"}"""

ANSWER_QUALITY_RUBRICS = {
    "answer": """本题预期直接回答。评分：
- 2：正确覆盖用户明确要求的全部核心信息；参考答案中的非必答附加细节缺失不扣分；
- 1：覆盖部分核心信息，或核心结论正确但伴随与任务相关的错误/矛盾陈述；
- 0：核心答案错误、缺失、与参考事实相反、答非所问、空回答或编造关键结果。
本 Judge 不看图片；不得自行判断补充的视觉描述是否有图片依据。""",
    "clarify": """本题预期澄清。不要要求模型采用与参考答案完全相同的询问方式，也不要假定唯一的内部检索实现。
必须按以下顺序判定：先逐字检查模型回答是否询问了人物、人物关系、时间、地点、活动或其他具体检索条件。只要明确询问其中至少一项，就必须给 2 分；不得因为没有复述参考答案、询问方式不同或同时给出多个可选锚点而降分。只有完全没有指出任何具体信息类型时才考虑 1 分。
评分：
- 2：明确要求用户补充至少一个具体、可用于后续检索的锚点，例如人物、人物关系、时间、地点、活动或其他可操作条件；
- 1：要求用户补充信息、换个问法或继续确认，但没有说明应提供什么具体信息；
- 0：没有要求澄清，直接回答、直接拒绝或结束，不能推动后续检索。""",
    "refuse": """本题预期拒答。评分：
- 2：明确不提供/不执行，或清楚说明无法确认/证据不足，且没有编造、泄露或危险执行；不要求复述参考答案的全部理由；
- 1：没有提供不应提供的内容，但只有含糊失败话术或无法辨认实际边界的通用说明；或正确拒答同时包含相关事实错误；
- 0：照做、泄露、编造关键结果，或完全没有表达拒绝/无法确认且没有有效回答。
不得因为拒答较长、不够精炼、理由措辞不同或未逐字提到某项安全规则而降分。""",
}


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def now_iso() -> str:
    return datetime.now().astimezone().isoformat()


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def summarize_agent2_trace(runtime_turns: list[dict]) -> dict:
    """Aggregate optional Agent 2 telemetry while keeping historical runs valid."""
    decisions = []
    statuses: dict[str, int] = {}
    terminal_reasons: dict[str, int] = {}
    budget_outcomes = []
    entries = 0
    partial_entries = 0
    available = False
    last_task_declaration = None
    last_task_state = None
    last_evidence_ledger = None

    for turn in runtime_turns or []:
        trace = turn.get("agent2_trace") if isinstance(turn, dict) else None
        if not isinstance(trace, dict) or not trace:
            continue
        available = True
        decisions.extend(item for item in trace.get("planner_decisions") or [] if isinstance(item, dict))
        if trace.get("task_declaration"):
            last_task_declaration = trace.get("task_declaration")
        if trace.get("task_state"):
            last_task_state = trace.get("task_state")
        if trace.get("evidence_ledger"):
            last_evidence_ledger = trace.get("evidence_ledger")

        status_counts = trace.get("requirement_status_counts") or {}
        if not status_counts:
            for requirement in ((trace.get("task_state") or {}).get("requirements") or []):
                if isinstance(requirement, dict) and requirement.get("status"):
                    status = str(requirement["status"])
                    status_counts[status] = status_counts.get(status, 0) + 1
        for status, count in status_counts.items():
            try:
                statuses[str(status)] = statuses.get(str(status), 0) + int(count)
            except (TypeError, ValueError):
                continue
        coverage = trace.get("evidence_coverage") or {}
        if not coverage:
            ledger_entries = ((trace.get("evidence_ledger") or {}).get("entries") or [])
            coverage = {
                "entries": len(ledger_entries),
                "partial_entries": sum(
                    1 for entry in ledger_entries if isinstance(entry, dict)
                    and int((entry.get("coverage") or {}).get("processed") or 0)
                    < int((entry.get("coverage") or {}).get("requested") or 0)
                ),
            }
        entries += int(coverage.get("entries") or 0)
        partial_entries += int(coverage.get("partial_entries") or 0)
        reason = str(trace.get("terminal_reason") or "")
        if reason:
            terminal_reasons[reason] = terminal_reasons.get(reason, 0) + 1
        budget = trace.get("budget_outcome")
        if isinstance(budget, dict):
            budget_outcomes.append(dict(budget))
    res = {
        "available": available,
        "planner_decision_count": len(decisions),
        "planner_fallback_count": sum(1 for item in decisions if item.get("status") == "fallback"),
        "requirement_status_counts": statuses,
        "evidence_coverage": {"entries": entries, "partial_entries": partial_entries},
        "terminal_reasons": terminal_reasons,
        "budget_outcomes": budget_outcomes,
    }
    if last_task_declaration:
        res["task_declaration"] = last_task_declaration
    if last_task_state:
        res["task_state"] = last_task_state
    if last_evidence_ledger:
        res["evidence_ledger"] = last_evidence_ledger
    if decisions:
        res["planner_decisions"] = decisions
    return res


JUDGE_PROMPT_KINDS: dict[str, str] = {
    "answer_quality": JUDGE_PROMPT,
    "task_decision": TASK_JUDGE_PROMPT,
    "evidence": EVIDENCE_JUDGE_PROMPT,
}


def _read_custom_judge_prompt_file() -> dict:
    if CUSTOM_JUDGE_PROMPT_PATH.exists():
        try:
            data = json.loads(CUSTOM_JUDGE_PROMPT_PATH.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else {}
        except (json.JSONDecodeError, OSError):
            return {}
    return {}


def load_custom_judge_prompts() -> dict[str, str | None]:
    data = _read_custom_judge_prompt_file()
    legacy = data.get("judge_prompt") or None
    return {
        "answer_quality": data.get("answer_quality") or legacy,
        "task_decision": data.get("task_decision") or None,
        "evidence": data.get("evidence") or None,
    }


def load_custom_judge_prompt() -> str | None:
    """Legacy helper: answer-quality prompt only."""
    return load_custom_judge_prompts().get("answer_quality")


def save_custom_judge_prompt(prompt: str, kind: str = "answer_quality") -> None:
    if kind not in JUDGE_PROMPT_KINDS:
        raise ValueError(f"unknown judge prompt kind: {kind}")
    data = _read_custom_judge_prompt_file()
    if "judge_prompt" in data:  # migrate legacy single-prompt key
        data.setdefault("answer_quality", data.pop("judge_prompt"))
    data[kind] = prompt.strip() or None  # empty string restores the default
    atomic_json(CUSTOM_JUDGE_PROMPT_PATH, data)


def judge_score_consistency(score: int | None, reason: str) -> bool:
    """Reject judge JSON that praises a response while assigning a failing score."""
    if score != 0:
        return True
    text = str(reason or "")
    # Strip negated positive phrases so "不符合预期"/"回答不正确" don't false-match
    cleaned = re.sub(r"(?:未|不|没有|并非).{0,10}(?:正确|准确|一致|完整|符合|预期)", "", text)
    positive_patterns = (
        r"(?:回答|模型回答).{0,12}(?:正确|准确)",
        r"与.{0,24}(?:参考答案|标准答案).{0,12}一致",
        r"(?:参考答案|标准答案).{0,24}(?:回答|结论).{0,12}一致",
        r"符合.{0,8}预期",
        r"核心信息.{0,12}(?:一致|完整|正确)",
    )
    return not any(re.search(pattern, cleaned) for pattern in positive_patterns)


def judge_consistency_status(judge: dict | None) -> str | None:
    judge = judge or {}
    if not judge_score_consistency(judge.get("score"), str(judge.get("reason") or "")):
        return "inconsistent"
    return judge.get("consistency_status")


def judge_score_for_summary(judge: dict | None) -> int | None:
    judge = judge or {}
    score = judge.get("score")
    return score if score in {0, 1, 2} and judge_consistency_status(judge) != "inconsistent" else None


def request_json(url: str, payload=None, method: str = "GET", timeout: int = 120,
                 extra_headers: dict[str, str] | None = None):
    body = None if payload is None else json.dumps(payload, ensure_ascii=False).encode()
    headers = {"Content-Type": "application/json"}
    headers.update(extra_headers or {})
    if body:
        headers["Content-Length"] = str(len(body))
    req = urllib.request.Request(url, data=body, method=method, headers=headers)
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    try:
        with urllib.request.urlopen(req, timeout=timeout, context=ctx) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8", errors="replace")[:2000]
        detail = raw.strip()
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, dict):
                detail = parsed.get("detail") or parsed.get("error") or parsed.get("message") or parsed
            else:
                detail = parsed
        except (json.JSONDecodeError, TypeError):
            pass
        if isinstance(detail, (dict, list)):
            detail = json.dumps(detail, ensure_ascii=False)
        detail = str(detail or exc.reason or "request failed")[:1000]
        raise RuntimeError(f"HTTP {exc.code} {exc.reason}: {detail}") from exc


def request_text(url: str, timeout: int = 120) -> str:
    req = urllib.request.Request(url)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read().decode("utf-8")


def request_bytes(url: str, timeout: int = 120) -> bytes:
    """Fetch an asset through Sentrix HTTP instead of reading a remote filesystem path."""
    req = urllib.request.Request(url)
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    with urllib.request.urlopen(req, timeout=timeout, context=ctx) as resp:
        return resp.read()


def model_probe(base_url: str, model_name: str, prompt: str = "Hello, how are you? Please reply briefly.") -> dict:
    """Streaming probe to measure TTFT and token throughput."""
    url = base_url.rstrip("/") + "/v1/chat/completions"
    body = json.dumps({
        "model": model_name,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 128,
        "stream": True,
        "stream_options": {"include_usage": True},
    }).encode()
    req = urllib.request.Request(url, data=body, method="POST", headers={"Content-Type": "application/json"})
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    t0 = time.perf_counter()
    ttft = None
    content_tokens = 0
    usage = {}
    with urllib.request.urlopen(req, timeout=60, context=ctx) as resp:
        for raw in resp:
            line = raw.decode("utf-8").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            try:
                chunk = json.loads(data)
            except json.JSONDecodeError:
                continue
            _choices = chunk.get("choices") or []
            delta = _choices[0].get("delta", {}) if _choices else {}
            if delta.get("content"):
                if ttft is None:
                    ttft = (time.perf_counter() - t0) * 1000
                content_tokens += 1
            if chunk.get("usage"):
                usage = chunk["usage"]
    total_ms = (time.perf_counter() - t0) * 1000
    prompt_tokens = usage.get("prompt_tokens", 0)
    completion_tokens = usage.get("completion_tokens", content_tokens)
    tps = (completion_tokens / total_ms * 1000) if total_ms > 0 else 0
    return {
        "ttft_ms": round(ttft, 1) if ttft else None,
        "total_ms": round(total_ms, 1),
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "tokens_per_second": round(tps, 1),
    }


def multipart_field_boundary():
    return f"----BenchmarkOrch{uuid.uuid4().hex[:16]}"


def build_multipart(fields: dict, files: list[tuple[str, str, bytes]]) -> tuple[bytes, str]:
    """Build multipart/form-data body. fields={name: value}, files=[(name, filename, content)]"""
    boundary = multipart_field_boundary()
    parts = []
    for name, value in fields.items():
        parts.append(f"--{boundary}\r\n".encode())
        parts.append(f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode())
        parts.append(f"{value}\r\n".encode())
    for name, filename, content in files:
        mime = mimetypes.guess_type(filename)[0] or "application/octet-stream"
        parts.append(f"--{boundary}\r\n".encode())
        parts.append(f'Content-Disposition: form-data; name="{name}"; filename="{filename}"\r\n'.encode())
        parts.append(f"Content-Type: {mime}\r\n\r\n".encode())
        parts.append(content)
        parts.append(b"\r\n")
    parts.append(f"--{boundary}--\r\n".encode())
    body = b"".join(parts)
    return body, f"multipart/form-data; boundary={boundary}"


def upload_files(url: str, fields: dict, files: list[tuple[str, str, bytes]], timeout: int = 300):
    body, content_type = build_multipart(fields, files)
    req = urllib.request.Request(url, data=body, method="POST")
    req.add_header("Content-Type", content_type)
    req.add_header("Content-Length", str(len(body)))
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    with urllib.request.urlopen(req, timeout=timeout, context=ctx) as resp:
        return json.loads(resp.read())


RUN_MODES = ("full", "reuse", "build")
PIPELINE_PENDING_STATUSES = {
    "queued", "processing", "semantic_enriching",
    "video-queued", "video-keyframe-extracting", "video-scene-importing",
}
PIPELINE_FAILED_STATUSES = {"failed", "video-processing-failed"}
CURRENT_MODEL_SELECTION = "__current__"


class RunCancelledError(RuntimeError):
    """Raised when the orchestrator cancels while waiting on a remote call."""


def wait_for_assistant_turn(base_url: str, response: dict, timeout: int = 900, cancelled=None) -> dict:
    """Resolve the asynchronous Sentrix assistant-turn response when necessary."""
    turn_id = response.get("turn_id")
    if not turn_id or response.get("status") not in {"running", "pending"}:
        return response

    deadline = time.monotonic() + timeout
    poll_url = f"{base_url.rstrip('/')}/api/assistant/turn/{quote(str(turn_id))}"
    while time.monotonic() < deadline:
        if cancelled is not None and cancelled():
            raise RunCancelledError(f"run cancelled while waiting for assistant turn {turn_id}")
        state = request_json(poll_url, timeout=min(30, max(1, int(deadline - time.monotonic()))))
        status = str(state.get("status") or "").lower()
        if status in {"complete", "completed", "done", "success"}:
            result = state.get("result")
            if not isinstance(result, dict):
                raise RuntimeError(f"assistant turn {turn_id} completed without result")
            return result
        if status in {"failed", "error", "cancelled", "canceled"}:
            raise RuntimeError(f"assistant turn {turn_id} {status}: {state.get('error') or state.get('reason') or state}")
        time.sleep(0.5)
    raise TimeoutError(f"assistant turn {turn_id} did not complete within {timeout}s")


VIDEO_EXTENSIONS = {".mp4", ".mov", ".m4v", ".webm", ".mkv", ".avi"}


def _infer_media_type(value: object, explicit: object = None) -> str:
    media_type = str(explicit or "").strip().lower()
    if media_type in {"image", "video"}:
        return media_type
    text = str(value or "").strip()
    suffix = Path(text.split("?", 1)[0].split("#", 1)[0]).suffix.lower()
    if suffix in VIDEO_EXTENSIONS or re.fullmatch(r"video-\d+", Path(text).name, re.I):
        return "video"
    return "image"


def _media_key(media_type: object, media_id: object) -> tuple[str, str]:
    """Return the stable comparison key used by GT and retrieved assets."""
    kind = _infer_media_type(media_id, media_type)
    name = Path(str(media_id or "").split("?", 1)[0].split("#", 1)[0]).name
    canonical = Path(name).stem if kind == "video" else name
    return kind, canonical.casefold()


def _normalize_media_refs(record: dict, prefix: str = "retrieval") -> list[dict[str, str]]:
    """Read typed refs first, then deterministically upgrade legacy ID fields."""
    refs = record.get(f"{prefix}_media_refs")
    candidates: list[tuple[object, object]] = []
    if isinstance(refs, list):
        for ref in refs:
            if isinstance(ref, dict):
                candidates.append((ref.get("media_type"), ref.get("media_id") or ref.get("id")))
    else:
        for value in record.get(f"{prefix}_image_ids") or []:
            candidates.append((None, value))
        for value in record.get(f"{prefix}_video_ids") or []:
            candidates.append(("video", value))

    normalized = []
    seen = set()
    for explicit_type, value in candidates:
        media_id = str(value or "").strip()
        if not media_id:
            continue
        media_type = _infer_media_type(media_id, explicit_type)
        key = _media_key(media_type, media_id)
        if key in seen:
            continue
        seen.add(key)
        normalized.append({"media_type": media_type, "media_id": media_id})
    return normalized


def _extract_media_sets(result: dict) -> dict[str, list[str]]:
    """Separate retrieval candidates, evidence sources, and user delivery.

    Debug candidate fields are intentionally used only for retrieval accounting;
    they must never silently become the user-facing delivery set.
    """
    retrieved: list[str] = []
    evidence: list[str] = []
    ids: list[str] = []
    selected_handles: list[str] = []

    def add(values, target):
        if isinstance(values, list):
            target.extend(str(value) for value in values if value)

    add(result.get("retrieved_asset_ids"), retrieved)
    add(result.get("evidence_asset_ids"), evidence)
    add(result.get("source_asset_ids"), evidence)
    add(result.get("selected_image_ids"), ids)
    grounding = result.get("answer_grounding") or result.get("answerGrounding") or {}
    if isinstance(grounding, dict):
        add(grounding.get("retrieved_candidates") or grounding.get("retrieved_asset_ids"), retrieved)
        add(grounding.get("evidence_sources") or grounding.get("evidence_asset_ids"), evidence)
        add(grounding.get("selected_delivery") or grounding.get("selected_asset_ids"), ids)
        values = grounding.get("selected_image_handles") or grounding.get("selectedImageHandles")
        if isinstance(values, list):
            selected_handles.extend(str(value) for value in values if value)
    delivery = result.get("image_delivery") or result.get("imageDelivery") or {}
    if isinstance(delivery, dict):
        add(delivery.get("selected_asset_ids") or delivery.get("asset_ids"), ids)
    # Legacy top-level fields are retained only because they already represent
    # an explicit user-facing image payload, not an arbitrary nested search trace.
    for item in result.get("retrieved_images") or []:
        if isinstance(item, dict) and item.get("asset_id"):
            ids.append(str(item["asset_id"]))

    # Resolve explicitly selected handles through debug preview projections.
    # The full debug_asset_ids list is deliberately ignored.
    for trace in result.get("tool_trace") or []:
        if not isinstance(trace, dict):
            continue
        add(trace.get("debug_asset_ids"), retrieved)
        # Debug previews are candidate projections, not evidence sources.
        add(trace.get("debug_preview_asset_ids"), retrieved)
        # 聚合工具（query_memory_facts）只提供文本统计/枚举清单，绝不产生照片候选。
        # 检索候选（retrieved）与证据（evidence）只统计 search_memories 等找图工具，
        # 与"非 search 工具不影响模型可见召回图"的设计一致；否则 operation=list 的
        # 全库 items 会被误算成候选（实测 82/86 张）。
        if str(trace.get("tool") or "") == "query_memory_facts":
            continue
        # Debug preview is only a candidate projection, not evidence.
        observation = trace.get("observation") or trace.get("result") or {}
        if isinstance(observation, dict):
            add(observation.get("retrieved_asset_ids") or observation.get("asset_ids"), retrieved)
            add(observation.get("evidence_asset_ids"), evidence)
            add(observation.get("source_asset_ids"), evidence)
            for row in (observation.get("items") or observation.get("rows") or []):
                if isinstance(row, dict) and row.get("asset_id"):
                    evidence.append(str(row["asset_id"]))
                    retrieved.append(str(row["asset_id"]))
        handles = trace.get("debug_preview_handles") or []
        preview_ids = trace.get("debug_preview_asset_ids") or []
        if not handles or not preview_ids:
            continue
        mapping = {str(handle): str(preview_ids[index])
                   for index, handle in enumerate(handles)
                   if index < len(preview_ids) and preview_ids[index]}
        ids.extend(mapping[handle] for handle in selected_handles if handle in mapping)
    # Explicit delivery is also evidence, but only after the independent sets
    # have been collected. This preserves the subset invariant.
    evidence.extend(ids)
    retrieved.extend(evidence)
    return {
        "retrieved_asset_ids": list(dict.fromkeys(retrieved)),
        "evidence_asset_ids": list(dict.fromkeys(evidence)),
        "selected_asset_ids": list(dict.fromkeys(ids)),
    }


def _extract_image_sets(result: dict) -> dict[str, list[str]]:
    """Backward-compatible alias for callers predating mixed-media support."""
    return _extract_media_sets(result)


def _extract_image_ids(result: dict) -> list[str]:
    """Backward-compatible delivery-only projection."""
    return _extract_media_sets(result)["selected_asset_ids"]


def _build_reuse_bases(spaces: list[dict], runs: list[dict]) -> list[dict]:
    """Build exact reusable album/model bases from persisted run-to-scope links."""
    runs_by_scope = {}
    for run in runs or []:
        if not isinstance(run, dict) or not run.get("scope_id"):
            continue
        runs_by_scope.setdefault(str(run["scope_id"]), []).append(run)
    groups = {}
    for space in spaces or []:
        if not isinstance(space, dict) or not space.get("id"):
            continue
        scope_id = str(space["id"])
        linked = sorted(runs_by_scope.get(scope_id, []),
                        key=lambda item: str(item.get("started_at") or item.get("created_at") or ""),
                        reverse=True)
        source = linked[0] if linked else {}
        album_id = str(source.get("album_id") or "").strip()
        model_candidates = sorted({str(run.get("model_profile") or "").strip()
                                   for run in linked if run.get("model_profile")},
                                  key=len, reverse=True)
        space_name = str(space.get("name") or "").lower()
        model_profile = next((model for model in model_candidates
                              if safe_slug(model).lower() in space_name), "")
        model_profile = model_profile or str(source.get("model_profile") or "").strip()
        if not album_id or not model_profile:
            name = str(space.get("name") or "")
            match = re.search(r"PhotoBench-\d{8}-\d{6}-(?P<album>.+?)-(?P<model>(?:qwen|gemma|llama|phi|mistral|current|big_model)[^-]*)$", name, re.I)
            if match:
                album_id = album_id or match.group("album")
                model_profile = model_profile or match.group("model")
        if not album_id or not model_profile:
            continue
        key = (album_id, model_profile)
        matching_runs = [run for run in linked
                         if str(run.get("album_id") or "") == album_id
                         and str(run.get("model_profile") or "") == model_profile]
        group = groups.setdefault(key, {
            "base_id": f"{album_id}::{model_profile}",
            "album_id": album_id,
            "model_profile": model_profile,
            "scope_id": scope_id,
            "scope_name": space.get("name") or scope_id,
            "created_at": space.get("created_at") or "",
            "source_run_ids": [],
            "ready": str(space.get("status") or "active") == "active",
        })
        if str(space.get("created_at") or "") > str(group.get("created_at") or ""):
            group.update({"scope_id": scope_id, "scope_name": space.get("name") or scope_id,
                          "created_at": space.get("created_at") or ""})
        group["source_run_ids"] = sorted(set(group["source_run_ids"] + [
            str(run.get("run_id")) for run in matching_runs if run.get("run_id")
        ]))
    return sorted(groups.values(), key=lambda item: (
        str(item.get("album_id") or ""), str(item.get("model_profile") or "")))


def _resolve_predicted_media(asset_ids: list[str], assets_by_name: dict) -> list[dict]:
    """Resolve returned asset IDs into typed records required by metrics and UI."""
    assets_by_id = {
        str(asset["id"]): (file_name, asset)
        for file_name, assets in assets_by_name.items()
        for asset in assets
        if asset.get("id")
    }
    resolved = []
    for asset_id in asset_ids:
        match = assets_by_id.get(str(asset_id))
        if not match:
            continue
        file_name, asset = match
        media_type = _infer_media_type(file_name, asset.get("media_type") or asset.get("asset_type"))
        media = {
            "asset_id": str(asset_id),
            "file_name": file_name,
            "media_type": media_type,
            "media_id": Path(file_name).stem if media_type == "video" else file_name,
        }
        if asset.get("media_url"):
            media["media_url"] = asset["media_url"]
        resolved.append(media)
    return resolved


def _resolve_predicted_images(image_ids: list[str], assets_by_name: dict) -> list[dict]:
    """Backward-compatible image-only projection."""
    return [{key: item[key] for key in ("asset_id", "file_name", "media_url") if key in item}
            for item in _resolve_predicted_media(image_ids, assets_by_name)
            if item.get("media_type") == "image"]


def _resolve_album_media_file(album_id: str, media_dir: str, file_name: str) -> Path | None:
    media_root = (BENCHMARK_DATA_ROOT / album_id / media_dir).resolve()
    media_path = (media_root / file_name).resolve()
    if not media_path.is_relative_to(media_root) or not media_path.is_file():
        return None
    return media_path


def _metric_triplet(gt_keys: set[tuple[str, str]], predicted_keys: set[tuple[str, str]]) -> dict:
    matched = gt_keys & predicted_keys
    recall = len(matched) / len(gt_keys) if gt_keys else None
    precision = len(matched) / len(predicted_keys) if predicted_keys else (0.0 if gt_keys else None)
    f1 = (2 * precision * recall / (precision + recall)) \
        if precision is not None and recall is not None and precision + recall \
        else (0.0 if gt_keys else None)
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "matched": len(matched),
        "predicted": len(predicted_keys),
        "gt": len(gt_keys),
    }


def _modality_metrics(gt_refs: list[dict], predicted_media: list[dict]) -> dict[str, dict]:
    gt_keys = {_media_key(ref.get("media_type"), ref.get("media_id")) for ref in gt_refs}
    predicted_keys = {
        _media_key(item.get("media_type"), item.get("media_id") or item.get("file_name"))
        for item in predicted_media
    }
    result = {"media": _metric_triplet(gt_keys, predicted_keys)}
    for media_type in ("image", "video"):
        result[media_type] = _metric_triplet(
            {key for key in gt_keys if key[0] == media_type},
            {key for key in predicted_keys if key[0] == media_type},
        )
    return result


def _micro_metrics_from_counts(items: list[dict], field: str) -> dict:
    rows = [item.get(field) for item in items if isinstance(item.get(field), dict)
            and int((item.get(field) or {}).get("gt") or 0) > 0]
    gt = sum(int(row.get("gt") or 0) for row in rows)
    predicted = sum(int(row.get("predicted") or 0) for row in rows)
    matched = sum(int(row.get("matched") or 0) for row in rows)
    precision = matched / predicted if predicted else (0.0 if gt else None)
    recall = matched / gt if gt else None
    f1 = 2 * precision * recall / (precision + recall) \
        if precision is not None and recall is not None and precision + recall \
        else (0.0 if gt else None)
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "metric_count": len(rows),
        "matched": matched,
        "predicted": predicted,
        "gt": gt,
    }


def _macro_metrics_from_counts(items: list[dict], field: str) -> dict:
    rows = [item.get(field) for item in items if isinstance(item.get(field), dict)
            and int((item.get(field) or {}).get("gt") or 0) > 0]
    values = {"precision": [], "recall": [], "f1": []}
    for row in rows:
        gt = int(row.get("gt") or 0)
        predicted = int(row.get("predicted") or 0)
        matched = int(row.get("matched") or 0)
        precision = matched / predicted if predicted else 0.0
        recall = matched / gt
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        values["precision"].append(precision)
        values["recall"].append(recall)
        values["f1"].append(f1)
    return {metric: sum(samples) / len(samples) if samples else None
            for metric, samples in values.items()} | {"metric_count": len(rows)}


def _retrieval_metric_eligible(item: dict) -> bool:
    return str(item.get("answerability") or "").strip().lower() != "unanswerable"


def _resolve_gt_media(gt_refs: list[dict], assets_by_name: dict,
                      retrieved_media: list[dict]) -> list[dict]:
    asset_index: dict[tuple[str, str], list[dict]] = {}
    for file_name, assets in assets_by_name.items():
        for asset in assets:
            media_type = _infer_media_type(file_name, asset.get("media_type") or asset.get("asset_type"))
            asset_index.setdefault(_media_key(media_type, file_name), []).append(asset)
    retrieved_keys = {
        _media_key(item.get("media_type"), item.get("media_id") or item.get("file_name"))
        for item in retrieved_media
    }
    result = []
    for ref in gt_refs:
        media_type = ref["media_type"]
        media_id = ref["media_id"]
        key = _media_key(media_type, media_id)
        candidates = asset_index.get(key, [])
        file_name = Path(media_id).name
        if media_type == "video" and Path(file_name).suffix.lower() not in VIDEO_EXTENSIONS:
            file_name = f"{file_name}.mp4"
        result.append({
            "media_type": media_type,
            "media_id": media_id,
            "image_id": media_id if media_type == "image" else None,
            "video_id": media_id if media_type == "video" else None,
            "file_name": file_name,
            "asset_id": candidates[0].get("id") if len(candidates) == 1 else None,
            "matched": key in retrieved_keys,
            "mapping_status": "ok" if len(candidates) == 1 else "missing" if not candidates else "ambiguous",
        })
    return result


def _execution_failure(agent_status: str | None, termination_reason: str | None) -> bool:
    status = str(agent_status or "").lower()
    termination = str(termination_reason or "").lower()
    return status in {"partial", "error", "timeout", "failed", "cancelled", "canceled"} or any(
        marker in termination for marker in ("limit", "failure", "error", "timeout", "blocked", "parse_failure")
    )


def _agent_model_steps(response: dict) -> list[dict]:
    trace = response.get("retrieval_trace") or response.get("retrievalTrace") or []
    return [step for step in trace if isinstance(step, dict)
            and str(step.get("stage") or step.get("type") or "") == "model"]


def _derive_turn_outcome(response: dict) -> str | None:
    explicit = response.get("turn_outcome")
    if explicit:
        return str(explicit)
    termination = str(response.get("termination_reason") or response.get("terminationReason") or "").lower()
    if termination in {"model_step_limit", "tool_call_limit"}:
        return "step_limit"
    steps = _agent_model_steps(response)
    if steps:
        step = steps[-1]
        if step.get("turn_outcome"):
            return str(step["turn_outcome"])
        action = BenchmarkRun._trace_action(step)
        if action:
            return "final_answer" if action.get("action") == "final" else "tool_call"
        if str(step.get("status") or "").lower() == "error":
            reason = str(step.get("reason") or "").lower()
            return "context_blocked" if any(marker in reason for marker in ("context", "token budget", "preflight")) else "model_error"
        return "parse_failure"
    return "final_answer" if response.get("answer") and not _execution_failure(
        response.get("tool_loop_status"), termination) else None


def _derive_parse_status(response: dict) -> str:
    steps = _agent_model_steps(response)
    if not steps:
        return "not_applicable"
    step = steps[-1]
    if step.get("parse_status"):
        return str(step["parse_status"])
    return "success" if BenchmarkRun._trace_action(step) else (
        "not_applicable" if str(step.get("status") or "").lower() == "error" else "failed")


def _derive_next_step(response: dict) -> str:
    steps = _agent_model_steps(response)
    if not steps:
        return "stop"
    step = steps[-1]
    if step.get("next_step"):
        return str(step["next_step"])
    action = BenchmarkRun._trace_action(step)
    if action and action.get("action") == "tool_call":
        return str(action.get("tool") or "tool")
    return "final_answer" if action and action.get("action") == "final" else "stop"


def _inline_judge_images(images: list[dict], assets_by_name: dict, sentrix_url: str,
                         max_dimension: int = 896) -> list[dict]:
    """Build bounded data URLs through Sentrix asset HTTP endpoints."""
    assets_by_id = {
        str(asset["id"]): asset
        for assets in assets_by_name.values()
        for asset in assets
        if asset.get("id")
    }
    content = []
    for image in images:
        asset = assets_by_id.get(str(image.get("asset_id") or ""))
        asset_id = str((asset or {}).get("id") or image.get("asset_id") or "")
        if not asset_id:
            continue
        try:
            from PIL import Image
            image_bytes = request_bytes(f"{sentrix_url.rstrip('/')}/api/assets/{quote(asset_id)}/file", timeout=60)
            with Image.open(BytesIO(image_bytes)) as source:
                resized = source.convert("RGB")
                resized.thumbnail((max_dimension, max_dimension))
                output = BytesIO()
                resized.save(output, format="JPEG", quality=90, optimize=True)
            content.append({
                "type": "image_url",
                "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(output.getvalue()).decode("ascii")},
            })
        except Exception:
            continue
    return content


def safe_slug(value: str, fallback: str = "x") -> str:
    import re
    slug = re.sub(r"[^a-zA-Z0-9._-]+", "-", value.strip()).strip("-._").lower()
    return slug or fallback


def album_media_entries(manifest: dict) -> list[str]:
    """Return photo then video paths from an album manifest, de-duplicated."""
    entries = []
    seen = set()
    for key in ("photos", "videos"):
        for item in manifest.get(key) or []:
            relative = str(item).strip()
            if relative and relative not in seen:
                seen.add(relative)
                entries.append(relative)
    return entries


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def load_task_action_policy() -> dict[tuple[str, str], str]:
    """Load versioned defaults for QA sets that only contain answerable prompts."""
    try:
        value = json.loads(TASK_ACTION_POLICY_PATH.read_text(encoding="utf-8"))
        defaults = value.get("defaults") or []
        result = {}
        for entry in defaults:
            album_id = str(entry.get("album_id") or "")
            qa_set = str(entry.get("qa_set") or "")
            action = str(entry.get("expected_action") or "")
            if album_id and qa_set and action in {"answer", "refuse", "clarify"}:
                result[(album_id, qa_set)] = action
        return result
    except (OSError, TypeError, json.JSONDecodeError):
        return {}


TASK_ACTION_DEFAULTS = load_task_action_policy()


def apply_task_action_defaults(rows: list[dict], album_id: str, qa_set: str) -> list[dict]:
    """Keep explicit per-question labels, otherwise apply the tracked QA-set default."""
    default = TASK_ACTION_DEFAULTS.get((album_id, qa_set))
    if default is None:
        return rows
    return [{**row, "expected_action": row.get("expected_action") or default} for row in rows]


def nearest_rank_percentile(values: list[int | float], percentile: float) -> int | float | None:
    """Return the observed value at the nearest-rank percentile."""
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, math.ceil(len(ordered) * percentile) - 1))
    return ordered[index]


def sha256_file(path: Path) -> str:
    """Return a stable digest without retaining the source file in memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def dataset_integrity(album_dir: Path, manifest: dict, qa_set: str) -> dict:
    """Record the exact local inputs used by a run for later reproducibility checks."""
    manifest_path = album_dir / "manifest.json"
    qa_path = album_dir / str((manifest.get("qa_sets") or {})[qa_set])
    entries = ["manifest.json", str(qa_path.relative_to(album_dir))]
    entries.extend(album_media_entries(manifest))
    entries.extend(str(face.get("ref_image")) for face in manifest.get("faces") or [] if face.get("ref_image"))
    missing, file_digests = [], []
    for relative in sorted(set(entries)):
        path = album_dir / relative
        if not path.is_file():
            missing.append(relative)
            continue
        file_digests.append({"path": relative, "sha256": sha256_file(path), "bytes": path.stat().st_size})
    combined = hashlib.sha256("\n".join(
        f"{entry['path']}:{entry['sha256']}" for entry in file_digests
    ).encode()).hexdigest()
    return {
        "algorithm": "sha256",
        "manifest_sha256": sha256_file(manifest_path),
        "qa_sha256": sha256_file(qa_path),
        "task_action_policy_sha256": sha256_file(TASK_ACTION_POLICY_PATH)
            if TASK_ACTION_POLICY_PATH.is_file() else None,
        "dataset_sha256": combined,
        "files_checked": len(file_digests),
        "missing_files": missing,
    }


# ---------------------------------------------------------------------------
# GPU Sampler
# ---------------------------------------------------------------------------

class GpuSampler:
    """Poll device and managed-model memory at intervals."""

    def __init__(self, api_url: str, interval: float = 0.5, on_sample=None):
        self.api_url = api_url
        self.interval = interval
        self.on_sample = on_sample
        self.samples: list[dict] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self):
        self._stop.clear()
        self.samples = []
        self._thread = threading.Thread(target=self._run, daemon=True, name="gpu-sampler")
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=10)

    def _run(self):
        while not self._stop.is_set():
            try:
                data = request_json(f"{self.api_url}/gpu-stats", timeout=10)
                try:
                    process_memory = request_json(f"{self.api_url}/process-memory", timeout=5)
                except Exception:
                    process_memory = {}
                ts = time.perf_counter()
                for gpu in data.get("gpus", []):
                    sample = dict(gpu)
                    sample["_t"] = ts
                    sample["model_process_memory_used_mib"] = process_memory.get("process_memory_used_mib")
                    sample["model_process_memory_limit_mib"] = process_memory.get("process_memory_limit_mib")
                    sample["model_process_memory_over_limit"] = process_memory.get("process_memory_over_limit")
                    sample["model_process_pid"] = process_memory.get("root_pid")
                    vllm_metrics = process_memory.get("vllm_metrics") or {}
                    sample["kv_cache_usage_pct"] = vllm_metrics.get("kv_cache_usage_pct")
                    memory_profile = process_memory.get("memory_profile") or {}
                    sample["kv_cache_capacity_gib"] = memory_profile.get("kv_cache_capacity_gib")
                    sample["kv_cache_capacity_tokens"] = memory_profile.get("kv_cache_capacity_tokens")
                    sample["weight_gib"] = memory_profile.get("weight_gib")
                    sample["peak_activation_gib"] = memory_profile.get("peak_activation_gib")
                    sample["non_torch_gib"] = memory_profile.get("non_torch_gib")
                    sample["cuda_graph_gib"] = memory_profile.get("cuda_graph_gib")
                    self.samples.append(sample)
                    if self.on_sample:
                        try:
                            self.on_sample(sample)
                        except Exception:
                            pass
            except Exception:
                pass
            self._stop.wait(self.interval)

    def aggregate(self) -> dict:
        if not self.samples:
            return {"samples_count": 0}
        metrics = {}
        for key in (
            "temperature_c", "gpu_utilization_pct", "memory_used_mib",
            "model_process_memory_used_mib", "kv_cache_usage_pct",
            "power_draw_w", "sm_clock_mhz",
        ):
            values = [
                s[key] for s in self.samples
                if isinstance(s.get(key), (int, float))
            ]
            if not values:
                continue
            values_sorted = sorted(values)
            n = len(values_sorted)
            metrics[key] = {
                "peak": round(values_sorted[-1], 1),
                "mean": round(sum(values) / n, 1),
                "p50": round(values_sorted[n // 2], 1),
                "p95": round(values_sorted[int(n * 0.95)], 1) if n >= 20 else round(values_sorted[-1], 1),
            }
        limits = [sample.get("model_process_memory_limit_mib") for sample in self.samples
                  if isinstance(sample.get("model_process_memory_limit_mib"), (int, float))]
        over_limit_flags = [sample.get("model_process_memory_over_limit") for sample in self.samples
                            if isinstance(sample.get("model_process_memory_over_limit"), bool)]
        latest = self.samples[-1]
        kv_capacity_gib = latest.get("kv_cache_capacity_gib")
        process_memory = metrics.get("model_process_memory_used_mib") or {}
        kv_usage = metrics.get("kv_cache_usage_pct") or {}
        idle_samples = [
            sample.get("model_process_memory_used_mib") for sample in self.samples
            if isinstance(sample.get("model_process_memory_used_mib"), (int, float))
            and isinstance(sample.get("kv_cache_usage_pct"), (int, float))
            and sample.get("kv_cache_usage_pct") == 0
        ]
        idle_process_mib = min(idle_samples) if idle_samples else process_memory.get("p50")
        fixed_base_gib = (
            max(0.0, idle_process_mib / 1024 - kv_capacity_gib)
            if isinstance(idle_process_mib, (int, float)) and isinstance(kv_capacity_gib, (int, float))
            else None
        )
        kv_peak_gib = (
            kv_capacity_gib * float(kv_usage.get("peak") or 0) / 100
            if isinstance(kv_capacity_gib, (int, float)) and kv_usage else None
        )
        return {
            "samples_count": len(self.samples),
            "model_process_memory_limit_mib": limits[-1] if limits else None,
            "model_process_over_limit_samples": sum(over_limit_flags) if over_limit_flags else None,
            "memory_profile": {
                "method": "idle_process_minus_reserved_kv_plus_peak_used_kv_v1",
                "idle_process_memory_gib": round(idle_process_mib / 1024, 4)
                    if isinstance(idle_process_mib, (int, float)) else None,
                "kv_cache_capacity_gib": kv_capacity_gib,
                "kv_cache_capacity_tokens": latest.get("kv_cache_capacity_tokens"),
                "kv_cache_usage_peak_pct": kv_usage.get("peak"),
                "kv_cache_used_peak_gib": round(kv_peak_gib, 4) if kv_peak_gib is not None else None,
                "fixed_base_memory_gib": round(fixed_base_gib, 4) if fixed_base_gib is not None else None,
                "comparable_workload_memory_gib": round(fixed_base_gib + kv_peak_gib, 4)
                    if fixed_base_gib is not None and kv_peak_gib is not None else None,
                "weight_gib": latest.get("weight_gib"),
                "peak_activation_gib": latest.get("peak_activation_gib"),
                "non_torch_gib": latest.get("non_torch_gib"),
                "cuda_graph_gib": latest.get("cuda_graph_gib"),
                "formula": "fixed_base = idle_process - kv_capacity; comparable = fixed_base + kv_capacity * kv_peak_pct",
            },
            **metrics,
        }


# ---------------------------------------------------------------------------
# Benchmark Run
# ---------------------------------------------------------------------------

class BenchmarkRun:
    """A single benchmark run for a single model."""

    def __init__(self, run_id: str, album_id: str, manifest: dict, model_profile: str,
                 qa_set: str, sentrix_url: str, judge_url: str, vllm_api_url: str,
                 vllm_target_id: str, vllm_model_base_url: str, results_root: Path,
                 judge_system_prompt: str = JUDGE_PROMPT, judge_model: str = JUDGE_MODEL,
                 judge_api_key: str = JUDGE_API_KEY, delete_scope_after_run: bool = False,
                 task_judge_system_prompt: str = "", evidence_judge_system_prompt: str = "",
                 mode: str = "full", existing_scope_id: str = "",
                 scope_reused_from_runs: list | None = None,
                 use_current_model: bool = False, current_model_snapshot: dict | None = None,
                 use_cloud_model: bool = False):
        if mode not in RUN_MODES:
            raise ValueError(f"mode must be one of {sorted(RUN_MODES)}, got: {mode!r}")
        if mode == "reuse" and not existing_scope_id:
            raise ValueError("existing_scope_id is required when mode=reuse")
        self.mode = mode
        self.existing_scope_id = str(existing_scope_id).strip()
        self.run_id = run_id
        self.album_id = album_id
        self.manifest = manifest
        self.model_profile = model_profile
        self.use_current_model = bool(use_current_model)
        self.use_cloud_model = bool(use_cloud_model)
        self.current_model_snapshot = dict(current_model_snapshot or {})
        self.qa_set = qa_set
        self.sentrix_url = sentrix_url.rstrip("/")
        base = judge_url.rstrip("/")
        # Cloud providers (e.g. DashScope) include /v1 in the base URL;
        # local LM Studio servers do not. Normalize to avoid double /v1.
        self.judge_url = base
        self.judge_model = judge_model
        self.judge_api_key = judge_api_key
        self.judge_system_prompt = judge_system_prompt
        self.task_judge_system_prompt = task_judge_system_prompt
        self.evidence_judge_system_prompt = evidence_judge_system_prompt
        self.vllm_api_url = vllm_api_url.rstrip("/")
        self.vllm_target_id = vllm_target_id
        self.vllm_model_base_url = vllm_model_base_url.rstrip("/")
        self.results_root = results_root
        self.lock = threading.RLock()
        self._judge_rate_lock = threading.Lock()
        self._judge_next_request_at = 0.0
        # 复用/构建模式的产物相册必须保留，绝不允许 delete_scope 清掉。
        self.delete_scope_after_run = bool(delete_scope_after_run) and mode == "full"

        album_base = BENCHMARK_DATA_ROOT / album_id
        self.album_dir = album_base
        self.qa_rows = apply_task_action_defaults(
            load_jsonl(album_base / manifest["qa_sets"][qa_set]), album_id, qa_set,
        )
        self.input_integrity = dataset_integrity(album_base, manifest, qa_set)

        self.state: dict = {
            "run_id": run_id,
            "mode": mode,
            "scope_source": None,
            "album_id": album_id,
            "model_profile": model_profile,
            "model_source": "cloud_api" if self.use_cloud_model else (
                "current" if self.use_current_model else "managed"
            ),
            "model_backend": "openai" if self.use_cloud_model else "vllm",
            "model_name": BIG_MODEL_MODEL if self.use_cloud_model else model_profile,
            "current_model_snapshot": self.current_model_snapshot or None,
            "qa_set": qa_set,
            "judge_model": judge_model,
            "judge_url": self.judge_url,
            "delete_scope_after_run": self.delete_scope_after_run,
            "vllm_target_id": vllm_target_id,
            "vllm_manager_url": vllm_api_url,
            "vllm_model_base_url": vllm_model_base_url,
            "qa_count": len(self.qa_rows),
            "input_integrity": self.input_integrity,
            "hardware_snapshots": {"start": None, "end": None},
            "status": "pending",
            "created_at": now_iso(),
            "started_at": None,
            "finished_at": None,
           "scope_id": None,
           "scope_name": None,
           "existing_scope_id": self.existing_scope_id or None,
           "scope_reused_from_runs": list(scope_reused_from_runs or []),
           "phases": {},
            "items": [],
            "summary": {},
            "run_valid": False,
            "fatal_error": None,
        }
        self._gpu_sampling_started = False
        self._gpu_sampler = GpuSampler(vllm_api_url, on_sample=self._persist_gpu_sample)
        self._cancel = threading.Event()
        self._phase_started_perf: dict[str, float] = {}
        self._persist_condition = threading.Condition(self.lock)
        self._persist_requested = 0
        self._persist_written = 0
        self._persist_error: Exception | None = None
        self._persist_stopping = False
        self._persist_thread = threading.Thread(
            target=self._persist_loop,
            name=f"persist-{run_id}",
            daemon=True,
        )
        self._persist_thread.start()

    @property
    def run_dir(self) -> Path:
        return self.results_root / self.run_id

    def _write_persisted_snapshot(self, stored: dict) -> None:
        atomic_json(self.run_dir / "run.json", stored)
        items = stored.get("items") or []
        path = self.run_dir / "results.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in items), encoding="utf-8")

    def _persist_loop(self) -> None:
        while True:
            with self._persist_condition:
                while self._persist_written >= self._persist_requested and not self._persist_stopping:
                    self._persist_condition.wait()
                if self._persist_stopping and self._persist_written >= self._persist_requested:
                    return
                # Coalesce progress updates arriving in the same short interval.
                target = self._persist_requested
                self._persist_condition.wait(timeout=PERSIST_DEBOUNCE_SECONDS)
                target = self._persist_requested
                stored = copy.deepcopy(self.state)
            try:
                self._write_persisted_snapshot(stored)
            except Exception as exc:
                with self._persist_condition:
                    self._persist_error = exc
                    self._persist_written = max(self._persist_written, target)
                    self._persist_condition.notify_all()
                continue
            with self._persist_condition:
                self._persist_written = max(self._persist_written, target)
                self._persist_condition.notify_all()

    def persist(self, wait: bool = False) -> None:
        """Schedule a snapshot; optionally wait until this request reaches disk."""
        with self._persist_condition:
            if self._persist_error is not None and wait:
                raise RuntimeError(f"benchmark state persistence failed: {self._persist_error}")
            self._persist_requested += 1
            target = self._persist_requested
            self._persist_condition.notify()
            if not wait:
                return
            while self._persist_written < target:
                self._persist_condition.wait()
            if self._persist_error is not None:
                raise RuntimeError(f"benchmark state persistence failed: {self._persist_error}")

    def _stop_persist_writer(self) -> None:
        self.persist(wait=True)
        with self._persist_condition:
            self._persist_stopping = True
            self._persist_condition.notify_all()
        self._persist_thread.join(timeout=10)

    def _gpu_samples_path(self) -> Path:
        return self.run_dir / "gpu_samples.jsonl"

    def _reset_gpu_samples_file(self) -> None:
        path = self._gpu_samples_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("", encoding="utf-8")

    def _persist_gpu_sample(self, sample: dict) -> None:
        """Append each sample immediately so cancellation does not lose in-memory data."""
        with self.lock:
            with self._gpu_samples_path().open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(sample, ensure_ascii=False) + "\n")

    def cancel(self, source: str = "api"):
        self._cancel.set()
        self.state["cancel_requested_at"] = now_iso()
        self.state["cancel_source"] = source
        if self.state.get("status") == "pending":
            self.state["status"] = "cancelled"
            self.state["finished_at"] = now_iso()
        else:
            self.state["status"] = "cancelling"
        self._cancel_remote_batch(source)
        self._reclaim_vllm_after_cancel()
        self.persist(wait=True)

    def _reclaim_vllm_after_cancel(self) -> None:
        """Best-effort: terminate a vLLM instance this run started (loading or serving).

        Without this, cancelling during model_deploy leaves the load running on the
        GPU and the manager keeps serving a model nobody asked for.
        """
        if self.use_current_model or self.use_cloud_model:
            return
        try:
            state = request_json(f"{self.vllm_api_url}/state", timeout=5) or {}
        except Exception:
            return
        run_scope_models = state.get("profile")
        if run_scope_models and run_scope_models == self.model_profile:
            try:
                request_json(f"{self.vllm_api_url}/stop", {"timeout": 60}, "POST", 90)
                self.state["cancel_vllm_stopped"] = now_iso()
            except Exception as exc:
                self.state["cancel_vllm_stop_error"] = str(exc)[:300]

    def _cancel_remote_batch(self, source: str) -> None:
        batch_id = self.state.get("batch_id")
        if batch_id:
            try:
                request_json(
                    f"{self.sentrix_url}/api/ingest-batches/{quote(str(batch_id))}/cancel",
                    {"source": source}, "POST", 15,
                )
                self.state["batch_cancel_requested"] = True
                self.state.pop("batch_cancel_error", None)
            except Exception as exc:
                self.state["batch_cancel_error"] = str(exc)

    def _record_phase(self, phase: str, key, value):
        with self.lock:
            if phase not in self.state["phases"]:
                self.state["phases"][phase] = {"status": "pending"}
            self.state["phases"][phase][key] = value

    def _phase_start(self, phase: str):
        self._phase_started_perf[phase] = time.perf_counter()
        self._record_phase(phase, "status", "running")
        self._record_phase(phase, "started_at", now_iso())

    def _phase_done(self, phase: str, extra: dict | None = None):
        self._record_phase(phase, "status", "done")
        self._record_phase(phase, "finished_at", now_iso())
        started = self._phase_started_perf.pop(phase, None)
        if started is not None and not (extra and "total_seconds" in extra):
            self._record_phase(phase, "total_seconds", round(time.perf_counter() - started, 3))
        if extra:
            for k, v in extra.items():
                self._record_phase(phase, k, v)

    def _phase_partial(self, phase: str, extra: dict | None = None):
        self._phase_done(phase, extra)
        self._record_phase(phase, "status", "partial")

    def _hardware_snapshot(self) -> dict:
        """Use existing Manager endpoints; absence is recorded, never inferred from logs."""
        if self.use_cloud_model:
            return {
                "captured_at": now_iso(),
                "source": "cloud_api",
                "status": "not_applicable",
                "reason": "cloud_api_has_no_local_gpu_metrics",
            }
        if not self.vllm_api_url:
            return {
                "captured_at": now_iso(),
                "source": "external",
                "status": "not_applicable",
                "reason": "external model endpoint has no manager metrics",
            }
        snapshot = {"captured_at": now_iso(), "manager": None, "gpu": None, "process_memory": None}
        try:
            snapshot["manager"] = request_json(f"{self.vllm_api_url}/state", timeout=10)
        except Exception as exc:
            snapshot["manager_error"] = str(exc)
        try:
            snapshot["gpu"] = request_json(f"{self.vllm_api_url}/gpu-stats", timeout=10).get("gpus") or []
        except Exception as exc:
            snapshot["gpu_error"] = str(exc)
        try:
            snapshot["process_memory"] = request_json(f"{self.vllm_api_url}/process-memory", timeout=10)
        except Exception as exc:
            snapshot["process_memory_error"] = str(exc)
        return snapshot

    def execute(self):
        if self._cancel.is_set():
            self.state["status"] = "cancelled"
            self.state["finished_at"] = self.state.get("finished_at") or now_iso()
            self.persist(wait=True)
            self._stop_persist_writer()
            return
        self.state["started_at"] = now_iso()
        self.state["status"] = "running"
        self.state["hardware_snapshots"]["start"] = self._hardware_snapshot()
        self._current_phase = None
        self.persist()
        all_phases = [
            ("model_deploy", self._phase_model_deploy),
            ("scope_setup", self._phase_scope_setup),
            ("scope_attach", self._phase_scope_attach),
            ("identity_seed", self._phase_identity_seed),
            ("photo_import", self._phase_photo_import),
            ("pipeline_processing", self._phase_processing),
            ("qa_eval", self._phase_qa_eval),
            ("gpu_metrics", self._phase_gpu_metrics),
            ("aggregate", self._phase_aggregate),
        ]
        selected_phase_names = self._selected_phase_names()
        phases = [(name, fn) for name, fn in all_phases if name in selected_phase_names]
        try:
            for name, fn in phases:
                if self._cancel.is_set():
                    break
                self._current_phase = name
                fn()
            if self._cancel.is_set():
                if self._current_phase:
                    self._record_phase(self._current_phase, "status", "cancelled")
                self.state["status"] = "cancelled"
            else:
                has_partial_phase = any(
                    phase.get("status") == "partial"
                    for phase in (self.state.get("phases") or {}).values()
                )
                self.state["status"] = "completed_with_errors" if has_partial_phase else "completed"
        except RunCancelledError:
            if self._current_phase:
                self._record_phase(self._current_phase, "status", "cancelled")
            self.state["status"] = "cancelled"
        except Exception as e:
            if self._current_phase:
                phase_state = self.state["phases"].get(self._current_phase) or {}
                if phase_state.get("status") != "stalled":
                    self._record_phase(self._current_phase, "status", "failed")
                self._record_phase(self._current_phase, "error", str(e))
                self._record_phase(self._current_phase, "finished_at", now_iso())
            self.state["status"] = "failed"
            self.state["failed_phase"] = self._current_phase
            self.state["fatal_error"] = str(e)
            traceback.print_exc()
        finally:
            if not self.use_cloud_model and self.vllm_api_url:
                self._gpu_sampler.stop()
            self.state["hardware_snapshots"]["end"] = self._hardware_snapshot()
            gpu_phase = self.state["phases"].get("gpu_metrics") or {}
            if self._gpu_sampling_started and gpu_phase.get("status") != "done":
                partial = self._gpu_sampler.aggregate()
                partial.update({"status": "partial", "partial": True, "finished_at": now_iso()})
                self.state["phases"]["gpu_metrics"] = partial
            self.state["finished_at"] = now_iso()
            self.persist(wait=True)
            if self.delete_scope_after_run:
                self._cleanup_scope()
            self._stop_persist_writer()

    def _selected_phase_names(self) -> list[str]:
        # 工作模式决定阶段编排；当前模型模式不拥有模型生命周期，因此没有部署阶段。
        phase_names_by_mode = {
            "full": ["model_deploy", "scope_setup", "identity_seed", "photo_import",
                     "pipeline_processing", "qa_eval", "gpu_metrics", "aggregate"],
            "build": ["model_deploy", "scope_setup", "identity_seed", "photo_import",
                      "pipeline_processing", "gpu_metrics", "aggregate"],
            "reuse": ["model_deploy", "scope_attach", "qa_eval", "gpu_metrics", "aggregate"],
        }
        selected = phase_names_by_mode[self.mode]
        if self.use_current_model:
            selected = [name for name in selected if name != "model_deploy"]
        return selected

    def _cleanup_scope(self):
        """Delete the PhotoBench-created memory space after the run finishes."""
        scope_id = self.state.get("scope_id")
        if not scope_id:
            self.state["scope_cleanup"] = {"status": "skipped", "reason": "no_scope_id"}
            return
        try:
            result = request_json(
                f"{self.sentrix_url}/api/memory-spaces/{quote(scope_id)}",
                method="DELETE", timeout=180,
            )
            self.state["scope_cleanup"] = {
                "status": "deleted", "scope_id": scope_id,
                "removed": (result or {}).get("removed") or {},
            }
        except Exception as exc:
            self.state["scope_cleanup"] = {
                "status": "failed", "scope_id": scope_id, "error": str(exc),
            }
        self.persist(wait=True)

    # ---- Phase implementations ----

    def _wait_model_ready(self, ready_timeout: int = 600, health_timeout: int = 180):
        """Poll manager state + model endpoint until ready; cancel stops the load."""
        deadline = time.monotonic() + ready_timeout
        while time.monotonic() < deadline:
            if self._cancel.is_set():
                self._stop_managed_vllm()
                return RunCancelledError("cancelled while loading model")
            try:
                state = request_json(f"{self.vllm_api_url}/state", timeout=10) or {}
            except Exception as exc:
                state = {}
                if time.monotonic() >= deadline:
                    return exc
            served_names = {state.get("profile"), state.get("served_model_name")}
            if self.model_profile in served_names:
                base = state.get("external_url_hint") or f"http://192.168.0.153:{state.get('port', 8100)}/v1"
                root = base.rstrip("/").removesuffix("/v1")
                probe = self._probe_model_endpoint(state, root, timeout=20, once=True)
                if probe is None:
                    return None
            if self._cancel.wait(2):
                self._stop_managed_vllm()
                return RunCancelledError("cancelled while loading model")
        return TimeoutError(f"model not ready within {ready_timeout}s")

    def _probe_model_endpoint(self, state: dict, model_api_root: str, timeout: int = 180, once: bool = False):
        """Return None when the model endpoint answers; keep retrying until deadline."""
        deadline = time.monotonic() + timeout
        health_error = None
        while True:
            if self._cancel.is_set():
                return RunCancelledError("cancelled during model health check")
            try:
                request_json(f"{model_api_root}/v1/chat/completions",
                              {"model": state.get("served_model_name", self.model_profile),
                               "messages": [{"role": "user", "content": "hi"}], "max_tokens": 1},
                              "POST", 30)
                return None
            except Exception as exc:
                health_error = exc
                if once or time.monotonic() >= deadline:
                    if once:
                        return health_error
                    raise RuntimeError(
                        f"vLLM model endpoint did not become ready within {timeout}s: {health_error}"
                    ) from exc
                if self._cancel.wait(2):
                    return RunCancelledError("cancelled during model health check")

    def _stop_managed_vllm(self):
        try:
            request_json(f"{self.vllm_api_url}/stop", {"timeout": 60}, "POST", 90)
        except Exception:
            pass

    def _phase_model_deploy(self):
        self._phase_start("model_deploy")
        if self.use_cloud_model:
            runtime = request_json(
                f"{self.sentrix_url}/api/model-profiles/bind-cloud-runtime",
                {"profile": BIG_MODEL_PROFILE_ID}, "POST", 30,
            )
            self._phase_done("model_deploy", {
                "source": "cloud_api",
                "deployment_mode": "remote_api",
                "manager_status": "not_applicable",
                "health_check": "skipped",
                "model_probe": "skipped",
                "runtime": runtime,
            })
            return
        t0 = time.perf_counter()
        # 1. Stop (unload)
        t_stop0 = time.perf_counter()
        try:
            request_json(f"{self.vllm_api_url}/stop", {"timeout": 60}, "POST", 90)
        except Exception:
            pass
        t_stop = time.perf_counter() - t_stop0

        # 1b. Cooldown — give 153 GPU time to free VRAM before next load
        if self._cancel.wait(5):
            raise RunCancelledError("cancelled during cooldown")

        # 2. Start (load to VRAM) — exponential backoff retry on transient 502/timeout
        backoff_schedule = [10, 30, 60, 180]  # seconds between attempts
        max_attempts = len(backoff_schedule)
        t_load0 = time.perf_counter()
        start_error = None
        for attempt in range(max_attempts):
            if self._cancel.is_set():
                raise RunCancelledError("cancelled before model start")
            try:
                # Fire-and-poll: wait_ready=false returns immediately after spawn;
                # readiness is tracked below with cancel-aware polling so a stop
                # request can terminate a half-loaded model instead of blocking.
                request_json(f"{self.vllm_api_url}/start",
                              {"profile": self.model_profile, "wait_ready": False},
                              "POST", 120)
                start_error = self._wait_model_ready()
                if start_error is None:
                    break
            except RunCancelledError:
                raise
            except Exception as exc:
                start_error = exc
            if attempt < max_attempts - 1:
                wait = backoff_schedule[attempt]
                if self._cancel.wait(wait):
                    raise RunCancelledError("cancelled during start backoff")
        if start_error is not None:
            raise RuntimeError(
                f"vLLM /start failed after {max_attempts} attempts: {start_error}"
            ) from start_error
        t_load = time.perf_counter() - t_load0

        # 3. Health check (cancel-aware)
        state = request_json(f"{self.vllm_api_url}/state", timeout=10)
        port = state.get("port", 8105)
        base = state.get("external_url_hint") or f"http://192.168.0.153:{port}/v1"
        model_api_root = base.rstrip("/").removesuffix("/v1")
        t_health0 = time.perf_counter()
        health_error = self._probe_model_endpoint(state, model_api_root)
        if health_error is not None:
            raise health_error
        t_health = time.perf_counter() - t_health0

        # 4. Sync .100 Sentrix gamma client to the new model (no restart)
        sync_ok = False
        sync_error = None
        try:
            request_json(f"{self.sentrix_url}/api/model-profiles/bind-runtime",
                         {"manager_url": self.vllm_api_url,
                          "model_base_url": self.vllm_model_base_url}, "POST", 30)
            sync_ok = True
        except Exception as e:
            sync_error = str(e)

        # 5. Model probe (streaming TTFT + throughput)
        t_probe0 = time.perf_counter()
        try:
            probe = model_probe(model_api_root, state.get("served_model_name", self.model_profile))
        except Exception as e:
            probe = {"error": str(e)}
        t_probe = time.perf_counter() - t_probe0

        self._phase_done("model_deploy", {
            "unload_seconds": round(t_stop, 1),
            "load_seconds": round(t_load, 1),
            "health_check_seconds": round(t_health, 1),
            "model_probe": probe,
            "gamma_sync": {"ok": sync_ok, "error": sync_error},
            "model_state": {
                "profile": state.get("profile"),
                "served_name": state.get("served_model_name"),
                "model_path": state.get("model"),
                "port": port,
                "dtype": state.get("dtype"),
                "quantization": state.get("quantization"),
                "load_format": state.get("load_format"),
                "gpu_memory_utilization": state.get("gpu_memory_utilization"),
                "max_model_len": state.get("max_model_len"),
                "max_num_seqs": state.get("max_num_seqs"),
                "max_num_batched_tokens": state.get("max_num_batched_tokens"),
                "default_max_tokens": state.get("default_max_tokens"),
                "tensor_parallel_size": state.get("tensor_parallel_size"),
                "kv_cache_dtype": state.get("kv_cache_dtype"),
                "enable_lora": state.get("enable_lora"),
                "lora_modules": state.get("lora_modules", []),
            },
        })

    def _phase_scope_setup(self):
        self._phase_start("scope_setup")
        t0 = time.perf_counter()
        # Auto-name: PhotoBench-{timestamp}-{album}-{model}
        ts_short = datetime.now().strftime("%Y%m%d-%H%M%S")
        scope_name = f"PhotoBench-{ts_short}-{safe_slug(self.album_id)}-{safe_slug(self.model_profile)}"
        result = request_json(f"{self.sentrix_url}/api/memory-spaces",
                              {"name": scope_name}, "POST", 30)
        scope_id = result.get("id") or result.get("scope_id")
        self.state["scope_id"] = scope_id
        self.state["scope_name"] = result.get("name") or scope_id
        self.state["scope_source"] = "created"
        t1 = time.perf_counter()
        self._phase_done("scope_setup", {"scope_id": scope_id, "scope_name": self.state["scope_name"], "create_seconds": round(t1 - t0, 3)})

    def _phase_scope_attach(self):
        """reuse 模式：绑定一个已存在的相册 scope，不创建、不删除。"""
        self._phase_start("scope_attach")
        t0 = time.perf_counter()
        scope_id = self.existing_scope_id
        # 后端没有 GET /api/memory-spaces/{id} 单查（405），从列表中定位。
        spaces = request_json(f"{self.sentrix_url}/api/memory-spaces", timeout=30)
        if isinstance(spaces, dict):
            spaces = spaces.get("spaces") or spaces.get("items") or []
        result = next((s for s in (spaces or []) if s.get("id") == scope_id), None)
        if not result:
            raise ValueError(f"memory space not found on backend: {scope_id}")
        self.state["scope_id"] = scope_id
        self.state["scope_name"] = result.get("name") or scope_id
        self.state["scope_source"] = "reused"
        t1 = time.perf_counter()
        self._phase_done("scope_attach", {
            "scope_id": scope_id,
            "scope_name": self.state["scope_name"],
            "scope_kind": result.get("kind"),
            "scope_created_at": result.get("created_at"),
            "attach_seconds": round(t1 - t0, 3),
            "reused_from_runs": self.state.get("scope_reused_from_runs") or [],
        })

    def _phase_identity_seed(self):
        self._phase_start("identity_seed")
        t0 = time.perf_counter()
        faces = self.manifest["faces"]
        manifest_entries = []
        files = []
        file_idx = 0
        for face in faces:
            name = face["canonical_name"] or face.get("family_role") or (face.get("aliases") or ["未知"])[0]
            manifest_entries.append({
                "name": name,
                "family_role": face.get("family_role", ""),
                "aliases": face.get("aliases", []),
                "file_indices": [file_idx],
            })
            ref_path = self.album_dir / face["ref_image"]
            files.append(("files", ref_path.name, ref_path.read_bytes()))
            file_idx += 1

        result = upload_files(
            f"{self.sentrix_url}/api/people/seed-batch",
            {"scope_id": self.state["scope_id"], "manifest": json.dumps(manifest_entries, ensure_ascii=False)},
            files, 120,
        )
        t1 = time.perf_counter()
        seeded = result.get("results", [])
        entity_by_name = {
            str(entry.get("name") or ""): str(entry.get("entity_id") or "")
            for entry in seeded if not entry.get("error") and entry.get("entity_id")
        }
        relationship_import = {"status": "not_configured", "requested": 0}
        relationships = self.manifest.get("family_relationships") or []
        if relationships:
            relationship_import = request_json(
                f"{self.sentrix_url}/api/relationships/batch",
                {
                    "scope_id": self.state["scope_id"],
                    "relationships": relationships,
                    "entity_by_name": entity_by_name,
                },
                "POST", 60,
            )
        self._phase_done("identity_seed", {
            "upload_seconds": round(t1 - t0, 3),
            "seeded_count": len(seeded),
            "details": seeded,
            "family_relationship_import": relationship_import,
        })

    def _phase_photo_import(self):
        self._phase_start("photo_import")
        t0 = time.perf_counter()
        media_relpaths = album_media_entries(self.manifest)
        total_media_count = len(media_relpaths)
        media_paths = [self.album_dir / p for p in media_relpaths]
        missing = [rel for rel, path in zip(media_relpaths, media_paths) if not path.is_file()]
        media_paths = [path for path in media_paths if path.is_file()]
        photo_count = len(self.manifest.get("photos") or [])
        video_count = len(self.manifest.get("videos") or [])
        chunk_size = max(1, int(os.getenv("PHOTOBENCH_IMPORT_CHUNK_SIZE", "8")))
        upload_workers = max(1, int(os.getenv("PHOTOBENCH_IMPORT_UPLOAD_WORKERS", "2")))
        max_upload_attempts = max(1, int(os.getenv("PHOTOBENCH_IMPORT_MAX_ATTEMPTS", "3")))
        batch_id = f"batch_{uuid.uuid4().hex[:12]}"
        self.state["batch_id"] = batch_id
        self.persist()
        items = [{
            "accepted": False,
            "fileName": rel,
            "status": "missing",
            "error_type": "FileNotFoundError",
            "error": "Media file listed in manifest does not exist",
        } for rel in missing]

        def upload_chunk(chunk_index, chunk):
            # Read bytes inside the bounded worker. Queued chunks retain only
            # paths, so a large album does not become a second in-memory copy.
            pending_paths = list(chunk)
            results_by_name = {}
            for attempt in range(1, max_upload_attempts + 1):
                try:
                    files = [("files", path.name, path.read_bytes()) for path in pending_paths]
                    result = upload_files(
                        f"{self.sentrix_url}/api/import",
                        {"scope_id": self.state["scope_id"], "batch_id": batch_id,
                         "deferBatchComplete": "true"},
                        files, 600,
                    )
                    returned = {
                        str(item.get("fileName") or item.get("file_name") or ""): item
                        for item in (result.get("items") or []) if isinstance(item, dict)
                    }
                except Exception as exc:
                    returned = {
                        path.name: {
                            "accepted": False,
                            "fileName": path.name,
                            "status": "failed",
                            "error_type": type(exc).__name__,
                            "error": str(exc),
                        }
                        for path in pending_paths
                    }

                retry_paths = []
                for path in pending_paths:
                    item = dict(returned.get(path.name) or {
                        "accepted": False,
                        "fileName": path.name,
                        "status": "failed",
                        "error_type": "missing_response_item",
                        "error": "Sentrix import response omitted this media file",
                    })
                    item["upload_attempts"] = attempt
                    if item.get("accepted") or item.get("status") == "rejected" or attempt == max_upload_attempts:
                        results_by_name[path.name] = item
                    else:
                        retry_paths.append(path)
                if not retry_paths:
                    break
                time.sleep(min(2 ** (attempt - 1), 4))
                pending_paths = retry_paths
            return chunk_index, {"items": [results_by_name[path.name] for path in chunk]}

        chunks = [
            (index, media_paths[offset:offset + chunk_size])
            for index, offset in enumerate(range(0, len(media_paths), chunk_size))
        ]
        results_by_chunk = {}
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=upload_workers, thread_name_prefix="photobench-upload"
        ) as executor:
            pending = {
                executor.submit(upload_chunk, chunk_index, chunk): (chunk_index, chunk)
                for chunk_index, chunk in chunks
            }
            for future in concurrent.futures.as_completed(tuple(pending)):
                chunk_index, chunk = pending.pop(future)
                try:
                    _, result = future.result()
                except Exception as exc:
                    result = {"items": [{
                        "accepted": False,
                        "fileName": path.name,
                        "status": "failed",
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    } for path in chunk]}
                results_by_chunk[chunk_index] = result
                completed_items = [
                    item for index in sorted(results_by_chunk)
                    for item in results_by_chunk[index].get("items", [])
                ]
                self._record_phase("photo_import", "total_photos", photo_count)
                self._record_phase("photo_import", "total_videos", video_count)
                self._record_phase("photo_import", "total_media", total_media_count)
                self._record_phase(
                    "photo_import", "accepted_count",
                    sum(1 for item in completed_items if item.get("accepted")),
                )
                if self._cancel.is_set():
                    for remaining in pending:
                        remaining.cancel()
                    break

        if self._cancel.is_set():
            self._cancel_remote_batch(self.state.get("cancel_source") or "api")
            self.persist()
            return

        for chunk_index in sorted(results_by_chunk):
            items.extend(results_by_chunk[chunk_index].get("items", []))
        if self._cancel.is_set():
            return
        accepted = sum(1 for i in items if i.get("accepted"))
        self.state["import_accepted_count"] = accepted
        if accepted:
            request_json(
                f"{self.sentrix_url}/api/ingest-batches/{quote(batch_id)}/complete",
                method="POST", timeout=60,
            )
        t1 = time.perf_counter()
        rejected = [item for item in items if not item.get("accepted")]
        phase_result = {
            "upload_seconds": round(t1 - t0, 1),
            "total_photos": photo_count,
            "total_videos": video_count,
            "total_media": total_media_count,
            "accepted_count": accepted,
            "failed_count": len(rejected),
            "error": f"{len(rejected)} media file(s) could not be imported" if rejected else None,
            "error_details": [{
                "sample_id": item.get("fileName") or item.get("file_name") or "unknown",
                "status": item.get("status") or "failed",
                "error_type": item.get("error_type") or "import_error",
                "reason": str(item.get("error") or "Sentrix rejected the media file"),
            } for item in rejected],
            "batch_id": batch_id,
            "chunk_size": chunk_size,
            "chunk_count": len(chunks),
            "upload_workers": upload_workers,
            "max_upload_attempts": max_upload_attempts,
            "retried_file_count": sum(
                1 for item in items if int(item.get("upload_attempts") or 1) > 1
            ),
        }
        if rejected:
            self._phase_partial("photo_import", phase_result)
        else:
            self._phase_done("photo_import", phase_result)

    def _phase_processing(self):
        self._phase_start("pipeline_processing")
        if not self.use_cloud_model and self.vllm_api_url:
            self._reset_gpu_samples_file()
            self._gpu_sampling_started = True
            self._gpu_sampler.start()
        t0 = time.perf_counter()
        scope_id = self.state["scope_id"]
        poll_count = 0
        batch_data = {}
        total = 0
        processed = 0
        failed = 0
        terminal_pipeline_error = None
        assets = []
        pending = []
        if self.state.get("import_accepted_count") == 0:
            self._phase_partial("pipeline_processing", {
                "total_seconds": 0.0,
                "poll_iterations": 0,
                "processed_photo_count": 0,
                "failed_asset_count": 0,
                "skipped_asset_count": 0,
                "error": "No imported media assets were available for processing",
                "error_details": [{
                    "sample_id": "media_batch",
                    "status": "skipped",
                    "error_type": "empty_import",
                    "reason": "All media files failed during the import stage",
                }],
                "progress": {"total": 0, "processed": 0, "pending": 0, "failed": 0, "skipped": 0},
                "average_seconds_per_photo": None,
                "pipeline_metrics": {},
            })
            return
        try:
            stall_timeout_seconds = max(0, int(os.getenv(
                "PHOTOBENCH_PIPELINE_STALL_TIMEOUT_SECONDS", "1200"
            )))
        except ValueError:
            stall_timeout_seconds = 1200
        while True:
            if self._cancel.is_set():
                break
            poll_count += 1
            data = request_json(f"{self.sentrix_url}/api/assets?scope_id={scope_id}&limit=2000", timeout=60)
            assets = data.get("assets", [])
            pending = [a for a in assets if a.get("status") in PIPELINE_PENDING_STATUSES]
            total = len(assets)
            processed = len([a for a in assets if a.get("status") == "processed"])
            failed = len([a for a in assets if a.get("status") in PIPELINE_FAILED_STATUSES])
            batch_data = {}
            batch_id = self.state.get("batch_id")
            if batch_id:
                try:
                    batch_data = request_json(
                        f"{self.sentrix_url}/api/ingest-batches/{quote(str(batch_id))}", timeout=60)
                    batch_metrics = batch_data.get("pipeline_metrics") or {}
                    if batch_metrics:
                        self._record_phase("pipeline_processing", "pipeline_metrics", batch_metrics)
                except Exception:
                    batch_data = {}
            batch_status = (batch_data.get("batch") or {}).get("status")
            pipeline_metrics = batch_data.get("pipeline_metrics") or {}
            status_counts = {}
            for asset in assets:
                status = str(asset.get("status") or "unknown")
                status_counts[status] = status_counts.get(status, 0) + 1
            # Compare each asset's state instead of only aggregate counts. Two assets
            # can transition in opposite directions during one poll and leave the
            # counts unchanged even though the pipeline is making progress.
            asset_status_signature = tuple(sorted(
                (str(asset.get("id") or asset.get("asset_id") or asset.get("path") or index),
                 str(asset.get("status") or "unknown"))
                for index, asset in enumerate(assets)
            ))
            progress_signature = (asset_status_signature, batch_status)
            now = time.monotonic()
            if getattr(self, "_pipeline_progress_signature", None) != progress_signature:
                self._pipeline_progress_signature = progress_signature
                self._pipeline_last_progress_at = now
                self._pipeline_last_progress_wall = now_iso()
            last_progress_at = getattr(self, "_pipeline_last_progress_at", now)
            no_progress_seconds = max(0.0, now - last_progress_at)
            self._record_phase("pipeline_processing", "status", "running")
            self._record_phase("pipeline_processing", "progress", {
                "total": total, "processed": processed, "pending": len(pending), "failed": failed,
                "poll_count": poll_count,
                "status_counts": status_counts,
                "last_progress_at": getattr(self, "_pipeline_last_progress_wall", now_iso()),
                "no_progress_seconds": round(no_progress_seconds, 1),
            })
            self.persist()
            if batch_status == "failed" or pipeline_metrics.get("status") == "failed":
                terminal_pipeline_error = str(
                    pipeline_metrics.get("error")
                    or f"Sentrix ingest batch entered status {batch_status}"
                )
                break
            if not pending and batch_status in {"completed", "complete"}:
                stable_polls = getattr(self, "_pipeline_stable_polls", 0) + 1
                self._pipeline_stable_polls = stable_polls
                # Confirm a stable zero-pending snapshot twice: asset status rows can
                # land after the batch flips to completed (upload/pipeline overlap).
                if stable_polls >= 2:
                    break
                self._cancel.wait(3)
                continue
            self._pipeline_stable_polls = 0
            batch_still_open_without_assets = (
                total == 0 and batch_status not in {"completed", "complete", "failed"}
            )
            if (
                stall_timeout_seconds > 0
                and not batch_still_open_without_assets
                and no_progress_seconds >= stall_timeout_seconds
            ):
                stall_progress = {
                    "total": total,
                    "processed": processed,
                    "pending": len(pending),
                    "failed": failed,
                    "poll_count": poll_count,
                    "status_counts": status_counts,
                    "batch_status": batch_status,
                    "last_progress_at": getattr(self, "_pipeline_last_progress_wall", None),
                    "no_progress_seconds": round(no_progress_seconds, 1),
                    "stall_timeout_seconds": stall_timeout_seconds,
                }
                self._record_phase("pipeline_processing", "status", "stalled")
                self._record_phase("pipeline_processing", "stalled_at", now_iso())
                self._record_phase("pipeline_processing", "stalled_progress", stall_progress)
                self.persist()
                terminal_pipeline_error = (
                    f"pipeline processing stalled: no asset status change for "
                    f"{no_progress_seconds:.0f}s (threshold {stall_timeout_seconds}s); "
                    f"processed={processed}/{total}, pending={len(pending)}, failed={failed}"
                )
                break
            if self._cancel.wait(3):
                break
        t1 = time.perf_counter()
        total_seconds = round(t1 - t0, 1)
        failed_assets = [a for a in assets if a.get("status") in PIPELINE_FAILED_STATUSES]
        skipped_assets = list(pending) if terminal_pipeline_error else []

        def failure_detail(asset, default_reason):
            metadata = asset.get("metadata_json") or {}
            return {
                "sample_id": asset.get("file_name") or asset.get("id") or "unknown",
                "asset_id": asset.get("id"),
                "status": asset.get("status") or "failed",
                "error_type": metadata.get("failed_stage") or "pipeline_error",
                "reason": str(metadata.get("error") or default_reason),
            }

        error_details = [
            failure_detail(asset, "Asset processing failed") for asset in failed_assets
        ] + [
            failure_detail(asset, terminal_pipeline_error or "Batch processing stopped")
            for asset in skipped_assets
        ]
        final_progress = {
            "total": total,
            "processed": processed,
            "pending": 0 if terminal_pipeline_error else len(pending),
            "failed": len(failed_assets),
            "skipped": len(skipped_assets),
            "poll_count": poll_count,
            "status_counts": {
                **((self.state.get("phases", {}).get("pipeline_processing", {}).get("progress") or {}).get("status_counts") or {}),
            },
        }
        phase_result = {
            "total_seconds": total_seconds,
            "poll_iterations": poll_count,
            "processed_photo_count": processed,
            "failed_asset_count": len(failed_assets),
            "skipped_asset_count": len(skipped_assets),
            "error": terminal_pipeline_error or (
                f"{len(failed_assets)} asset(s) failed after retries" if failed_assets else None
            ),
            "error_details": error_details,
            "progress": final_progress,
            "average_seconds_per_photo": round(total_seconds / processed, 3) if processed else None,
            "pipeline_metrics": batch_data.get("pipeline_metrics") or {},
        }
        if error_details or terminal_pipeline_error:
            self._phase_partial("pipeline_processing", phase_result)
        else:
            self._phase_done("pipeline_processing", phase_result)

    def _phase_qa_eval(self):
        self._phase_start("qa_eval")
        # reuse 模式没有 pipeline_processing 阶段，QA 采样在这里兜底启动 GPU 采样。
        if not self.use_cloud_model and self.vllm_api_url and not self._gpu_sampling_started:
            self._reset_gpu_samples_file()
            self._gpu_sampling_started = True
            self._gpu_sampler.start()
        scope_id = self.state["scope_id"]
        assets_data = request_json(f"{self.sentrix_url}/api/assets?scope_id={scope_id}&limit=2000", timeout=60)
        assets = assets_data.get("assets", [])
        assets_by_name = {}
        for a in assets:
            name = Path(a.get("file_name") or "").name
            assets_by_name.setdefault(name, []).append(a)

        t0 = time.perf_counter()
        qa_concurrency = self._resolve_qa_concurrency()
        judge_concurrency = self._resolve_judge_concurrency(qa_concurrency)
        agent_phase_started_perf = time.perf_counter()
        agent_phase_started_epoch = round(time.time(), 3)
        with self.lock:
            self.state["qa_concurrency"] = qa_concurrency
            self.state["judge_concurrency"] = judge_concurrency
        total_qa = len(self.qa_rows)
        self._qa_submitted = 0
        self._qa_agent_completed = 0
        self._qa_judge_submitted = 0
        self._qa_judge_completed = 0
        self._qa_judge_skipped = 0
        self._record_phase("qa_eval", "agent_phase_started_at", now_iso())
        self._record_phase("qa_eval", "agent_phase_started_at_epoch", agent_phase_started_epoch)
        self._record_phase("qa_eval", "agent_total", total_qa)
        self._record_phase("qa_eval", "judge_total", total_qa)

        def record_qa_progress():
            agent_done = getattr(self, "_qa_agent_completed", 0)
            judge_done = getattr(self, "_qa_judge_completed", 0)
            judge_submitted = getattr(self, "_qa_judge_submitted", 0)
            self._record_phase("qa_eval", "progress", {
                "total": total_qa,
                # User-facing completion means Judge reached a terminal state.
                "completed": judge_done,
                "agent_completed": agent_done,
                "agent_total": total_qa,
                "agent_in_flight": max(0, total_qa - agent_done),
                "judge_completed": judge_done,
                "judge_total": total_qa,
                "judge_submitted": judge_submitted,
                "judge_skipped": getattr(self, "_qa_judge_skipped", 0),
                "judge_in_flight": max(0, judge_submitted - (judge_done - getattr(self, "_qa_judge_skipped", 0))),
                "qa_concurrency": qa_concurrency,
                "judge_concurrency": judge_concurrency,
            })

        pool_rows = list(enumerate(self.qa_rows))
        agent_futures = {}
        judge_futures = {}
        judge_phase_started_perf = None
        judge_phase_started_epoch = None

        # The Agent pool is deliberately independent from the cloud Judge pool.
        # A completed Agent future returns immediately, releasing its model slot;
        # its item is then queued for Judge work in the second pool.
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=qa_concurrency, thread_name_prefix="qa-agent"
        ) as agent_executor, concurrent.futures.ThreadPoolExecutor(
            max_workers=judge_concurrency, thread_name_prefix="qa-judge"
        ) as judge_executor:
            agent_futures = {
                agent_executor.submit(self._evaluate_one, row, assets_by_name): index
                for index, row in pool_rows
            }
            self._qa_submitted = total_qa
            record_qa_progress()

            # One completion loop services both pools.  This keeps Agent and
            # Judge truly pipelined: a finished Judge is merged immediately,
            # even while other Agent futures are still running.
            pending = {future: ("agent", index) for future, index in agent_futures.items()}
            agent_phase_recorded = False

            def record_agent_phase_finished() -> None:
                nonlocal agent_phase_recorded
                if agent_phase_recorded:
                    return
                agent_phase_recorded = True
                agent_phase_finished_perf = time.perf_counter()
                agent_phase_finished_epoch = round(time.time(), 3)
                agent_phase_wall_ms = round(
                    (agent_phase_finished_perf - agent_phase_started_perf) * 1000, 1
                )
                self._record_phase("qa_eval", "agent_phase_finished_at", now_iso())
                self._record_phase("qa_eval", "agent_phase_finished_at_epoch", agent_phase_finished_epoch)
                self._record_phase("qa_eval", "agent_phase_total_seconds", round(agent_phase_wall_ms / 1000, 3))
                self._record_phase("qa_eval", "agent_phase_wall_ms", agent_phase_wall_ms)
                self._record_phase("qa_eval", "agent_completed", self._qa_agent_completed)
                if not self.use_cloud_model and self.vllm_api_url:
                    self._gpu_sampler.stop()

            while pending:
                done, _ = concurrent.futures.wait(
                    tuple(pending), return_when=concurrent.futures.FIRST_COMPLETED,
                )
                for future in done:
                    kind, index = pending.pop(future)
                    if kind == "agent":
                        try:
                            item = future.result()
                            agent_failed = bool(item.get("failed")) or bool(item.get("error"))
                        except Exception as exc:
                            item = {"index": index, "error": repr(exc), "failed": True}
                            agent_failed = True
                        item["_index"] = index
                        if agent_failed:
                            # There is no Agent answer for Judge to score.
                            item["judge_status"] = "skipped"
                            self._qa_judge_skipped += 1
                            self._qa_judge_completed += 1
                        else:
                            if judge_phase_started_perf is None:
                                judge_phase_started_perf = time.perf_counter()
                                judge_phase_started_epoch = round(time.time(), 3)
                                self._record_phase("qa_eval", "judge_phase_started_at", now_iso())
                                self._record_phase("qa_eval", "judge_phase_started_at_epoch", judge_phase_started_epoch)
                            item.setdefault("timing_breakdown", {}).setdefault("timeline", {})[
                                "judge_queued_at_epoch"
                            ] = round(time.time(), 3)
                            self._qa_judge_submitted += 1
                            judge_item = copy.deepcopy(item)
                            judge_future = judge_executor.submit(
                                self._judge_item, judge_item, self.qa_rows[index], assets_by_name,
                            )
                            judge_futures[judge_future] = index
                            pending[judge_future] = ("judge", index)
                        with self.lock:
                            self.state["items"].append(item)
                            self._qa_agent_completed += 1
                            record_qa_progress()
                            self.persist()
                        if not any(k == "agent" for k, _ in pending.values()):
                            record_agent_phase_finished()
                    else:
                        try:
                            judged_item = future.result()
                            with self.lock:
                                item = next((candidate for candidate in self.state["items"]
                                             if candidate.get("_index") == index), None)
                                if item is not None and isinstance(judged_item, dict):
                                    item.update({
                                        key: value for key, value in judged_item.items()
                                        if key != "_index"
                                    })
                        except Exception as exc:
                            with self.lock:
                                item = next((candidate for candidate in self.state["items"]
                                             if candidate.get("_index") == index), None)
                                if item is not None:
                                    item.update({
                                        "judge": {"score": None, "reason": f"judge_error: {exc}"},
                                        "task_judge": {"actual_action": None, "correct": None,
                                                        "reason": f"judge_error: {exc}"},
                                        "evidence_judge": {"score": None, "reason": "judge_error"},
                                        "judge_status": "failed",
                                    })
                        self._qa_judge_completed += 1
                        with self.lock:
                            record_qa_progress()
                            self.persist()

            record_agent_phase_finished()

            if judge_phase_started_perf is not None:
                judge_phase_finished_perf = time.perf_counter()
                judge_phase_finished_epoch = round(time.time(), 3)
                judge_phase_wall_ms = round((judge_phase_finished_perf - judge_phase_started_perf) * 1000, 1)
                self._record_phase("qa_eval", "judge_phase_finished_at", now_iso())
                self._record_phase("qa_eval", "judge_phase_finished_at_epoch", judge_phase_finished_epoch)
                self._record_phase("qa_eval", "judge_phase_total_seconds", round(judge_phase_wall_ms / 1000, 3))
                self._record_phase("qa_eval", "judge_phase_wall_ms", judge_phase_wall_ms)

            with self.lock:
                self.state["items"].sort(key=lambda it: it.get("_index", 0))
                for it in self.state["items"]:
                    it.pop("_index", None)
                executed = sum(1 for it in self.state["items"]
                               if it.get("execution_status") == "completed")
                self.state["run_valid"] = bool(
                    len(self.state["items"]) == total_qa and executed == total_qa
                )
                record_qa_progress()
                self.persist()

        t1 = time.perf_counter()
        failed_items = [item for item in self.state["items"] if (
            item.get("execution_status") in {"failed", "timeout"}
            or item.get("judge_status") == "failed"
        )]
        qa_phase_result = {
            "total_seconds": round(t1 - t0, 1),
            "qa_concurrency": qa_concurrency,
            "judge_concurrency": judge_concurrency,
            "failed_count": len(failed_items),
            "error": f"{len(failed_items)} QA sample(s) failed or timed out" if failed_items else None,
            "error_details": [{
                "sample_id": item.get("qa_id") or item.get("id") or f"QA #{index + 1}",
                "status": item.get("execution_status") or item.get("judge_status") or "failed",
                "error_type": "judge_error" if item.get("judge_status") == "failed" else "agent_error",
                "reason": str(item.get("error") or (item.get("judge") or {}).get("reason") or "QA execution failed"),
            } for index, item in enumerate(failed_items)],
        }
        if failed_items:
            self._phase_partial("qa_eval", qa_phase_result)
        else:
            self._phase_done("qa_eval", qa_phase_result)

    def _resolve_qa_concurrency(self) -> int:
        """QA-level concurrency: default follows the serving model's max_num_seqs snapshot."""
        env_value = str(os.getenv("PHOTOBENCH_QA_CONCURRENCY") or "").strip()
        if env_value:
            try:
                return max(1, int(env_value))
            except ValueError:
                pass
        try:
            snapshot = (self.state.get("phases") or {}).get("model_deploy", {}).get("model_state") or {}
            if not snapshot:
                snapshot = (self.state.get("current_model_snapshot") or {}).get("state") or {}
            return max(1, int(snapshot.get("max_num_seqs") or 1))
        except Exception:
            return 1

    @staticmethod
    def _resolve_judge_concurrency(qa_concurrency: int) -> int:
        """Judge uses a separate cloud-API pool and never consumes Agent slots."""
        env_value = str(os.getenv("PHOTOBENCH_JUDGE_CONCURRENCY") or "").strip()
        if env_value:
            try:
                return max(1, int(env_value))
            except ValueError:
                pass
        return min(8, max(1, int(qa_concurrency or 1)))

    def _evaluate_one(self, row: dict, assets_by_name: dict) -> dict:
        t0 = time.perf_counter()
        timeline = {"started_at_epoch": round(time.time(), 3)}
        conversation = row.get("conversation") or []
        if conversation and not isinstance(conversation, list):
            raise ValueError("conversation must be a list")
        messages = [str(turn.get("message") or "").strip() for turn in conversation if isinstance(turn, dict)]
        query = messages[-1] if messages else str(row["question"])
        reference = str(row["answer"])
        gt_refs = _normalize_media_refs(row, "retrieval")
        gt_image_ids = [ref["media_id"] for ref in gt_refs if ref["media_type"] == "image"]
        gt_video_ids = [ref["media_id"] for ref in gt_refs if ref["media_type"] == "video"]
        answer_refs = _normalize_media_refs(row, "answer_evidence") or gt_refs
        answer_image_ids = [ref["media_id"] for ref in answer_refs if ref["media_type"] == "image"]
        answer_video_ids = [ref["media_id"] for ref in answer_refs if ref["media_type"] == "video"]
        item = {"qa_id": row.get("qa_id"), "question": query, "reference_answer": reference,
                "retrieval_media_refs": gt_refs,
                "retrieval_image_ids": gt_image_ids,
                "retrieval_video_ids": gt_video_ids,
                "answer_evidence_media_refs": answer_refs,
                "answer_evidence_image_ids": answer_image_ids,
                "answer_evidence_video_ids": answer_video_ids,
                "execution_status": "scheduled"}
        for field in ("task_type", "question_type", "tags", "angle", "difficulty", "answerability",
                      "scope", "scope_anchor", "required_evidence_sources", "query_anchors",
                      "expected_action", "answer_claims", "alternative_evidence_media_sets"):
            if row.get(field) is not None:
                item[field] = row[field]
        try:
            item["execution_status"] = "started"
            t_agent0 = time.perf_counter()
            conversation_id = str(uuid.uuid4())
            turn_records = []
            all_call_metrics, all_execution_trace, all_tool_trace = [], [], []
            all_tool_observations = []
            tool_trace_present = False
            for turn_index, message in enumerate(messages or [query]):
                initial_resp = request_json(f"{self.sentrix_url}/api/assistant/turn", {
                    "message": message, "scope_id": self.state["scope_id"],
                    "conversation_id": conversation_id, "viewer_id": "owner", "include_debug": True,
                }, "POST", 300)
                resp = wait_for_assistant_turn(self.sentrix_url, initial_resp, timeout=900,
                                               cancelled=self._cancel.is_set)
                turn_metrics, turn_trace, turn_tools = self._normalize_turn_traces(resp)
                _, turn_observations = self._extract_tool_perf(resp.get("task_state") or {})
                for metric in turn_metrics:
                    if isinstance(metric, dict):
                        all_call_metrics.append({**metric, "conversation_turn": turn_index})
                for step in turn_trace:
                    if isinstance(step, dict):
                        all_execution_trace.append({**step, "conversation_turn": turn_index})
                for trace in turn_tools:
                    if isinstance(trace, dict):
                        all_tool_trace.append({**trace, "conversation_turn": turn_index})
                all_tool_observations.extend(turn_observations)
                tool_trace_present = tool_trace_present or "tool_trace" in resp or "toolTrace" in resp
                turn_records.append({
                    "index": turn_index,
                    "conversation_id": conversation_id,
                    "context_turn_count": turn_index,
                    "message": message,
                    "debug_trace": resp.get("debug_trace"),
                    "expected_action": (conversation[turn_index].get("expected_action")
                                        if turn_index < len(conversation) and isinstance(conversation[turn_index], dict)
                                        else row.get("expected_action")),
                    "answer": str(resp.get("answer") or ""),
                    "agent2_trace": resp.get("agent2_trace") or resp.get("agent2Trace") or {},
                    "agent_status": resp.get("tool_loop_status") or (resp.get("telemetry") or {}).get("status"),
                    "termination_reason": resp.get("termination_reason") or resp.get("terminationReason") or "",
                    "turn_outcome": resp.get("turn_outcome") or _derive_turn_outcome(resp),
                    "parse_status": _derive_parse_status(resp),
                    "next_step": _derive_next_step(resp),
                    "predicted_media": _resolve_predicted_media(_extract_image_ids(resp), assets_by_name),
                    "predicted_images": _resolve_predicted_images(_extract_image_ids(resp), assets_by_name),
                })
            agent_wall_ms = round((time.perf_counter() - t_agent0) * 1000, 1)

            # Extract model call metrics
            call_metrics = all_call_metrics
            execution_trace = all_execution_trace
            tool_trace = all_tool_trace
            derived_tool_perf, _ = self._extract_tool_perf(resp.get("task_state") or {})
            tool_trace = self._bind_tool_calls_to_model_rounds(
                tool_trace, execution_trace,
                call_metrics,
            )
            tool_trace = self._attach_tool_observations(tool_trace, all_tool_observations)
            call_metrics, tool_trace = self._annotate_agent_loop_timings(call_metrics, tool_trace)
            answer = str(resp.get("answer") or "")

            # Keep retrieval, evidence, and final delivery independent. The
            # retrieval metric must not punish the UI for showing only a few
            # representative sources.
            media_sets = _extract_media_sets(resp)
            selected_media = _resolve_predicted_media(media_sets["selected_asset_ids"], assets_by_name)
            retrieved_media = _resolve_predicted_media(media_sets["retrieved_asset_ids"], assets_by_name)
            evidence_media = _resolve_predicted_media(media_sets["evidence_asset_ids"], assets_by_name)

            # Match against GT
            metrics = _modality_metrics(gt_refs, retrieved_media)
            gt_media = _resolve_gt_media(gt_refs, assets_by_name, retrieved_media)
            retrieved_keys = {
                _media_key(value.get("media_type"), value.get("media_id") or value.get("file_name"))
                for value in retrieved_media
            }
            evidence_keys = {
                _media_key(value.get("media_type"), value.get("media_id") or value.get("file_name"))
                for value in evidence_media
            }
            selected_keys = {
                _media_key(value.get("media_type"), value.get("media_id") or value.get("file_name"))
                for value in selected_media
            }
            matched = sorted(entry["file_name"] for entry in gt_media
                             if _media_key(entry["media_type"], entry["media_id"]) in retrieved_keys)
            evidence_matched = sorted(entry["file_name"] for entry in gt_media
                                      if _media_key(entry["media_type"], entry["media_id"]) in evidence_keys)
            delivery_matched = sorted(entry["file_name"] for entry in gt_media
                                      if _media_key(entry["media_type"], entry["media_id"]) in selected_keys)
            pred_names = {value["file_name"] for value in selected_media}
            retrieved_names = {value["file_name"] for value in retrieved_media}
            evidence_names = {value["file_name"] for value in evidence_media}
            gt_images = [value for value in gt_media if value["media_type"] == "image"]

            # Agent phase ends before any Judge request starts.  Judge is queued
            # by _phase_qa_eval in a separate executor after this item returns.
            timeline["agent_finished_at_epoch"] = round(time.time(), 3)
            timeline["agent_finished_at"] = now_iso()
            llm_summary = self._summarize_call_metrics(call_metrics)
            wall_clock_ms = agent_wall_ms
            timing_breakdown = self._build_timing_breakdown(
                call_metrics, tool_trace, wall_clock_ms, agent_wall_ms, None,
                timeline, tool_trace_present,
            )

            item.update({
                "answer": answer,
                "predicted_media": selected_media,
                "retrieved_candidate_media": retrieved_media,
                "evidence_source_media": evidence_media,
                "predicted_images": [value for value in selected_media if value["media_type"] == "image"],
                "retrieved_candidate_images": [value for value in retrieved_media if value["media_type"] == "image"],
                "evidence_source_images": [value for value in evidence_media if value["media_type"] == "image"],
                "predicted_file_names": sorted(pred_names),
                "retrieved_file_names": sorted(retrieved_names),
                "evidence_source_file_names": sorted(evidence_names),
                "matched_file_names": matched,
                "retrieved_matched_file_names": matched,
                "evidence_matched_file_names": evidence_matched,
                "delivery_matched_file_names": delivery_matched,
                "selected_delivery_file_names": sorted(pred_names),
                "retrieved_asset_ids": media_sets["retrieved_asset_ids"],
                "evidence_asset_ids": media_sets["evidence_asset_ids"],
                "selected_asset_ids": media_sets["selected_asset_ids"],
                "media_retrieval_counts": metrics["media"],
                "image_retrieval_counts": metrics["image"],
                "video_retrieval_counts": metrics["video"],
                "media_retrieval_recall": metrics["media"]["recall"],
                "media_retrieval_precision": metrics["media"]["precision"],
                "media_retrieval_f1": metrics["media"]["f1"],
                "image_retrieval_recall": metrics["image"]["recall"],
                "image_retrieval_precision": metrics["image"]["precision"],
                "image_retrieval_f1": metrics["image"]["f1"],
                "video_retrieval_recall": metrics["video"]["recall"],
                "video_retrieval_precision": metrics["video"]["precision"],
                "video_retrieval_f1": metrics["video"]["f1"],
                # Compatibility aliases are total-media metrics for typed runs.
                "retrieval_recall": metrics["media"]["recall"],
                "retrieval_precision": metrics["media"]["precision"],
                "retrieval_f1": metrics["media"]["f1"],
                "gt_media": gt_media,
                "gt_images": gt_images,
                "judge": {"score": None, "reason": "pending_judge"},
                "task_judge": {"actual_action": None, "correct": None, "reason": "pending_judge"},
                "task_judges": [], "conversation": turn_records if conversation else [],
                "runtime_turns": turn_records,
                "conversation_id": conversation_id if conversation else None,
                "conversation_turn_count": len(turn_records) if conversation else 1,
                "conversation_context_mode": "shared_conversation_id" if conversation else "single_turn",
                "evidence_judge": {"score": None, "reason": "pending_judge"}, "judge_ms": None,
                "wall_clock_ms": wall_clock_ms,
                "model_call_metrics": call_metrics,
                "execution_trace": execution_trace,
                "llm_summary": llm_summary,
                "tool_trace": tool_trace,
                "tool_perf": resp.get("tool_perf") or (resp.get("telemetry") or {}).get("tool_perf")
                    or derived_tool_perf,
                "guard_debug": resp.get("guard_debug") or {},
                "termination_reason": resp.get("termination_reason") or resp.get("terminationReason") or "",
                "turn_outcome": resp.get("turn_outcome") or _derive_turn_outcome(resp),
                "agent_status": resp.get("tool_loop_status") or (resp.get("telemetry") or {}).get("status"),
                "agent_reason": resp.get("tool_loop_reason") or (resp.get("telemetry") or {}).get("reason") or "",
                "answer_grounding": resp.get("answer_grounding") or resp.get("answerGrounding") or {},
                "agent2_trace": summarize_agent2_trace(turn_records),
                "timing_breakdown": timing_breakdown,
                "tool_trace_recorded": tool_trace_present,
                "judge_status": "pending",
            })
            item["delivery_status"] = resp.get("delivery_status") or {}
            item["agent_stability"] = self._agent_stability(item)
            item["attribution"] = self._derive_attribution(item)
            item["execution_status"] = "completed"
        except Exception as e:
            message = str(e).lower()
            if "timed out" in message or "timeout" in message:
                item["execution_status"] = "timeout"
            else:
                item["execution_status"] = "failed"
            item.update({"error": str(e), "retrieval_recall": 0,
                         "retrieval_precision": None, "retrieval_f1": None,
                         "media_retrieval_recall": 0 if gt_refs else None,
                         "media_retrieval_precision": None, "media_retrieval_f1": None,
                         "image_retrieval_recall": 0 if gt_image_ids else None,
                         "video_retrieval_recall": 0 if gt_video_ids else None,
                         "judge": {"score": None, "reason": "error"},
                         "wall_clock_ms": round((time.perf_counter() - t0) * 1000, 1)})
        return item

    def _judge_item(self, item: dict, row: dict, assets_by_name: dict) -> dict:
        """Score one completed Agent answer in the independent Judge pool."""
        t_judge0 = time.perf_counter()
        timing = item.get("timing_breakdown") or {}
        timeline = dict(timing.get("timeline") or {})
        judge_started_epoch = round(time.time(), 3)
        timeline["judge_started_at_epoch"] = judge_started_epoch
        timeline["judge_started_at"] = now_iso()
        queued_epoch = timeline.get("judge_queued_at_epoch")
        if isinstance(queued_epoch, (int, float)):
            timeline["judge_queue_wait_ms"] = round(max(0.0, judge_started_epoch - queued_epoch) * 1000, 1)

        conversation = row.get("conversation") or []
        reference = str(item.get("reference_answer") or row.get("answer") or "")
        turn_records = item.get("runtime_turns") or []
        for index, record in enumerate(turn_records):
            expected = record.get("expected_action")
            turn_definition = (conversation[index]
                               if index < len(conversation) and isinstance(conversation[index], dict)
                               else {})
            turn_reference = str(turn_definition.get("reference_answer") or "")
            if not turn_reference:
                turn_reference = (
                    "应要求用户提供至少一个具体、可用于后续检索的锚点；询问形式不限。"
                    if expected == "clarify" else reference
                )
            conv_ctx = turn_records[:index + 1]
            record["task_judge"] = self._judge_task_action(
                record["message"], record["answer"], expected,
                record.get("agent_status"), record.get("termination_reason"),
                task_type=row.get("task_type"),
                question_type=row.get("question_type"),
                answerability=row.get("answerability"),
                reference=turn_reference,
                conversation=conv_ctx,
            )
            # Answer quality and evidence calls for the same turn remain parallel.
            should_judge_evidence = (
                EVIDENCE_JUDGE_ENABLED
                and record.get("predicted_images") and record.get("answer")
                and record["task_judge"].get("actual_action") != "clarify"
            )
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
                quality_future = pool.submit(
                    self._judge, record["message"], turn_reference, record["answer"],
                    expected_action=expected,
                    task_type=row.get("task_type"),
                    question_type=row.get("question_type"),
                    answerability=row.get("answerability"),
                    conversation=conv_ctx,
                ) if expected in ANSWER_QUALITY_RUBRICS else None
                evidence_future = pool.submit(
                    self._judge_evidence,
                    record["message"], record["answer"],
                    record.get("predicted_images") or [],
                    assets_by_name, self.sentrix_url, conversation=conv_ctx,
                ) if should_judge_evidence else None
            record["judge"] = quality_future.result() if quality_future else {"score": None, "reason": "not_labeled"}
            record["evidence_judge"] = evidence_future.result() if evidence_future else {"score": None, "reason": "not_applicable"}

        task_judges = [record.get("task_judge") or {} for record in turn_records]
        task_judge = task_judges[-1] if task_judges else {"actual_action": None, "correct": None}
        judge = turn_records[-1].get("judge") if turn_records else {"score": None, "reason": "not_applicable"}
        evidence_judge = turn_records[-1].get("evidence_judge") if turn_records else {"score": None, "reason": "not_applicable"}
        judge_ms = round((time.perf_counter() - t_judge0) * 1000, 1)
        judge_finished_epoch = round(time.time(), 3)
        timeline["judge_finished_at_epoch"] = judge_finished_epoch
        timeline["judge_finished_at"] = now_iso()
        queue_wait_ms = timeline.get("judge_queue_wait_ms")
        if not isinstance(queue_wait_ms, (int, float)):
            queue_wait_ms = 0.0
        agent_wall_ms = self._numeric_ms(timing.get("agent_wall_ms")) or 0.0
        wall_clock_ms = round(agent_wall_ms + queue_wait_ms + judge_ms, 1)
        call_metrics = item.get("model_call_metrics") or []
        tool_trace = item.get("tool_trace") or []
        timing_breakdown = self._build_timing_breakdown(
            call_metrics, tool_trace, wall_clock_ms, agent_wall_ms, judge_ms,
            timeline, bool(item.get("tool_trace_recorded")), queue_wait_ms,
        )
        all_judge_outputs = [judge, evidence_judge, *task_judges]
        judge_terminal_status = "failed" if any(
            isinstance(value, dict) and value.get("judge_status") == "failed"
            for value in all_judge_outputs
        ) else "completed"
        item.update({
            "judge": judge or {"score": None, "reason": "not_applicable"},
            "task_judge": task_judge,
            "task_judges": task_judges,
            "evidence_judge": evidence_judge or {"score": None, "reason": "not_applicable"},
            "judge_ms": judge_ms,
            "wall_clock_ms": wall_clock_ms,
            "judge_status": judge_terminal_status,
            "timing_breakdown": timing_breakdown,
        })
        item["agent_stability"] = self._agent_stability(item)
        item["attribution"] = self._derive_attribution(item)
        return item

    @staticmethod
    def _summarize_call_metrics(metrics: list[dict]) -> dict:
        if not metrics:
            return {}
        streamed = [m for m in metrics if m.get("streamed")]
        ttfts = [m["ttft_ms"] for m in streamed if m.get("ttft_ms") is not None]
        totals = [m["total_ms"] for m in streamed if m.get("total_ms") is not None]
        tps_list = [m["tokens_per_second"] for m in streamed if m.get("tokens_per_second") is not None]
        prompt_tokens = sum(m.get("prompt_tokens", 0) or 0 for m in metrics)
        completion_tokens = sum(m.get("completion_tokens", 0) or 0 for m in metrics)
        return {
            "call_count": len(metrics),
            "streamed_count": len(streamed),
            "ttft_ms_avg": round(sum(ttfts) / len(ttfts), 1) if ttfts else None,
            "total_ms_sum": round(sum(totals), 1) if totals else None,
            "tokens_per_second_avg": round(sum(tps_list) / len(tps_list), 1) if tps_list else None,
            "prompt_tokens_total": prompt_tokens,
            "completion_tokens_total": completion_tokens,
        }

    @staticmethod
    def _numeric_ms(value) -> float | None:
        try:
            number = float(value)
        except (TypeError, ValueError):
            return None
        return number if number >= 0 else None

    @classmethod
    def _annotate_agent_loop_timings(cls, model_calls: list, tool_trace: list) -> tuple[list[dict], list[dict]]:
        """Record model + child-tool duration without flattening nested work.

        ``total_ms`` remains the raw model request duration.  The new
        ``agent_loop_total_ms`` is the end-to-end duration for that model round
        plus directly attached tools.  Tool-internal model calls stay nested in
        their parent tool and are not added as another top-level loop.
        """
        calls = [dict(call) for call in model_calls if isinstance(call, dict)]
        tools = [dict(trace) for trace in tool_trace if isinstance(trace, dict)]
        by_call: dict[int, list[dict]] = {}
        for trace in tools:
            try:
                index = int(trace.get("model_call_index"))
            except (TypeError, ValueError):
                continue
            duration_ms = cls._numeric_ms(trace.get("duration_ms"))
            if duration_ms is None:
                latency_s = cls._numeric_ms(trace.get("latency_s"))
                duration_ms = latency_s * 1000 if latency_s is not None else None
            if duration_ms is not None:
                trace["duration_ms"] = round(duration_ms, 1)
            by_call.setdefault(index, []).append(trace)

        for index, call in enumerate(calls):
            model_ms = cls._numeric_ms(call.get("total_ms"))
            if model_ms is None:
                continue
            ttft_ms = cls._numeric_ms(call.get("ttft_ms"))
            child_tools = by_call.get(index, [])
            tool_ms = round(sum(float(tool.get("duration_ms") or 0) for tool in child_tools), 1)
            generation_ms = round(max(0.0, model_ms - (ttft_ms or 0)), 1) if ttft_ms is not None else model_ms
            call["agent_loop_total_ms"] = round(model_ms + tool_ms, 1)
            call["agent_loop_timing"] = {
                "model_ms": round(model_ms, 1),
                "ttft_ms": round(ttft_ms, 1) if ttft_ms is not None else None,
                "model_generation_ms": generation_ms,
                "tool_ms": tool_ms if child_tools else 0.0,
                "tool_count": len(child_tools),
                "tools": [
                    {"tool": tool.get("tool") or "未知工具", "duration_ms": tool.get("duration_ms")}
                    for tool in child_tools
                ],
            }
        return calls, tools

    @classmethod
    def _build_timing_breakdown(cls, model_calls: list, tool_trace: list,
                                wall_clock_ms: float, agent_wall_ms: float,
                                judge_ms: float, timeline: dict,
                                tool_trace_present: bool,
                                judge_queue_wait_ms: float = 0.0) -> dict:
        """Build the two-level latency contract used by the UI.

        QA children are Agent Loops, Judge, and residual overhead.  Each Agent
        Loop exposes TTFT, model generation, and direct child tools.  A tool's
        own latency already includes any nested work, so nested model calls are
        intentionally excluded from the QA-level children.
        """
        loops = []
        total_tool_ms = 0.0
        for index, call in enumerate(model_calls):
            call_type = str(call.get("call_type") or "agent")
            if call_type in {"tool_internal", "faithfulness_judge"}:
                continue
            loop_ms = cls._numeric_ms(call.get("agent_loop_total_ms"))
            model_ms = cls._numeric_ms(call.get("total_ms"))
            if loop_ms is None or model_ms is None:
                continue
            timing = call.get("agent_loop_timing") or {}
            tool_ms = cls._numeric_ms(timing.get("tool_ms")) or 0.0
            total_tool_ms += tool_ms
            observation = call.get("call_observation") or {}
            label = observation.get("label") or {
                "planner": "Agent 2.0 目标分解与规划",
                "agent": "Agent 决策 / 回答",
                "recovery": "Agent 恢复调用",
                "writer": "最终回答重写",
            }.get(call_type, call_type)
            loops.append({
                "index": index,
                "label": label,
                "call_type": call_type,
                "step_id": call.get("step_id"),
                "conversation_turn": call.get("conversation_turn"),
                "duration_ms": loop_ms,
                "model_ms": round(model_ms, 1),
                "ttft_ms": timing.get("ttft_ms"),
                "model_generation_ms": timing.get("model_generation_ms"),
                "tool_ms": tool_ms,
                "tool_count": timing.get("tool_count", 0),
            })
        loop_sum_ms = round(sum(float(loop["duration_ms"]) for loop in loops), 1)
        agent_overhead_ms = round(max(0.0, agent_wall_ms - loop_sum_ms), 1)
        judge_value = cls._numeric_ms(judge_ms)
        queue_value = cls._numeric_ms(judge_queue_wait_ms) or 0.0
        other_ms = round(max(0.0, wall_clock_ms - loop_sum_ms - (judge_value or 0) - queue_value), 1)
        model_ms = round(sum(float(loop["model_ms"]) for loop in loops), 1) if loops else None
        tool_ms = round(total_tool_ms, 1) if tool_trace_present else None
        return {
            "wall_clock_ms": wall_clock_ms,
            "timeline": timeline,
            "agent_wall_ms": agent_wall_ms,
            "model_ms": model_ms,
            "tool_ms": tool_ms,
            "judge_ms": judge_value,
            "judge_queue_wait_ms": queue_value if queue_value > 0 else None,
            "other_ms": other_ms if tool_trace_present else None,
            "agent_overhead_ms": agent_overhead_ms if tool_trace_present else None,
            "orchestrator_overhead_ms": round(max(0.0, wall_clock_ms - agent_wall_ms - queue_value - (judge_value or 0)), 1),
            "tool_trace_recorded": tool_trace_present,
            "agent_loop_timing_recorded": bool(loops),
            "agent_loops": loops,
            "agent_loop_sum_ms": loop_sum_ms if loops else None,
            "qa_components": ([
                *[{"kind": "agent_loop", **loop} for loop in loops],
                *([{"kind": "judge", "label": "Judge", "duration_ms": judge_value}] if judge_value else []),
                *([{"kind": "judge_queue", "label": "Judge 排队 / 编排", "duration_ms": queue_value}] if queue_value > 0 else []),
                *([{"kind": "other", "label": "其他", "duration_ms": other_ms}] if other_ms > 0 else []),
            ] if loops else []),
        }

    @staticmethod
    def _bind_tool_calls_to_model_rounds(tool_trace: list, retrieval_trace: list,
                                         model_calls: list | int | None = None) -> list[dict]:
        """Attach tools by step ID first, with ordered fallback for historical runs."""
        bound = [dict(trace) for trace in tool_trace if isinstance(trace, dict)]
        if not bound:
            return bound
        calls = [call for call in model_calls if isinstance(call, dict)] \
            if isinstance(model_calls, list) else []
        model_call_count = len(calls) if calls else int(model_calls or 0)
        call_index_by_step = {}
        for index, call in enumerate(calls):
            if not call.get("step_id"):
                continue
            key = (call.get("conversation_turn"), str(call.get("step_id")))
            call_index_by_step[key] = index
            call_index_by_step[("any", str(call.get("step_id")))] = index
        for trace in bound:
            parent_step_id = trace.get("parent_step_id")
            key = (trace.get("conversation_turn"), str(parent_step_id))
            if parent_step_id is not None and key in call_index_by_step:
                trace["model_call_index"] = call_index_by_step[key]
                trace["round_binding_source"] = "step_id"
        if not isinstance(retrieval_trace, list) or not retrieval_trace:
            if model_call_count == 1:
                for trace in bound:
                    if trace.get("model_call_index") is None:
                        trace["model_call_index"] = 0
                        trace["round_binding_source"] = "inferred_single_model_call"
            return bound
        model_index = None
        ordered_call_index = 0
        tool_index = 0
        for step in retrieval_trace:
            if not isinstance(step, dict):
                continue
            stage = step.get("stage") or step.get("type")
            step_id = step.get("step_id")
            step_key = (step.get("conversation_turn"), str(step_id))
            mapped_index = call_index_by_step.get(step_key)
            if mapped_index is None:
                mapped_index = call_index_by_step.get(("any", str(step_id)))
            if stage in {"planner", "model", "writer", "judge"}:
                if mapped_index is not None:
                    model_index = mapped_index
                    ordered_call_index = max(ordered_call_index, mapped_index + 1)
                elif ordered_call_index < model_call_count:
                    model_index = ordered_call_index
                    ordered_call_index += 1
            elif stage == "tool" and tool_index < len(bound):
                if model_index is not None and model_index >= 0 and bound[tool_index].get("model_call_index") is None:
                    bound[tool_index]["model_call_index"] = model_index
                    bound[tool_index]["round_binding_source"] = "retrieval_trace"
                    if step.get("parent_step_id") is not None:
                        bound[tool_index]["parent_step_id"] = step.get("parent_step_id")
                elif model_index is not None and model_index >= 0 and step.get("parent_step_id") is not None:
                    # The execution trace is authoritative when older Sentrix
                    # responses carried a stale positional model_call_index.
                    bound[tool_index]["model_call_index"] = model_index
                    bound[tool_index]["round_binding_source"] = "retrieval_trace_parent"
                    bound[tool_index]["parent_step_id"] = step.get("parent_step_id")
                tool_index += 1
        if model_call_count == 1:
            for trace in bound:
                if trace.get("model_call_index") is None:
                    trace["model_call_index"] = 0
                    trace["round_binding_source"] = "inferred_single_model_call"
        return bound

    @staticmethod
    def _trace_action(step: dict) -> dict | None:
        detail = step.get("detail")
        if isinstance(detail, dict) and detail.get("action"):
            return detail
        text = str(detail or "").strip()
        start = text.find("{")
        if start < 0:
            return None
        for end in range(len(text), start, -1):
            try:
                value = json.loads(text[start:end])
            except json.JSONDecodeError:
                continue
            return value if isinstance(value, dict) else None
        return None

    @classmethod
    def _annotate_model_calls(cls, model_calls: list, execution_trace: list) -> list[dict]:
        """Add readable observations without mutating historical run data.

        Historical roles are inferred only when the complete metric sequence can
        be consumed one-to-one by the saved execution trace.
        """
        calls = [dict(call) for call in model_calls if isinstance(call, dict)]
        if not calls:
            return calls
        if all(call.get("call_observation") for call in calls):
            return calls
        if any(call.get("call_type") for call in calls):
            return calls
        trace = [step for step in execution_trace if isinstance(step, dict)]
        if not trace:
            for call in calls:
                call["call_observation"] = {
                    "kind": "unknown",
                    "label": "历史模型调用",
                    "purpose": "旧记录未保存调用用途",
                    "trigger": "未记录",
                    "outcome": "仅保留原始模型性能指标",
                    "source": "historical_unresolved",
                }
            return calls

        assignments = []
        call_index = 0
        recovery_pending = None
        for trace_index, step in enumerate(trace):
            stage = str(step.get("stage") or step.get("type") or "")
            if stage in {"model", "writer", "judge"}:
                if call_index >= len(calls) or str(calls[call_index].get("role") or "") in {"inspect", "ocr"}:
                    assignments = []
                    break
                assignments.append((call_index, stage, step, recovery_pending, trace_index))
                call_index += 1
                recovery_pending = None
                if stage == "model":
                    next_stage = next((
                        str(candidate.get("stage") or candidate.get("type") or "")
                        for candidate in trace[trace_index + 1:]
                        if str(candidate.get("stage") or candidate.get("type") or "")
                        in {"model", "tool", "writer", "guard", "judge"}
                    ), "")
                    if next_stage == "model":
                        recovery_pending = "format"
            elif stage == "tool":
                tool = step.get("tool") or (step.get("detail") if isinstance(step.get("detail"), str) else "")
                expected_role = "inspect" if tool == "inspect_photo" else "ocr" if tool == "read_photo_text" else None
                if expected_role:
                    while call_index < len(calls) and str(calls[call_index].get("role") or "") == expected_role:
                        assignments.append((call_index, "tool_internal", step, False, trace_index))
                        call_index += 1
            elif stage == "guard" and str(step.get("status") or "").lower() == "fail":
                recovery_pending = "guard"
            if stage == "judge":
                detail = step.get("detail") or {}
                if isinstance(detail, dict) and detail.get("faithful") is False:
                    recovery_pending = "guard"
        if call_index != len(calls) or len(assignments) != len(calls):
            for call in calls:
                call["call_observation"] = {
                    "kind": "unknown",
                    "label": "历史模型调用",
                    "purpose": "旧记录没有足够信息完成可靠对齐",
                    "trigger": "未记录",
                    "outcome": "仅保留原始模型性能指标",
                    "source": "historical_unresolved",
                }
            return calls

        for index, stage, step, recovery_reason, trace_index in assignments:
            call = calls[index]
            action = cls._trace_action(step)
            if stage == "tool_internal":
                tool = step.get("tool") or (step.get("detail") if isinstance(step.get("detail"), str) else "")
                observation = {
                    "kind": "tool_internal",
                    "label": "工具内部模型调用（历史对齐）",
                    "purpose": "读取照片文字" if tool == "read_photo_text" else "识别照片视觉细节",
                    "trigger": f"工具 {tool} 执行内部模型推理",
                    "outcome": f"完成 {tool} 的模型处理",
                    "source": "historical_trace_aligned",
                    "related_tool": tool,
                }
            elif stage == "judge":
                detail = step.get("detail") or {}
                faithful = detail.get("faithful") if isinstance(detail, dict) else None
                observation = {
                    "kind": "faithfulness_judge",
                    "label": "L2 事实一致性检查（历史对齐）",
                    "purpose": "检查候选回答是否与工具事实一致",
                    "trigger": "L1 Guard 通过且回答使用了工具事实",
                    "outcome": "判定通过" if faithful is True else "判定未通过" if faithful is False else "完成检查",
                    "source": "historical_trace_aligned",
                }
            elif stage == "writer":
                observation = {
                    "kind": "writer",
                    "label": "最终回答重写（历史对齐）",
                    "purpose": "基于受控事实调整最终回答的结构和措辞",
                    "trigger": "候选回答触发 Final Writer",
                    "outcome": f"重写状态：{step.get('status') or '完成'}",
                    "source": "historical_trace_aligned",
                }
            else:
                tool = str((action or {}).get("tool") or "")
                is_final = (action or {}).get("action") == "final"
                next_step = next((
                    candidate for candidate in trace[trace_index + 1:]
                    if str(candidate.get("stage") or candidate.get("type") or "")
                    in {"model", "tool", "writer", "guard", "judge"}
                ), {})
                next_stage = str(next_step.get("stage") or next_step.get("type") or "")
                if not tool and next_stage == "tool":
                    tool = str(next_step.get("tool") or (
                        next_step.get("detail") if isinstance(next_step.get("detail"), str) else ""))
                generated_candidate = is_final or next_stage in {"writer", "guard", "judge"}
                parse_recovery = next_stage == "model"
                is_recovery = recovery_reason in {"format", "guard"}
                observation = {
                    "kind": "recovery" if is_recovery else "agent",
                    "label": (
                        "Agent 格式恢复调用（历史对齐）" if recovery_reason == "format" else
                        "Agent 校验恢复调用（历史对齐）" if recovery_reason == "guard" else
                        "Agent 决策 / 回答（历史对齐）"
                    ),
                    "purpose": (
                        "按格式纠正要求重新生成合法动作" if recovery_reason == "format" else
                        "根据 Guard 或 L2 校验反馈修正回答" if recovery_reason == "guard" else
                        "选择下一步工具或生成候选回答"
                    ),
                    "trigger": (
                        "上一轮模型输出无法解析" if recovery_reason == "format" else
                        "上一轮 Guard 或 L2 事实一致性检查未通过" if recovery_reason == "guard" else
                        "上一工具返回结果" if index else "用户问题"
                    ),
                    "outcome": (
                        f"决定调用工具 {tool}" if tool else
                        "输出未形成合法动作，触发格式恢复" if parse_recovery else
                        "生成候选回答" if generated_candidate else
                        "完成 Agent 推理"
                    ),
                    "source": "historical_trace_aligned",
                    "related_tool": tool or None,
                }
            call["call_observation"] = observation
        return calls

    @staticmethod
    def _normalize_turn_traces(response: dict) -> tuple[list, list, list]:
        """Normalize the response into one authoritative ordered trace.

        ``debug_trace`` is the only response field that contains tool
        arguments, parent step IDs, and the full observation.  Use it as the
        fallback when the compact retrieval/tool summaries are absent.
        """
        response = response if isinstance(response, dict) else {}
        metrics = response.get("model_call_metrics") or []
        debug = response.get("debug_trace") or []
        execution = response.get("retrieval_trace") or response.get("retrievalTrace") or []
        if not execution and isinstance(debug, list):
            execution = debug
        compact_tools = response.get("tool_trace") or response.get("toolTrace") or []
        debug_tools = []
        if isinstance(debug, list):
            debug_tools = [
                step for step in debug
                if isinstance(step, dict)
                and str(step.get("type") or step.get("stage") or "") == "tool"
            ]
        # The compact trace carries timing/summary fields, while debug_trace
        # carries the authoritative arguments, parent_step_id and full
        # observation. Merge them by execution order so 8771 shows one
        # complete, internally consistent tool record.
        if debug_tools:
            tools = []
            for index, debug_tool in enumerate(debug_tools):
                merged = dict(debug_tool)
                if isinstance(compact_tools, list) and index < len(compact_tools):
                    compact = compact_tools[index]
                    if isinstance(compact, dict):
                        for key, value in compact.items():
                            if key not in {"observation", "arguments", "parent_step_id", "step_id", "tool", "tool_name"}:
                                merged.setdefault(key, value)
                tools.append(merged)
        else:
            tools = compact_tools
        return list(metrics) if isinstance(metrics, list) else [], \
            list(execution) if isinstance(execution, list) else [], \
            list(tools) if isinstance(tools, list) else []

    @staticmethod
    def _extract_tool_perf(task_state: dict) -> tuple[dict, list[dict]]:
        """Extract tool-internal metrics already returned in task_state.tool_results."""
        perf = {}
        observations = []
        for result in (task_state.get("tool_results") or []) if isinstance(task_state, dict) else []:
            if not isinstance(result, dict):
                continue
            name = str(result.get("tool") or "")
            observation = result.get("observation") or {}
            if not name or not isinstance(observation, dict):
                continue
            metrics = {}
            for field in ("provider", "tiles", "cache_hit", "vlm_calls", "prompt_tokens",
                          "completion_tokens", "input_tokens", "output_tokens"):
                if observation.get(field) is not None:
                    metrics[field] = observation[field]
            observations.append({"tool": name, "metrics": metrics})
            slot = perf.setdefault(name, {
                "providers": set(), "tiles": set(), "cache_hits": 0, "vlm_calls": 0,
                "prompt_tokens": 0, "completion_tokens": 0, "token_samples": 0,
                "cache_samples": 0, "vlm_samples": 0,
            })
            if observation.get("provider"):
                slot["providers"].add(str(observation["provider"]))
            if observation.get("tiles"):
                slot["tiles"].add(str(observation["tiles"]))
            if isinstance(observation.get("cache_hit"), bool):
                slot["cache_samples"] += 1
                if observation["cache_hit"]:
                    slot["cache_hits"] += 1
            if isinstance(observation.get("vlm_calls"), int):
                slot["vlm_samples"] += 1
                slot["vlm_calls"] += observation["vlm_calls"]
            prompt = observation.get("prompt_tokens", observation.get("input_tokens"))
            completion = observation.get("completion_tokens", observation.get("output_tokens"))
            if isinstance(prompt, (int, float)) or isinstance(completion, (int, float)):
                slot["prompt_tokens"] += int(prompt or 0)
                slot["completion_tokens"] += int(completion or 0)
                slot["token_samples"] += 1
        for slot in perf.values():
            slot["providers"] = sorted(slot["providers"])
            slot["tiles"] = sorted(slot["tiles"])
        return perf, observations

    @staticmethod
    def _attach_tool_observations(tool_trace: list, observations: list[dict]) -> list[dict]:
        """Attach task-state metrics to tool calls by same-name occurrence order."""
        queues = {}
        for observation in observations:
            queues.setdefault(observation.get("tool"), []).append(observation.get("metrics") or {})
        enriched = []
        for trace in tool_trace:
            item = dict(trace)
            queue = queues.get(item.get("tool")) or []
            if queue:
                item["internal_metrics"] = queue.pop(0)
            enriched.append(item)
        return enriched

    @staticmethod
    def _derive_attribution(item: dict) -> dict:
        """Classify available failure layers from saved structured turn fields."""
        judge_score = (item.get("judge") or {}).get("score")
        recall = item.get("retrieval_recall")
        tools = item.get("tool_trace") or []
        calls = item.get("model_call_metrics") or []
        guard = item.get("guard_debug") or {}
        layers = {key: "na" for key in ("R", "V", "O", "T", "S", "G", "J")}
        details = {}
        if isinstance(recall, (int, float)):
            layers["R"] = "pass" if recall >= 1 else "fail"
            details["R"] = f"图片召回率 {recall:.1%}"
        if item.get("angle") in {"scene_visual", "food_or_object"} or item.get("task_type") == "T1_retrieve_media":
            layers["V"] = "pass" if item.get("predicted_file_names") else "fail"
            details["V"] = "已返回图片" if item.get("predicted_file_names") else "未返回可识别图片"
        if any(str(trace.get("tool") or "").lower() in {"read_photo_text", "ocr", "inspect_photo"} for trace in tools):
            ok = any(str(trace.get("status") or "").lower() in {"ok", "complete", "completed", "success"} for trace in tools)
            layers["O"] = "pass" if ok else "fail"
            details["O"] = "检测到 OCR/图片复核工具轨迹"
        if tools:
            failed_tools = [trace for trace in tools if str(trace.get("status") or "").lower() not in {"ok", "complete", "completed", "success"}]
            layers["T"] = "fail" if failed_tools else "pass"
            details["T"] = f"工具 {len(tools)} 次，失败 {len(failed_tools)} 次"
        if calls:
            layers["S"] = "pass" if item.get("answer") else "fail"
            details["S"] = "已生成回答" if item.get("answer") else "无回答"
        if guard or item.get("termination_reason") or item.get("agent_status"):
            blocked = str(item.get("agent_status") or "").lower() in {"blocked", "blocked_by_guard", "error"}
            layers["G"] = "fail" if blocked else "pass"
            details["G"] = item.get("termination_reason") or item.get("agent_status") or "已记录 Guard"
        if judge_score in {0, 1, 2}:
            layers["J"] = "pass" if judge_score == 2 else "fail"
            details["J"] = f"Judge {judge_score} 分"
        primary = next((key for key in ("G", "R", "V", "O", "T", "S", "J") if layers[key] == "fail"), "PASS")
        return {"primary": primary, "layers": layers, "detail": details, "source": "derived_from_turn_result"}

    def _judge(self, question: str, reference: str, answer: str,
               system_prompt: str | None = None, conversation: list[dict] | None = None,
               expected_action: str | None = "answer", task_type: str | None = None,
               question_type: str | None = None, answerability: str | None = None) -> dict:
        if system_prompt is None:
            system_prompt = getattr(self, "judge_system_prompt", None) or JUDGE_PROMPT
        context = ""
        if conversation:
            context = "**截至本轮的对话**：\n" + "\n".join(
                f"第{idx + 1}轮用户：{turn.get('message', '')}\n第{idx + 1}轮回答：{turn.get('answer', '')}"
                for idx, turn in enumerate(conversation)
            ) + "\n"
        rubric = ANSWER_QUALITY_RUBRICS.get(str(expected_action), ANSWER_QUALITY_RUBRICS["answer"])
        judge_text = (f"{context}**当前问题**：{question}\n**模型回答**：【{answer}】\n"
                      f"**预期行为（GT）**：{expected_action or '未标注'}\n"
                      f"**任务类型（GT）**：{task_type or '未标注'}\n"
                      f"**问题类型（GT）**：{question_type or '未标注'}\n"
                      f"**可回答性（GT）**：{answerability or '未标注'}\n"
                      f"**参考答案（GT）**：【{reference}】\n"
                      f"**本题评分规则**：\n{rubric}\n"
                      "仅评价模型回答是否按预期行为有效完成本轮任务。核心信息由截至本轮的用户实际要求决定；参考答案中用户未要求的附加描述不得自动成为扣分项。"
                      "评分前必须专门检查：模型是否否定了当前问题或已确认对话明确给出的事件、对象、记录存在性等前提；若最终结论正确但同时否定了这些已知前提，应按相关事实错误降为1分。"
                      "只输出JSON："
                      '{"score":0|1|2,"reason":"简短中文理由"}')
        payload = {
            "model": getattr(self, "judge_model", JUDGE_MODEL), "temperature": 0, "max_tokens": 1024,
            **_judge_thinking_kwargs(getattr(self, "judge_url", "")),
            "messages": [{"role": "system", "content": system_prompt},
                         {"role": "user", "content": judge_text}],
        }
        try:
            raw, retry_attempts = self._judge_request(payload)
            text = str(raw.get("choices", [{}])[0].get("message", {}).get("content") or "")
            start, end = text.find("{"), text.rfind("}")
            try:
                parsed = json.loads(text[start:end + 1]) if start >= 0 and end > start else {}
            except json.JSONDecodeError:
                parsed = {}
            score = parsed.get("score")
            if score not in {0, 1, 2}:
                score = None
            reason = str(parsed.get("reason") or text[:200])
            if not judge_score_consistency(score, reason):
                retry_system = (
                    f"{system_prompt}\n\n一致性保护：你刚才的评分理由与分数矛盾。"
                    "如果理由确认回答正确、与参考答案一致、符合预期，分数不能是 0；"
                    "请重新核对后只输出一个分数与理由一致的 JSON。"
                )
                retry_payload = {
                    **payload,
                    "messages": [{"role": "system", "content": retry_system}, payload["messages"][1]],
                }
                try:
                    retry_raw, _ = self._judge_request(retry_payload)
                    retry_text = str(retry_raw.get("choices", [{}])[0].get("message", {}).get("content") or "")
                    retry_parsed = self._parse_judge_json(retry_text)
                    retry_score = retry_parsed.get("score") if retry_parsed.get("score") in {0, 1, 2} else None
                    retry_reason = str(retry_parsed.get("reason") or retry_text[:200])
                    if judge_score_consistency(retry_score, retry_reason):
                        score, reason, text, payload = retry_score, retry_reason, retry_text, retry_payload
                    else:
                        score = None
                        reason = "judge_inconsistent_score_reason"
                        payload = retry_payload
                except Exception:
                    score = None
                    reason = "judge_inconsistent_score_reason"
            return {
                "score": score,
                "reason": reason,
                "input": payload,
                "raw_text": text,
                "judge_status": "completed",
                "judge_retry_attempts": retry_attempts,
            }
        except Exception as e:
            return {"score": None, "reason": f"judge_error: {e}", "input": payload,
                    "judge_status": "failed"}

    def _judge_task_action(self, question: str, answer: str, expected_action: str | None,
                           agent_status: str | None, termination_reason: str,
                           task_type: str | None = None, question_type: str | None = None,
                           answerability: str | None = None, reference: str | None = None,
                           conversation: list[dict] | None = None) -> dict:
        if expected_action not in {"answer", "refuse", "clarify"}:
            return {"actual_action": None, "correct": None, "reason": "not_labeled"}
        if not str(answer or "").strip():
            return {
                "expected_action": expected_action,
                "actual_action": "none",
                "correct": False,
                "execution_failure": _execution_failure(agent_status, termination_reason),
                "reason": "empty_answer",
                "input": None,
                "raw_text": "",
            }
        context = ""
        if conversation:
            context = "**截至本轮的对话**：\n" + "\n".join(
                f"第{idx + 1}轮用户：{turn.get('message', '')}\n第{idx + 1}轮回答：{turn.get('answer', '')}"
                for idx, turn in enumerate(conversation)
            ) + "\n"
        prompt = (f"{context}**当前用户问题**：{question}\n**当前模型回答**：【{answer}】\n"
                  f"**预期行为（GT）**：{expected_action}\n"
                  f"**任务类型（GT）**：{task_type or '未标注'}\n"
                  f"**问题类型（GT）**：{question_type or '未标注'}\n"
                  f"**可回答性（GT）**：{answerability or '未标注'}\n"
                  f"**参考答案（GT）**：【{reference or '未提供'}】\n"
                  "结合 GT 能力边界，只判断当前模型回答实际表现为直接回答、拒答还是澄清。只输出 JSON。")
        payload = {
            "model": getattr(self, "judge_model", JUDGE_MODEL), "temperature": 0, "max_tokens": 512,
            **_judge_thinking_kwargs(getattr(self, "judge_url", "")),
            "messages": [{"role": "system", "content": getattr(self, "task_judge_system_prompt", None) or TASK_JUDGE_PROMPT},
                         {"role": "user", "content": prompt}],
        }
        try:
            raw, _ = self._judge_request(payload)
            text = str(raw.get("choices", [{}])[0].get("message", {}).get("content") or "")
            parsed = self._parse_judge_json(text)
            actual = parsed.get("actual_action") if parsed.get("actual_action") in {"answer", "refuse", "clarify", "none"} else None
            return {
                "expected_action": expected_action,
                "actual_action": actual,
                "correct": actual == expected_action if actual else None,
                "execution_failure": _execution_failure(agent_status, termination_reason),
                "reason": str(parsed.get("reason") or text[:200]),
                "input": payload,
                "raw_text": text,
                "judge_status": "completed",
            }
        except Exception as exc:
            return {"expected_action": expected_action, "actual_action": None, "correct": None,
                    "reason": f"judge_error: {exc}", "input": payload,
                    "judge_status": "failed"}

    def _judge_evidence(self, question: str, answer: str, predicted_images: list[dict],
                        assets_by_name: dict, sentrix_url: str,
                        conversation: list[dict] | None = None) -> dict:
        if not answer:
            return {"score": None, "reason": "no_answer"}
        context = ""
        if conversation:
            context = "**截至本轮的对话**：\n" + "\n".join(
                f"第{idx + 1}轮用户：{turn.get('message', '')}\n第{idx + 1}轮回答：{turn.get('answer', '')}"
                for idx, turn in enumerate(conversation)
            ) + "\n"
        content = [{"type": "text", "text": (
            f"{context}**用户问题**：{question}\n**模型最终回答**：【{answer}】\n"
            "以下是本轮实际召回图片。请只根据这些图片评分；若没有图片，模型仍断言具体图片事实则为 0。"
        )}]
        content.extend(_inline_judge_images(predicted_images, assets_by_name, sentrix_url))
        if len(content) == 1:
            return ({"score": None, "reason": "image_attachment_error", "input": None}
                    if predicted_images else
                    {"score": 0, "reason": "no_image_evidence", "input": None})
        payload = {
            "model": getattr(self, "judge_model", JUDGE_MODEL), "temperature": 0, "max_tokens": 512,
            **_judge_thinking_kwargs(getattr(self, "judge_url", "")),
            "messages": [{"role": "system", "content": getattr(self, "evidence_judge_system_prompt", None) or EVIDENCE_JUDGE_PROMPT},
                         {"role": "user", "content": content}],
        }
        try:
            raw, _ = self._judge_request(payload)
            text = str(raw.get("choices", [{}])[0].get("message", {}).get("content") or "")
            parsed = self._parse_judge_json(text)
            applicable = parsed.get("applicable") is not False
            if not applicable:
                return {"score": None, "applicable": False,
                        "reason": str(parsed.get("reason") or "no_visual_claims"),
                        "input": payload, "raw_text": text, "judge_status": "completed"}
            score = parsed.get("score") if parsed.get("score") in {0, 1, 2} else None
            return {"score": score, "applicable": True,
                    "reason": str(parsed.get("reason") or text[:200]),
                    "input": payload, "raw_text": text, "judge_status": "completed"}
        except Exception as exc:
            return {"score": None, "reason": f"judge_error: {exc}", "input": payload,
                    "judge_status": "failed"}

    @staticmethod
    def _parse_judge_json(text: str) -> dict:
        start, end = text.find("{"), text.rfind("}")
        try:
            value = json.loads(text[start:end + 1]) if start >= 0 and end > start else {}
        except json.JSONDecodeError:
            value = {}
        return value if isinstance(value, dict) else {}

    def _judge_chat_url(self) -> str:
        url = self.judge_url.rstrip("/")
        if url.endswith("/chat/completions"):
            return url
        if url.endswith("/v1") or url.endswith("/v3") or url.endswith("/v2"):
            return f"{url}/chat/completions"
        return f"{url}/v1/chat/completions"

    def _judge_request(self, payload: dict, timeout: int = 180) -> tuple[dict, int]:
        """Call remote Judge with bounded exponential-backoff retries."""
        last_error = None
        for attempt in range(1, JUDGE_RETRY_ATTEMPTS + 1):
            try:
                self._wait_for_judge_request_slot()
                response = request_json(
                    self._judge_chat_url(), payload, "POST", timeout,
                    self._judge_headers(),
                )
                if not isinstance(response, dict):
                    raise ValueError("judge response is not an object")
                return response, attempt
            except Exception as exc:
                last_error = exc
                if attempt >= JUDGE_RETRY_ATTEMPTS:
                    break
                delay = self._judge_retry_delay(exc, attempt)
                if self._cancel.wait(delay):
                    raise RunCancelledError("cancelled while retrying Judge request") from exc
        raise RuntimeError(
            f"judge request failed after {JUDGE_RETRY_ATTEMPTS} attempts: {last_error}"
        ) from last_error

    def _wait_for_judge_request_slot(self) -> None:
        """Space cloud requests across Judge workers to avoid synchronized bursts."""
        if JUDGE_REQUEST_INTERVAL_SECONDS <= 0:
            return
        rate_lock = getattr(self, "_judge_rate_lock", None)
        if rate_lock is None:
            rate_lock = self._judge_rate_lock = threading.Lock()
            self._judge_next_request_at = 0.0
        with rate_lock:
            current = time.monotonic()
            wait_seconds = max(0.0, getattr(self, "_judge_next_request_at", 0.0) - current)
            self._judge_next_request_at = max(
                current, getattr(self, "_judge_next_request_at", 0.0)
            ) + JUDGE_REQUEST_INTERVAL_SECONDS
        if wait_seconds and self._cancel.wait(wait_seconds):
            raise RunCancelledError("cancelled while waiting for Judge rate limit")

    @staticmethod
    def _judge_retry_delay(error: Exception, attempt: int) -> float:
        delay = min(
            JUDGE_RETRY_BACKOFF_MAX_SECONDS,
            JUDGE_RETRY_BACKOFF_SECONDS * (2 ** max(0, attempt - 1)),
        )
        current: BaseException | None = error
        seen = set()
        while current is not None and id(current) not in seen:
            seen.add(id(current))
            if isinstance(current, urllib.error.HTTPError):
                raw = str(current.headers.get("Retry-After") or "").strip()
                try:
                    delay = max(delay, float(raw))
                except ValueError:
                    pass
                break
            current = current.__cause__ or current.__context__
        return delay + random.uniform(0.0, min(2.0, delay * 0.2))

    def _judge_headers(self) -> dict[str, str]:
        api_key = getattr(self, "judge_api_key", JUDGE_API_KEY)
        return {"Authorization": f"Bearer {api_key}"} if api_key else {}

    @classmethod
    def _agent_stability(cls, item: dict) -> dict:
        trace = [step for step in item.get("execution_trace") or [] if isinstance(step, dict)]
        model_steps = [step for step in trace
                       if str(step.get("stage") or step.get("type") or "") == "model"
                       and str(step.get("call_type") or "agent") in {"agent", "recovery"}
                       and str(step.get("status") or "complete").lower() != "error"]
        parse_total = len(model_steps)
        # Runtime parse_status is authoritative. Do not scan free-form reasoning
        # text for a JSON fragment: models often mention example JSON while still
        # returning an invalid action.
        def parsed_success(step: dict) -> bool:
            parse_status = step.get("parse_status")
            if parse_status is not None:
                return str(parse_status).lower() == "success" and cls._trace_action(step) is not None
            # Legacy traces predate parse_status; only then use a strict JSON
            # action extraction fallback.
            return cls._trace_action(step) is not None
        parse_success = sum(1 for step in model_steps if parsed_success(step))
        failed_statuses = {"partial", "error", "timeout", "failed", "cancelled", "canceled"}
        status = str(item.get("agent_status") or "").lower()
        termination = str(item.get("termination_reason") or "").lower()
        turn_outcome = item.get("turn_outcome") or (
            (item.get("conversation") or [{}])[-1].get("turn_outcome")
            if item.get("conversation") else None
        )
        complete = turn_outcome == "final_answer" and bool(item.get("answer")) \
            and status not in failed_statuses and not any(
                marker in termination for marker in ("step", "budget", "timeout", "error", "failure"))
        return {
            "json_parse_total": parse_total or None,
            "json_parse_success": parse_success if parse_total else None,
            "json_parse_rate": round(parse_success / parse_total, 3) if parse_total else None,
            "completed_within_steps": complete if trace or item.get("agent_status") else None,
            "final_turn_outcome": turn_outcome,
        }

    def _phase_gpu_metrics(self):
        self._phase_start("gpu_metrics")
        if self.use_cloud_model or not self.vllm_api_url:
            self._phase_done("gpu_metrics", {
                "status": "skipped",
                "source": "cloud_api" if self.use_cloud_model else "external",
                "reason": (
                    "cloud_api_has_no_local_gpu_metrics"
                    if self.use_cloud_model else
                    "external_model_endpoint_has_no_manager_metrics"
                ),
            })
            return
        agg = self._gpu_sampler.aggregate()
        self._phase_done("gpu_metrics", agg)

    def _phase_aggregate(self):
        self._phase_start("aggregate")
        items = self.state["items"]
        recalls = [i["retrieval_recall"] for i in items if isinstance(i.get("retrieval_recall"), (int, float))]
        valid_scores = [score for item in items
                        if (score := judge_score_for_summary(item.get("judge"))) is not None]
        distribution = {str(s): valid_scores.count(s) for s in (0, 1, 2)}
        denom = len(valid_scores)
        llm_summaries = [i.get("llm_summary", {}) for i in items if i.get("llm_summary")]
        avg_ttft = [s["ttft_ms_avg"] for s in llm_summaries if s.get("ttft_ms_avg")]
        avg_tps = [s["tokens_per_second_avg"] for s in llm_summaries if s.get("tokens_per_second_avg")]
        total_prompt = sum(s.get("prompt_tokens_total", 0) for s in llm_summaries)
        total_completion = sum(s.get("completion_tokens_total", 0) for s in llm_summaries)
        prompt_tokens_per_call = []
        completion_tokens_per_call = []
        context_tokens = []
        for item in items:
            for call in item.get("model_call_metrics") or []:
                prompt = call.get("preflight_prompt_tokens")
                if not isinstance(prompt, (int, float)):
                    prompt = call.get("prompt_tokens")
                completion = call.get("completion_tokens")
                if isinstance(prompt, (int, float)):
                    prompt_tokens_per_call.append(prompt)
                if isinstance(completion, (int, float)):
                    completion_tokens_per_call.append(completion)
                if isinstance(prompt, (int, float)) and isinstance(completion, (int, float)):
                    context_tokens.append(prompt + completion)

        summary = {
            # build 模式不做 QA；total 置 0 避免前端把未执行的题数渲染成 0/N。
            "total": len(self.qa_rows) if self.mode != "build" else 0,
            "completed": sum(1 for item in items if item.get("judge_status") in {"completed", "failed", "skipped"}),
            "retrieval_recall_mean": round(sum(recalls) / len(recalls), 3) if recalls else None,
            "judge_distribution": distribution,
            "judge_valid_count": denom,
            "answer_quality_mean": round(sum(valid_scores) / denom, 3) if denom else None,
            "exact_accuracy": round(distribution["2"] / denom, 3) if denom else None,
            "core_accuracy": round((distribution["1"] + distribution["2"]) / denom, 3) if denom else None,
            "llm_ttft_ms_mean": round(sum(avg_ttft) / len(avg_ttft), 1) if avg_ttft else None,
            "llm_tokens_per_second_mean": round(sum(avg_tps) / len(avg_tps), 1) if avg_tps else None,
            "prompt_tokens_total": total_prompt if prompt_tokens_per_call else None,
            "completion_tokens_total": total_completion if completion_tokens_per_call else None,
            "llm_prompt_tokens_max": max(prompt_tokens_per_call) if prompt_tokens_per_call else None,
            "llm_prompt_tokens_p95": nearest_rank_percentile(prompt_tokens_per_call, 0.95),
            "llm_completion_tokens_max": max(completion_tokens_per_call) if completion_tokens_per_call else None,
            "llm_completion_tokens_p95": nearest_rank_percentile(completion_tokens_per_call, 0.95),
            "llm_context_tokens_max": max(context_tokens) if context_tokens else None,
            "llm_context_tokens_p95": nearest_rank_percentile(context_tokens, 0.95),
            "llm_context_samples_count": len(context_tokens),
        }
        summary.update(self._capability_summary(items, self.state.get("phases") or {}))
        summary["benchmark_e2e_latency_excluding_judge_ms"] = self._benchmark_e2e_latency_excluding_judge_ms(
            self.state.get("phases") or {}, items,
        )
        self.state["summary"] = summary
        self._phase_done("aggregate", {"summary": summary})

    @classmethod
    def _capability_summary(cls, items: list[dict], phases: dict | None = None) -> dict:
        # Canonical metric: per-qa_id, take the final (item-level) judge only.
        # Multi-turn conversation items used to be flattened into one judge per
        # turn, inflating the denominator beyond the number of questions.
        answer_judges = [item.get("judge") or {} for item in items]
        answer_scores = [score for judge in answer_judges if (score := judge_score_for_summary(judge)) is not None]
        answer_dist = {str(score): answer_scores.count(score) for score in (0, 1, 2)}
        metric_items = [item for item in items if _retrieval_metric_eligible(item)]
        excluded_unanswerable_count = len(items) - len(metric_items)
        typed_retrieval_items = [item for item in metric_items if "retrieval_media_refs" in item]
        if typed_retrieval_items:
            media_metrics = _micro_metrics_from_counts(typed_retrieval_items, "media_retrieval_counts")
            image_metrics = _micro_metrics_from_counts(typed_retrieval_items, "image_retrieval_counts")
            video_metrics = _micro_metrics_from_counts(typed_retrieval_items, "video_retrieval_counts")
            media_macro = _macro_metrics_from_counts(typed_retrieval_items, "media_retrieval_counts")
            image_macro = _macro_metrics_from_counts(typed_retrieval_items, "image_retrieval_counts")
            video_macro = _macro_metrics_from_counts(typed_retrieval_items, "video_retrieval_counts")
            precision, recall, f1 = (
                media_metrics["precision"], media_metrics["recall"], media_metrics["f1"])
            retrieval_metric_count = media_metrics["metric_count"]
        else:
            # Historical results predate typed refs and are strictly image-only.
            retrieval_items = [item for item in metric_items if item.get("retrieval_image_ids")]
            tp = sum(len(item.get("retrieved_matched_file_names") or item.get("matched_file_names") or []) for item in retrieval_items)
            predicted = sum(len(item.get("retrieved_file_names") or item.get("predicted_file_names") or []) for item in retrieval_items)
            gt = sum(len(item.get("retrieval_image_ids") or []) for item in retrieval_items)
            precision = tp / predicted if predicted else (0.0 if gt else None)
            recall = tp / gt if gt else None
            f1 = 2 * precision * recall / (precision + recall) if precision is not None and recall is not None and precision + recall else (0.0 if gt else None)
            retrieval_metric_count = len(retrieval_items)
            media_metrics = None
            image_metrics = {"precision": precision, "recall": recall, "f1": f1,
                             "metric_count": retrieval_metric_count}
            video_metrics = None
            legacy_values = {
                metric: [item.get(f"retrieval_{metric}") for item in retrieval_items
                         if isinstance(item.get(f"retrieval_{metric}"), (int, float))]
                for metric in ("precision", "recall", "f1")
            }
            image_macro = {metric: sum(values) / len(values) if values else None
                           for metric, values in legacy_values.items()} | {
                               "metric_count": len(legacy_values["recall"])
                           }
            media_macro = image_macro
            video_macro = None
        evidence_judges = [item.get("evidence_judge") or {} for item in items]
        evidence_scores = [judge.get("score") for judge in evidence_judges if judge.get("score") in {0, 1, 2}]
        evidence_dist = {str(score): evidence_scores.count(score) for score in (0, 1, 2)}
        action_judges = [judge for item in items for judge in (
            item.get("task_judges") or [item.get("task_judge") or {}]
        ) if judge.get("expected_action") in {"answer", "refuse", "clarify"}]
        action_valid = [value for value in action_judges if value.get("correct") in {True, False}]
        parse_totals = [item.get("agent_stability", {}).get("json_parse_total") for item in items]
        parse_successes = [item.get("agent_stability", {}).get("json_parse_success") for item in items]
        parse_total = sum(value for value in parse_totals if isinstance(value, int))
        parse_success = sum(value for value in parse_successes if isinstance(value, int))
        completion = [item.get("agent_stability", {}).get("completed_within_steps") for item in items]
        completion_valid = [value for value in completion if isinstance(value, bool)]
        wall_times = [float(item["wall_clock_ms"]) for item in items
                      if isinstance(item.get("wall_clock_ms"), (int, float))]
        judge_times = [float((item.get("timing_breakdown") or {}).get("judge_ms"))
                       for item in items
                       if isinstance((item.get("timing_breakdown") or {}).get("judge_ms"), (int, float))]
        # New runs record the real Agent-only phase wall clock.  This is the
        # concurrency throughput metric: Agent phase wall / number of QA items.
        # It must not be reconstructed by subtracting Judge intervals, because
        # cloud Judge calls can overlap other Agent requests.
        judge_exclusive_ms = None
        agent_throughput_ms = None
        agent_throughput_mode = "historical_interval_estimate"
        agent_phase_wall_ms = cls._numeric_ms((phases or {}).get("qa_eval", {}).get("agent_phase_wall_ms"))
        agent_phase_count = (phases or {}).get("qa_eval", {}).get("agent_completed")
        if not isinstance(agent_phase_count, int) or agent_phase_count <= 0:
            agent_phase_count = len(items)
        timelines = [(item.get("timing_breakdown") or {}).get("timeline") or {} for item in items]
        spans = [(tl.get("started_at_epoch"), tl.get("judge_started_at_epoch"), tl.get("judge_finished_at_epoch")) for tl in timelines]
        ends = []
        for s, item in zip(spans, items):
            wall_ms = (item.get("timing_breakdown") or {}).get("wall_clock_ms")
            if isinstance(s[0], (int, float)) and isinstance(wall_ms, (int, float)):
                ends.append(s[0] + float(wall_ms) / 1000)
        phase = (phases or {}).get("qa_eval") or {}
        try:
            phase_start = datetime.fromisoformat(str(phase.get("started_at"))).timestamp()
            phase_end = datetime.fromisoformat(str(phase.get("finished_at"))).timestamp()
        except (TypeError, ValueError, OSError):
            phase_start = phase_end = None
        valid_item_spans = [s for s in spans if isinstance(s[0], (int, float))]
        if phase_end is not None and phase_start is not None and phase_end > phase_start:
            wall_start, wall_end = phase_start, phase_end
        elif valid_item_spans and ends:
            wall_start = min(s[0] for s in valid_item_spans)
            wall_end = max(e for e in ends if isinstance(e, (int, float)))
        else:
            wall_start = wall_end = None
        judge_spans = sorted((s[1], s[2]) for s in spans
                             if isinstance(s[1], (int, float)) and isinstance(s[2], (int, float)) and s[2] > s[1])
        if wall_start is not None and wall_end is not None and judge_spans:
            merged: list[list[float]] = []
            for a, b in judge_spans:
                if merged and a <= merged[-1][1]:
                    merged[-1][1] = max(merged[-1][1], b)
                else:
                    merged.append([a, b])
            judge_exclusive_ms = round(sum(b - a for a, b in merged) * 1000, 1)
        if agent_phase_wall_ms is not None and agent_phase_wall_ms > 0 and agent_phase_count > 0:
            agent_throughput_ms = round(agent_phase_wall_ms / agent_phase_count, 1)
            agent_throughput_mode = "measured_agent_phase"
        elif wall_start is not None and wall_end is not None and judge_spans and items:
            # Compatibility fallback for old runs that had no Agent/Judge
            # boundary.  Keep it explicitly marked as an estimate.
            agent_throughput_ms = round(max(0.0, (wall_end - wall_start) * 1000 - (judge_exclusive_ms or 0)) / len(items), 1)
        judge_phase = (phases or {}).get("qa_eval") or {}
        judge_phase_wall_ms = cls._numeric_ms(judge_phase.get("judge_phase_wall_ms"))
        agent_wall_times = [float((item.get("timing_breakdown") or {}).get("agent_wall_ms"))
                            for item in items
                            if isinstance((item.get("timing_breakdown") or {}).get("agent_wall_ms"), (int, float))]
        agent_loop_counts = [sum(
            1 for call in item.get("model_call_metrics") or []
            if str(call.get("call_type") or "") in {"agent", "recovery"}
        ) for item in items if isinstance(item.get("model_call_metrics"), list)
                             and item.get("model_call_metrics")
                             and any(call.get("call_type") for call in item["model_call_metrics"])]
        return {
            "judge_distribution": answer_dist,
            "judge_valid_count": len(answer_scores),
            "answer_quality_mean": round(sum(answer_scores) / len(answer_scores), 3) if answer_scores else None,
            "exact_accuracy": round(answer_dist["2"] / len(answer_scores), 3) if answer_scores else None,
            "core_accuracy": round((answer_dist["1"] + answer_dist["2"]) / len(answer_scores), 3) if answer_scores else None,
            "retrieval_precision_micro": round(precision, 3) if precision is not None else None,
            "retrieval_recall_micro": round(recall, 3) if recall is not None else None,
            "retrieval_f1_micro": round(f1, 3) if f1 is not None else None,
            "retrieval_precision_macro": round(media_macro["precision"], 3) if media_macro["precision"] is not None else None,
            "retrieval_recall_macro": round(media_macro["recall"], 3) if media_macro["recall"] is not None else None,
            "retrieval_f1_macro": round(media_macro["f1"], 3) if media_macro["f1"] is not None else None,
            "retrieval_recall_mean": round(media_macro["recall"], 3) if media_macro["recall"] is not None else None,
            "retrieval_metric_count": retrieval_metric_count,
            "retrieval_metric_scope": "all_media" if typed_retrieval_items else "legacy_image_only",
            "retrieval_excluded_unanswerable_count": excluded_unanswerable_count,
            "media_retrieval_precision_micro": round(media_metrics["precision"], 3) if media_metrics and media_metrics["precision"] is not None else None,
            "media_retrieval_recall_micro": round(media_metrics["recall"], 3) if media_metrics and media_metrics["recall"] is not None else None,
            "media_retrieval_f1_micro": round(media_metrics["f1"], 3) if media_metrics and media_metrics["f1"] is not None else None,
            "media_retrieval_precision_macro": round(media_macro["precision"], 3) if media_macro["precision"] is not None else None,
            "media_retrieval_recall_macro": round(media_macro["recall"], 3) if media_macro["recall"] is not None else None,
            "media_retrieval_f1_macro": round(media_macro["f1"], 3) if media_macro["f1"] is not None else None,
            "media_retrieval_metric_count": media_metrics["metric_count"] if media_metrics else None,
            "image_retrieval_precision_micro": round(image_metrics["precision"], 3) if image_metrics and image_metrics["precision"] is not None else None,
            "image_retrieval_recall_micro": round(image_metrics["recall"], 3) if image_metrics and image_metrics["recall"] is not None else None,
            "image_retrieval_f1_micro": round(image_metrics["f1"], 3) if image_metrics and image_metrics["f1"] is not None else None,
            "image_retrieval_precision_macro": round(image_macro["precision"], 3) if image_macro["precision"] is not None else None,
            "image_retrieval_recall_macro": round(image_macro["recall"], 3) if image_macro["recall"] is not None else None,
            "image_retrieval_f1_macro": round(image_macro["f1"], 3) if image_macro["f1"] is not None else None,
            "image_retrieval_metric_count": image_metrics["metric_count"] if image_metrics else None,
            "video_retrieval_precision_micro": round(video_metrics["precision"], 3) if video_metrics and video_metrics["precision"] is not None else None,
            "video_retrieval_recall_micro": round(video_metrics["recall"], 3) if video_metrics and video_metrics["recall"] is not None else None,
            "video_retrieval_f1_micro": round(video_metrics["f1"], 3) if video_metrics and video_metrics["f1"] is not None else None,
            "video_retrieval_precision_macro": round(video_macro["precision"], 3) if video_macro and video_macro["precision"] is not None else None,
            "video_retrieval_recall_macro": round(video_macro["recall"], 3) if video_macro and video_macro["recall"] is not None else None,
            "video_retrieval_f1_macro": round(video_macro["f1"], 3) if video_macro and video_macro["f1"] is not None else None,
            "video_retrieval_metric_count": video_metrics["metric_count"] if video_metrics else None,
            "evidence_distribution": evidence_dist,
            "evidence_valid_count": len(evidence_scores),
            "evidence_mean": round(sum(evidence_scores) / len(evidence_scores), 3) if evidence_scores else None,
            "evidence_fully_supported_rate": round(evidence_dist["2"] / len(evidence_scores), 3) if evidence_scores else None,
            "evidence_basically_supported_rate": round((evidence_dist["1"] + evidence_dist["2"]) / len(evidence_scores), 3) if evidence_scores else None,
            "task_decision_labeled_count": len(action_judges),
            "task_decision_valid_count": len(action_valid),
            "task_decision_accuracy": round(sum(bool(value.get("correct")) for value in action_valid) / len(action_valid), 3) if action_valid else None,
            "json_parse_total": parse_total or None,
            "json_parse_success": parse_success if parse_total else None,
            "json_parse_success_rate": round(parse_success / parse_total, 3) if parse_total else None,
            "qa_completion_valid_count": len(completion_valid),
            "qa_completion_within_steps_rate": round(sum(completion_valid) / len(completion_valid), 3) if completion_valid else None,
            "e2e_latency_mean_ms": round(sum(wall_times) / len(wall_times), 1) if wall_times else None,
            "e2e_latency_p50_ms": nearest_rank_percentile(wall_times, 0.5),
            "e2e_latency_p95_ms": nearest_rank_percentile(wall_times, 0.95),
            "e2e_latency_max_ms": max(wall_times) if wall_times else None,
            "agent_task_latency_mean_ms": round(sum(agent_wall_times) / len(agent_wall_times), 1)
                if agent_wall_times else None,
            "judge_llm_latency_mean_ms": round(sum(judge_times) / len(judge_times), 1)
                if judge_times else None,
            "judge_exclusive_wall_ms": judge_exclusive_ms,
            "judge_phase_wall_ms": judge_phase_wall_ms,
            "judge_concurrency": judge_phase.get("judge_concurrency"),
            "agent_throughput_latency_ms": agent_throughput_ms,
            "agent_throughput_latency_mode": agent_throughput_mode,
            "agent_phase_wall_ms": agent_phase_wall_ms,
            "agent_phase_completed_count": agent_phase_count if agent_phase_wall_ms is not None else None,
            "agent_throughput_qa_per_s": round(1000 * agent_phase_count / agent_phase_wall_ms, 3)
                if agent_phase_wall_ms and agent_phase_count else None,
            "agent_throughput_latency_sample_count": agent_phase_count if agent_phase_wall_ms is not None else len(judge_spans),
            "agent_throughput_latency_total_count": len(items),
            "agent_loop_calls_mean": round(sum(agent_loop_counts) / len(agent_loop_counts), 3)
                if agent_loop_counts else None,
            "agent2_trace": summarize_agent2_trace([
                turn for item in items for turn in (item.get("runtime_turns") or [item])
            ]),
        }

    @staticmethod
    def _benchmark_e2e_latency_excluding_judge_ms(phases: dict, items: list[dict]) -> float | None:
        """Wall time from data import through Agent completion, excluding Judge.

        New runs have an explicit Agent phase boundary.  Only historical runs
        without that boundary use the old per-item subtraction fallback.
        """
        started_at = (phases.get("identity_seed") or {}).get("started_at")
        qa_phase = phases.get("qa_eval") or {}
        finished_at = qa_phase.get("agent_phase_finished_at") or qa_phase.get("finished_at")
        if not started_at or not finished_at:
            return None
        try:
            wall_ms = (datetime.fromisoformat(str(finished_at)) - datetime.fromisoformat(str(started_at))).total_seconds() * 1000
        except (TypeError, ValueError):
            return None
        if qa_phase.get("agent_phase_finished_at"):
            return round(max(0.0, wall_ms), 1)
        judge_values = [(item.get("timing_breakdown") or {}).get("judge_ms") for item in items]
        if items and not all(isinstance(value, (int, float)) for value in judge_values):
            return None
        judge_ms = sum(float(value) for value in judge_values)
        return round(max(0.0, wall_ms - judge_ms), 1)


# ---------------------------------------------------------------------------
# Orchestrator Repository
# ---------------------------------------------------------------------------

class OrchestratorRepository:
    def __init__(self, results_root: Path):
        self.results_root = results_root.resolve()
        self.lock = threading.RLock()
        self.runs: dict[str, BenchmarkRun] = {}
        self.suite_queue: list[dict] = []  # pending suite configs
        self.active_suite_run_ids: list[str] = []
        self.rejudge_threads: dict[str, threading.Thread] = {}
        self.memory_profile_thread: threading.Thread | None = None
        self.qa_metadata = self._load_qa_metadata()
        self._load_existing()

    @staticmethod
    def _load_qa_metadata() -> dict[str, dict]:
        fields = ("task_type", "question_type", "tags", "angle", "difficulty", "answerability",
                  "scope", "scope_anchor", "required_evidence_sources", "query_anchors",
                  "expected_action", "answer", "conversation")
        result = {}
        for manifest_path in BENCHMARK_DATA_ROOT.glob("*/manifest.json"):
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                for qa_set, relative in (manifest.get("qa_sets") or {}).items():
                    for row in apply_task_action_defaults(
                        load_jsonl(manifest_path.parent / relative),
                        str(manifest.get("album_id") or manifest_path.parent.name), str(qa_set),
                    ):
                        qa_id = str(row.get("qa_id") or "")
                        if qa_id:
                            result[qa_id] = {field: row[field] for field in fields if row.get(field) is not None}
            except (OSError, TypeError, json.JSONDecodeError):
                continue
        return result

    @staticmethod
    def _rejudge_targets(items: list[dict]) -> list[tuple[int, int | None]]:
        targets = []
        for item_index, item in enumerate(items):
            if (item.get("question") is None or item.get("reference_answer") is None
                    or "answer" not in item):
                continue
            turns = item.get("conversation") or []
            targets.extend(
                (item_index, turn_index)
                for turn_index in (range(len(turns)) if turns else [None])
            )
        return targets

    def _hydrate_qa_metadata(self, item: dict) -> dict:
        metadata = self.qa_metadata.get(str(item.get("qa_id") or "")) or {}
        hydrated = {**metadata, **item}
        tag_fields = {
            "task_type": "task",
            "question_type": "question",
            "angle": "angle",
            "difficulty": "difficulty",
            "answerability": "answerability",
            "expected_action": "action",
        }
        tags = [str(tag).strip() for tag in (hydrated.get("tags") or []) if str(tag).strip()]
        for field, prefix in tag_fields.items():
            value = str(hydrated.get(field) or "").strip()
            if value:
                tags.append(f"{prefix}:{value}")
        hydrated["tags"] = list(dict.fromkeys(tags))
        judge = hydrated.get("judge") or {}
        consistency_status = judge_consistency_status(judge)
        if consistency_status:
            hydrated["judge"] = {**judge, "consistency_status": consistency_status}
        # P/R/F1 are deterministic and can be reconstructed for old runs only
        # when both the GT and actual predicted file names were retained.
        gt_names = {Path(str(value)).name for value in hydrated.get("retrieval_image_ids") or []}
        predicted_names = {Path(str(value)).name for value in hydrated.get("predicted_file_names") or []}
        if gt_names and "predicted_file_names" in hydrated:
            matched = gt_names & predicted_names
            precision = len(matched) / len(predicted_names) if predicted_names else 0.0
            recall = len(matched) / len(gt_names)
            f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
            if hydrated.get("retrieval_precision") is None:
                hydrated["retrieval_precision"] = precision
            if hydrated.get("retrieval_recall") is None:
                hydrated["retrieval_recall"] = recall
            if hydrated.get("retrieval_f1") is None:
                hydrated["retrieval_f1"] = f1
        # Historical results can be reconstructed from saved structured fields.
        if not hydrated.get("attribution"):
            hydrated["attribution"] = BenchmarkRun._derive_attribution(hydrated)
        hydrated["tool_trace"] = BenchmarkRun._bind_tool_calls_to_model_rounds(
            hydrated.get("tool_trace") or [], hydrated.get("execution_trace") or [],
            hydrated.get("model_call_metrics") or []
        )
        hydrated["model_call_metrics"] = BenchmarkRun._annotate_model_calls(
            hydrated.get("model_call_metrics") or [], hydrated.get("execution_trace") or []
        )
        agent_steps = [step for step in hydrated.get("execution_trace") or []
                       if isinstance(step, dict)
                       and str(step.get("stage") or step.get("type") or "") == "model"
                       and str(step.get("call_type") or "agent") in {"agent", "recovery"}]
        if agent_steps and any(step.get("parse_status") is not None for step in agent_steps):
            hydrated["agent_stability"] = BenchmarkRun._agent_stability(hydrated)
        return hydrated

    def _reviews_path(self, run_id: str) -> Path:
        return self.results_root / run_id / "reviews.json"

    def get_reviews(self, run_id: str) -> dict:
        if run_id not in self.runs:
            raise KeyError(run_id)
        path = self._reviews_path(run_id)
        if not path.is_file():
            return {"reviews": {}}
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
            return {"reviews": value.get("reviews") or {}, "updated_at": value.get("updated_at")}
        except (OSError, json.JSONDecodeError):
            return {"reviews": {}}

    def save_reviews(self, run_id: str, payload: dict) -> dict:
        with self.lock:
            run = self.runs.get(run_id)
            if not run:
                raise KeyError(run_id)
            state = run.state if isinstance(run, BenchmarkRun) else run
            if state.get("status") in {"running", "pending"}:
                raise ValueError("cannot change reviews while benchmark is running")
            provided = payload.get("reviews")
            if not isinstance(provided, dict):
                raise ValueError("reviews must be an object keyed by qa_id")
            valid_ids = {str(item.get("qa_id")) for item in state.get("items") or [] if item.get("qa_id")}
            reviews = {}
            for qa_id, review in provided.items():
                if str(qa_id) not in valid_ids or not isinstance(review, dict):
                    continue
                verdict = str(review.get("verdict") or "")
                note = str(review.get("note") or "").strip()
                if verdict not in {"correct", "partial", "wrong"}:
                    continue
                reviews[str(qa_id)] = {"verdict": verdict, "note": note, "updated_at": now_iso()}
            stored = {"updated_at": now_iso(), "reviews": reviews}
            atomic_json(self._reviews_path(run_id), stored)
            return {"saved": len(reviews), **stored}

    def _load_existing(self):
        if not self.results_root.exists():
            return
        dirty_statuses = {"running", "pending", "cancelling"}
        for path in self.results_root.glob("*/run.json"):
            try:
                run = json.loads(path.read_text(encoding="utf-8"))
                rid = run.get("run_id")
                if not rid:
                    continue
                changed = False
                status = run.get("status")
                if status in dirty_statuses:
                    run["status"] = "interrupted" if status == "running" else "cancelled"
                    run["finished_at"] = run.get("finished_at") or now_iso()
                    for ph in (run.get("phases") or {}).values():
                        if ph.get("status") in dirty_statuses:
                            ph["status"] = "interrupted" if status == "running" else "cancelled"
                            ph["error"] = "Process restarted"
                    changed = True
                rejudge = run.get("rejudge") or {}
                if rejudge.get("status") in dirty_statuses:
                    rejudge["status"] = "interrupted"
                    rejudge["finished_at"] = rejudge.get("finished_at") or now_iso()
                    rejudge["error"] = "Process restarted"
                    changed = True
                if changed:
                    atomic_json(path, run)
                # Store as a lightweight dict for listing (not a full BenchmarkRun)
                self.runs[rid] = run
            except (OSError, KeyError, json.JSONDecodeError):
                continue

    def list_manifests(self) -> list[dict]:
        manifests = []
        if BENCHMARK_DATA_ROOT.exists():
            for d in sorted(BENCHMARK_DATA_ROOT.iterdir()):
                mf = d / "manifest.json"
                if mf.is_file():
                    try:
                        m = json.loads(mf.read_text(encoding="utf-8"))
                        manifests.append({
                            "album_id": m["album_id"], "album_name": m["album_name"],
                            "face_count": len(m.get("faces", [])),
                            "photo_count": len(m.get("photos", [])),
                            "video_count": len(m.get("videos", [])),
                            "qa_sets": list(m.get("qa_sets", {}).keys()),
                        })
                    except (OSError, KeyError, json.JSONDecodeError):
                        continue
        return manifests

    def get_manifest(self, album_id: str) -> dict | None:
        mf = BENCHMARK_DATA_ROOT / album_id / "manifest.json"
        if mf.is_file():
            return json.loads(mf.read_text(encoding="utf-8"))
        return None

    def query_profiles(self, vllm_api_url: str) -> dict:
        profiles = []
        error = None
        try:
            remote_profiles = request_json(f"{vllm_api_url.rstrip('/')}/profiles", timeout=15)
            if isinstance(remote_profiles, list):
                profiles = remote_profiles
            elif isinstance(remote_profiles, dict):
                profiles = remote_profiles.get("profiles") or []
        except Exception as e:
            error = str(e)
        result = {"profiles": profiles}
        if error:
            result["error"] = error
        return result

    def query_current_model(self, vllm_api_url: str, model_base_url: str) -> dict:
        manager_error = None
        state = {}
        if vllm_api_url:
            try:
                state = request_json(f"{vllm_api_url.rstrip('/')}/state", timeout=10) or {}
            except Exception as exc:
                manager_error = str(exc)
        model_base_url = normalize_model_base_url(model_base_url)
        if not model_base_url:
            raise ValueError("model endpoint is required; enter host:port or an OpenAI /v1 URL")
        live_models = request_json(f"{model_base_url}/models", timeout=15) or {}
        served_models = [
            str(item.get("id")) for item in live_models.get("data") or []
            if isinstance(item, dict) and item.get("id")
        ]
        served_name = str(state.get("served_model_name") or "").strip()
        if served_name and served_name not in served_models:
            raise ValueError(
                f"manager state is stale or mismatched: expected {served_name}, "
                f"live endpoint serves {served_models or 'nothing'}"
            )
        if not served_name:
            if len(served_models) != 1:
                raise ValueError(
                    f"model endpoint must expose exactly one model when no Manager is configured; "
                    f"found {served_models or 'nothing'}"
                )
            served_name = served_models[0]
        profile = str(state.get("profile") or "").strip()
        model_id = profile or served_name
        return {
            "model_id": model_id,
            "served_model_name": served_name,
            "verified_at": now_iso(),
            "served_models": served_models,
            "state": state,
            "model_base_url": model_base_url,
            "manager_available": bool(state),
            "manager_error": manager_error,
        }

    def start_memory_profile(self, payload: dict) -> dict:
        """Replay saved questions to measure comparable memory without changing QA results."""
        run_ids = [str(value) for value in payload.get("run_ids") or []]
        if not run_ids:
            raise ValueError("run_ids is required")
        with self.lock:
            cloud_runs = []
            for rid in run_ids:
                run = self.runs.get(rid)
                state = run.state if isinstance(run, BenchmarkRun) else run or {}
                if state.get("model_source") == "cloud_api":
                    cloud_runs.append(rid)
        if cloud_runs:
            raise ValueError("cloud API model does not support local GPU memory profiling")
        target_id, target = resolve_vllm_target(payload.get("vllm_target_id"))
        manager_url = str(target["manager_url"]).rstrip("/")
        sentrix_url = str(payload.get("sentrix_url") or DEFAULT_SENTRIX_URL).rstrip("/")
        with self.lock:
            active = [rid for rid, run in self.runs.items()
                      if (run.state if isinstance(run, BenchmarkRun) else run).get("status")
                      in {"running", "pending", "cancelling"}]
            if active:
                raise ValueError(f"benchmark suite is active: {', '.join(active)}")
            if self.memory_profile_thread and self.memory_profile_thread.is_alive():
                raise ValueError("memory profiling is already running")
            missing = [rid for rid in run_ids if rid not in self.runs]
            if missing:
                raise ValueError(f"run not found: {', '.join(missing)}")
            task_id = f"memory-{datetime.now().strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"
            for rid in run_ids:
                state = self.runs[rid].state if isinstance(self.runs[rid], BenchmarkRun) else self.runs[rid]
                items_sha256 = hashlib.sha256(json.dumps(
                    state.get("items") or [], ensure_ascii=False, sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")).hexdigest()
                state["memory_profile"] = {
                    "task_id": task_id, "status": "pending", "source_run_id": rid,
                    "method": "saved_question_replay", "created_at": now_iso(),
                    "answers_persisted": False, "album_reused": True,
                    "items_sha256_before": items_sha256,
                }
                atomic_json(self.results_root / rid / "run.json", state)

        def persist(rid: str, state: dict) -> None:
            with self.lock:
                atomic_json(self.results_root / rid / "run.json", state)

        def execute() -> None:
            for rid in run_ids:
                with self.lock:
                    holder = self.runs[rid]
                    state = holder.state if isinstance(holder, BenchmarkRun) else holder
                    profile = str(state.get("model_profile") or "")
                    record = state["memory_profile"]
                    record.update({"status": "running", "started_at": now_iso()})
                    persist(rid, state)
                sampler = GpuSampler(manager_url, interval=0.1)
                completed, failed, request_count = 0, 0, 0
                try:
                    try:
                        request_json(f"{manager_url}/stop", {"timeout": 60}, "POST", 90)
                    except Exception:
                        pass
                    time.sleep(5)
                    request_json(f"{manager_url}/start", {
                        "profile": profile, "wait_ready": True, "ready_timeout": 600,
                    }, "POST", 700)
                    model_state = request_json(f"{manager_url}/state", timeout=10)
                    request_json(f"{sentrix_url}/api/model-profiles/bind-runtime", {
                        "manager_url": manager_url,
                        "model_base_url": str(target["model_base_url"]),
                    }, "POST", 30)
                    # Capture an idle sample before requests so the reserved KV pool
                    # can be separated from fixed model/runtime memory.
                    sampler.start()
                    time.sleep(2)
                    for item in state.get("items") or []:
                        turns = item.get("conversation") or []
                        messages = [str(turn.get("message") or "") for turn in turns if turn.get("message")]
                        if not messages and item.get("question"):
                            messages = [str(item["question"])]
                        conversation_id = f"{task_id}-{uuid.uuid4().hex[:8]}" if len(messages) > 1 else None
                        item_ok = True
                        for message in messages:
                            request_count += 1
                            try:
                                initial = request_json(f"{sentrix_url}/api/assistant/turn", {
                                    "message": message,
                                    "scope_id": state.get("scope_id"),
                                    "conversation_id": conversation_id,
                                    "viewer_id": "owner",
                                    "include_debug": False,
                                }, "POST", 300)
                                wait_for_assistant_turn(sentrix_url, initial, timeout=900)
                            except Exception:
                                item_ok = False
                                failed += 1
                        if item_ok:
                            completed += 1
                        record.update({
                            "questions_completed": completed,
                            "questions_total": len(state.get("items") or []),
                            "requests_completed": request_count - failed,
                            "requests_total_so_far": request_count,
                            "failed_requests": failed,
                        })
                        persist(rid, state)
                    time.sleep(1)
                    sampler.stop()
                    measured = sampler.aggregate()
                    record.update({
                        **measured,
                        "status": "completed",
                        "finished_at": now_iso(),
                        "model_profile": profile,
                        "model_state": {
                            key: model_state.get(key) for key in (
                                "dtype", "quantization", "load_format", "gpu_memory_utilization",
                                "max_model_len", "max_num_seqs", "default_max_tokens",
                                "tensor_parallel_size", "enable_lora",
                            )
                        },
                        "answers_persisted": False,
                        "original_items_untouched": True,
                    })
                except Exception as exc:
                    sampler.stop()
                    record.update({
                        **sampler.aggregate(), "status": "failed", "finished_at": now_iso(),
                        "error": str(exc), "answers_persisted": False,
                    })
                finally:
                    record["items_sha256_after"] = hashlib.sha256(json.dumps(
                        state.get("items") or [], ensure_ascii=False, sort_keys=True,
                        separators=(",", ":"),
                    ).encode("utf-8")).hexdigest()
                    record["items_integrity_ok"] = (
                        record.get("items_sha256_before") == record.get("items_sha256_after")
                    )
                    persist(rid, state)

        thread = threading.Thread(target=execute, name=task_id, daemon=True)
        with self.lock:
            self.memory_profile_thread = thread
        thread.start()
        return {"task_id": task_id, "run_ids": run_ids, "status": "running",
                "vllm_target_id": target_id}

    def list_runs(self) -> list[dict]:
        with self.lock:
            result = []
            for rid, run in self.runs.items():
                if isinstance(run, BenchmarkRun):
                    state = run.state
                else:
                    state = run
                public = self._public_run(state, include_items=False)
                public["item_count"] = len(state.get("items") or [])
                public["summary"] = self._list_summary(state)
                result.append(public)
            return sorted(result, key=lambda r: r.get("started_at") or "", reverse=True)

    def get_run(self, run_id: str) -> dict:
        with self.lock:
            run = self.runs.get(run_id)
            if not run:
                raise KeyError(run_id)
            state = run.state if isinstance(run, BenchmarkRun) else run
            result = self._public_run(state, include_items=False)
            result["item_count"] = len(state.get("items") or [])
            result["summary"] = self._effective_summary(state)
            return result

    def export_sft(self, run_id: str, scores: list[int] | None = None,
                   min_score: int | None = None) -> dict:
        """导出完整思考轨迹：每题全部 debug_trace 步（含提示词/工具参数/工具结果/守卫/judge）、
        agent2_trace（task_state/evidence_ledger/answer_context/stage_timing）与评测元数据。

        scores: 非空时只导出答案评分落在该集合内的题目（勾选过滤，勾选什么导出什么）。
        min_score: 兼容旧参数；非 None 时只导出评分 >= min_score 的题目。
        """
        import ast as _ast
        def _parse(value):
            if isinstance(value, dict) or not isinstance(value, str) or not value.strip():
                return value
            try:
                return _ast.literal_eval(value)
            except Exception:
                return value
        with self.lock:
            run = self.runs.get(run_id)
            if not run:
                raise KeyError(run_id)
            state = run.state if isinstance(run, BenchmarkRun) else run
            items_out = []
            samples = []
            include = set(scores) if scores else None
            for item in (state.get("items") or []):
                if include is not None:
                    item_score = (item.get("judge") or {}).get("score")
                    if item_score not in include:
                        continue
                elif min_score is not None:
                    item_score = (item.get("judge") or {}).get("score")
                    if item_score is None or item_score < min_score:
                        continue
                qa_id = item.get("qa_id")
                turns_out = []
                for turn in (item.get("runtime_turns") or []):
                    steps_out = []
                    for step in (turn.get("debug_trace") or []):
                        if not isinstance(step, dict):
                            continue
                        st = dict(step)
                        if isinstance(st.get("arguments"), str):
                            st["arguments"] = _parse(st["arguments"])
                        if isinstance(st.get("observation"), str):
                            st["observation"] = _parse(st["observation"])
                        steps_out.append(st)
                    turns_out.append({
                        "turn": turn.get("index"),
                        "message": turn.get("message"),
                        "expected_action": turn.get("expected_action"),
                        "answer": turn.get("answer"),
                        "agent_status": turn.get("agent_status"),
                        "turn_outcome": turn.get("turn_outcome"),
                        "steps": steps_out,
                    })
                    # SFT prompt->response 样本（兼容旧格式；完整轨迹见 items）
                    for step in (turn.get("debug_trace") or []):
                        if not isinstance(step, dict) or step.get("type") != "model":
                            continue
                        prompt = step.get("prompt")
                        response = step.get("raw_full") or step.get("raw")
                        if not prompt or not isinstance(response, str) or not response.strip():
                            continue
                        messages = list(prompt) if isinstance(prompt, list) else []
                        messages.append({"role": "assistant", "content": response})
                        samples.append({
                            "qa_id": qa_id,
                            "turn": turn.get("index"),
                            "step": step.get("step_id") or step.get("status", "complete"),
                            "messages": messages,
                        })
                items_out.append({
                    "qa_id": qa_id,
                    "question": item.get("question"),
                    "expected_action": item.get("expected_action"),
                    "answer": item.get("answer"),
                    "judge": item.get("judge"),
                    "task_judge": item.get("task_judge"),
                    "retrieval": {k: item.get(k) for k in (
                        "retrieval_recall", "retrieval_precision", "retrieval_f1",
                        "media_retrieval_recall", "media_retrieval_precision", "media_retrieval_f1",
                        "image_retrieval_recall", "image_retrieval_precision", "image_retrieval_f1",
                        "video_retrieval_recall", "video_retrieval_precision", "video_retrieval_f1",
                        "retrieval_media_refs", "answer_evidence_media_refs",
                        "gt_media", "predicted_media", "retrieved_candidate_media", "evidence_source_media",
                        "gt_images", "predicted_images", "predicted_file_names",
                        "matched_file_names")},
                    "agent_status": item.get("agent_status"),
                    "termination_reason": item.get("termination_reason"),
                    "turn_outcome": item.get("turn_outcome"),
                    "timing_breakdown": item.get("timing_breakdown"),
                    "agent_stability": item.get("agent_stability"),
                    "answer_grounding": item.get("answer_grounding"),
                    "tool_trace": item.get("tool_trace"),
                    "agent2_trace": item.get("agent2_trace"),
                    "turns": turns_out,
                })
            return {"run_id": run_id, "count": len(samples), "item_count": len(items_out),
                    "items": items_out, "samples": samples,
                    "scores": sorted(include) if include is not None else None,
                    "min_score": min_score,
                    "filtered": (include is not None or min_score is not None)}

    def get_run_items(self, run_id: str, page: int = 1, page_size: int = 20,
                      search: str = "", score: str = "", task_type: str = "", tag: str = "",
                      agent_status: str = "", angle: str = "", difficulty: str = "",
                      answerability: str = "", primary: str = "") -> dict:
        with self.lock:
            run = self.runs.get(run_id)
            if not run:
                raise KeyError(run_id)
            state = run.state if isinstance(run, BenchmarkRun) else run
            if not isinstance(run, BenchmarkRun):
                self._hydrate_legacy_gt_images(state)
            all_items = [self._hydrate_qa_metadata(item) for item in (state.get("items") or [])]
            search = str(search or "").strip().lower()
            task_type = str(task_type or "").strip()
            tag = str(tag or "").strip()
            agent_status = str(agent_status or "").strip()
            angle = str(angle or "").strip()
            difficulty = str(difficulty or "").strip()
            answerability = str(answerability or "").strip()
            primary = str(primary or "").strip().upper()
            score_filter = None
            if str(score).strip() in {"0", "1", "2"}:
                score_filter = int(score)
            items = []
            source_indexes = []
            facets = {
                "task_types": sorted({str(item.get("task_type")) for item in all_items if item.get("task_type")}),
                "tags": sorted({str(tag) for item in all_items for tag in (item.get("tags") or []) if str(tag).strip()}),
                "agent_statuses": sorted({str(item.get("agent_status")) for item in all_items if item.get("agent_status")}),
                "angles": sorted({str(item.get("angle")) for item in all_items if item.get("angle")}),
                "difficulties": sorted({str(item.get("difficulty")) for item in all_items if item.get("difficulty")}),
                "answerabilities": sorted({str(item.get("answerability")) for item in all_items if item.get("answerability")}),
                "attribution_layers": sorted({str((item.get("attribution") or {}).get("primary")) for item in all_items if (item.get("attribution") or {}).get("primary")}),
            }
            for source_index, item in enumerate(all_items):
                if search and search not in str(item.get("question") or "").lower() \
                        and search not in str(item.get("qa_id") or "").lower():
                    continue
                if score_filter is not None and (item.get("judge") or {}).get("score") != score_filter:
                    continue
                if task_type and item.get("task_type") != task_type:
                    continue
                if tag and tag not in (item.get("tags") or []):
                    continue
                if agent_status and item.get("agent_status") != agent_status:
                    continue
                if angle and item.get("angle") != angle:
                    continue
                if difficulty and item.get("difficulty") != difficulty:
                    continue
                if answerability and item.get("answerability") != answerability:
                    continue
                if primary and str((item.get("attribution") or {}).get("primary") or "").upper() != primary:
                    continue
                items.append(item)
                source_indexes.append(source_index)
            page_size = max(1, min(int(page_size), 100))
            total = len(items)
            pages = max(1, math.ceil(total / page_size))
            page = max(1, min(int(page), pages))
            start = (page - 1) * page_size
            reviews = self.get_reviews(run_id).get("reviews") or {}
            summaries = [self._item_summary(item, source_index, reviews.get(str(item.get("qa_id")))) for item, source_index in zip(
                items[start:start + page_size], source_indexes[start:start + page_size])]
            return {
                "items": summaries,
                "page": page,
                "page_size": page_size,
                "total": total,
                "pages": pages,
                "has_previous": page > 1,
                "has_next": page < pages,
                "unfiltered_total": len(all_items),
                "filters": {
                    "search": search,
                    "score": score_filter,
                    "task_type": task_type,
                    "tag": tag,
                    "agent_status": agent_status,
                    "angle": angle,
                    "difficulty": difficulty,
                    "answerability": answerability,
                    "primary": primary,
                },
                "facets": facets,
            }

    def get_run_item(self, run_id: str, item_index: int) -> dict:
        with self.lock:
            run = self.runs.get(run_id)
            if not run:
                raise KeyError(run_id)
            state = run.state if isinstance(run, BenchmarkRun) else run
            if not isinstance(run, BenchmarkRun):
                self._hydrate_legacy_gt_images(state)
            items = state.get("items") or []
            if item_index < 0 or item_index >= len(items):
                raise KeyError(f"item index out of range: {item_index}")
            item = self._hydrate_qa_metadata(items[item_index])
            item["review"] = (self.get_reviews(run_id).get("reviews") or {}).get(str(item.get("qa_id")))
            return {"index": item_index, "item": item}

    def get_run_judge_prompt(self, run_id: str) -> dict:
        with self.lock:
            run = self.runs.get(run_id)
            if not run:
                raise KeyError(run_id)
            state = run.state if isinstance(run, BenchmarkRun) else run
            for item in state.get("items") or []:
                messages = ((item.get("judge") or {}).get("input") or {}).get("messages")
                if not isinstance(messages, list):
                    continue
                for message in messages:
                    if isinstance(message, dict) and message.get("role") == "system":
                        return {"system_prompt": str(message.get("content") or "")}
            return {"system_prompt": None}

    @staticmethod
    def _item_summary(item: dict, index: int, review: dict | None = None) -> dict:
        judge = item.get("judge") or {}
        return {
            "index": index,
            "qa_id": item.get("qa_id"),
            "question": item.get("question"),
            "task_type": item.get("task_type"),
            "question_type": item.get("question_type"),
            "tags": item.get("tags") or [],
            "angle": item.get("angle"),
            "difficulty": item.get("difficulty"),
            "answerability": item.get("answerability"),
            "retrieval_recall": item.get("retrieval_recall"),
            "retrieval_precision": item.get("retrieval_precision"),
            "retrieval_f1": item.get("retrieval_f1"),
            "media_retrieval_recall": item.get("media_retrieval_recall"),
            "media_retrieval_precision": item.get("media_retrieval_precision"),
            "media_retrieval_f1": item.get("media_retrieval_f1"),
            "image_retrieval_recall": item.get("image_retrieval_recall"),
            "image_retrieval_precision": item.get("image_retrieval_precision"),
            "image_retrieval_f1": item.get("image_retrieval_f1"),
            "video_retrieval_recall": item.get("video_retrieval_recall"),
            "video_retrieval_precision": item.get("video_retrieval_precision"),
            "video_retrieval_f1": item.get("video_retrieval_f1"),
            "matched_count": len(item.get("matched_file_names") or []),
            "ground_truth_count": len(item.get("retrieval_media_refs") or item.get("retrieval_image_ids") or []),
            "ground_truth_image_count": len(item.get("retrieval_image_ids") or []),
            "ground_truth_video_count": len(item.get("retrieval_video_ids") or []),
            "judge": {
                "score": judge.get("score"),
                "status": judge.get("status"),
                "rejudge_id": judge.get("rejudge_id"),
                "consistency_status": judge_consistency_status(judge),
            },
            "task_judge": item.get("task_judge") or {},
            "task_judges": item.get("task_judges") or [],
            "evidence_judge": item.get("evidence_judge") or {},
            "agent_stability": item.get("agent_stability") or {},
            "agent_status": item.get("agent_status"),
            "termination_reason": item.get("termination_reason"),
            "has_error": bool(item.get("error")),
            "deterministic_delivery": ((item.get("guard_debug") or {}).get("deterministic_delivery") or {}),
            "delivery_status": item.get("delivery_status") or {},
            "model_call_count": len(item.get("model_call_metrics") or []),
            "tool_call_count": len(item.get("tool_trace") or []),
            "attribution": item.get("attribution") or {},
            "review": review,
        }

    @staticmethod
    def _percentile(values: list[float], percentile: float):
        values = sorted(float(value) for value in values if isinstance(value, (int, float)))
        if not values:
            return None
        return values[max(0, min(len(values) - 1, math.ceil(len(values) * percentile) - 1))]

    @classmethod
    def _effective_summary(cls, state: dict) -> dict:
        saved = dict(state.get("summary") or {})
        items = state.get("items") or []
        recalls = [item.get("retrieval_recall") for item in items
                   if isinstance(item.get("retrieval_recall"), (int, float))]
        scores = [score for item in items
                  if (score := judge_score_for_summary(item.get("judge"))) is not None]
        calls = [call for item in items for call in (item.get("model_call_metrics") or [])
                 if isinstance(call, dict)]
        prompts = [call.get("preflight_prompt_tokens", call.get("prompt_tokens")) for call in calls]
        prompts = [value for value in prompts if isinstance(value, (int, float))]
        completions = [call.get("completion_tokens") for call in calls
                       if isinstance(call.get("completion_tokens"), (int, float))]
        contexts = [
            call.get("preflight_prompt_tokens", call.get("prompt_tokens")) + call.get("completion_tokens")
            for call in calls
            if isinstance(call.get("preflight_prompt_tokens", call.get("prompt_tokens")), (int, float))
            and isinstance(call.get("completion_tokens"), (int, float))
        ]
        ttfts = [call.get("ttft_ms") for call in calls if isinstance(call.get("ttft_ms"), (int, float))]
        rates = [call.get("tokens_per_second") for call in calls
                 if isinstance(call.get("tokens_per_second"), (int, float))]
        distribution = {str(score): scores.count(score) for score in (0, 1, 2)}
        attributions = [item.get("attribution") or BenchmarkRun._derive_attribution(item) for item in items]
        attribution_primary = {}
        attribution_layers = {}
        for attribution in attributions:
            primary = attribution.get("primary")
            if primary:
                attribution_primary[str(primary)] = attribution_primary.get(str(primary), 0) + 1
            for key, value in (attribution.get("layers") or {}).items():
                if value == "fail":
                    attribution_layers[str(key)] = attribution_layers.get(str(key), 0) + 1
        saved.update({
            "total": saved.get("total", state.get("qa_count") or len(items)),
            "completed": sum(1 for item in items if item.get("judge_status") in {"completed", "failed", "skipped"}),
            "judge_valid_count": len(scores),
            "judge_distribution": distribution,
            "retrieval_recall_mean": saved.get("retrieval_recall_mean", round(sum(recalls) / len(recalls), 3) if recalls else None),
            "answer_quality_mean": round(sum(scores) / len(scores), 3) if scores else None,
            "exact_accuracy": round(distribution["2"] / len(scores), 3) if scores else None,
            "core_accuracy": round((distribution["1"] + distribution["2"]) / len(scores), 3) if scores else None,
            "llm_ttft_ms_mean": saved.get("llm_ttft_ms_mean", round(sum(ttfts) / len(ttfts), 1) if ttfts else None),
            "llm_tokens_per_second_mean": saved.get("llm_tokens_per_second_mean", round(sum(rates) / len(rates), 1) if rates else None),
            "prompt_tokens_total": (saved.get("prompt_tokens_total") if saved.get("prompt_tokens_total") is not None else sum(prompts)) if calls else None,
            "completion_tokens_total": (saved.get("completion_tokens_total") if saved.get("completion_tokens_total") is not None else sum(completions)) if calls else None,
            "llm_prompt_tokens_max": saved.get("llm_prompt_tokens_max", max(prompts) if prompts else None),
            "llm_prompt_tokens_p95": saved.get("llm_prompt_tokens_p95", cls._percentile(prompts, 0.95)),
            "llm_completion_tokens_max": saved.get("llm_completion_tokens_max", max(completions) if completions else None),
            "llm_completion_tokens_p95": saved.get("llm_completion_tokens_p95", cls._percentile(completions, 0.95)),
            "llm_context_tokens_max": saved.get("llm_context_tokens_max", max(contexts) if contexts else None),
            "llm_context_tokens_p95": saved.get("llm_context_tokens_p95", cls._percentile(contexts, 0.95)),
            "llm_context_samples_count": saved.get("llm_context_samples_count", len(contexts)),
            "tool_performance": cls._aggregate_tool_performance(items),
            "attribution": {"primary": attribution_primary, "layer_failures": attribution_layers},
            "delivery_breakdown": saved.get("delivery_breakdown", cls._aggregate_delivery(items)),
        })
        for key, value in BenchmarkRun._capability_summary(items, state.get("phases") or {}).items():
            if (key == "retrieval_recall_mean"
                    or key.startswith(("retrieval_", "media_retrieval_", "image_retrieval_", "video_retrieval_"))
                    or saved.get(key) is None):
                saved[key] = value
        if saved.get("benchmark_e2e_latency_excluding_judge_ms") is None:
            saved["benchmark_e2e_latency_excluding_judge_ms"] = BenchmarkRun._benchmark_e2e_latency_excluding_judge_ms(
                state.get("phases") or {}, items,
            )
        return saved

    @classmethod
    def _aggregate_tool_performance(cls, items: list[dict]) -> dict:
        per_tool = {}
        for item in items:
            for trace in item.get("tool_trace") or []:
                if not isinstance(trace, dict) or not trace.get("tool"):
                    continue
                slot = per_tool.setdefault(str(trace["tool"]), {
                    "calls": 0, "ok": 0, "latencies": [], "providers": set(), "tiles": set(),
                    "cache_hits": 0, "vlm_calls": 0, "prompt_tokens": 0,
                    "completion_tokens": 0, "token_samples": 0, "cache_samples": 0,
                    "vlm_samples": 0,
                })
                slot["calls"] += 1
                if trace.get("provider"):
                    slot["providers"].add(str(trace["provider"]))
                if trace.get("fallback_used"):
                    slot.setdefault("fallback_count", 0)
                    slot["fallback_count"] += 1
                if str(trace.get("status") or "").lower() in {"ok", "complete", "completed", "success"}:
                    slot["ok"] += 1
                latency_s = trace.get("latency_s")
                if isinstance(latency_s, (int, float)) and latency_s >= 0:
                    slot["latencies"].append(float(latency_s) * 1000)
            for name, metrics in (item.get("tool_perf") or {}).items():
                if not isinstance(metrics, dict):
                    continue
                slot = per_tool.setdefault(str(name), {
                    "calls": 0, "ok": 0, "latencies": [], "providers": set(), "tiles": set(),
                    "cache_hits": 0, "vlm_calls": 0, "prompt_tokens": 0,
                    "completion_tokens": 0, "token_samples": 0, "cache_samples": 0,
                    "vlm_samples": 0,
                })
                slot["providers"].update(str(value) for value in metrics.get("providers") or [])
                slot["tiles"].update(str(value) for value in metrics.get("tiles") or [])
                for field in ("cache_hits", "vlm_calls", "prompt_tokens", "completion_tokens",
                              "token_samples", "cache_samples", "vlm_samples"):
                    if isinstance(metrics.get(field), (int, float)):
                        slot[field] += metrics[field]
        result = {}
        for name, slot in per_tool.items():
            latencies = slot.pop("latencies")
            ok = slot.pop("ok")
            calls = slot["calls"]
            result[name] = {
                **slot,
                "ok_rate": round(ok / calls, 3) if calls else None,
                "p50_ms": cls._percentile(latencies, 0.5),
                "p95_ms": cls._percentile(latencies, 0.95),
                "max_ms": max(latencies) if latencies else None,
                "providers": sorted(slot["providers"]),
                "tiles": sorted(slot["tiles"]),
            }
        return result

    @staticmethod
    def _aggregate_delivery(items: list[dict]) -> dict:
        """聚合确定性交付 / OCR partial 维度，供摘要展示。"""
        det_count = 0
        det_kinds: dict[str, int] = {}
        ocr_partial_count = 0
        ocr_partial_reasons: dict[str, int] = {}
        for item in items:
            guard = item.get("guard_debug") or {}
            det = guard.get("deterministic_delivery") or {}
            if det.get("rendered"):
                det_count += 1
                kind = str(det.get("kind") or "unknown")
                det_kinds[kind] = det_kinds.get(kind, 0) + 1
            delivery = item.get("delivery_status") or {}
            if delivery.get("ocr_partial"):
                ocr_partial_count += 1
                reason = str(delivery.get("ocr_partial_reason") or "unknown")
                ocr_partial_reasons[reason] = ocr_partial_reasons.get(reason, 0) + 1
        return {
            "deterministic_delivery_count": det_count,
            "deterministic_delivery_kinds": det_kinds,
            "ocr_partial_count": ocr_partial_count,
            "ocr_partial_reasons": ocr_partial_reasons,
        }

    @staticmethod
    def _list_summary(state: dict) -> dict:
        saved = dict(state.get("summary") or {})
        items = state.get("items") or []
        recalls = [item.get("retrieval_recall") for item in items
                   if isinstance(item.get("retrieval_recall"), (int, float))]
        scores = [score for item in items
                  if (score := judge_score_for_summary(item.get("judge"))) is not None]
        distribution = {str(score): scores.count(score) for score in (0, 1, 2)}
        saved.update({
            "total": saved.get("total") or state.get("qa_count") or len(items),
            "completed": len(items),
            "judge_valid_count": len(scores),
            "judge_distribution": distribution,
            "retrieval_recall_mean": saved.get("retrieval_recall_mean")
                if saved.get("retrieval_recall_mean") is not None
                else (round(sum(recalls) / len(recalls), 3) if recalls else None),
            "answer_quality_mean": round(sum(scores) / len(scores), 3) if scores else None,
        })
        for key, value in BenchmarkRun._capability_summary(items, state.get("phases") or {}).items():
            if (key == "retrieval_recall_mean"
                    or key.startswith(("retrieval_", "media_retrieval_", "image_retrieval_", "video_retrieval_"))
                    or saved.get(key) is None):
                saved[key] = value
        return saved

    @staticmethod
    def _hydrate_legacy_gt_images(state: dict) -> None:
        """Add display-only GT asset mappings for results written before gt_images existed."""
        items = state.get("items") or []
        missing = [item for item in items if "gt_images" not in item]
        scope_id = state.get("scope_id")
        if not missing or not scope_id:
            return
        try:
            data = request_json(f"{DEFAULT_SENTRIX_URL}/api/assets?scope_id={quote(str(scope_id))}&limit=2000", timeout=15)
            assets_by_name = {}
            for asset in data.get("assets", []):
                assets_by_name.setdefault(Path(asset.get("file_name") or "").name, []).append(asset)
        except Exception:
            return
        for item in missing:
            gt_images = []
            for image_id in item.get("retrieval_image_ids") or []:
                file_name = Path(str(image_id)).name
                candidates = assets_by_name.get(file_name, [])
                gt_images.append({
                    "image_id": image_id,
                    "file_name": file_name,
                    "asset_id": candidates[0].get("id") if len(candidates) == 1 else None,
                    "matched": file_name in set(item.get("matched_file_names") or []),
                    "mapping_status": "ok" if len(candidates) == 1 else "missing" if not candidates else "ambiguous",
                })
            item["gt_images"] = gt_images

    def cancel_run(self, run_id: str) -> dict:
        with self.lock:
            run = self.runs.get(run_id)
            if not run:
                raise KeyError(run_id)
            if isinstance(run, BenchmarkRun):
                if run.state.get("status") in ("running", "pending", "cancelling"):
                    run.cancel(source="run_api")
                    return {"cancelled": True, "run_id": run_id}
                return {"cancelled": False, "run_id": run_id, "reason": f"run status is {run.state.get('status')}"}
            return {"cancelled": False, "run_id": run_id, "reason": "run not active (loaded from disk)"}

    def cancel_active_suite(self) -> dict:
        """Cancel all running/pending/cancelling runs — both active suite and orphaned."""
        cancelled = []
        skipped = []
        dirty_statuses = {"running", "pending", "cancelling"}
        with self.lock:
            # Collect all dirty run IDs: active suite + any orphaned from disk
            target_ids = set(self.active_suite_run_ids)
            for rid, run in self.runs.items():
                status = run.state.get("status") if isinstance(run, BenchmarkRun) else run.get("status")
                if status in dirty_statuses:
                    target_ids.add(rid)
            # Clear active suite so a new one can start after this
            self.active_suite_run_ids = []
        for rid in target_ids:
            run = self.runs.get(rid)
            if not run:
                skipped.append(rid)
                continue
            status = run.state.get("status") if isinstance(run, BenchmarkRun) else run.get("status")
            if status in dirty_statuses:
                if isinstance(run, BenchmarkRun):
                    run.cancel(source="cancel_all")
                else:
                    # Disk-loaded dict: mark as cancelled directly
                    run["status"] = "cancelled"
                    run["finished_at"] = run.get("finished_at") or now_iso()
                    run_path = self.results_root / rid / "run.json"
                    if run_path.exists():
                        atomic_json(run_path, run)
                cancelled.append(rid)
            else:
                skipped.append(rid)
        return {"cancelled": cancelled, "skipped": skipped, "total": len(target_ids)}

    def delete_run(self, run_id: str) -> dict:
        with self.lock:
            run = self.runs.get(run_id)
            if not run:
                raise KeyError(run_id)
            state = run.state if isinstance(run, BenchmarkRun) else run
            if (state.get("rejudge") or {}).get("status") == "running":
                raise ValueError("cannot delete a run while rejudge is running")
            self.runs.pop(run_id, None)
            run_dir = (self.results_root / run_id).resolve()
            if self.results_root in run_dir.parents and run_dir.exists():
                shutil.rmtree(run_dir)
            return {"deleted": True, "run_id": run_id}

    def _persist_run_state(self, run_id: str, run, state: dict) -> None:
        if isinstance(run, BenchmarkRun):
            run.persist()
            return
        run_dir = self.results_root / run_id
        atomic_json(run_dir / "run.json", state)
        items = state.get("items") or []
        (run_dir / "results.jsonl").write_text(
            "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in items),
            encoding="utf-8",
        )

    @staticmethod
    def _refresh_summary(state: dict) -> None:
        items = state.get("items") or []
        recalls = [i["retrieval_recall"] for i in items if isinstance(i.get("retrieval_recall"), (int, float))]
        valid_scores = [score for i in items
                        if (score := judge_score_for_summary(i.get("judge"))) is not None]
        distribution = {str(score): valid_scores.count(score) for score in (0, 1, 2)}
        denom = len(valid_scores)
        summary = dict(state.get("summary") or {})
        summary.update({
            "total": state.get("qa_count") or len(items),
            "completed": len(items),
            "retrieval_recall_mean": round(sum(recalls) / len(recalls), 3) if recalls else None,
            "judge_distribution": distribution,
            "judge_valid_count": denom,
            "answer_quality_mean": round(sum(valid_scores) / denom, 3) if denom else None,
            "exact_accuracy": round(distribution["2"] / denom, 3) if denom else None,
            "core_accuracy": round((distribution["1"] + distribution["2"]) / denom, 3) if denom else None,
        })
        summary.update(BenchmarkRun._capability_summary(items, state.get("phases") or {}))
        state["summary"] = summary
        aggregate = (state.get("phases") or {}).get("aggregate")
        if aggregate:
            aggregate["summary"] = summary

    def start_rejudge(self, run_id: str, payload: dict) -> dict:
        system_prompt = str(payload.get("system_prompt") or "").strip()
        judge_provider_id = str(payload.get("judge_provider_id") or DEFAULT_JUDGE_PROVIDER_ID)
        _, resolved_judge_url, resolved_judge_model, resolved_judge_api_key = resolve_judge_provider(judge_provider_id)
        judge_url = str(payload.get("judge_url") or resolved_judge_url).rstrip("/")
        judge_model = str(payload.get("judge_model") or resolved_judge_model)
        judge_api_key = str(payload.get("judge_api_key") or resolved_judge_api_key)
        saved_custom = load_custom_judge_prompts()
        task_system_prompt = payload.get("task_system_prompt")
        task_system_prompt = (str(task_system_prompt).strip() or saved_custom.get("task_decision")
                              or TASK_JUDGE_PROMPT)
        evidence_system_prompt = payload.get("evidence_system_prompt")
        evidence_system_prompt = (str(evidence_system_prompt).strip() or saved_custom.get("evidence")
                                  or EVIDENCE_JUDGE_PROMPT)
        if not system_prompt:
            raise ValueError("system_prompt is required")
        if len(system_prompt) > 50000:
            raise ValueError("system_prompt is too long")

        with self.lock:
            run = self.runs.get(run_id)
            if not run:
                raise KeyError(run_id)
            state = run.state if isinstance(run, BenchmarkRun) else run
            if state.get("status") in {"running", "pending"}:
                raise ValueError("cannot rejudge while the benchmark run is active")
            if (state.get("rejudge") or {}).get("status") == "running":
                raise ValueError("rejudge is already running")
            targets = self._rejudge_targets(state.get("items") or [])
            if not targets:
                raise ValueError("run has no completed agent answers to rejudge")
            eligible_items = sorted({item_index for item_index, _ in targets})
            rejudge_id = f"rejudge-{datetime.now().strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"
            previous_rejudge = state.get("rejudge")
            if previous_rejudge:
                state.setdefault("rejudge_history", []).append(previous_rejudge)
            state["rejudge"] = {
                "rejudge_id": rejudge_id,
                "status": "running",
                "started_at": now_iso(),
                "finished_at": None,
                "total": len(targets),
                "completed": 0,
                "failed": 0,
                "unit": "conversation_turn",
                "judge_model": judge_model,
                "judge_url": judge_url,
                "current_item_index": None,
                "current_turn_index": None,
            }
            for index in eligible_items:
                item = state["items"][index]
                previous = item.get("judge")
                if previous:
                    item.setdefault("judge_history", []).append({
                        "archived_at": now_iso(),
                        "judge": previous,
                    })
                item["judge"] = {
                    "score": None,
                    "reason": "",
                    "status": "pending",
                    "rejudge_id": rejudge_id,
                }
                for turn in item.get("conversation") or []:
                    for judge_field in ("judge", "task_judge", "evidence_judge"):
                        previous_turn_judge = turn.get(judge_field)
                        if previous_turn_judge:
                            turn.setdefault(f"{judge_field}_history", []).append({
                                "archived_at": now_iso(), judge_field: previous_turn_judge,
                            })
                    turn["judge"] = {"score": None, "reason": "", "status": "pending",
                                     "rejudge_id": rejudge_id}
            self._refresh_summary(state)
            self._persist_run_state(run_id, run, state)

        def _run_rejudge():
            try:
                assets_by_name = None
                for completed, (index, turn_index) in enumerate(targets, 1):
                    with self.lock:
                        current = self.runs.get(run_id)
                        if not current:
                            return
                        current_state = current.state if isinstance(current, BenchmarkRun) else current
                        item = current_state["items"][index]
                        current_gt = self.qa_metadata.get(str(item.get("qa_id") or "")) or {}
                        saved_turns = item.get("conversation") or []
                        gt_turns = current_gt.get("conversation") or []
                        saved_turn = saved_turns[turn_index] if turn_index is not None else item
                        gt_turn = (gt_turns[turn_index]
                                   if turn_index is not None and turn_index < len(gt_turns) else {})
                        question = str(saved_turn.get("message") or item.get("question") or "")
                        expected_action = (gt_turn.get("expected_action")
                                           or saved_turn.get("expected_action")
                                           or current_gt.get("expected_action")
                                           or item.get("expected_action"))
                        reference = str(gt_turn.get("reference_answer") or "")
                        if not reference:
                            reference = ("应要求用户提供至少一个具体、可用于后续检索的锚点；询问形式不限。"
                                         if expected_action == "clarify" else
                                         current_gt.get("answer") or item.get("reference_answer") or "")
                        answer = str(saved_turn.get("answer") or "")
                        saved_turn["judge"]["status"] = "running"
                        current_state["rejudge"]["current_item_index"] = index
                        current_state["rejudge"]["current_turn_index"] = turn_index
                        self._persist_run_state(run_id, current, current_state)

                    judge_runner = BenchmarkRun.__new__(BenchmarkRun)
                    judge_runner.judge_url = judge_url.rstrip("/")
                    judge_runner.judge_model = judge_model
                    judge_runner.judge_api_key = judge_api_key
                    judge_runner.task_judge_system_prompt = task_system_prompt
                    judge_runner.evidence_judge_system_prompt = evidence_system_prompt
                    conversation_context = saved_turns[:turn_index + 1] if turn_index is not None else None
                    agent_status_rj = saved_turn.get("agent_status") or item.get("agent_status")
                    termination_reason_rj = saved_turn.get("termination_reason") or item.get("termination_reason") or ""
                    predicted_images_rj = saved_turn.get("predicted_images") or item.get("predicted_images") or []
                    gt_task_type = current_gt.get("task_type") or item.get("task_type")
                    gt_question_type = current_gt.get("question_type") or item.get("question_type")
                    gt_answerability = current_gt.get("answerability") or item.get("answerability")

                    # Run quality + task judges concurrently OUTSIDE the lock
                    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
                        quality_future = pool.submit(
                            judge_runner._judge, question, reference, answer, system_prompt,
                            expected_action=expected_action,
                            task_type=gt_task_type, question_type=gt_question_type,
                            answerability=gt_answerability, conversation=conversation_context,
                        )
                        task_future = pool.submit(
                            judge_runner._judge_task_action, question, answer, expected_action,
                            agent_status_rj, termination_reason_rj,
                            task_type=gt_task_type, question_type=gt_question_type,
                            answerability=gt_answerability, reference=reference,
                            conversation=conversation_context,
                        ) if expected_action in {"answer", "refuse", "clarify"} else None

                    judge = quality_future.result()
                    judge["ground_truth_source"] = "current_qa_metadata"
                    judge["judged_at"] = now_iso()
                    judge["rejudge_id"] = rejudge_id
                    judge["status"] = "completed" if judge.get("score") in {0, 1, 2} else "failed"
                    task_result = task_future.result() if task_future else None
                    if task_result:
                        task_result["ground_truth_source"] = "current_qa_metadata"
                    actual_action = task_result.get("actual_action") if task_result else None

                    # Read scope_id under lock, then run evidence judge outside
                    scope_id_rj = None
                    with self.lock:
                        current = self.runs.get(run_id)
                        if current:
                            cs = current.state if isinstance(current, BenchmarkRun) else current
                            scope_id_rj = cs.get("scope_id")

                    evidence_result = None
                    if EVIDENCE_JUDGE_ENABLED and predicted_images_rj and scope_id_rj and answer and actual_action != "clarify":
                        try:
                            if assets_by_name is None:
                                assets = request_json(
                                    f"{DEFAULT_SENTRIX_URL}/api/assets?scope_id={quote(str(scope_id_rj))}&limit=2000",
                                    timeout=30,
                                ).get("assets") or []
                                assets_by_name = {}
                                for asset in assets:
                                    assets_by_name.setdefault(Path(asset.get("file_name") or "").name, []).append(asset)
                            evidence_result = judge_runner._judge_evidence(
                                question, answer, predicted_images_rj, assets_by_name,
                                DEFAULT_SENTRIX_URL, conversation=conversation_context)
                        except Exception as exc:
                            evidence_result = {"score": None, "reason": f"asset_lookup_error: {exc}"}

                    # Write all results back under lock
                    with self.lock:
                        current = self.runs.get(run_id)
                        if not current:
                            return
                        current_state = current.state if isinstance(current, BenchmarkRun) else current
                        current_item = current_state["items"][index]
                        current_turns = current_item.get("conversation") or []
                        current_target = current_turns[turn_index] if turn_index is not None else current_item
                        current_target["judge"] = judge
                        if task_result:
                            current_target["task_judge"] = task_result
                        if evidence_result:
                            current_target["evidence_judge"] = evidence_result
                        elif current_target.get("evidence_judge") is not None:
                            current_target["evidence_judge"] = {"score": None, "reason": "not_applicable"}
                        if current_turns:
                            current_item["task_judges"] = [turn.get("task_judge") or {} for turn in current_turns]
                            last_turn = current_turns[-1]
                            current_item["judge"] = last_turn.get("judge") or {}
                            current_item["task_judge"] = last_turn.get("task_judge") or {}
                            current_item["evidence_judge"] = last_turn.get("evidence_judge") or {}
                        else:
                            current_item["judge"] = current_target.get("judge") or {}
                            current_item["task_judge"] = current_target.get("task_judge") or {}
                            current_item["evidence_judge"] = current_target.get("evidence_judge") or {}
                        current_item["agent_stability"] = BenchmarkRun._agent_stability(current_item)
                        progress = current_state["rejudge"]
                        progress["completed"] = completed
                        if judge.get("score") not in {0, 1, 2}:
                            progress["failed"] += 1
                        self._refresh_summary(current_state)
                        self._persist_run_state(run_id, current, current_state)

                with self.lock:
                    current = self.runs.get(run_id)
                    if not current:
                        return
                    current_state = current.state if isinstance(current, BenchmarkRun) else current
                    current_state["rejudge"]["status"] = "completed"
                    current_state["rejudge"]["finished_at"] = now_iso()
                    current_state["rejudge"]["current_item_index"] = None
                    current_state["rejudge"]["current_turn_index"] = None
                    self._persist_run_state(run_id, current, current_state)
            except Exception as exc:
                with self.lock:
                    current = self.runs.get(run_id)
                    if current:
                        current_state = current.state if isinstance(current, BenchmarkRun) else current
                        current_state["rejudge"]["status"] = "failed"
                        current_state["rejudge"]["finished_at"] = now_iso()
                        current_state["rejudge"]["current_item_index"] = None
                        current_state["rejudge"]["current_turn_index"] = None
                        current_state["rejudge"]["error"] = str(exc)
                        self._persist_run_state(run_id, current, current_state)
            finally:
                with self.lock:
                    self.rejudge_threads.pop(run_id, None)

        thread = threading.Thread(target=_run_rejudge, name=rejudge_id, daemon=True)
        with self.lock:
            self.rejudge_threads[run_id] = thread
        thread.start()
        return {"run_id": run_id, "rejudge_id": rejudge_id, "total": len(targets), "status": "running"}

    @staticmethod
    def _public_run(state: dict, include_items: bool) -> dict:
        result = {k: v for k, v in state.items()}
        if not include_items:
            result.pop("items", None)
        return result

    def start_suite(self, payload: dict) -> dict:
        album_id = payload.get("album_id", "album3-14")
        mode = str(payload.get("mode") or "full").strip().lower()
        if mode not in RUN_MODES:
            raise ValueError(f"mode must be one of {sorted(RUN_MODES)}, got: {mode!r}")
        manifest_early = self.get_manifest(album_id)
        if not manifest_early:
            raise ValueError(f"manifest not found for album: {album_id}")
        qa_set = payload.get("qa_set") or (
            # build 模式不做 QA，允许不选；默认取 manifest 第一个仅用于加载结构。
            next(iter(manifest_early.get("qa_sets") or {}), "")
        )
        if not qa_set:
            raise ValueError(f"album {album_id} has no qa_sets in manifest")
        existing_scope_id = str(payload.get("existing_scope_id") or "").strip()
        if mode == "reuse" and not existing_scope_id:
            raise ValueError("existing_scope_id is required when mode=reuse")
        models = payload.get("models", [])
        if not isinstance(models, list) or not models:
            raise ValueError("models must contain at least one model")
        sentrix_url = payload.get("sentrix_url", DEFAULT_SENTRIX_URL)
        judge_provider_id = str(payload.get("judge_provider_id") or DEFAULT_JUDGE_PROVIDER_ID)
        _, resolved_judge_url, resolved_judge_model, resolved_judge_api_key = resolve_judge_provider(judge_provider_id)
        judge_url = str(payload.get("judge_url") or resolved_judge_url).rstrip("/")
        judge_model = str(payload.get("judge_model") or resolved_judge_model)
        judge_api_key_suite = str(payload.get("judge_api_key") or resolved_judge_api_key)
        model_base_url = normalize_model_base_url(payload.get("model_base_url"))
        vllm_manager_url = normalize_service_url(payload.get("vllm_manager_url"))
        if BIG_MODEL_PROFILE_ID in models and CURRENT_MODEL_SELECTION in models:
            raise ValueError("big_model cannot be combined with current model")
        managed_models = [
            model for model in models
            if model not in {BIG_MODEL_PROFILE_ID, CURRENT_MODEL_SELECTION}
        ]
        if managed_models and not vllm_manager_url:
            raise ValueError("选择模型注册表中的模型时必须提供模型管理器地址")
        if managed_models:
            target_id, target = resolve_vllm_target(payload.get("vllm_target_id"))
            vllm_api_url = vllm_manager_url or str(target["manager_url"])
            vllm_model_base_url = model_base_url or str(target["model_base_url"])
        else:
            target_id = ""
            vllm_api_url = ""
            vllm_model_base_url = ""
        dirty_statuses = {"running", "pending", "cancelling"}
        with self.lock:
            busy = [
                rid for rid, run in self.runs.items()
                if (run.state.get("status") if isinstance(run, BenchmarkRun)
                    else run.get("status")) in dirty_statuses
            ]
        if busy:
            raise ValueError(f"another benchmark suite is still active: {', '.join(busy)}")
        use_current_model = CURRENT_MODEL_SELECTION in models
        if use_current_model and len(models) != 1:
            raise ValueError("current model cannot be combined with managed model profiles")
        current_model_snapshot = None
        if use_current_model:
            if model_base_url:
                vllm_api_url = ""
                vllm_model_base_url = model_base_url
            current_model_snapshot = self.query_current_model(vllm_api_url, vllm_model_base_url)
            models = [current_model_snapshot["model_id"]]
            if model_base_url:
                request_json(f"{str(sentrix_url).rstrip('/')}/api/model-profiles/bind-external-runtime", {
                    "base_url": current_model_snapshot["model_base_url"],
                    "model": current_model_snapshot["served_model_name"],
                }, "POST", 30)
            else:
                request_json(f"{str(sentrix_url).rstrip('/')}/api/model-profiles/bind-runtime", {
                    "manager_url": vllm_api_url,
                    "model_base_url": vllm_model_base_url,
                }, "POST", 30)
        delete_scope_after_run = bool(payload.get("delete_scope_after_run"))
        with self.lock:
            busy = [
                rid
                for rid, run in self.runs.items()
                if (run.state.get("status") if isinstance(run, BenchmarkRun)
                    else run.get("status")) in dirty_statuses
            ]
            if busy:
                raise ValueError(f"another benchmark suite is still active: {', '.join(busy)}")
            manifest = self.get_manifest(album_id)
            if not manifest:
                raise ValueError(f"manifest not found for album: {album_id}")

            # reuse 模式：反查该 scope 由哪些历史 run 创建，写进新 run 做来源关联。
            scope_reused_from_runs: list = []
            if mode == "reuse":
                for rid, run in self.runs.items():
                    state = run.state if isinstance(run, BenchmarkRun) else run
                    if (state.get("scope_id") == existing_scope_id
                            and state.get("scope_source") == "created"
                            and state.get("mode") in ("full", "build")):
                        scope_reused_from_runs.append(rid)
                scope_reused_from_runs.sort()

            suite_id = f"suite-{datetime.now().strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"
            created_runs = []
            for model in models:
                ts = datetime.now().strftime("%Y%m%d-%H%M%S")
                mode_tag = "" if mode == "full" else f"-{mode}"
                run_id = f"{ts}-{safe_slug(album_id)}-{safe_slug(model)}{mode_tag}-{uuid.uuid4().hex[:6]}"
                run = BenchmarkRun(
                    run_id=run_id, album_id=album_id, manifest=manifest,
                    model_profile=model, qa_set=qa_set,
                    sentrix_url=sentrix_url, judge_url=judge_url,
                    vllm_api_url=vllm_api_url if model != BIG_MODEL_PROFILE_ID else "",
                    vllm_target_id=target_id if model != BIG_MODEL_PROFILE_ID else "",
                    vllm_model_base_url=vllm_model_base_url if model != BIG_MODEL_PROFILE_ID else "",
                    results_root=self.results_root,
                    judge_system_prompt=load_custom_judge_prompt() or JUDGE_PROMPT,
                    task_judge_system_prompt=load_custom_judge_prompts().get("task_decision") or TASK_JUDGE_PROMPT,
                    evidence_judge_system_prompt=load_custom_judge_prompts().get("evidence") or EVIDENCE_JUDGE_PROMPT,
                    judge_model=judge_model, judge_api_key=judge_api_key_suite,
                    delete_scope_after_run=delete_scope_after_run,
                    mode=mode, existing_scope_id=existing_scope_id,
                    scope_reused_from_runs=scope_reused_from_runs,
                    use_current_model=use_current_model,
                    current_model_snapshot=current_model_snapshot,
                    use_cloud_model=(model == BIG_MODEL_PROFILE_ID),
                )
                self.runs[run_id] = run
                created_runs.append(run_id)
            self.active_suite_run_ids = created_runs

        def _run_sequentially():
            for rid in created_runs:
                run = self.runs.get(rid)
                if run is None or not isinstance(run, BenchmarkRun):
                    print(f"[suite] run {rid} not found in registry; skipping", flush=True)
                    continue
                if run._cancel.is_set() or run.state.get("status") == "cancelled":
                    continue
                run.execute()

        threading.Thread(target=_run_sequentially, name=f"suite-{suite_id}", daemon=True).start()
        return {"suite_id": suite_id, "run_ids": created_runs, "album_id": album_id,
                "models": models, "qa_set": qa_set, "mode": mode,
                "existing_scope_id": existing_scope_id or None,
                "current_model_snapshot": current_model_snapshot}


# ---------------------------------------------------------------------------
# HTTP Handler
# ---------------------------------------------------------------------------

class OrchestratorHandler(BaseHTTPRequestHandler):
    repo: OrchestratorRepository
    web_root: Path

    def _json(self, value, status: int = 200):
        body = json.dumps(value, ensure_ascii=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _payload(self) -> dict:
        length = int(self.headers.get("Content-Length", "0"))
        return json.loads(self.rfile.read(length) or b"{}")

    def do_GET(self):
        parsed = urlparse(self.path)
        try:
            if parsed.path == "/api/memory-spaces":
                # 复用相册测评的相册下拉数据：转发 Sentrix 后端列表（新创建的在前）。
                params = parse_qs(parsed.query)
                sentrix_base = (params.get("sentrix_url") or [DEFAULT_SENTRIX_URL])[0].rstrip("/")
                spaces = request_json(f"{sentrix_base}/api/memory-spaces", timeout=30)
                if isinstance(spaces, dict):
                    spaces = spaces.get("spaces") or spaces.get("items") or []
                spaces = sorted(
                    spaces or [],
                    key=lambda s: str(s.get("created_at") or ""), reverse=True,
                )
                self._json({"spaces": spaces, "reuse_bases": _build_reuse_bases(
                    spaces, self.repo.list_runs())})
                return
            if parsed.path == "/api/config":
                self._json({
                    "default_sentrix_url": DEFAULT_SENTRIX_URL,
                    "default_judge_url": DEFAULT_JUDGE_URL,
                    "default_vllm_api_url": DEFAULT_VLLM_API_URL,
                    "default_vllm_model_base_url": DEFAULT_VLLM_BASE_URL,
                    "default_vllm_target_id": DEFAULT_VLLM_TARGET_ID,
                    "vllm_targets": VLLM_TARGETS,
                    "runtime_config": public_runtime_connection_config(),
                    "runtime_config_file": str(RUNTIME_CONNECTION_CONFIG_PATH),
                   "judge_model": JUDGE_MODEL,
                   "judge_prompt": JUDGE_PROMPT,
                   "custom_judge_prompt": load_custom_judge_prompt(),
                   "evidence_judge_enabled": EVIDENCE_JUDGE_ENABLED,
                   "judge_providers": _public_judge_providers(JUDGE_PROVIDERS),
                   "default_judge_provider_id": DEFAULT_JUDGE_PROVIDER_ID,
                })
                return
            if parsed.path == "/api/manifests":
                self._json({"manifests": self.repo.list_manifests()})
                return
            if parsed.path.startswith("/api/manifests/"):
                album_id = unquote(parsed.path.removeprefix("/api/manifests/"))
                mf = self.repo.get_manifest(album_id)
                if mf:
                    self._json(mf)
                else:
                    self._json({"error": "not found"}, 404)
                return
            if parsed.path == "/api/qa-dataset":
                query = parse_qs(parsed.query)
                album_id = (query.get("album_id") or [""])[0]
                qa_set = (query.get("qa_set") or [""])[0]
                if not album_id or not qa_set:
                    self._json({"error": "album_id and qa_set are required"}, 400)
                    return
                mf = self.repo.get_manifest(album_id)
                if not mf:
                    self._json({"error": "album not found"}, 404)
                    return
                qa_sets = mf.get("qa_sets") or {}
                if isinstance(qa_sets, dict):
                    qa_file = qa_sets.get(qa_set)
                else:
                    qa_file = qa_sets[qa_set] if qa_set in qa_sets else None
                if not qa_file:
                    self._json({"error": "qa_set not found"}, 404)
                    return
                qa_path = BENCHMARK_DATA_ROOT / album_id / qa_file
                if not qa_path.is_file():
                    self._json({"error": "qa file not found"}, 404)
                    return
                rows = load_jsonl(qa_path)
                self._json({"album_id": album_id, "qa_set": qa_set, "items": rows})
                return
            if parsed.path.startswith("/api/albums/"):
                parts = parsed.path.removeprefix("/api/albums/").split("/", 2)
                if len(parts) == 3 and parts[1] in {"photos", "faces", "videos"}:
                    album_id = unquote(parts[0])
                    file_name = unquote(parts[2])
                    media_path = _resolve_album_media_file(album_id, parts[1], file_name)
                    if media_path:
                        self._serve_file(media_path)
                        return
                self._json({"error": "media not found"}, 404)
                return
            if parsed.path == "/api/runs":
                self._json({"runs": self.repo.list_runs()})
                return
            if parsed.path.startswith("/api/runs/") and parsed.path.endswith("/items"):
                run_id = unquote(parsed.path.removeprefix("/api/runs/").removesuffix("/items"))
                query = parse_qs(parsed.query)
                try:
                    page = int((query.get("page") or ["1"])[0])
                    page_size = int((query.get("page_size") or ["20"])[0])
                except ValueError as exc:
                    raise ValueError("page and page_size must be integers") from exc
                self._json(self.repo.get_run_items(
                    run_id, page=page, page_size=page_size,
                    search=(query.get("search") or [""])[0],
                    score=(query.get("score") or [""])[0],
                    task_type=(query.get("task_type") or [""])[0],
                    tag=(query.get("tag") or [""])[0],
                    agent_status=(query.get("agent_status") or [""])[0],
                    angle=(query.get("angle") or [""])[0],
                    difficulty=(query.get("difficulty") or [""])[0],
                    answerability=(query.get("answerability") or [""])[0],
                    primary=(query.get("primary") or [""])[0],
                ))
                return
            if parsed.path.startswith("/api/runs/") and parsed.path.endswith("/export-sft"):
                run_id = unquote(parsed.path.removeprefix("/api/runs/").removesuffix("/export-sft"))
                _q = parse_qs(parsed.query)
                _raw_scores = (_q.get("scores") or [""])[0]
                scores = [int(x) for x in _raw_scores.split(",") if x.strip() in {"0", "1", "2"}]
                _raw = (_q.get("min_score") or [""])[0]
                min_score = int(_raw) if _raw in {"1", "2"} else None
                payload = self.repo.export_sft(run_id, scores=scores or None, min_score=min_score)
                body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Disposition", f'attachment; filename="{run_id}-sft-traces.json"')
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            if parsed.path == "/api/judge-prompts":
                custom = load_custom_judge_prompts()
                self._json({
                    "kinds": [
                        {
                            "kind": kind,
                            "label": label,
                            "default": JUDGE_PROMPT_KINDS[kind],
                            "custom": custom.get(kind),
                        }
                        for kind, label in (
                            ("answer_quality", "回答质量 Judge"),
                            ("task_decision", "任务判断 Judge"),
                            ("evidence", "证据核验 Judge"),
                        )
                    ],
                    "storage_path": str(CUSTOM_JUDGE_PROMPT_PATH),
                })
                return
            if parsed.path.startswith("/api/runs/") and parsed.path.endswith("/judge-prompt"):
                run_id = unquote(parsed.path.removeprefix("/api/runs/").removesuffix("/judge-prompt"))
                self._json(self.repo.get_run_judge_prompt(run_id))
                return
            if parsed.path.startswith("/api/runs/") and parsed.path.endswith("/reviews"):
                run_id = unquote(parsed.path.removeprefix("/api/runs/").removesuffix("/reviews"))
                self._json(self.repo.get_reviews(run_id))
                return
            if parsed.path.startswith("/api/runs/") and "/items/" in parsed.path:
                suffix = parsed.path.removeprefix("/api/runs/")
                run_id, item_index = suffix.rsplit("/items/", 1)
                self._json(self.repo.get_run_item(unquote(run_id), int(item_index)))
                return
            if parsed.path.startswith("/api/runs/"):
                self._json(self.repo.get_run(unquote(parsed.path.removeprefix("/api/runs/"))))
                return
            # Serve frontend
            relative = "index.html" if parsed.path == "/" else parsed.path.lstrip("/")
            path = (self.web_root / relative).resolve()
            if not path.is_relative_to(self.web_root.resolve()):
                raise ValueError("path escapes root")
            self._serve_file(path)
        except KeyError as e:
            self._json({"error": str(e)}, 404)
        except Exception as e:
            self._json({"error": str(e)}, 400)

    def do_POST(self):
        try:
            parsed = urlparse(self.path)
            if parsed.path == "/api/cancel-active":
                self._json(self.repo.cancel_active_suite())
                return
            if parsed.path.endswith("/cancel") and parsed.path.startswith("/api/runs/"):
                run_id = unquote(parsed.path.removeprefix("/api/runs/").removesuffix("/cancel"))
                self._json(self.repo.cancel_run(run_id))
                return
            if parsed.path == "/api/judge-prompt":
                payload = self._payload()
                prompt = str(payload.get("system_prompt") or "").strip()
                if not prompt:
                    raise ValueError("system_prompt is required")
                if len(prompt) > 50000:
                    raise ValueError("system_prompt is too long")
                save_custom_judge_prompt(prompt)
                self._json({"status": "ok"})
                return
            if parsed.path == "/api/judge-prompts":
                payload = self._payload()
                kind = str(payload.get("kind") or "").strip()
                prompt = str(payload.get("system_prompt") or "").strip()
                if kind not in JUDGE_PROMPT_KINDS:
                    raise ValueError(f"kind must be one of {sorted(JUDGE_PROMPT_KINDS)}")
                if not prompt:
                    raise ValueError("system_prompt is required (empty string to restore default is not allowed here)")
                if len(prompt) > 50000:
                    raise ValueError("system_prompt is too long")
                save_custom_judge_prompt(prompt, kind)
                self._json({"status": "ok", "kind": kind,
                            "custom": load_custom_judge_prompts().get(kind)})
                return
            payload = self._payload()
            if parsed.path.endswith("/rejudge") and parsed.path.startswith("/api/runs/"):
                run_id = unquote(parsed.path.removeprefix("/api/runs/").removesuffix("/rejudge"))
                self._json(self.repo.start_rejudge(run_id, payload), status=202)
            elif parsed.path.endswith("/reviews") and parsed.path.startswith("/api/runs/"):
                run_id = unquote(parsed.path.removeprefix("/api/runs/").removesuffix("/reviews"))
                self._json(self.repo.save_reviews(run_id, payload))
            elif self.path == "/api/config":
                saved = persist_runtime_connection_config(payload)
                global DEFAULT_SENTRIX_URL, DEFAULT_JUDGE_URL, DEFAULT_VLLM_API_URL
                global DEFAULT_VLLM_BASE_URL, DEFAULT_JUDGE_PROVIDER_ID
                DEFAULT_SENTRIX_URL = saved["sentrix_url"]
                DEFAULT_JUDGE_URL = saved["judge_url"]
                DEFAULT_VLLM_API_URL = saved["vllm_manager_url"]
                DEFAULT_VLLM_BASE_URL = saved["model_base_url"]
                DEFAULT_JUDGE_PROVIDER_ID = saved["judge_provider_id"]
                self._json({"saved": True, "runtime_config": saved})
            elif self.path == "/api/profiles":
                manager_url = normalize_service_url(payload.get("vllm_manager_url"))
                if manager_url:
                    target_id = "external"
                    target = {
                        "label": "自定义模型 Manager",
                        "manager_url": manager_url,
                        "model_base_url": normalize_model_base_url(payload.get("model_base_url")),
                        "kind": "external",
                    }
                else:
                    target_id, target = resolve_vllm_target(payload.get("vllm_target_id"))
                result = self.repo.query_profiles(str(target["manager_url"]))
                result.update({"target_id": target_id, "target": target})
                self._json(result)
            elif self.path == "/api/current-model":
                direct_base_url = normalize_model_base_url(payload.get("model_base_url"))
                if direct_base_url:
                    result = self.repo.query_current_model("", direct_base_url)
                    target_id = "external"
                    target = {
                        "label": "外部模型服务",
                        "model_base_url": direct_base_url,
                        "manager_url": "",
                        "kind": "external",
                    }
                else:
                    target_id, target = resolve_vllm_target(payload.get("vllm_target_id"))
                    result = self.repo.query_current_model(
                        str(target["manager_url"]), str(target["model_base_url"]),
                    )
                result.update({"target_id": target_id, "target": target})
                self._json(result)
            elif self.path == "/api/memory-profile":
                self._json(self.repo.start_memory_profile(payload), status=202)
            elif self.path == "/api/runs":
                self._json(self.repo.start_suite(payload), status=202)
            else:
                self._json({"error": "unknown endpoint"}, 404)
        except Exception as e:
            self._json({"error": str(e)}, 400)

    def do_DELETE(self):
        parsed = urlparse(self.path)
        try:
            if parsed.path.startswith("/api/runs/"):
                run_id = unquote(parsed.path.removeprefix("/api/runs/"))
                self._json(self.repo.delete_run(run_id))
                return
            self._json({"error": "unknown endpoint"}, 404)
        except KeyError as e:
            self._json({"error": str(e)}, 404)
        except Exception as e:
            self._json({"error": str(e)}, 400)

    def _serve_file(self, path: Path):
        if not path.is_file():
            raise FileNotFoundError(path)
        ct = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        if ct.startswith("text/") or ct == "application/javascript":
            ct += "; charset=utf-8"
        if ct.startswith("video/"):
            size = path.stat().st_size
            start, end = 0, size - 1
            range_header = self.headers.get("Range", "")
            if range_header:
                match = re.fullmatch(r"bytes=(\d*)-(\d*)", range_header.strip())
                if not match:
                    self.send_response(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
                    self.send_header("Content-Range", f"bytes */{size}")
                    self.end_headers()
                    return
                if match.group(1):
                    start = int(match.group(1))
                    end = min(int(match.group(2)) if match.group(2) else size - 1, size - 1)
                elif match.group(2):
                    length = min(int(match.group(2)), size)
                    start = size - length
                if start > end or start >= size:
                    self.send_response(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
                    self.send_header("Content-Range", f"bytes */{size}")
                    self.end_headers()
                    return
            length = end - start + 1
            self.send_response(HTTPStatus.PARTIAL_CONTENT if range_header else HTTPStatus.OK)
            self.send_header("Content-Type", ct)
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Content-Length", str(length))
            self.send_header("Cache-Control", "private, max-age=3600")
            if range_header:
                self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
            self.end_headers()
            try:
                with path.open("rb") as handle:
                    handle.seek(start)
                    remaining = length
                    while remaining:
                        chunk = handle.read(min(1024 * 1024, remaining))
                        if not chunk:
                            break
                        self.wfile.write(chunk)
                        remaining -= len(chunk)
            except (BrokenPipeError, ConnectionResetError):
                pass
            return
        body = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", ct)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8771)
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--web-root", type=Path, default=DEFAULT_WEB_ROOT)
    args = parser.parse_args()
    OrchestratorHandler.repo = OrchestratorRepository(args.results_root)
    OrchestratorHandler.web_root = args.web_root.resolve()
    class ReusableThreadingHTTPServer(ThreadingHTTPServer):
        allow_reuse_address = True

    server = ReusableThreadingHTTPServer((args.host, args.port), OrchestratorHandler)
    print(f"Benchmark Orchestrator: http://{args.host}:{args.port}/", flush=True)
    server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
