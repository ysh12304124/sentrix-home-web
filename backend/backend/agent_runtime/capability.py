"""Capability Matrix 运行时读取（Phase H H8 — Capability-Aware Tool Use）。

从 configs/tool_capability_matrix.json 读取工具-子能力实测状态，
为工具描述提供能力提示、为 tool_policy 提供就绪状态决策输入。

判定只反映真实 benchmark 数据（inspect_capability_benchmark + L2 eval），
不做样本凑数；样本不足的一律 experimental/untested，禁止“1.0 (n=2) 判 ready”。
"""

from __future__ import annotations

import json
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent
_MATRIX_PATH = _ROOT / "configs" / "tool_capability_matrix.json"

_CACHE: dict | None = None


def _load() -> dict:
    global _CACHE
    if _CACHE is None:
        try:
            _CACHE = json.loads(_MATRIX_PATH.read_text(encoding="utf-8"))
        except Exception:
            _CACHE = {}
    return _CACHE


def reload() -> None:
    global _CACHE
    _CACHE = None
    _load()


def judge_status(*, n: int, support_rate: float | None, false_confident_rate: float | None = None) -> str:
    """按实测样本量与支持率判定就绪状态。

    - n==0          -> untested
    - 1<=n<5        -> experimental（样本不足，一律不得 ready）
    - n>=5 且支持率>=0.7 且误自信率<=0.3 -> ready
    - 其余          -> limited
    """
    if not n:
        return "untested"
    if n < 5:
        return "experimental"
    if support_rate is None:
        return "limited"
    if support_rate >= 0.7 and (false_confident_rate is None or false_confident_rate <= 0.3):
        return "ready"
    return "limited"


def sub_capability_status(tool: str, capability: str) -> str:
    return ((_load().get(tool) or {}).get(capability) or {}).get("status") or "untested"


def tool_capability_summary(tool: str) -> str:
    """工具-子能力状态摘要（拼进 tool description 用）。

    只标出“就绪”的子能力，保持极短；没有就绪能力时返回空串。
    未就绪的能力不逐个列举，避免工具描述过长干扰模型动作解析。
    """
    entry = _load().get(tool) or {}
    caps = {k: v for k, v in entry.items()
            if isinstance(v, dict) and k not in ("note",)}
    if not caps:
        return ""
    ready = sorted(name for name, v in caps.items()
                   if (v.get("status") or "untested") == "ready")
    if not ready:
        return ""
    return "可靠能力：" + "、".join(ready)
