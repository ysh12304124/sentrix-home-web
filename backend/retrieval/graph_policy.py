"""Deterministic policy for deciding when graph retrieval is useful.

The graph channel is a recall source for questions whose answer depends on a
path (time order, relations, causes, or multiple facts).  It must not be
enabled for ordinary photo lookup in ``auto`` mode: that would add latency and
generic neighbours without supplying a graph-specific signal.  ``on`` is kept
as an explicit ablation mode and enables the channel for every question.
"""

from __future__ import annotations

import os
import re
from typing import Any


_PATTERNS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("multi_hop", (
        r"多跳", r"分别.*(?:谁|哪里|什么)", r"先.*再", r"之后.*谁", r"前后.*(?:谁|什么)",
        r"(?:first|then|after that|before that).*(?:who|what|where)",
    )),
    ("causal", (
        r"原因", r"为什么", r"导致", r"因为", r"影响", r"因果", r"(?:why|cause|caused|because|result)",
    )),
    ("temporal", (
        r"(?:之前|之后|以后|以前|先后|顺序|当时|期间|同一天|后来|时间线|时序)",
        r"(?:before|after|during|timeline|chronological|sequence)",
    )),
    ("relationship", (
        r"(?:关系|一起|同框|同一个|相关|联系|参与|出席|陪同)",
        r"(?:relationship|together|with whom|related|attended)",
    )),
    ("cross_media", (
        r"(?:视频.*照片|照片.*视频|图.*视频|视频.*图|片段)",
        r"(?:video.*photo|photo.*video|cross.?media|clip)",
    )),
)


def _mode() -> str:
    value = str(os.getenv("SENTRIX_GRAPH_RETRIEVAL_MODE", "auto")).strip().lower()
    return value if value in {"off", "auto", "on"} else "auto"


def _query_text(question: str, query: Any | None) -> str:
    parts = [str(question or "").strip()]
    if query is not None:
        parts.append(str(getattr(query, "whole_query", "") or "").strip())
        for facet in getattr(query, "facets", None) or []:
            parts.append(str(getattr(facet, "surface_text", "") or "").strip())
    return " ".join(dict.fromkeys(part for part in parts if part)).lower()


def graph_retrieval_policy(question: str, *, filters: Any | None = None,
                           query: Any | None = None) -> dict:
    """Return the route contract persisted into retrieval telemetry.

    The result contains no model judgement and is therefore stable across
    evaluations.  ``filters.time_bounds`` is also a temporal signal even when
    the natural-language wording has already been normalized away by query
    parsing.
    """
    mode = _mode()
    text = _query_text(question, query)
    intent = "ordinary"
    matched_signal = ""
    for candidate, patterns in _PATTERNS:
        match = next((pattern for pattern in patterns if re.search(pattern, text, re.IGNORECASE)), None)
        if match:
            intent, matched_signal = candidate, match
            break
    if intent == "ordinary" and filters is not None and (
        getattr(filters, "time_bounds", None) is not None
        or getattr(filters, "annual_time_window", None) is not None
    ):
        intent, matched_signal = "temporal", "hard_time_filter"

    enabled = mode == "on" or (mode == "auto" and intent != "ordinary")
    if mode == "off":
        reason = "disabled_by_mode"
    elif enabled:
        reason = f"semantic_intent:{intent}" if intent != "ordinary" else "enabled_by_mode"
    else:
        reason = "ordinary_query_auto_mode"
    return {
        "enabled": enabled,
        "mode": mode,
        "intent": intent,
        "reason": reason,
        "matched_signal": matched_signal or None,
        "policy_version": "graph-route-v2",
    }
