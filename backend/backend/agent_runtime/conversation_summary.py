"""D3 — Conversation Summary。

- ``build_conversation_summary``：确定性压缩（可审计，零额外模型成本）。
- ``summarize_with_model``：可选 12B 摘要（后台线程调用，不阻塞回答交付）。

Summary 只服务当前会话，可随会话删除；不等于 Core Memory，不自动成为家庭事实。
"""

from __future__ import annotations

import json

SUMMARY_SYSTEM = (
    "你是 Sentrix 家庭记忆助手的对话摘要器。请把下面的对话压缩成 4-8 条要点，"
    "用中文、每条一行，开头用短横线。只保留：用户目标/正在整理的话题、已确认的事实、"
    "未解决的问题、用户提到但还没完成的事。不要编造对话里没有的内容。"
)


def _msg_text(content) -> str:
    if isinstance(content, dict):
        return str(content.get("text") or content.get("content") or "")
    if isinstance(content, str):
        return content
    return ""


def build_conversation_summary(messages: list, *, max_chars: int = 900) -> str:
    """确定性摘要：最近 14 条消息压缩为 用户/助手 要点行（可审计）。"""
    lines: list[str] = []
    for msg in (messages or [])[-14:]:
        role = "用户" if msg.get("role") == "user" else "助手"
        text = _msg_text(msg.get("content")).strip().replace("\n", " ")
        if not text:
            continue
        lines.append(f"{role}：{text[:120]}")
    if not lines:
        return ""
    prefix = "（本会话历史要点）"
    body = "\n".join(lines)[:max_chars]
    return f"{prefix}\n{body}"


def summarize_with_model(chat_fn, messages: list, *, max_chars: int = 900) -> str:
    """12B 生成结构化摘要；失败时回退确定性压缩。"""
    try:
        lines = []
        for msg in (messages or [])[-40:]:
            role = "用户" if msg.get("role") == "user" else "助手"
            text = _msg_text(msg.get("content")).strip().replace("\n", " ")
            if text:
                lines.append(f"{role}：{text[:160]}")
        if not lines:
            return ""
        raw = chat_fn([
            {"role": "system", "content": SUMMARY_SYSTEM},
            {"role": "user", "content": "\n".join(lines)},
        ])
        text = (raw or "").strip()
        if len(text) < 20:
            raise ValueError("empty summary")
        return text[:max_chars]
    except Exception:
        return build_conversation_summary(messages, max_chars=max_chars)
