"""L2 模型评审 guard — 12B Observation Faithfulness Judge（对照实验 18 例：bad_recall 90.9% / good_precision 100%）。

分层：L1 FinalGuard（确定性规则，零容忍项兜底）→ L2 本模块（语义级编造/矛盾/漏报判定）。
L2 只对"有工具调用结果"的 turn 在 L1 通过后执行；判定 unfaithful 时返回问题清单交由 runtime 走修正步。
"""

from __future__ import annotations

import json
import re

JUDGE_SYSTEM = """你是 Sentrix 家庭记忆助手的“事实一致性评审”。你的任务是核对模型的最终回答是否忠实于工具观察。

工具观察是唯一可信事实来源。请判断最终回答是否存在以下问题：
1. fabrication（编造）：回答断言了观察中没有的事实或细节。例如观察说“照片中没有猫”，回答却说“猫是白色的”。
2. contradiction（矛盾）：回答与观察直接冲突。例如观察说“多云天气”，回答说“晴天阳光充足”。
3. omission（漏报/否认）：工具返回了结果（total>0 或有观察），回答却说“没有找到/没找到/未找到/不存在”。total>0 时只能说“找到候选但未确认”，不能说“没有找到”。
4. certainty_upgrade（过度声称）：观察/检索只是 candidate_only 或 unknown，回答却说“确认/确定是”。
5. missing_disclosure（缺口未披露）：检索只是 partial/candidate，回答没说明“还不能确认”而直接当成确定事实。

特别注意：
- search_memories 返回 total=0（无候选）时，回答里出现“找到了候选/找到了一些相关的候选照片”就是编造。
- search_memories 返回 total>0（有候选）时，回答里出现“没有找到”就是漏报，即使加了“目前/抱歉”之类的词。

允许的情况（不要误报）：
- 回答忠实复述观察（包括如实说“没有猫/没有人/无法判断”）。
- 检索为空（total=0）时如实回答“没有找到”。
- 回答基于 inspect_photo 观察描述照片内容。

只输出一个 JSON 对象，不要 markdown、不要多余文字：
{"faithful": true 或 false, "problems": [{"type": "...", "detail": "..."}], "reason": "一句话理由"}"""


def parse_verdict(raw: str) -> dict | None:
    raw = (raw or "").strip()
    if not raw:
        return None
    text = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.I)
    text = re.sub(r"\s*```\s*$", "", text)
    start = text.find("{")
    if start < 0:
        return None
    depth, end = 0, None
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                end = i + 1
                break
    if end is None:
        return None
    try:
        return json.loads(text[start:end])
    except Exception:
        return None


def judge_faithfulness(chat_fn, *, query: str, tool_results: list, answer: str) -> tuple[bool, list]:
    """返回 (faithful, problems)。任何异常都降级为放行（L1 已兜底）。"""
    try:
        obs_lines = []
        for tr in tool_results or []:
            compact = {k: tr.get(k) for k in (
                "tool", "total", "satisfaction", "blocked", "inspect_text", "certainty")}
            if compact.get("tool") and compact["tool"] == "inspect_photo" and compact.get("satisfaction") is None:
                compact.pop("satisfaction", None)
            obs_lines.append("- " + json.dumps(compact, ensure_ascii=False))
        user = (
            f"用户问题：{query}\n"
            f"工具观察：\n" + ("\n".join(obs_lines) or "(无)") + "\n"
            f"模型最终回答：{answer}"
        )
        raw = chat_fn([
            {"role": "system", "content": JUDGE_SYSTEM},
            {"role": "user", "content": user},
        ])
        verdict = parse_verdict(raw)
        if not verdict or "faithful" not in verdict:
            return True, []
        if verdict.get("faithful") is False:
            problems = [f"judge_{p.get('type', 'issue')}:{p.get('detail', '')}"
                        for p in (verdict.get("problems") or [])] or ["judge_unfaithful"]
            return False, problems
        return True, []
    except Exception:
        return True, []
