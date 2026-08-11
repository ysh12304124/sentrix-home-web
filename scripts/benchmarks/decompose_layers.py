#!/usr/bin/env python3
"""Phase F v2 F9 — QA 分层归因 R/V/O/T/S/G/J + Tool 性能聚合。

对 QA run 的每一行做确定性分层归因（不调用模型）：
  R — Retrieval   ：gold 证据图是否进入检索结果（evidence.recall）
  V — Visual      ：需要看图的问题是否调用了视觉工具且召回达标
  O — OCR         ：需要读文字的问题是否调用 read_photo_text
  T — Tool Seq    ：工具序列是否完整（检索→证据工具→final），是否存在 premature final
  S — Synthesis   ：证据已找到但 final 是否出错（judge wrong / jargon 泄漏）
  G — Guard       ：是否被 guard 拦截（blocked_by_guard）
  J — Judge       ：judge 是否成功给出判定

primary = 第一个失败的层级；全部通过 = PASS。

用法（被 run_qa_benchmark.py 调用，也可独立对 qa_result.json 做离线归因）：
  decompose_row(row) -> {"primary":..., "layers":{...}, "detail":{...}}
  aggregate_tool_perf(rows) -> {tool: {calls, ok_rate, p50, p95, providers, ...}}
"""
from __future__ import annotations

import re
import statistics
from collections import defaultdict

SEARCH_TOOLS = {"search_memories", "query_memory_facts", "search_conversation_history"}
VISUAL_TOOLS = {"inspect_photo", "read_photo_text", "get_original_photos", "get_result_page"}

# 需要读文字/数字的问题信号（问题或 gold 中出现）
OCR_RE = re.compile(
    r"电话|价格|多少钱|号码|编号|菜单|招牌|字|年份|日期|数字|元|块|号|价|多少岁|数量|第\s*几张"
)
# 需要看图的问题信号
VISUAL_RE = re.compile(
    r"合影|穿|颜色|场景|图案|拿着|戴|坐|站|摆|沙雕|火把|气球|衣服|动作|表情|身上|手里|哪张|哪一"
    r"张|菜单|招牌|照片中|图中|画面"
)
# 检索 jargon 泄漏（与 evaluate_answer_style.py 保持一致的超集）
JARGON_TERMS = [
    "候选照片", "候选", "partial_support", "candidate_only", "full_support", "no_match",
    "匹配程度", "检索结果", "相似候选", "query_satisfaction", "条件已确认", "部分确认",
    "相似匹配", "关键词的相似", "基于关键词", "相似度", "top", "得分",
]

_LAYER_ORDER = ["R", "V", "O", "T", "S", "G", "J"]


def _has(tools, names) -> bool:
    return any(t in names for t in (tools or []))


def decompose_row(row: dict) -> dict:
    qid = row.get("qa_id") or ""
    question = (row.get("question") or "").strip()
    gold = (row.get("gold_answer") or "").strip()
    answer = (row.get("answer") or "").strip()
    tools = list(row.get("tools") or [])
    status = row.get("status") or ""
    ev = row.get("evidence") or {}
    recall = ev.get("recall")
    has_gold = bool(ev.get("has_gold"))
    judge = row.get("judge") or {}
    guard = row.get("guard_debug") or {}
    verdict = (judge or {}).get("verdict")

    text = question + gold
    need_ocr = bool(OCR_RE.search(text))
    need_visual = bool(VISUAL_RE.search(question)) or has_gold
    recall_ok = (recall or 0) >= 0.5
    a_norm = re.sub(r"\s+", "", answer)

    layers = {}
    detail = {}

    # R
    if has_gold:
        layers["R"] = "pass" if recall_ok else "fail"
        detail["R"] = f"gold={len(ev.get('gold') or [])} recall={recall}"
    else:
        layers["R"] = "na"
        detail["R"] = "无 gold 证据（unanswerable）"

    # V
    if need_visual:
        called_visual = _has(tools, VISUAL_TOOLS)
        if called_visual and recall_ok:
            layers["V"] = "pass"
            detail["V"] = "已调视觉工具且召回达标"
        elif called_visual:
            layers["V"] = "fail"
            detail["V"] = "已调视觉工具但证据召回不足"
        else:
            layers["V"] = "fail"
            detail["V"] = "需要看图但未调用视觉工具"
    else:
        layers["V"] = "na"
        detail["V"] = "无需看图"

    # O
    if need_ocr:
        if "read_photo_text" in tools:
            layers["O"] = "pass"
            detail["O"] = "已调用 read_photo_text"
        else:
            layers["O"] = "fail"
            detail["O"] = "问题含文字/数字信号但未调用 read_photo_text"
    else:
        layers["O"] = "na"
        detail["O"] = "无需 OCR"

    # T
    if status in ("error", "timeout"):
        layers["T"] = "fail"
        detail["T"] = f"状态 {status}"
    elif has_gold and not _has(tools, SEARCH_TOOLS):
        layers["T"] = "fail"
        detail["T"] = "有 gold 证据但未检索"
    elif has_gold and recall_ok and not (_has(tools, VISUAL_TOOLS) or "read_photo_text" in tools):
        empty_ans = not answer or bool(re.search(r"无法|不能确定|没有找到|如果需要|建议", answer))
        if empty_ans:
            layers["T"] = "fail"
            detail["T"] = "premature final：证据已召回但未继续解析即结束"
        else:
            layers["T"] = "pass"
            detail["T"] = "直接由记忆/检索给出答案"
    else:
        layers["T"] = "pass"
        detail["T"] = "工具序列完整"
    if "T" not in layers:
        layers["T"] = "pass"

    # S
    if recall_ok:
        jargon = [t for t in JARGON_TERMS if t in a_norm]
        if verdict == "wrong":
            layers["S"] = "fail"
            detail["S"] = f"证据已召回但 judge=wrong（jargon={jargon}）"
        elif jargon:
            layers["S"] = "fail"
            detail["S"] = f"final 泄漏检索 jargon: {jargon[:3]}"
        elif verdict in ("correct", "partial"):
            layers["S"] = "pass"
            detail["S"] = f"judge={verdict}"
        elif verdict == "error" or not judge.get("judged"):
            layers["S"] = "na"
            detail["S"] = "judge 未判定"
        else:
            layers["S"] = "pass"
            detail["S"] = f"judge={verdict}"
    else:
        layers["S"] = "na"
        detail["S"] = "证据未召回（先归因 R）"

    # G
    if status == "blocked_by_guard":
        layers["G"] = "fail"
        detail["G"] = f"guard 拦截: {guard.get('reason') or guard.get('l1_codes') or ''}"
    elif guard.get("l1_codes") or guard.get("reason"):
        layers["G"] = "pass"
        detail["G"] = "guard 参与但未误拦"
    else:
        layers["G"] = "na"
        detail["G"] = "guard 未触发"

    # J
    if judge is None:
        layers["J"] = "na"
        detail["J"] = "未启用 judge"
    elif judge.get("judged"):
        layers["J"] = "pass"
        detail["J"] = f"verdict={verdict}"
    else:
        layers["J"] = "fail"
        detail["J"] = "judge 调用失败"

    primary = next((L for L in _LAYER_ORDER if layers.get(L) == "fail"), None)
    return {"primary": primary or "PASS", "layers": layers, "detail": detail}


def decompose_summary(rows: list[dict]) -> dict:
    prim = defaultdict(int)
    layer_fail = defaultdict(int)
    layer_pass = defaultdict(int)
    layer_na = defaultdict(int)
    for r in rows:
        d = r.get("decom") or decompose_row(r)
        prim[d["primary"]] += 1
        for L, st in (d.get("layers") or {}).items():
            if st == "fail":
                layer_fail[L] += 1
            elif st == "pass":
                layer_pass[L] += 1
            else:
                layer_na[L] += 1
    return {
        "primary": dict(prim),
        "layer_fail": dict(layer_fail),
        "layer_pass": dict(layer_pass),
        "layer_na": dict(layer_na),
    }


def aggregate_tool_perf(rows: list[dict]) -> dict:
    """聚合 run 级工具性能：calls / ok / p50 / p95 / provider / tiles。

    数据来自每行 telemetry.tool_trace（耗时+状态）与 telemetry.tool_perf（OCR provider 等）。
    """
    per = {}
    for r in rows:
        tel = r.get("telemetry") or {}
        for t in tel.get("tool_trace") or []:
            name = t.get("tool") or ""
            if not name:
                continue
            slot = per.setdefault(name, {"calls": 0, "ok": 0, "lat_ms": [], "providers": set(), "tiles": set(), "cache_hits": 0, "vlm_calls": 0})
            slot["calls"] += 1
            if (t.get("status") or "") == "ok":
                slot["ok"] += 1
            lat = t.get("latency_s")
            if isinstance(lat, (int, float)) and lat >= 0:
                slot["lat_ms"].append(round(lat * 1000, 1))
        # 行级 tool_perf（含 OCR provider/cache，可选）
        for name, stats in (tel.get("tool_perf") or {}).items():
            slot = per.setdefault(name, {"calls": 0, "ok": 0, "lat_ms": [], "providers": set(), "tiles": set(), "cache_hits": 0, "vlm_calls": 0})
            slot["providers"].update(stats.get("providers") or [])
            slot["tiles"].update(stats.get("tiles") or [])
            slot["cache_hits"] += stats.get("cache_hits") or 0
            slot["vlm_calls"] += stats.get("vlm_calls") or 0
    out = {}
    for name, s in per.items():
        lat = s["lat_ms"]
        out[name] = {
            "calls": s["calls"],
            "ok_rate": round(s["ok"] / max(1, s["calls"]), 3),
            "p50_ms": round(statistics.median(lat), 1) if lat else None,
            "p95_ms": round(sorted(lat)[int(len(lat) * 0.95) - 1], 1) if len(lat) >= 2 else (round(lat[0], 1) if lat else None),
            "max_ms": round(max(lat), 1) if lat else None,
            "providers": sorted(s["providers"]),
            "tiles": sorted(s["tiles"]),
            "cache_hits": s["cache_hits"],
            "vlm_calls": s["vlm_calls"],
        }
    return out
