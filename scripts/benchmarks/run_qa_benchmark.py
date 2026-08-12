#!/usr/bin/env python3
"""Sentrix QA 自动测评 — 对齐 4174 真实生产链路。

一键测评入口：
  python run_qa_benchmark.py \
    --qa /Users/rm001/Downloads/album3/qa/full-album3.jsonl \
    --base http://192.168.0.153:4174 --scope album3 \
    --out ~/Downloads/sentrix_qa_report

行为：
  - 通过 4174 的 /api/assistant/turn 逐题提问（真实 tool_loop + 8100 12B + sentrix.db）
  - 轮询 /api/assistant/turn/{id} 直至 complete/error
  - 用 8100 12B 作为 Judge 对比“标准答案 vs agent 答案”（answerable/unanswerable 两种模式）
  - 证据比对：agent 本轮 tool_results/result_set 里的 asset 文件名 vs QA 标准证据图片
  - 输出 JSON + HTML 并列报告（问题 / 标准答案 / agent 答案 / 判定 / 工具 / 证据图）

只读，不写数据库，不改生产行为。
"""

import argparse
import concurrent.futures
import hashlib
import json
import re
import sys
import time
import urllib.request
import urllib.error
from collections import Counter
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from backend.agent_runtime.answer_nucleus import build_nucleus
from decompose_layers import decompose_row, decompose_summary, aggregate_tool_perf

DEFAULT_QA = "/Users/rm001/Downloads/album3/qa/full-album3.jsonl"
DEFAULT_BASE = "http://192.168.0.153:4174"
DEFAULT_SCOPE = "album3"
DEFAULT_JUDGE_BASE = "http://192.168.0.153:8100/v1"
TURN_TIMEOUT_S = 300
POLL_INTERVAL_S = 1.5

JUDGE_SYSTEM = """你是 Sentrix 家庭记忆助手的 QA 自动评分员。给定一个用户问题、标准答案(ground truth)和助手实际给出的回答，判断助手回答与标准答案在事实上是否一致。

评分规则：
- correct：助手回答的核心事实与标准答案一致，没有事实错误，允许措辞不同。
- partial：方向正确但缺少关键细节、或只答对一部分、或包含一处次要偏差。
- wrong：核心事实错误、关键细节编造、或回答与标准答案冲突。

关键规则：
1. 对 answerable 的问题：标准答案是一段实质事实。
   - 助手回答包含标准答案的核心事实（允许措辞不同）→ correct；核心事实完整但附带“还不能完全确认/看起来是”这类不确定性 → correct。
   - 助手给出部分核心事实或方向正确但缺关键细节 → partial。
   - 助手回答完全没有任何核心事实，只是“无法确认/没有找到/如果需要可以再看”这类回避 → wrong（这是回答失败），即使语气诚恳。
   - 助手把“只是候选/未确认”说成“确认/确定是”（编造或确定性升级）→ wrong。
2. 对 unanswerable（无法回答）的问题：标准答案是“无法确定/没有足够信息”。助手如实说明无法确定/信息不足 → correct；助手编造答案或强行给结论 → wrong。
3. 若助手只回答出标准答案的一部分（例如只答了地点没答人物，或多张照片只找到一张），判 partial。
4. 助手回答简洁直接、不重复检索过程，是加分项，不应因此扣分。

只输出一个 JSON 对象，不要 markdown、不要多余文字：
{"verdict": "correct" 或 "partial" 或 "wrong", "reason": "一句话理由"}

只输出一个 JSON 对象，不要 markdown、不要多余文字：
{"verdict": "correct" 或 "partial" 或 "wrong", "reason": "一句话理由"}"""


# macOS 会把系统级 HTTP 代理注入 urllib；本地/LAN 直连必须显式禁用代理。
_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def http_json(method, url, payload=None, timeout=30, retries=2):
    last = None
    for attempt in range(retries + 1):
        try:
            body = json.dumps(payload).encode() if payload is not None else None
            req = urllib.request.Request(url, data=body, method=method,
                                         headers={"Content-Type": "application/json"})
            with _OPENER.open(req, timeout=timeout) as resp:
                return json.loads(resp.read().decode())
        except Exception as exc:
            last = exc
            time.sleep(2 * (attempt + 1))
    raise last
    body = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(url, data=body, method=method,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode())


def health(base):
    try:
        return http_json("GET", f"{base}/api/health", timeout=20)
    except Exception as exc:
        return {"status": "error", "error": str(exc)}


def hardware(base):
    """QA 硬件快照：GPU/CPU/内存 + 当前模型（/api/hardware，全容错）。"""
    try:
        return http_json("GET", f"{base}/api/hardware", timeout=20)
    except Exception as exc:
        return {"error": str(exc)}


_ROW_OCR_STATS: list[dict] = []


def _collect_ocr_stats(result: dict) -> dict:
    """从单题结果收集 OCR provider 使用情况（供 summary 聚合）。"""
    stats = {"providers": [], "latency_s": [], "conf": [], "fallback": 0}
    seen = set()
    for tr in (result.get("task_state") or {}).get("tool_results") or []:
        if tr.get("tool") != "read_photo_text":
            continue
        key = (tr.get("tool_call_id"), tr.get("provider"))
        if key in seen:
            continue
        seen.add(key)
        if tr.get("provider"):
            stats["providers"].append(tr["provider"])
        if tr.get("confidence") is not None:
            stats["conf"].append(float(tr["confidence"]))
        if tr.get("fallback_used"):
            stats["fallback"] += 1
    for t in (result.get("tool_trace") or []):
        if t.get("tool") == "read_photo_text" and t.get("latency_s"):
            stats["latency_s"].append(float(t["latency_s"]))
    return stats


def _pctile(vals: list, q: float):
    if not vals:
        return None
    a = sorted(vals)
    i = min(len(a) - 1, max(0, int(round(len(a) * q)) - 1))
    return round(a[i], 2)


def aggregate_ocr_stats() -> dict:
    """聚合所有题的 OCR provider 使用率 / 延迟分位 / fallback rate。"""
    prov = {}
    lat = []
    conf = []
    fallback = 0
    rows_with_ocr = 0
    for st in _ROW_OCR_STATS:
        if st["providers"]:
            rows_with_ocr += 1
        lat.extend(st["latency_s"])
        conf.extend(st["conf"])
        fallback += st["fallback"]
        for p in st["providers"]:
            prov[p] = prov.get(p, 0) + 1
    total = sum(prov.values())
    return {
        "rows_with_ocr": rows_with_ocr,
        "provider_usage": prov,
        "small_share": round(prov.get("small", 0) / total, 3) if total else None,
        "latency_p50_s": _pctile(lat, 0.5),
        "latency_p95_s": _pctile(lat, 0.95),
        "confidence_avg": round(sum(conf) / len(conf), 3) if conf else None,
        "fallback_count": fallback,
        "call_total": total,
    }


def start_turn(base, message, scope_id):
    return http_json("POST", f"{base}/api/assistant/turn",
                     {"message": message, "conversation_id": None,
                      "scope_id": scope_id, "viewer_id": "owner"}, timeout=30)


def poll_turn(base, turn_id, timeout_s=TURN_TIMEOUT_S):
    t0 = time.time()
    last = None
    while time.time() - t0 < timeout_s:
        try:
            last = http_json("GET", f"{base}/api/assistant/turn/{turn_id}", timeout=20)
        except Exception:
            time.sleep(POLL_INTERVAL_S)
            continue
        if last.get("status") in ("complete", "error"):
            return last
        time.sleep(POLL_INTERVAL_S)
    return {"status": "timeout", "turn_id": turn_id, "result": None}


def load_assets(base, scope_id):
    """asset_id -> file_name 映射（用于证据图片比对）。"""
    mapping = {}
    try:
        data = http_json("GET", f"{base}/api/assets?scope_id={scope_id}&limit=2000", timeout=40)
        for item in data.get("assets", []):
            mapping[item["id"]] = item.get("file_name") or ""
    except Exception as exc:
        print(f"[warn] 拉取 assets 映射失败: {exc}")
    return mapping


def collect_evidence(result, asset_map):
    """从 task_state.tool_results / samples / result_set 汇总 agent 证据 asset_id 集合。"""
    asset_ids = []
    task_state = result.get("task_state") or {}
    for tr in task_state.get("tool_results") or []:
        obs = tr.get("observation") or {}
        for s in (obs.get("samples") or tr.get("samples") or []):
            if s and s.get("asset_id"):
                asset_ids.append(s["asset_id"])
        for key in ("asset_ids", "evidence_asset_ids", "image_ids"):
            for aid in (obs.get(key) or tr.get(key) or []):
                if aid:
                    asset_ids.append(aid)
    seen, names = set(), []
    for aid in asset_ids:
        if aid in seen:
            continue
        seen.add(aid)
        names.append({"asset_id": aid, "file_name": asset_map.get(aid, "")})
    return names


def normalize_name(name):
    base = Path(name or "").name
    return re.sub(r"\.[a-z]+$", "", base, flags=re.I)


def evidence_score(agent_evidence, gold_image_ids):
    gold = {normalize_name(x) for x in gold_image_ids if x}
    agent = {normalize_name(x["file_name"]) for x in agent_evidence if x.get("file_name")}
    if not gold:
        return {"has_gold": False, "recall": None, "precision": None, "hit": None,
                "overlap": [], "gold": [], "agent": []}
    overlap = sorted(gold & agent)
    recall = len(overlap) / len(gold) if gold else 0.0
    precision = len(overlap) / len(agent) if agent else 0.0
    return {"has_gold": True, "recall": round(recall, 3), "precision": round(precision, 3),
            "hit": bool(overlap), "overlap": overlap,
            "gold": sorted(gold), "agent": sorted(agent)}


def judge_answer(question, gold, agent, answerable, judge_base):
    if not agent or not agent.strip():
        return {"verdict": "wrong", "reason": "agent 没有给出答案", "judged": False}
    prompt = (
        f"用户问题：{question}\n\n"
        f"标准答案：{gold}\n\n"
        f"助手回答：{agent}\n\n"
        f"这个问题是否可回答：{'是' if answerable else '否，标准答案为信息不足'}\n\n"
        f"请给出判定 JSON。"
    )
    payload = {
        "model": "gemma4-12b-it",
        "messages": [{"role": "system", "content": JUDGE_SYSTEM},
                     {"role": "user", "content": prompt}],
        "temperature": 0.0,
        "max_tokens": 300,
    }
    try:
        data = http_json("POST", f"{judge_base}/chat/completions", payload, timeout=120)
        text = (data.get("choices") or [{}])[0].get("message", {}).get("content", "")
        match = re.search(r"\{.*\}", text, re.S)
        parsed = json.loads(match.group(0)) if match else {}
        verdict = parsed.get("verdict")
        if verdict not in ("correct", "partial", "wrong"):
            verdict = "wrong"
        return {"verdict": verdict, "reason": parsed.get("reason", text[:120]), "judged": True}
    except Exception as exc:
        return {"verdict": "error", "reason": f"judge 调用失败: {exc}", "judged": False}


def _error_row(qa, message, latency):
    row = {"qa_id": qa.get("qa_id", ""), "question": qa.get("question", ""),
           "gold_answer": qa.get("answer", ""), "answer": "", "status": "error",
           "reason": "", "tools": [], "latency_s": latency, "telemetry": {},
           "task_type": qa.get("task_type", ""), "angle": qa.get("angle", ""),
           "difficulty": qa.get("difficulty", ""), "answerability": qa.get("answerability", ""),
           "gold_evidence_ids": qa.get("answer_evidence_image_ids", []),
           "agent_evidence": [], "evidence": {"has_gold": False, "recall": None,
           "precision": None, "hit": None, "overlap": [], "gold": [], "agent": []},
           "judge": None, "guard_debug": {}, "error": message}
    row["decom"] = decompose_row(row)
    return row


def run_one(qa, base, scope_id, asset_map, judge_base, judge_enabled, idx, total):
    print(f"[{idx}/{total}] {qa['qa_id']} {qa['question'][:50]}...", flush=True)
    started = time.time()
    try:
        start = start_turn(base, qa["question"], scope_id)
        turn_id = start.get("turn_id")
        if not turn_id:
            return _error_row(qa, f"no turn_id: {start}", 0)
    except Exception as exc:
        return _error_row(qa, f"start turn failed: {exc}", round(time.time() - started, 2))

    poll = poll_turn(base, turn_id)
    latency = round(time.time() - started, 2)
    result = poll.get("result") or {} if isinstance(poll.get("result"), dict) else {}
    answer = (result.get("answer") or "").strip()
    status = result.get("tool_loop_status") or poll.get("status")
    tools = [t.get("tool") for t in (result.get("tool_trace") or []) if t.get("tool")]
    telemetry = result.get("telemetry") or {}
    if telemetry and result.get("tool_trace"):
        telemetry["tool_trace"] = result.get("tool_trace")
    # F9：行级 tool_perf（OCR provider / confidence / exact / cache，来自 task_state.tool_results）
    tp = {}
    for tr in (result.get("task_state") or {}).get("tool_results") or []:
        name = tr.get("tool") or ""
        if not name:
            continue
        slot = tp.setdefault(name, {"providers": set(), "confidences": [],
                                    "exact_counts": [], "cache_hits": 0})
        if tr.get("provider"):
            slot["providers"].add(tr.get("provider"))
        if tr.get("confidence") is not None:
            slot["confidences"].append(round(float(tr["confidence"]), 3))
        if isinstance(tr.get("exact_values"), list):
            slot["exact_counts"].append(len(tr["exact_values"]))
        if tr.get("cache_hit"):
            slot["cache_hits"] += 1
    for slot in tp.values():
        slot["providers"] = sorted(slot["providers"])
    if tp:
        telemetry["tool_perf"] = tp
    # Phase H H6：本行 OCR provider 记录（供 summary 聚合 small/VLM 使用率）
    _ROW_OCR_STATS.append(_collect_ocr_stats(result))
    evidence = collect_evidence(result, asset_map)
    score = evidence_score(evidence, qa.get("answer_evidence_image_ids") or [])
    judge = judge_answer(qa["question"], qa.get("answer") or "", answer,
                         qa.get("answerability") == "answerable", judge_base) if judge_enabled else None
    try:
        _nuc = build_nucleus(result.get("task_state") or {}, qa.get("question", ""))
        nucleus = {v.kind: {"value": v.display or str(v.value), "unit": v.unit,
                            "certainty": v.certainty}
                   for v in _nuc.values
                   if v.kind in ("count", "date", "first", "last", "result_total",
                                 "boolean", "price", "phone", "year")}
    except Exception:
        nucleus = {}
    row = {
        "qa_id": qa["qa_id"],
        "question": qa["question"],
        "gold_answer": qa.get("answer", ""),
        "answer": answer,
        "status": status,
        "nucleus": nucleus,
        "reason": result.get("tool_loop_reason") or "",
        "tools": tools,
        "latency_s": latency,
        "telemetry": telemetry,
        "task_type": qa.get("task_type", ""),
        "angle": qa.get("angle", ""),
        "difficulty": qa.get("difficulty", ""),
        "answerability": qa.get("answerability", ""),
        "gold_evidence_ids": qa.get("answer_evidence_image_ids", []),
        "agent_evidence": evidence,
        "evidence": score,
        "judge": judge,
        "guard_debug": result.get("guard_debug") or {},
        "ocr_texts": [tr.get("ocr_text") for tr in (result.get("task_state") or {}).get("tool_results") or []
                      if tr.get("tool") == "read_photo_text" and tr.get("ocr_text")],
        "error": None,
    }
    row["decom"] = decompose_row(row)
    return row


def summarize(rows, health_data):
    n = len(rows)
    errored = [r for r in rows if r.get("error")]
    statuses = Counter(r.get("status") or "error" for r in rows)
    avg_latency = round(sum(r.get("latency_s") or 0 for r in rows) / max(1, n), 1)
    judged = [r for r in rows if (r.get("judge") or {}).get("judged")]
    verdicts = Counter((r.get("judge") or {}).get("verdict") for r in judged)
    ev_has = [r for r in rows if (r.get("evidence") or {}).get("has_gold")]
    ev_hit = sum(1 for r in ev_has if (r.get("evidence") or {}).get("hit"))
    ev_recall = round(sum((r.get("evidence") or {}).get("recall") or 0 for r in ev_has) / max(1, len(ev_has)), 3)
    tool_usage = Counter()
    for r in rows:
        for t in r.get("tools") or []:
            tool_usage[t] += 1
    return {
        "total": n, "errored": len(errored), "statuses": dict(statuses),
        "avg_latency_s": avg_latency,
        "judged": len(judged), "verdicts": dict(verdicts),
        "evidence_questions": len(ev_has), "evidence_hit": ev_hit,
        "evidence_recall_avg": ev_recall,
        "tool_usage": dict(tool_usage),
        "decom": decompose_summary(rows),
        "tool_perf": aggregate_tool_perf(rows),
        "ocr": aggregate_ocr_stats(),
        "health": health_data,
    }


def render_html(rows, summary, meta):
    badge = {"complete": "ok", "partial": "warn", "blocked_by_guard": "warn",
             "error": "bad", "timeout": "bad"}
    verdict_badge = {"correct": "ok", "partial": "warn", "wrong": "bad", "error": "bad"}
    health = summary.get("health") or {}
    tools_html = "".join(
        f'<span class="tool">{t.get("name")}<em>{t.get("readiness")}</em></span>'
        for t in (health.get("agent") or {}).get("tools") or [])
    cards = ""
    cards += f'<div class="card"><b>{summary["total"]}</b><span>题目</span></div>'
    cards += f'<div class="card"><b>{summary["statuses"].get("complete", 0)}</b><span>complete</span></div>'
    cards += f'<div class="card"><b>{summary["avg_latency_s"]}s</b><span>平均耗时</span></div>'
    cards += f'<div class="card"><b>{summary["verdicts"].get("correct", 0)}/{summary["judged"]}</b><span>judge correct</span></div>'
    cards += f'<div class="card"><b>{summary["evidence_hit"]}/{summary["evidence_questions"]}</b><span>证据命中</span></div>'
    cards += f'<div class="card"><b>{summary["evidence_recall_avg"]}</b><span>证据召回均值</span></div>'

    table_rows = []
    for i, r in enumerate(rows, 1):
        j = r.get("judge") or {}
        ev = r.get("evidence") or {}
        gold_imgs = r.get("gold_evidence_ids") or []
        gold_names = [Path(x).name for x in gold_imgs]
        agent_names = [(x.get("file_name") or x.get("asset_id") or "?") for x in r.get("agent_evidence") or []]
        overlap = set(ev.get("overlap") or [])
        def img_tags(names, mark_hit):
            if not names:
                return '<span class="muted">—</span>'
            return "".join(
                f'<span class="evimg{" hit" if mark_hit and normalize_name(n) in overlap else ""}">'
                f'{Path(n).name}</span>' for n in names[:10])
        status_cls = badge.get(r.get("status"), "warn")
        verdict_cls = verdict_badge.get(j.get("verdict"), "warn")
        ev_metrics = ("—" if not ev.get("has_gold") else
                      f'R{ev.get("recall")} P{ev.get("precision")}{" ✓" if ev.get("hit") else ""}')
        table_rows.append(f"""
<tr>
  <td class="num">{i}</td>
  <td class="q"><div class="qid">{r['qa_id']}</div><div>{r['question']}</div>
      <div class="tags"><span>{r.get('task_type','')}</span><span>{r.get('angle','')}</span><span>{r.get('difficulty','')}</span><span>{r.get('answerability','')}</span></div></td>
  <td class="gold">{r.get('gold_answer','')}<div class="evline">证据: {img_tags(gold_names, False)}</div></td>
  <td class="agent"><div class="status {status_cls}">{r.get('status')}</div>{r.get('answer','') or '<span class=muted>(无答案)</span>'}
      <div class="evline">证据: {img_tags(agent_names, True)}</div>
      <div class="tools">{", ".join(r.get("tools") or []) or "—"}</div>
      <div class="meta">latency {r.get('latency_s')}s · {ev_metrics}</div></td>
  <td class="judge"><div class="verdict {verdict_cls}">{j.get('verdict','—') if j else 'skip'}</div><small>{j.get('reason','')}</small></td>
</tr>""")
    health_model = ((health.get("models") or {}).get("vlm") or {})
    return f"""<!doctype html><html lang="zh"><head><meta charset="utf-8">
<title>Sentrix QA Benchmark — {meta['qa_file']}</title>
<style>
 body{{font-family:-apple-system,'PingFang SC',sans-serif;margin:0;background:#f6f7f9;color:#1f2328}}
 header{{background:#111418;color:#fff;padding:20px 28px}}
 header h1{{margin:0 0 6px;font-size:20px}}
 header .sub{{color:#9aa4af;font-size:13px}}
 .wrap{{padding:20px 28px}}
 .cards{{display:grid;grid-template-columns:repeat(6,1fr);gap:12px;margin:16px 0}}
 .card{{background:#fff;border:1px solid #e3e6ea;border-radius:10px;padding:14px;text-align:center}}
 .card b{{display:block;font-size:22px}}
 .card span{{color:#6b7280;font-size:12px}}
 .health{{background:#fff;border:1px solid #e3e6ea;border-radius:10px;padding:14px;margin:16px 0;font-size:13px;color:#374151}}
 .health .tool{{margin-right:10px}} .health em{{color:#9ca3af;margin-left:4px;font-style:normal}}
 table{{width:100%;border-collapse:collapse;background:#fff;border:1px solid #e3e6ea;border-radius:10px;overflow:hidden}}
 th,td{{padding:10px 12px;vertical-align:top;text-align:left;font-size:13px;border-bottom:1px solid #eef0f3}}
 th{{background:#fafbfc;font-weight:600}}
 td.q,.agent{{width:24%}} td.gold{{width:20%}} td.judge{{width:14%}}
 .qid{{color:#9ca3af;font-size:11px;margin-bottom:2px}}
 .tags span{{background:#eef2f6;border-radius:6px;padding:1px 6px;font-size:11px;margin-right:4px;color:#4b5563}}
 .status,.verdict{{display:inline-block;border-radius:6px;padding:1px 8px;font-size:12px;margin-bottom:4px}}
 .ok{{background:#e7f6ec;color:#137a3b}} .warn{{background:#fdf3e0;color:#a05a00}} .bad{{background:#fde8e8;color:#b42318}}
 .evline{{margin-top:6px;font-size:12px;color:#6b7280}}
 .evimg{{display:inline-block;background:#eef2f6;border-radius:4px;padding:1px 5px;margin:2px;font-size:11px}}
 .evimg.hit{{background:#d8f0e0;color:#137a3b}}
 .tools{{margin-top:6px;font-size:11px;color:#9ca3af}}
 .meta{{margin-top:4px;font-size:11px;color:#9ca3af}}
 .muted{{color:#b3b9c2}}
</style></head><body>
<header><h1>Sentrix QA 自动测评报告</h1>
<div class="sub">{meta['timestamp']} · QA: {meta['qa_file']} · 端点: {meta['base']} · scope: {meta['scope']} · Judge: {meta['judge_base']}</div></header>
<div class="wrap">
<div class="cards">{cards}</div>
<div class="health">生产 agent 状态: profile={((health.get('agent') or {{}}).get('profile'))} · tools: {tools_html}
 · VLM: {health_model.get('name','')} @ {health_model.get('endpoint','')} · db: {((health.get('models') or {{}}).get('llm') or {{}}).get('base_url','')}</div>
<table><thead><tr><th>#</th><th>问题</th><th>标准答案</th><th>Agent 回答（4174 实测）</th><th>Judge</th></tr></thead>
<tbody>{''.join(table_rows)}</tbody></table>
</div></body></html>"""


def main():
    ap = argparse.ArgumentParser(description="Sentrix QA 自动测评（对齐 4174 生产链路）")
    ap.add_argument("--qa", default=DEFAULT_QA, help="QA jsonl 路径")
    ap.add_argument("--base", default=DEFAULT_BASE, help="4174 或后端 base URL")
    ap.add_argument("--scope", default=DEFAULT_SCOPE, help="memory scope")
    ap.add_argument("--judge-base", default=DEFAULT_JUDGE_BASE, help="12B judge OpenAI 兼容端点")
    ap.add_argument("--no-judge", action="store_true", help="跳过 LLM judge")
    ap.add_argument("--concurrency", type=int, default=1, help="并发 turn 数（默认 1，GPU 共享）")
    ap.add_argument("--limit", type=int, default=0, help="只跑前 N 题（0=全部）")
    ap.add_argument("--out", default="~/Downloads/sentrix_qa_report", help="输出目录")
    ap.add_argument("--tag", default="qa", help="run 标签（如 phasee、baseline）")
    ap.add_argument("--note", default="", help="run 备注")
    ap.add_argument("--no-upload", action="store_true", help="不上传 QA Dashboard")
    args = ap.parse_args()

    qa_path = Path(args.qa).expanduser()
    if not qa_path.is_file():
        sys.exit(f"QA 文件不存在: {qa_path}")
    rows_in = [json.loads(line) for line in qa_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if args.limit:
        rows_in = rows_in[:args.limit]
    out_dir = Path(args.out).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    # G8：manifest / checksum —— QA 数据集在跑前/跑后必须一致，不一致拒绝 run
    qa_md5 = hashlib.md5(qa_path.read_bytes()).hexdigest()
    manifest_path = out_dir / "manifest.json"
    if manifest_path.is_file():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception:
            manifest = {}
        expected = manifest.get("qa_checksum_md5")
        if expected and expected != qa_md5:
            sys.exit(
                f"[拒绝] QA manifest 校验失败：当前文件 md5={qa_md5}，"
                f"manifest 记录={expected}（{manifest.get('qa_file','')}）。"
                "QA 数据集被修改过，为保持基准可比性已终止。如确需更换数据集，请先更新 manifest.json。")
    else:
        manifest = {
            "qa_file": str(qa_path),
            "qa_checksum_md5": qa_md5,
            "cases": len(rows_in),
            "scope_id": args.scope,
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2),
                                 encoding="utf-8")
        print(f"[manifest] 首次运行，已记录 QA 数据集 md5={qa_md5}")
    # 跑完后再次校验，防止运行期间数据集被改动
    _after_md5 = hashlib.md5(qa_path.read_bytes()).hexdigest()
    if _after_md5 != qa_md5:
        sys.exit(f"[拒绝] QA 文件在运行期间被修改（{qa_md5} -> {_after_md5}），结果作废。")

    print("=" * 90)
    print(f"Sentrix QA 自动测评 | base={args.base} scope={args.scope} qa={qa_path}")
    h = health(args.base)
    if h.get("status") != "ok":
        print(f"[warn] 4174 health 非 ok: {h.get('status', h.get('error'))}")
    else:
        print(f"health ok | profile={h['agent']['profile']} tools={[t['name'] for t in h['agent']['tools']]} "
              f"| vlm={h['models']['vlm']['name']}")
    asset_map = load_assets(args.base, args.scope)
    print(f"asset 映射: {len(asset_map)} 张（scope={args.scope}）")

    hw_start = hardware(args.base)
    if hw_start.get("gpu"):
        g = hw_start["gpu"][0]
        print(f"硬件快照: {g['name']} VRAM {g['memory_used_mib']}/{g['memory_total_mib']} MiB "
              f"util={g['utilization_percent']}% temp={g['temperature_c']}C")
    else:
        print(f"硬件快照: 不可用（{hw_start.get('error', '无 GPU/仅系统信息')}）")

    results = []
    if args.concurrency <= 1:
        for i, qa in enumerate(rows_in, 1):
            results.append(run_one(qa, args.base, args.scope, asset_map,
                                   args.judge_base, not args.no_judge, i, len(rows_in)))
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as ex:
            futures = {ex.submit(run_one, qa, args.base, args.scope, asset_map,
                                 args.judge_base, not args.no_judge, i, len(rows_in)): qa
                       for i, qa in enumerate(rows_in, 1)}
            for fut in concurrent.futures.as_completed(futures):
                results.append(fut.result())
        results.sort(key=lambda r: r.get("qa_id", ""))

    summary = summarize(results, h)
    hw_end = hardware(args.base)
    meta = {"timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "qa_file": str(qa_path), "base": args.base, "scope": args.scope,
            "judge_base": args.judge_base, "tag": args.tag,
            "hardware_start": hw_start, "hardware_end": hw_end}
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + args.tag
    run_dir = out_dir / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    # 前端需要 file_name -> asset_id（load_assets 返回 asset_id -> file_name）
    asset_map_rev = {Path(fn).name.lower(): aid for aid, fn in asset_map.items()}
    run_payload = {"meta": meta, "summary": summary, "rows": results, "asset_map": asset_map_rev}
    (run_dir / "qa_result.json").write_text(json.dumps(run_payload, ensure_ascii=False, indent=2),
                                            encoding="utf-8")
    run_meta = {"run_id": run_id, "tag": args.tag, "created_at": meta["timestamp"],
                "note": args.note, "branch_153": "", "profile": (h.get("agent") or {}).get("profile", ""),
                "qa_file": str(qa_path), "qa_checksum_md5": qa_md5, "cases": len(rows_in)}
    (run_dir / "run_meta.json").write_text(json.dumps(run_meta, ensure_ascii=False, indent=2),
                                           encoding="utf-8")
    # 最新副本（兼容旧工具）
    json_path = out_dir / "qa_result.json"
    json_path.write_text(json.dumps(run_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    uploaded = False
    if not args.no_upload:
        try:
            up = http_json("POST", f"{args.base}/api/qa/runs/upload",
                           {"run_id": run_id, "meta": meta, "summary": summary,
                            "rows": results, "asset_map": asset_map_rev,
                            "tag": args.tag, "note": args.note,
                            "profile": run_meta["profile"],
                            "qa_checksum_md5": qa_md5}, timeout=120)
            uploaded = up.get("status") == "ok"
            print(f"Dashboard 上传: {up.get('status')} ({up.get('run_id')})")
        except Exception as exc:
            print(f"[warn] Dashboard 上传失败: {exc}")

    print()
    print("-" * 90)
    print(f"SUMMARY: {summary['total']} 题 | complete={summary['statuses'].get('complete',0)} "
          f"| avg={summary['avg_latency_s']}s | judge={summary['verdicts']} "
          f"| evidence hit={summary['evidence_hit']}/{summary['evidence_questions']} recall={summary['evidence_recall_avg']}")
    print(f"工具使用: {summary['tool_usage']}")
    print(f"run 归档: {run_dir}")
    if uploaded:
        print(f"QA Dashboard: {args.base}/qa")
    else:
        print("QA Dashboard: 未上传（用 --no-upload 关闭此提示）")
    return 0 if summary["errored"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
