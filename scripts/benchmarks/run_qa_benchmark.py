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
import json
import re
import sys
import time
import urllib.request
import urllib.error
from collections import Counter
from datetime import datetime
from pathlib import Path

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
1. 对 answerable 的问题：标准答案是一段实质事实。若助手回答属于“没有找到/未找到/无法确认/没有相关记录/查不到/无法回答”，一律判 wrong（这是检索/回答失败），即使语气诚恳。
2. 对 unanswerable（无法回答）的问题：标准答案是“无法确定/没有足够信息”。助手如实说明无法确定/信息不足 → correct；助手编造答案或强行给结论 → wrong。
3. 若助手只回答出标准答案的一部分（例如只答了地点没答人物，或多张照片只找到一张），判 partial。

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
    return {"qa_id": qa.get("qa_id", ""), "question": qa.get("question", ""),
            "gold_answer": qa.get("answer", ""), "answer": "", "status": "error",
            "reason": "", "tools": [], "latency_s": latency, "telemetry": {},
            "task_type": qa.get("task_type", ""), "angle": qa.get("angle", ""),
            "difficulty": qa.get("difficulty", ""), "answerability": qa.get("answerability", ""),
            "gold_evidence_ids": qa.get("answer_evidence_image_ids", []),
            "agent_evidence": [], "evidence": {"has_gold": False, "recall": None,
            "precision": None, "hit": None, "overlap": [], "gold": [], "agent": []},
            "judge": None, "guard_debug": {}, "error": message}


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
    evidence = collect_evidence(result, asset_map)
    score = evidence_score(evidence, qa.get("answer_evidence_image_ids") or [])
    judge = judge_answer(qa["question"], qa.get("answer") or "", answer,
                         qa.get("answerability") == "answerable", judge_base) if judge_enabled else None
    return {
        "qa_id": qa["qa_id"],
        "question": qa["question"],
        "gold_answer": qa.get("answer", ""),
        "answer": answer,
        "status": status,
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
        "error": None,
    }


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
    args = ap.parse_args()

    qa_path = Path(args.qa).expanduser()
    if not qa_path.is_file():
        sys.exit(f"QA 文件不存在: {qa_path}")
    rows_in = [json.loads(line) for line in qa_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if args.limit:
        rows_in = rows_in[:args.limit]
    out_dir = Path(args.out).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

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
    meta = {"timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "qa_file": str(qa_path), "base": args.base, "scope": args.scope,
            "judge_base": args.judge_base}
    json_path = out_dir / "qa_result.json"
    json_path.write_text(json.dumps({"meta": meta, "summary": summary, "rows": results},
                                    ensure_ascii=False, indent=2), encoding="utf-8")
    html_path = out_dir / "qa_report.html"
    html_path.write_text(render_html(results, summary, meta), encoding="utf-8")

    print()
    print("-" * 90)
    print(f"SUMMARY: {summary['total']} 题 | complete={summary['statuses'].get('complete',0)} "
          f"| avg={summary['avg_latency_s']}s | judge={summary['verdicts']} "
          f"| evidence hit={summary['evidence_hit']}/{summary['evidence_questions']} recall={summary['evidence_recall_avg']}")
    print(f"工具使用: {summary['tool_usage']}")
    print(f"JSON: {json_path}")
    print(f"HTML: {html_path}")
    return 0 if summary["errored"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
