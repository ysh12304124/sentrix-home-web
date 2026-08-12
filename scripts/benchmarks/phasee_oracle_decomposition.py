#!/usr/bin/env python3
"""Phase E — 26 题 Oracle Decomposition（离线归因，不碰运行时代码）。

层级：
  R — Retrieval：GT 图是否进入检索 Top-K（来自 qa_result.json evidence 字段）
  V — Visual Oracle：GT 图 + inspect_photo 真实 prompt → 8100 VLM 能否答对
  O — OCR Oracle：GT 图 + OCR prompt → 8100 VLM 文字读取基线（read_photo_text spike）
  T — Tool Selection/Sequence：实际 tools 序列 vs 期望路径
  S — Synthesis：agent final 是否保留 gold 关键实体
  G — Guard：guard_debug 是否拦截/误拦
  J — Judge：verdict 与人工判定是否一致（人工判定表内嵌）

只读：下载 GT 图到本地缓存，调用 8100 只读推理，不写数据库、不改生产行为。
"""
from __future__ import annotations

import argparse
import base64
import json
import re
import sys
import time
import urllib.request
from datetime import datetime
from pathlib import Path

_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))

DEFAULT_BASE = "http://192.168.0.153:4174"
DEFAULT_VLM = "http://192.168.0.153:8100/v1"

INSPECT_PROMPT = """观察这张照片，输出 JSON：
{{"observation": "一句话描述", "certainty": "supported|uncertain"}}
问题：{question}"""

OCR_PROMPT = """请读出这张照片中的所有文字（包括招牌、菜单、价格、数字、电话）。输出 JSON：
{{"text_regions": [{{"text": "...", "confidence": 0.96}}], "full_text": "..."}}
如果看不清文字，full_text 输出空字符串。"""

# 每题 gold answer 的关键实体（人工标注，仅用于离线归因匹配，不进入运行时代码/prompt）
KEY_ENTITIES = {
    "validation-album3-012-q01": {"place": ["如是海度假村", "昌黎", "秦皇岛"]},
    "validation-album3-012-q02": {"num": ["两张", "2张", "2 张"], "text": ["大头儿子", "小头爸爸"]},
    "validation-album3-012-q06": {"unanswerable": True},
    "validation-album3-012-q08": {"color": ["黄色", "镂空"]},
    "validation-album3-012-q03": {"text": ["大头儿子", "小头爸爸"]},
    "validation-album3-024-q01": {"date": ["2022", "6月23日", "6 月 23 日"]},
    "validation-album3-024-q04": {"num": ["三张", "3张", "3 张"], "place": ["江宁路", "普陀"]},
    "validation-album3-024-q08": {"unanswerable": True},
    "validation-album3-024-q05": {"text": ["白色背心", "背心"], "color": ["白色"]},
    "validation-album3-024-q02": {"text": ["大圣葱油拌面", "大圣"]},
    "validation-album3-024-q07": {"num": ["22048084", "22048085"]},
    "validation-album3-026-q01": {"num": ["1974"]},
    "validation-album3-026-q02": {"text": ["顶呱呱"], "num": ["两", "2"]},
    "validation-album3-026-q08": {"unanswerable": True},
    "validation-album3-026-q03": {"num": ["34"]},
    "validation-album3-026-q06": {"num": ["10"]},
    "validation-album3-026-q07": {"num": ["8"]},
    "validation-album3-040-q01": {"text": ["兔子"]},
    "validation-album3-040-q02": {"num": ["5", "五"]},
    "validation-album3-040-q04": {"place": ["永年", "邯郸"], "num": ["5", "五"], "text": ["兔子"]},
    "validation-album3-047-q01": {"place": ["清迈", "夜间动物园", "Chiang Mai"]},
    "validation-album3-047-q02": {"num": ["4", "四"]},
    "validation-album3-047-q04": {"text": ["棕榈", "棕榈树"]},
    "validation-album3-047-q08": {"unanswerable": True},
    "validation-album3-047-q03": {"text": ["火把"], "obj": ["扇形"]},
    "validation-album3-047-q07": {"num": ["210440", "21点04分"], "text": ["火把"]},
}

# 期望 Tool 路径（人工标注，仅归因用）
EXPECTED_TOOLS = {
    "validation-album3-012-q01": ["search_memories"],
    "validation-album3-012-q02": ["search_memories"],
    "validation-album3-012-q06": ["search_memories", "inspect_photo"],
    "validation-album3-012-q08": ["search_memories", "inspect_photo"],
    "validation-album3-012-q03": ["search_memories", "inspect_photo"],
    "validation-album3-024-q01": ["query_memory_facts", "search_memories"],
    "validation-album3-024-q04": ["search_memories"],
    "validation-album3-024-q08": ["search_memories", "inspect_photo"],
    "validation-album3-024-q05": ["search_memories", "inspect_photo"],
    "validation-album3-024-q02": ["search_memories", "inspect_photo"],
    "validation-album3-024-q07": ["search_memories", "inspect_photo"],
    "validation-album3-026-q01": ["search_memories", "inspect_photo"],
    "validation-album3-026-q02": ["search_memories"],
    "validation-album3-026-q08": ["search_memories", "inspect_photo"],
    "validation-album3-026-q03": ["search_memories", "inspect_photo"],
    "validation-album3-026-q06": ["search_memories", "inspect_photo"],
    "validation-album3-026-q07": ["search_memories", "inspect_photo"],
    "validation-album3-040-q01": ["search_memories", "inspect_photo"],
    "validation-album3-040-q02": ["search_memories"],
    "validation-album3-040-q04": ["search_memories"],
    "validation-album3-047-q01": ["search_memories"],
    "validation-album3-047-q02": ["search_memories"],
    "validation-album3-047-q04": ["search_memories", "inspect_photo"],
    "validation-album3-047-q08": ["search_memories", "inspect_photo"],
    "validation-album3-047-q03": ["search_memories", "inspect_photo"],
    "validation-album3-047-q07": ["search_memories", "inspect_photo"],
}

VISUAL_REQUIRED = {k for k, v in EXPECTED_TOOLS.items() if "inspect_photo" in v}


def http_json(method, url, payload=None, timeout=60):
    body = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(url, data=body, method=method,
                                 headers={"Content-Type": "application/json"})
    with _OPENER.open(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode())


def fetch_bytes(url, timeout=90):
    req = urllib.request.Request(url, method="GET")
    with _OPENER.open(req, timeout=timeout) as resp:
        return resp.read()


def load_asset_map(base, scope_id):
    mapping = {}
    try:
        data = http_json("GET", f"{base}/api/assets?scope_id={scope_id}&limit=2000")
        for item in data.get("assets", []):
            fn = item.get("file_name") or ""
            key = Path(fn).name.lower()
            mapping.setdefault(key, item["id"])
    except Exception as exc:
        print(f"[warn] assets 映射失败: {exc}")
    return mapping


def vlm_chat(vlm_base, system, user, image_b64=None, timeout=180):
    payload = {
        "model": "gemma4-12b-it",
        "messages": [{"role": "system", "content": system},
                     {"role": "user", "content": user}],
        "max_tokens": 300,
        "temperature": 0.0,
    }
    if image_b64:
        payload["messages"][1]["content"] = [
            {"type": "text", "text": user},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}},
        ]
    try:
        data = http_json("POST", f"{vlm_base}/chat/completions", payload, timeout=timeout)
        return (data.get("choices") or [{}])[0].get("message", {}).get("content", "")
    except Exception as exc:
        return f"__ERROR__ {exc}"


def extract_json_obj(text):
    if not text or text.startswith("__ERROR__"):
        return {"_error": text}
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        return {"_raw": text[:200]}
    try:
        return json.loads(text[start:end + 1])
    except Exception:
        return {"_raw": text[start:end + 1][:200]}


def norm_digits(text):
    # 归一化全角/半角数字与常见量词，用于数字匹配
    t = str(text or "")
    for ch in "元人民币￥¥:：.个张套杯份":
        t = t.replace(ch, " ")
    t = re.sub(r"\s+", "", t)
    return t


def match_entities(text, ents):
    hits = {}
    for kind, values in ents.items():
        found = []
        for v in values:
            if v.isdigit():
                if re.search(rf"(?<!\d){v}(?!\d)", norm_digits(text) or ""):
                    found.append(v)
            elif v and v.lower() in (text or "").lower():
                found.append(v)
        if found:
            hits[kind] = found
    return hits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qa-result", required=True, help="qa_result.json 路径")
    ap.add_argument("--base", default=DEFAULT_BASE)
    ap.add_argument("--vlm", default=DEFAULT_VLM)
    ap.add_argument("--out", default="~/Downloads/sentrix_qa_report/phasee_qa_decomposition.json")
    ap.add_argument("--cache", default="~/Downloads/sentrix_qa_report/cache/assets")
    ap.add_argument("--no-vlm", action="store_true", help="跳过 VLM oracle（纯静态归因）")
    ap.add_argument("--ocr-only-ids", default="", help="逗号分隔：只对指定 qa_id 跑 OCR oracle")
    args = ap.parse_args()

    result = json.loads(Path(args.qa_result).expanduser().read_text(encoding="utf-8"))
    rows = result["rows"]
    scope_id = result.get("meta", {}).get("scope") or "album3-v2"
    cache = Path(args.cache).expanduser()
    cache.mkdir(parents=True, exist_ok=True)
    asset_map = load_asset_map(args.base, scope_id)
    print(f"assets 映射: {len(asset_map)} 个文件（scope={scope_id}）")

    ocr_only = {x for x in args.ocr_only_ids.split(",") if x}
    analysis = []
    for i, r in enumerate(rows, 1):
        qid = r["qa_id"]
        ents = KEY_ENTITIES.get(qid, {})
        gold = r.get("gold_answer") or ""
        answer = r.get("answer") or ""
        ev = r.get("evidence") or {}
        tools = r.get("tools") or []
        judge = r.get("judge") or {}
        guard = r.get("guard_debug") or {}
        gold_ev = r.get("gold_evidence_ids") or []

        # ---- R 层 ----
        r_pass = ev.get("has_gold", False) and ev.get("hit", False)
        r_recall = ev.get("recall")
        if ents.get("unanswerable"):
            r_pass = True  # unanswerable 题无 GT 图，答案=如实否认，不构成检索失败
            r_recall = "N/A"

        # ---- V 层 Oracle（GT 图 + inspect 真实 prompt）----
        v_res = {"ran": False, "pass": None, "observation": "", "certainty": "", "error": None}
        if not args.no_vlm and gold_ev and qid in VISUAL_REQUIRED and not ents.get("unanswerable"):
            v_res["ran"] = True
            obs_all = []
            for gpath in gold_ev[:2]:
                fn = Path(gpath).name.lower()
                aid = asset_map.get(fn)
                if not aid:
                    v_res["error"] = f"asset 未找到: {fn}"
                    continue
                dest = cache / f"{qid}_{fn.replace('.', '_')}.jpg"
                try:
                    if not dest.is_file():
                        raw = fetch_bytes(f"{args.base}/api/assets/{aid}/file")
                        dest.write_bytes(raw)
                    b64 = base64.b64encode(dest.read_bytes()).decode()
                    sys_msg = "你是照片复核助手。严格按要求的 JSON 输出。"
                    raw_out = vlm_chat(args.vlm, sys_msg,
                                       INSPECT_PROMPT.format(question=r["question"]), b64)
                    parsed = extract_json_obj(raw_out)
                    obs = parsed.get("observation") or parsed.get("_raw") or raw_out
                    obs_all.append(str(obs)[:300])
                except Exception as exc:
                    v_res["error"] = str(exc)
            v_res["observation"] = " || ".join(obs_all)
            hits = match_entities(v_res["observation"], ents)
            v_res["hit_kinds"] = hits
            v_res["pass"] = bool(hits) or bool(match_entities(v_res["observation"],
                                                             {"text": list(ents.get("text", [])),
                                                              "num": list(ents.get("num", [])),
                                                              "place": list(ents.get("place", [])),
                                                              "color": list(ents.get("color", [])),
                                                              "obj": list(ents.get("obj", []))}))

        # ---- O 层 OCR Oracle（只对 OCR/数字/食物价格类）----
        o_res = {"ran": False, "pass": None, "full_text": "", "error": None}
        ocr_angles = {"ocr_named_entity", "food_or_object"}
        if not args.no_vlm and qid in VISUAL_REQUIRED and (r.get("angle") in ocr_angles or qid in ocr_only) and gold_ev:
            o_res["ran"] = True
            texts = []
            for gpath in gold_ev[:2]:
                fn = Path(gpath).name.lower()
                aid = asset_map.get(fn)
                if not aid:
                    continue
                dest = cache / f"{qid}_{fn.replace('.', '_')}.jpg"
                try:
                    if not dest.is_file():
                        raw = fetch_bytes(f"{args.base}/api/assets/{aid}/file")
                        dest.write_bytes(raw)
                    b64 = base64.b64encode(dest.read_bytes()).decode()
                    raw_out = vlm_chat(args.vlm, "你是 OCR 助手，严格按 JSON 输出。", OCR_PROMPT, b64)
                    parsed = extract_json_obj(raw_out)
                    full = parsed.get("full_text") or ""
                    regions = [x.get("text", "") for x in parsed.get("text_regions") or []]
                    texts.append((full + " " + " ".join(regions)).strip()[:500])
                except Exception as exc:
                    o_res["error"] = str(exc)
            o_res["full_text"] = " || ".join(texts)
            hits = match_entities(o_res["full_text"], ents)
            o_res["hit_kinds"] = hits
            o_res["pass"] = bool(hits)

        # ---- T 层 ----
        expected = EXPECTED_TOOLS.get(qid, [])
        needs_inspect = qid in VISUAL_REQUIRED
        inspected = any("inspect" in t for t in tools)
        t_pass = True
        t_note = ""
        if needs_inspect and not inspected:
            t_pass = False
            t_note = "需要视觉复核但未调用 inspect_photo"
        if not needs_inspect and inspected and r.get("status") != "complete":
            t_note = t_note or "非视觉题调用了 inspect"
        t_missing = [t for t in expected if t.split(".")[0] not in " ".join(tools)]

        # ---- S 层（final 是否保留 gold 关键实体，确定性）----
        s_hits = match_entities(answer, ents) if not ents.get("unanswerable") else {}
        # unanswerable 题：判断是否如实说无法确认
        if ents.get("unanswerable"):
            deny = bool(re.search(r"无法确认|不确定|没有足够|看不出来|无法判断|查不到|没有找到|无法识别", answer))
            s_pass = deny
            s_note = "如实否认" if s_pass else "未如实说明无法确认（疑似编造）"
        else:
            s_pass = bool(s_hits)
            s_note = f"命中 {s_hits}" if s_hits else "final 未包含 gold 关键实体"

        # ---- G 层 ----
        g_blocked = r.get("status") == "blocked_by_guard"
        g_note = ""
        if guard:
            g_note = json.dumps(guard, ensure_ascii=False)[:200]
        elif g_blocked:
            g_note = "status=blocked_by_guard 但无 guard_debug"

        # ---- J 层 ----
        j_verdict = judge.get("verdict")

        # ---- primary root cause ----
        if r.get("status") == "error" or r.get("status") == "timeout":
            primary = "D"
            label = "Data/Error"
        elif not r_pass:
            primary = "R"
            label = "Retrieval"
        elif not t_pass:
            primary = "T"
            label = "Tool Selection/Sequence"
        elif v_res.get("ran") and v_res.get("pass") is False:
            primary = "V"
            label = "Visual/OCR"
        elif o_res.get("ran") and o_res.get("pass") is False and not v_res.get("pass"):
            primary = "V"  # 归入视觉读取能力
            label = "Visual/OCR"
        elif not s_pass:
            primary = "S"
            label = "Synthesis/Final"
        elif g_blocked:
            primary = "G"
            label = "Guard"
        elif j_verdict == "wrong" and s_pass and r_pass:
            primary = "J"
            label = "Judge/Noise"
        else:
            primary = "PASS"
            label = "Pass"

        analysis.append({
            "qa_id": qid, "question": r["question"], "gold_answer": gold,
            "answer": answer, "status": r.get("status"),
            "task_type": r.get("task_type"), "angle": r.get("angle"),
            "answerability": r.get("answerability"),
            "tools": tools,
            "expected_tools": expected, "t_pass": t_pass, "t_note": t_note,
            "r_pass": r_pass, "r_recall": r_recall,
            "v_oracle": v_res, "o_oracle": o_res,
            "s_pass": s_pass, "s_note": s_note,
            "g_blocked": g_blocked, "g_note": g_note,
            "judge": j_verdict, "judge_reason": judge.get("reason", ""),
            "primary": primary, "primary_label": label,
        })
        print(f"[{i:02d}/{len(rows)}] {qid} primary={primary} R={'PASS' if r_pass else 'FAIL'} "
              f"V={v_res.get('pass') if v_res.get('ran') else '—'} "
              f"O={o_res.get('pass') if o_res.get('ran') else '—'} "
              f"T={'PASS' if t_pass else 'FAIL'} S={'PASS' if s_pass else 'FAIL'}")

    from collections import Counter
    primary_counter = Counter(a["primary"] for a in analysis)
    r_fail = [a["qa_id"] for a in analysis if not a["r_pass"]]
    v_fail = [a["qa_id"] for a in analysis if a["v_oracle"].get("ran") and a["v_oracle"].get("pass") is False]
    t_fail = [a["qa_id"] for a in analysis if not a["t_pass"]]
    s_fail = [a["qa_id"] for a in analysis if not a["s_pass"]]
    summary = {
        "total": len(analysis),
        "primary_counter": dict(primary_counter),
        "r_fail": r_fail, "v_fail": v_fail, "t_fail": t_fail, "s_fail": s_fail,
    }
    out_path = Path(args.out).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "meta": {"created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                 "source": str(Path(args.qa_result).expanduser()),
                 "base": args.base, "vlm": args.vlm,
                 "note": "Phase E Oracle Decomposition（离线归因）"},
        "summary": summary, "rows": analysis,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    md = out_path.with_suffix(".md")
    lines = ["# Phase E — 26 题 Oracle Decomposition", "",
             f"- 数据源: {Path(args.qa_result).expanduser()}",
             f"- VLM: {args.vlm}",
             f"- Primary 归因: {json.dumps(primary_counter, ensure_ascii=False)}", ""]
    for a in analysis:
        lines += [f"### {a['qa_id']} [{a['primary_label']}]", 
                  f"- Q: {a['question']}",
                  f"- Gold: {a['gold_answer']}",
                  f"- Agent: {a['answer']}" if len(a['answer']) < 200 else f"- Agent: {a['answer'][:200]}…",
                  f"- Tools: {a['tools']} | 期望: {a['expected_tools']} | T={'PASS' if a['t_pass'] else 'FAIL'}",
                  f"- R={'PASS' if a['r_pass'] else 'FAIL'}(recall={a['r_recall']}) | "
                  f"V={a['v_oracle'].get('pass') if a['v_oracle'].get('ran') else '—'} | "
                  f"O={a['o_oracle'].get('pass') if a['o_oracle'].get('ran') else '—'} | "
                  f"S={'PASS' if a['s_pass'] else 'FAIL'} | G={'blocked' if a['g_blocked'] else 'ok'} | Judge={a['judge']}",
                  f"- V observation: {a['v_oracle'].get('observation', '')[:150]}" if a['v_oracle'].get('ran') else "",
                  f"- OCR text: {a['o_oracle'].get('full_text', '')[:150]}" if a['o_oracle'].get('ran') else "",
                  f"- S note: {a['s_note']}", f"- G note: {a['g_note']}", ""]
    md.write_text("\n".join(lines), encoding="utf-8")
    print()
    print(f"SUMMARY: {json.dumps(summary, ensure_ascii=False)}")
    print(f"输出: {out_path}")
    print(f"Markdown: {md}")


if __name__ == "__main__":
    main()
