"""Run the repository's 40 structured QA cases against video memory scopes."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import sqlite3
import time
import urllib.request
from collections import Counter
from pathlib import Path


def api_json(url, method="GET", payload=None, timeout=180):
    data = None if payload is None else json.dumps(payload, ensure_ascii=False).encode()
    request = urllib.request.Request(url, data=data, method=method, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode())


def query_turn(api, message, scope):
    started = time.perf_counter()
    start = api_json(f"{api}/api/assistant/turn", "POST", {"message": message, "scope_id": scope})
    turn_id = start["turn_id"]
    while True:
        result = api_json(f"{api}/api/assistant/turn/{turn_id}")
        if result.get("status") != "running":
            return result, round(time.perf_counter() - started, 2)
        time.sleep(2)


def iso_date(value):
    text = str(value or "")
    match = re.search(r"(20\d{2})[-/年](\d{1,2})[-/月](\d{1,2})", text)
    return dt.date(int(match.group(1)), int(match.group(2)), int(match.group(3))) if match else None


def ground_truth(db, scope, case):
    conn = sqlite3.connect(db)
    conn.row_factory = sqlite3.Row
    assets = conn.execute("SELECT media_type, captured_at, captured_location FROM assets WHERE scope_id=?", (scope,)).fetchall()
    observations = conn.execute(
        "SELECT o.captured_at, o.place FROM observations o JOIN assets a ON a.id=o.asset_id WHERE a.scope_id=?",
        (scope,),
    ).fetchall()
    reference = case.get("reference") or {}
    media = reference.get("media_type")
    rows = assets if not media else [row for row in assets if row["media_type"] == media]
    place = reference.get("place")
    if place:
        rows = [row for row in observations if place in str(row["place"] or "")]
    time_range = (reference.get("time_range") or {})
    start = iso_date(time_range.get("start"))
    end = iso_date(time_range.get("end"))
    if start or end:
        rows = [row for row in rows if (value := iso_date(row["captured_at"])) and (not start or value >= start) and (not end or value <= end)]
    query = case["query"]
    if "明哥" in query or "乐乐" in query:
        return {"value": 0, "kind": case["answer_type"]}
    if case["answer_type"] in {"count", "exists"}:
        if "照片" in query and not media and not place:
            rows = [row for row in rows if row["media_type"] == "image"]
        value = len(rows)
        return {"value": value, "kind": case["answer_type"]}
    if case["answer_type"] in {"first_occurrence", "last_occurrence"}:
        values = sorted((iso_date(row["captured_at"]) for row in rows if iso_date(row["captured_at"])), reverse=case["answer_type"] == "last_occurrence")
        return {"value": values[0].isoformat() if values else None, "kind": case["answer_type"]}
    if case["answer_type"] == "grouped_list":
        group_by = (reference.get("aggregation") or {}).get("group_by")
        if group_by == "month":
            values = Counter(iso_date(row["captured_at"]).strftime("%Y-%m") for row in rows if iso_date(row["captured_at"]))
        else:
            values = Counter(str(row["captured_location"] or "未标注地点") for row in rows)
        return {"value": dict(values), "kind": case["answer_type"]}
    return {"value": 0, "kind": case["answer_type"]}


def grade(answer, expected):
    answer = str(answer or "")
    value = expected["value"]
    if expected["kind"] == "count":
        return bool(re.search(rf"(?<!\d){re.escape(str(value))}(?!\d)", answer)) if value else bool(re.search(r"没有|未找到|不存在|0", answer))
    if expected["kind"] == "exists":
        positive = not bool(re.search(r"没有|未找到|不存在|查无", answer))
        return positive if value else not positive
    if expected["kind"] in {"first_occurrence", "last_occurrence"}:
        return bool(value and value[:4] in answer and value[5:7].lstrip("0") in answer) if value else bool(re.search(r"没有|未找到|不存在", answer))
    if expected["kind"] == "grouped_list":
        groups = list(value)
        return all(group in answer or f"{group[:4]}年{int(group[5:]):d}月" in answer for group in groups) if groups else bool(re.search(r"没有|未找到|不存在", answer))
    return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--api", required=True)
    parser.add_argument("--db", required=True)
    parser.add_argument("--cases", required=True)
    parser.add_argument("--scope", action="append", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    cases = json.loads(Path(args.cases).read_text(encoding="utf-8"))["cases"]
    results = []
    for scope in args.scope:
        for index, case in enumerate(cases, 1):
            response, latency = query_turn(args.api, case["query"], scope)
            result = response.get("result") or {}
            answer = result.get("answer") or response.get("answer") or ""
            expected = ground_truth(args.db, scope, case)
            passed = grade(answer, expected)
            record = {"scope": scope, "id": case["query"], "query": case["query"], "answer": answer,
                      "expected": expected, "status": response.get("status"), "passed": passed,
                      "latency_s": latency, "tool_trace": result.get("tool_trace") or result.get("toolTrace") or []}
            results.append(record)
            print(f"[{scope}] {index:02d}/40 {'PASS' if passed else 'FAIL'} {latency:.1f}s {case['query']} -> {answer[:80]}", flush=True)
    summary = {}
    for scope in args.scope:
        scoped = [item for item in results if item["scope"] == scope]
        summary[scope] = {"total": len(scoped), "passed": sum(item["passed"] for item in scoped),
                          "pass_rate": round(sum(item["passed"] for item in scoped) / max(1, len(scoped)), 4),
                          "avg_latency_s": round(sum(item["latency_s"] for item in scoped) / max(1, len(scoped)), 2)}
    output = {"summary": summary, "results": results}
    Path(args.out).write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
