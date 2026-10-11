"""Run the 40+40 HippoVlog annotations against the WorldMM endpoint."""

from __future__ import annotations

import json
import re
import time
import urllib.error
import urllib.request
from pathlib import Path


ANNOTATIONS = Path("/mnt/d/datasets/HippoVlog/annotations/questions_worldmm.jsonl")
API = "http://192.168.0.200:8091/api/worldmm/qa"
SCOPES = {"BpVmNB3eKdM": "hippo-qa-BpVmNB3eKdM", "Ei7hTKr8Ins": "hippo-qa-Ei7hTKr8Ins"}
OUT = Path("HippoVlog-Bp-Ei-worldmm-annotation-qa.json")


def call(payload, timeout=300):
    request = urllib.request.Request(API, data=json.dumps(payload, ensure_ascii=False).encode(), method="POST", headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode())


def parse_letter(value, options):
    text = str(value or "")
    match = re.search(r"(?:answer_letter|letter|答案|选项)\s*[=:：]?\s*[\"'*` ]*([ABCD])\b", text, re.I)
    if match and match.group(1).upper() in options:
        return match.group(1).upper()
    match = re.match(r"\s*[\[【(* ]*([ABCD])\b", text, re.I)
    if match and match.group(1).upper() in options:
        return match.group(1).upper()
    for key, option in options.items():
        if str(option).lower() in text.lower():
            return key
    return None


def main():
    rows = [json.loads(line) for line in ANNOTATIONS.read_text(encoding="utf-8").splitlines() if line.strip() and json.loads(line)["video_id"] in SCOPES]
    results = []
    for index, question in enumerate(rows, 1):
        started = time.perf_counter()
        try:
            response = call({"scope_id": SCOPES[question["video_id"]], "question": question["question_text"], "options": question["options"]})
            error = None
        except Exception as exc:
            response = {"answer": "", "error": str(exc)}
            error = str(exc)
        predicted = parse_letter(response.get("answer"), question["options"])
        correct = predicted == question["correct_answer"]
        item = {
            **question,
            "scope": SCOPES[question["video_id"]],
            "predicted_answer": predicted,
            "model_answer": response.get("answer"),
            "model_explanation": response.get("explanation"),
            "correct": correct,
            "latency_s": round(time.perf_counter() - started, 2),
            "error": error,
            "evidence": response.get("evidence") or [],
            "retrieval": response.get("retrieval") or {},
            "retrieval_timing": response.get("retrieval_timing") or {},
            "reason": ("回答与标注答案一致；检索证据见 evidence。" if correct else "回答与标注答案不一致；对照 annotation_explanation 与检索 evidence 可定位是音频、视觉或时间对齐不足。"),
        }
        results.append(item)
        OUT.write_text(json.dumps({"results": results}, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[{question['video_id']}] {index:02d}/{len(rows)} {'PASS' if correct else 'FAIL'} {item['latency_s']:.1f}s expected={question['correct_answer']} predicted={predicted}", flush=True)
    summary = {}
    for video_id in SCOPES:
        scoped = [item for item in results if item["video_id"] == video_id]
        by_category = {}
        for category in sorted({item["category"] for item in scoped}):
            group = [item for item in scoped if item["category"] == category]
            by_category[category] = {"total": len(group), "passed": sum(item["correct"] for item in group), "pass_rate": round(sum(item["correct"] for item in group) / max(1, len(group)), 4)}
        summary[video_id] = {"total": len(scoped), "passed": sum(item["correct"] for item in scoped), "pass_rate": round(sum(item["correct"] for item in scoped) / max(1, len(scoped)), 4), "avg_latency_s": round(sum(item["latency_s"] for item in scoped) / max(1, len(scoped)), 2), "categories": by_category}
    OUT.write_text(json.dumps({"system": "WorldMM-style caption/audio/episodic/semantic/visual late fusion", "summary": summary, "results": results}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
