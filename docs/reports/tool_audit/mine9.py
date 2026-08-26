import json
from collections import Counter
PATH = "docs/reports/tool_audit/results_134140.jsonl"
rows = []
with open(PATH) as f:
    for line in f:
        line = line.strip()
        if line:
            rows.append(json.loads(line))

# score=2 in recovery group -- are these "evidence_insufficient" (correctly unanswerable)?
score2_qtypes = Counter()
score0_recovery_qtypes = Counter()
for r in rows:
    gd = r.get("guard_debug") or {}
    ra = gd.get("recovery_attempts", 0)
    score = (r.get("judge") or {}).get("score")
    if ra and ra > 0:
        if score == 2:
            score2_qtypes[r.get("question_type")] += 1
        if score == 0:
            score0_recovery_qtypes[r.get("question_type")] += 1

print("score=2 (correct) in recovery group, by question_type:", score2_qtypes)
print("score=0 (wrong) in recovery group, by question_type:", score0_recovery_qtypes)

# inspect_photo observation "uncertain/无法确认/无法定位" rate
insp_total = 0
insp_uncertain = 0
for r in rows:
    for turn in (r.get("runtime_turns") or []):
        for step in (turn.get("debug_trace") or []):
            if step.get("type") == "tool" and step.get("tool") == "inspect_photo":
                insp_total += 1
                obs = step.get("observation") or {}
                if obs.get("certainty") == "uncertain" or "blocked" in obs:
                    insp_uncertain += 1
print(f"\ninspect_photo calls: {insp_total}, uncertain/blocked: {insp_uncertain} ({insp_uncertain/insp_total:.0%})")

# read_photo_text ocr_failed rate
rpt_total = 0
rpt_failed = 0
for r in rows:
    for turn in (r.get("runtime_turns") or []):
        for step in (turn.get("debug_trace") or []):
            if step.get("type") == "tool" and step.get("tool") == "read_photo_text":
                rpt_total += 1
                obs = step.get("observation") or {}
                if obs.get("reason") == "ocr_failed" or obs.get("status") == "partial":
                    rpt_failed += 1
print(f"read_photo_text calls: {rpt_total}, ocr_failed/partial: {rpt_failed}")

# inspect_photo unknown_handle failures (handle resolution bugs)
unknown_handle = 0
for r in rows:
    for turn in (r.get("runtime_turns") or []):
        for step in (turn.get("debug_trace") or []):
            if step.get("type") == "tool" and step.get("tool") == "inspect_photo":
                obs = step.get("observation") or {}
                if "unknown_handle" in (obs.get("blocked") or []):
                    unknown_handle += 1
                    print("  unknown_handle case, args:", step.get("arguments"))
print(f"inspect_photo unknown_handle failures: {unknown_handle}")
