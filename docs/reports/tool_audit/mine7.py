import json, re
PATH = "docs/reports/tool_audit/results_134140.jsonl"
rows = []
with open(PATH) as f:
    for line in f:
        line = line.strip()
        if line:
            rows.append(json.loads(line))

# Reclassify "parse failures": strip common prefixes/fences like the real runtime would, then try again
def try_parse(raw):
    s = raw.strip()
    s = re.sub(r"^thought\s*\n", "", s)
    s = re.sub(r"^```(json)?", "", s).strip()
    s = re.sub(r"```$", "", s).strip()
    try:
        return json.loads(s)
    except Exception:
        return None

real_malformed = []
benign_prefixed = 0
for r in rows:
    qid = r["qa_id"]
    for turn in (r.get("runtime_turns") or []):
        for step in (turn.get("debug_trace") or []):
            if step.get("type") != "model":
                continue
            raw = step.get("raw_full") or step.get("raw") or ""
            if try_parse(raw) is None:
                real_malformed.append((qid, raw[:150]))
            elif raw.strip().startswith("thought"):
                benign_prefixed += 1

print(f"benign 'thought\\n{{json}}' prefixed steps (real parser handles fine): {benign_prefixed}")
print(f"genuinely malformed (not valid JSON even after strip): {len(real_malformed)}")
for x in real_malformed:
    print("  ", x)

# recovery / guard stats
recovery_counts = []
fabrication_flags = 0
qa_with_recovery = []
for r in rows:
    gd = r.get("guard_debug") or {}
    ra = gd.get("recovery_attempts", 0)
    recovery_counts.append(ra)
    if ra and ra > 0:
        qa_with_recovery.append(r["qa_id"])
    for j in (gd.get("judge") or []):
        if not j.get("faithful") and "judge_fabrication" in (j.get("problems") or []):
            fabrication_flags += 1

from collections import Counter
print("\nrecovery_attempts distribution:", Counter(recovery_counts))
print("QA ids with recovery_attempts>=1:", qa_with_recovery)
print("total judge_fabrication flags (across all judge calls, pre+post recovery):", fabrication_flags)

# For QAs with recovery, check: judge score after recovery, and whether pre-recovery final was closer to reference
for qid in qa_with_recovery:
    r = next(x for x in rows if x["qa_id"] == qid)
    print(f"\n--- {qid} --- score={r.get('judge',{}).get('score')} ref={r.get('reference_answer')!r}")
    print("final used:", (r.get('answer') or '')[:150])
