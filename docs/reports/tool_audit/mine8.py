import json
from collections import Counter
PATH = "docs/reports/tool_audit/results_134140.jsonl"
rows = []
with open(PATH) as f:
    for line in f:
        line = line.strip()
        if line:
            rows.append(json.loads(line))

recovery_scores = []
no_recovery_scores = []
for r in rows:
    gd = r.get("guard_debug") or {}
    ra = gd.get("recovery_attempts", 0)
    score = (r.get("judge") or {}).get("score")
    if ra and ra > 0:
        recovery_scores.append(score)
    else:
        no_recovery_scores.append(score)

print("recovery group score dist:", Counter(recovery_scores), "n=", len(recovery_scores))
print("no-recovery group score dist:", Counter(no_recovery_scores), "n=", len(no_recovery_scores))

# how many recovery-final answers are a "找不到/无法确认" style bail-out
bail_words = ["抱歉","无法确认","没有找到","无法找到","不确定","没能找到","没有相关"]
bail_count = 0
for r in rows:
    gd = r.get("guard_debug") or {}
    ra = gd.get("recovery_attempts", 0)
    if ra and ra > 0:
        ans = r.get("answer") or ""
        if any(w in ans for w in bail_words):
            bail_count += 1
print(f"\nof {len(recovery_scores)} recovery-triggered QAs, {bail_count} final answers are a bail-out ('找不到/无法确认'-style)")

# condition_summary "all unknown" check in search_memories observations
unknown_heavy = 0
total_search_calls = 0
for r in rows:
    for turn in (r.get("runtime_turns") or []):
        for step in (turn.get("debug_trace") or []):
            if step.get("type") == "tool" and step.get("tool") == "search_memories":
                total_search_calls += 1
                obs = step.get("observation") or {}
                preview = obs.get("preview") or []
                if preview:
                    unknowns = sum(1 for p in preview for v in (p.get("condition_summary") or {}).values() if v == "unknown")
                    total_conds = sum(len(p.get("condition_summary") or {}) for p in preview)
                    if total_conds and unknowns / total_conds > 0.8:
                        unknown_heavy += 1
print(f"\nsearch_memories calls with >80% condition_summary=='unknown' across preview: {unknown_heavy}/{total_search_calls}")
