import json, re
from collections import Counter, defaultdict

PATH = "docs/reports/tool_audit/results_134140.jsonl"
rows = []
with open(PATH) as f:
    for line in f:
        line = line.strip()
        if line:
            rows.append(json.loads(line))

by_qid = {r["qa_id"]: r for r in rows}

def iter_model_steps(row):
    for turn in (row.get("runtime_turns") or []):
        for step in (turn.get("debug_trace") or []):
            if step.get("type") == "model":
                yield step

# Case 1: album3v4-009 filters.place = "婚礼舞台" -- what happened next? did it retry/correct? what's the final answer & judge score?
for qid in ["album3v4-009", "album3v4-025", "album3v4-008", "album3v4-051"]:
    r = by_qid[qid]
    print(f"===== {qid} =====")
    print("question:", r.get("question"))
    print("judge score:", (r.get("judge") or {}).get("score"), "| reason:", (r.get("judge") or {}).get("reason", "")[:150])
    print("answer:", (r.get("answer") or "")[:200])
    print("termination_reason:", r.get("termination_reason"), "| agent_status:", r.get("agent_status"))
    steps = list(iter_model_steps(r))
    print(f"total model steps: {len(steps)}")
    for i, s in enumerate(steps):
        raw = (s.get("raw_full") or s.get("raw") or "")[:200]
        print(f"  [{i}] {raw}")
    print()
