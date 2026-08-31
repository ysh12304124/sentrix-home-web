import json
PATH = "docs/reports/tool_audit/results_134140.jsonl"
rows = []
with open(PATH) as f:
    for line in f:
        line = line.strip()
        if line:
            rows.append(json.loads(line))
by_qid = {r["qa_id"]: r for r in rows}

for qid in ["album3v4-008", "album3v4-025", "album3v4-051"]:
    r = by_qid[qid]
    print(f"===== {qid} =====")
    print("Q:", r.get("question"))
    print("REF:", r.get("reference_answer"))
    for turn in (r.get("runtime_turns") or []):
        for step in (turn.get("debug_trace") or []):
            if step.get("type") == "tool":
                print(f"  TOOL={step.get('tool')} args={json.dumps(step.get('arguments'), ensure_ascii=False)}")
                print(f"    observation={json.dumps(step.get('observation'), ensure_ascii=False)[:1200]}")
            if step.get("type") == "judge":
                print(f"  JUDGE faithful={step.get('faithful')} problems={step.get('problems')} debug={json.dumps(step.get('debug'), ensure_ascii=False)[:400]}")
    print()
