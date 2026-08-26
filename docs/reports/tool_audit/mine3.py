import json
PATH = "docs/reports/tool_audit/results_134140.jsonl"
rows = []
with open(PATH) as f:
    for line in f:
        line = line.strip()
        if line:
            rows.append(json.loads(line))
by_qid = {r["qa_id"]: r for r in rows}

for qid in ["album3v4-008", "album3v4-025", "album3v4-051", "album3v4-009"]:
    r = by_qid[qid]
    print(f"===== {qid} =====")
    print("guard_debug:", json.dumps(r.get("guard_debug"), ensure_ascii=False))
    print("turn_outcome:", json.dumps(r.get("turn_outcome"), ensure_ascii=False)[:800])
    print()
