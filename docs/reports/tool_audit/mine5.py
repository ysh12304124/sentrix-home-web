import json
PATH = "docs/reports/tool_audit/results_134140.jsonl"
rows = []
with open(PATH) as f:
    for line in f:
        line = line.strip()
        if line:
            rows.append(json.loads(line))
by_qid = {r["qa_id"]: r for r in rows}

def iter_steps(row):
    for turn in (row.get("runtime_turns") or []):
        for step in (turn.get("debug_trace") or []):
            yield step

for qid in ["album3v4-008", "album3v4-025", "album3v4-051"]:
    r = by_qid[qid]
    print(f"===== {qid} =====")
    for i, s in enumerate(iter_steps(r)):
        t = s.get("type")
        if t == "tool_result" or t == "observation" or "observation" in s or "result" in s:
            print(f"  [{i}] type={t} keys={list(s.keys())}")
            print("     ", json.dumps(s, ensure_ascii=False)[:1500])
    print()
