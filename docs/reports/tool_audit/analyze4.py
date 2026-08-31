import json

PATH = "docs/reports/tool_audit/results_134140.jsonl"

rows = []
with open(PATH) as f:
    for line in f:
        line = line.strip()
        if line:
            rows.append(json.loads(line))

r = rows[0]
print("=== execution_trace sample ===")
print(json.dumps(r.get("execution_trace"), ensure_ascii=False, indent=2)[:4000])
print("\n=== conversation sample (truncated) ===")
print(json.dumps(r.get("conversation"), ensure_ascii=False, indent=2)[:3000])
print("\n=== runtime_turns sample ===")
print(json.dumps(r.get("runtime_turns"), ensure_ascii=False, indent=2)[:3000])
