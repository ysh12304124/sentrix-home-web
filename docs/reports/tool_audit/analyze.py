import json
from collections import Counter, defaultdict

PATH = "docs/reports/tool_audit/results_134140.jsonl"

PERSON_TOOLS = {"get_core_memory", "get_person_memory", "get_person_profile"}

rows = []
with open(PATH) as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))

print(f"total rows: {len(rows)}")

tool_call_counter = Counter()
person_tool_calls = []
place_pollution = []
retry_heavy = []
low_cap_inspect = []
judge_fail_ids = []
tool_seq_by_qid = {}

for r in rows:
    qid = r.get("qa_id")
    trace = r.get("tool_trace") or []
    seq = []
    for step in trace:
        # tool_trace entries shape unknown yet; inspect first
        pass
    tool_seq_by_qid[qid] = trace

# Print a sample of tool_trace structure
sample = rows[0]
print("---- sample tool_trace ----")
print(json.dumps(sample.get("tool_trace"), ensure_ascii=False, indent=2)[:3000])
print("---- sample judge ----")
print(json.dumps(sample.get("judge"), ensure_ascii=False, indent=2)[:1000])
print("---- sample guard_debug ----")
print(json.dumps(sample.get("guard_debug"), ensure_ascii=False, indent=2)[:1500])
