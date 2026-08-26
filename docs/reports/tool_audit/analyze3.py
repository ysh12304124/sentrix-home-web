import json, re
from collections import Counter, defaultdict

PATH = "docs/reports/tool_audit/results_134140.jsonl"

rows = []
with open(PATH) as f:
    for line in f:
        line = line.strip()
        if line:
            rows.append(json.loads(line))

# inspect a few tool_trace entries for inspect_photo to see fields (question passed, observation)
count = 0
for r in rows:
    for step in (r.get("tool_trace") or []):
        if step.get("tool") == "inspect_photo":
            count += 1
            if count <= 3:
                print(json.dumps(step, ensure_ascii=False, indent=2)[:2000])
                print("-----")
print("total inspect_photo steps:", count)
