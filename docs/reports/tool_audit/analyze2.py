import json
from collections import Counter, defaultdict

PATH = "docs/reports/tool_audit/results_134140.jsonl"
PERSON_TOOLS = {"get_core_memory", "get_person_memory", "get_person_profile"}

rows = []
with open(PATH) as f:
    for line in f:
        line = line.strip()
        if line:
            rows.append(json.loads(line))

print(f"total={len(rows)}")

tool_counter = Counter()
tool_error_counter = Counter()
person_tool_usage = []
qtype_counter = Counter()
judge_score_by_hastool = defaultdict(list)

no_tool_calls = 0
retry_denials = []  # dedup/duplicate denials
denied_reasons = Counter()

person_tool_seen_qids = []
place_like_filters = []

for r in rows:
    qid = r.get("qa_id")
    qtype = r.get("question_type") or r.get("task_type")
    qtype_counter[qtype] += 1
    trace = r.get("tool_trace") or []
    if not trace:
        no_tool_calls += 1
    seq = []
    for step in trace:
        tool = step.get("tool")
        status = step.get("status")
        reason = step.get("reason") or ""
        error = step.get("error") or ""
        seq.append(tool)
        tool_counter[tool] += 1
        if status not in ("ok", None):
            tool_error_counter[(tool, status)] += 1
        if "duplicate" in reason or "denied" in str(status) or "denied" in reason:
            denied_reasons[reason[:80]] += 1
        if tool in PERSON_TOOLS:
            person_tool_seen_qids.append((qid, tool, status, reason))
    judge = r.get("judge") or {}
    score = judge.get("score")
    judge_score_by_hastool[tuple(sorted(set(seq)))].append(score)

print("\n=== Tool call frequency (all steps across 100 QA) ===")
for t, c in tool_counter.most_common():
    print(f"  {t}: {c}")

print("\n=== Tool non-ok statuses ===")
for (t, s), c in tool_error_counter.most_common(30):
    print(f"  {t} -> {s}: {c}")

print(f"\n=== QA items with ZERO tool calls: {no_tool_calls} ===")

print(f"\n=== Person-memory tools (get_core_memory/get_person_memory/get_person_profile) usage ===")
print(f"  total calls across 100 QA: {len(person_tool_seen_qids)}")
for item in person_tool_seen_qids[:20]:
    print("  ", item)

print("\n=== question_type distribution ===")
for k, v in qtype_counter.most_common():
    print(f"  {k}: {v}")
