import json
PATH = "docs/reports/tool_audit/results_134140.jsonl"
rows = []
with open(PATH) as f:
    for line in f:
        line = line.strip()
        if line:
            rows.append(json.loads(line))
by_qid = {r["qa_id"]: r for r in rows}

# find a qid where inspect_photo unknown_handle happened, print full tool sequence with result_set_ids
for r in rows:
    hit = False
    seq = []
    for turn in (r.get("runtime_turns") or []):
        for step in (turn.get("debug_trace") or []):
            if step.get("type") == "tool":
                tool = step.get("tool")
                obs = step.get("observation") or {}
                info = {"tool": tool, "args": step.get("arguments")}
                if tool == "search_memories":
                    info["result_set_id"] = obs.get("result_set_id")
                    info["total"] = obs.get("total")
                if tool == "inspect_photo":
                    info["blocked"] = obs.get("blocked")
                    info["summary"] = obs.get("summary")
                seq.append(info)
                if tool == "inspect_photo" and "unknown_handle" in (obs.get("blocked") or []):
                    hit = True
    if hit:
        print(f"=== {r['qa_id']} ===  Q: {r.get('question')[:60]}")
        for s in seq:
            print("  ", s)
        print()
