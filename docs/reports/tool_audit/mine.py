import json, re
from collections import Counter, defaultdict

PATH = "docs/reports/tool_audit/results_134140.jsonl"

rows = []
with open(PATH) as f:
    for line in f:
        line = line.strip()
        if line:
            rows.append(json.loads(line))

PERSON_TOOLS = {"get_core_memory", "get_person_memory", "get_person_profile"}

def iter_debug_steps(row):
    for turn in (row.get("runtime_turns") or []):
        for step in (turn.get("debug_trace") or []):
            yield turn, step

# ---- 1. tool_call action inventory (real tool + args, not just tool_trace summary) ----
tool_call_counter = Counter()
person_tool_calls = []
place_pollution = []          # filters.place containing non-place words (heuristic: contains 婚礼/聚餐/生日/活动 etc, or long phrase)
facts_vs_search_confusion = [] # query_memory_facts called with a query that looks like it needs search, or vice versa
unknown_tool_or_parse_fail = []
duplicate_denied = []
low_cap_visual_calls = []      # inspect_photo called, question implies people_count/ocr_number/small_object
retry_costs = defaultdict(int) # qid -> extra model steps beyond first tool_call decision

ACTIVITY_WORDS = ["婚礼","聚餐","生日","旅游","聚会","仪式","春游","毕业","比赛","演出","活动","过年","春节","聚","宴","party","游玩"]

for r in rows:
    qid = r.get("qa_id")
    qtext = r.get("question") or ""
    for turn, step in iter_debug_steps(r):
        raw = step.get("raw_full") or step.get("raw") or ""
        stype = step.get("type")
        if stype != "model":
            continue
        try:
            action = json.loads(raw)
        except Exception:
            unknown_tool_or_parse_fail.append((qid, raw[:120]))
            continue
        act = action.get("action")
        if act == "tool_call":
            tool = action.get("tool")
            args = action.get("arguments") or {}
            tool_call_counter[tool] += 1
            if tool in PERSON_TOOLS:
                person_tool_calls.append((qid, tool, args))
            filt = args.get("filters") or {}
            place = filt.get("place")
            if place and any(w in place for w in ACTIVITY_WORDS):
                place_pollution.append((qid, tool, place, qtext[:40]))
            if tool == "inspect_photo":
                q = (args.get("question") or "")
                if re.search(r"(几个人|多少人|人数)", q):
                    low_cap_visual_calls.append((qid, "people_count", q, qtext[:40]))
                if re.search(r"(号码|价格|多少钱|电话|车牌|门牌|编号)", q):
                    low_cap_visual_calls.append((qid, "ocr_number_via_inspect", q, qtext[:40]))
        elif act is None:
            unknown_tool_or_parse_fail.append((qid, raw[:120]))

print("=== real tool_call counter (from debug_trace, includes retries) ===")
for t, c in tool_call_counter.most_common():
    print(f"  {t}: {c}")

print(f"\n=== person-memory tool calls (get_core_memory/get_person_memory/get_person_profile): {len(person_tool_calls)} ===")
for x in person_tool_calls[:20]:
    print("  ", x)

print(f"\n=== filters.place polluted with activity words: {len(place_pollution)} ===")
for x in place_pollution:
    print("  ", x)

print(f"\n=== JSON parse failures / unknown action in model step: {len(unknown_tool_or_parse_fail)} ===")
for x in unknown_tool_or_parse_fail[:15]:
    print("  ", x)

print(f"\n=== inspect_photo called for low-capability sub-tasks (people_count / ocr_number): {len(low_cap_visual_calls)} ===")
for x in low_cap_visual_calls:
    print("  ", x)
