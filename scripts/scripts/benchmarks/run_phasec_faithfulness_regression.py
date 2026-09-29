"""Phase C/C10 — Faithfulness v2 线上回归指标（打 8091 生产，scope=全部相册）。

输出：
  - 每个 P0 场景的 tool_loop_status / guard_debug.recovery_attempts
  - Guard Recovery Success = 有恢复步数且最终 complete 的比例
  - Hard fact wrong final 计数（exists 类若 blocked_by_guard 或回答与事实矛盾记为失败）
  - candidate_only -> full match 计数（C8 分层回答应出现"候选/不能完全确认"披露）
"""

import json
import sys
import time
import urllib.request

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8091"

QUESTIONS = [
    # (id, question, 期望)
    ("q03_exists", "2023年5月拍过照片吗？", "exists"),
    ("p0_1_place", "去年去过哪里？", "place"),
    ("p0_2_meal", "这两年吃过什么？", "meal"),
    ("c8_layered", "去年十月爬山的照片山上有雪吗？", "search_inspect"),
    ("p0_4_time", "去年春天去了哪里？", "place"),
]


def post(path, payload):
    req = urllib.request.Request(BASE + path, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=20) as r:
        return json.loads(r.read().decode())


def get(path):
    with urllib.request.urlopen(BASE + path, timeout=20) as r:
        return json.loads(r.read().decode())


def run_turn(message):
    start = post("/api/assistant/turn", {"message": message, "conversation_id": None,
                                         "scope_id": "", "viewer_id": "owner"})
    turn_id = start["turn_id"]
    t0 = time.time()
    done = None
    while time.time() - t0 < 240:
        try:
            poll = get(f"/api/assistant/turn/{turn_id}")
        except Exception:
            time.sleep(1)
            continue
        if poll.get("status") in ("complete", "error"):
            done = poll.get("result") or {}
            break
        time.sleep(1)
    return done or {"tool_loop_status": "timeout", "answer": ""}


def main():
    print("=" * 90)
    print(f"C10 Faithfulness v2 线上回归（{BASE}）")
    rows = []
    for qid, question, kind in QUESTIONS:
        r = run_turn(question)
        gd = r.get("guard_debug") or {}
        answer = (r.get("answer") or "").strip()
        status = r.get("tool_loop_status")
        recovered = gd.get("recovery_attempts", 0) > 0
        disclosure = any(w in answer for w in ("候选", "不能完全确认", "无法完全", "还不能确认", "不确定", "部分确认"))
        rows.append({
            "id": qid, "kind": kind, "status": status,
            "recovery_attempts": gd.get("recovery_attempts", 0),
            "recovered_ok": recovered and status == "complete",
            "l1_codes": gd.get("l1_codes", []),
            "answer_len": len(answer),
            "disclosure": disclosure,
            "answer": answer[:200],
        })
        print(f"[{qid}] {question}")
        print(f"  status={status} recovery={rows[-1]['recovery_attempts']} l1={rows[-1]['l1_codes']} disclosure={disclosure}")
        print(f"  answer: {answer[:200]}")

    completes = [r for r in rows if r["status"] == "complete"]
    recoverable = [r for r in rows if r["recovery_attempts"] > 0]
    print("=" * 90)
    print(f"completion: {len(completes)}/{len(rows)}")
    print(f"guard recovery success: {sum(1 for r in recoverable if r['recovered_ok'])}/{len(recoverable)}"
          f" ({100 * sum(1 for r in recoverable if r['recovered_ok']) / max(len(recoverable), 1):.0f}%)")
    print(f"candidate/full-assert disclosure present in search_inspect: "
          f"{[r['id'] for r in rows if r['kind'] == 'search_inspect' and r['disclosure']]}")


if __name__ == "__main__":
    main()
