#!/usr/bin/env python3
"""Replay representative questions and verify final writer authority."""
import json, sys, time, urllib.request, subprocess
from pathlib import Path

BASE = (sys.argv[1] if len(sys.argv) > 1 else "http://192.168.0.153:8091").rstrip("/")
QA = Path(sys.argv[2] if len(sys.argv) > 2 else "services/photobench/data/album3-max/qa/album3-max-100qa.jsonl")
IDS = {"052","097","058","086","001","048","095","006","007","063","088","025","079"}

def call(path, payload=None, timeout=900):
    cmd = ["curl", "-sS", "--max-time", str(timeout), BASE + path]
    if payload is not None:
        cmd += ["-H", "Content-Type: application/json",
                "--data-binary", json.dumps(payload, ensure_ascii=False)]
    last = ""
    for _ in range(5):
        proc = subprocess.run(cmd, capture_output=True, text=True)
        last = proc.stderr or proc.stdout
        if proc.returncode == 0:
            try:
                return json.loads(proc.stdout)
            except json.JSONDecodeError:
                pass
        time.sleep(2)
    raise RuntimeError(last)

def turn(message, conversation_id):
    initial = call("/api/assistant/turn", {"message":message,"scope_id":"album_ca0cc0ddda3a","conversation_id":conversation_id,"viewer_id":"owner","include_debug":True}, 300)
    if initial.get("status") not in {"running","pending"}:
        return initial
    tid = initial.get("turn_id")
    deadline = time.time() + 900
    while time.time() < deadline:
        state = call("/api/assistant/turn/" + str(tid), None, 120)
        if str(state.get("status","")).lower() in {"complete","completed","done","success"}:
            return state.get("result") or {}
        if str(state.get("status","")).lower() in {"failed","error","cancelled","canceled"}:
            return {"error":state}
        time.sleep(.5)
    return {"error":"timeout","turn_id":tid}

rows=[]
for line in QA.read_text(encoding="utf-8").splitlines():
    row=json.loads(line)
    qid=str(row.get("qa_id","")).split("-")[-1].zfill(3)
    if qid in IDS:
        rows.append((qid, str(row.get("question") or ""), row.get("answer")))
for qid, question, reference in rows:
    result=turn(question, "p53-"+qid+"-"+str(int(time.time())))
    trace=result.get("agent2_trace") or {}
    print(json.dumps({"qa_id":qid,"question":question,"reference":reference,
        "answer":result.get("answer"),"writer_output":result.get("writer_output") or trace.get("writer_output"),
        "termination":trace.get("terminal_reason") or result.get("termination_reason"),
        "grounding":result.get("answer_grounding"),"image_delivery":result.get("image_delivery"),
        "tool_count":len(result.get("tool_trace") or [])},ensure_ascii=False), flush=True)
