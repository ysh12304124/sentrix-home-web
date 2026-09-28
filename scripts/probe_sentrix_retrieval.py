#!/usr/bin/env python3
"""Sentrix 检索健康探测（防复发护栏，2026-08-22 检索降级事故后建立）。

两级探测，改动 8091 / 向量库 / 模型后必跑：

  Level 1（默认，秒级，不需要主模型）：
      GET /api/health 的 memory.vectorIndex —— 断言 qdrant 可用、未降级、
      无锁失败记录；已有检索历史时同时断言 p95 < 阈值。
      对应事故根因①：僵尸进程持 qdrant 锁 → 静默降级 SQLite 全表扫。

  Level 2（--live，约 2 分钟，需要 8100 主模型在线）：
      发 5 个固定问题走 /api/assistant/turn，从 tool_trace 抽每次
      search_memories 的 latency_s，断言全部 < 阈值（默认 3s）。
      对应事故根因②：融合层 N+1 全表解码（修复后单次应 <2s）。

用法：
  python3 scripts/probe_sentrix_retrieval.py                     # Level 1
  python3 scripts/probe_sentrix_retrieval.py --live              # Level 1+2
  python3 scripts/probe_sentrix_retrieval.py --host 127.0.0.1:8091 --threshold-ms 3000

非零退出码 = 探测失败；CI/脚本可直接串联。
"""

import argparse
import json
import sys
import time
import urllib.error
import urllib.request

# 固定 5 题：自然记忆问题，只验证检索时延与链路健康，不验证召回质量。
PROBE_QUESTIONS = [
    "帮我找找去年婚礼相关的照片",
    "最近有没有和家人一起出去玩的照片",
    "找几张有小朋友的照片",
    "有没有拍过风景或者山水的照片",
    "帮我看看聚会吃饭的照片有哪些",
]

TURN_TIMEOUT_S = 150
POLL_INTERVAL_S = 3


def _get_json(url: str, timeout: int = 10):
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _post_json(url: str, payload: dict, timeout: int = 15):
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}, method="POST"
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def probe_health(base: str, threshold_ms: float) -> bool:
    """Level 1：health 的 vectorIndex 状态断言。"""
    print("== Level 1: health vectorIndex ==")
    try:
        health = _get_json(f"{base}/api/health")
    except Exception as exc:
        print(f"[FAIL] /api/health 不可达: {exc}")
        return False
    index = (health.get("memory") or {}).get("vectorIndex") or {}
    if not index:
        print("[FAIL] health.memory.vectorIndex 缺失")
        return False

    ok = True
    lock = index.get("qdrant_lock") or {}
    if lock.get("ts"):
        print(f"[FAIL] qdrant 锁失败记录: {lock}")
        ok = False
    else:
        print(f"[ok] qdrant_lock 无失败记录 (held_by_process={lock.get('held_by_process')})")

    if index.get("degraded"):
        print(f"[FAIL] 检索处于降级状态 degraded_since={index.get('degraded_since')}")
        ok = False
    elif index.get("qdrant_available"):
        print(
            f"[ok] qdrant 可用 collections={index.get('collections')} "
            f"points={index.get('points')}"
        )
    else:
        print("[FAIL] qdrant 不可用（qdrant_available=false）")
        ok = False

    total = index.get("total_searches") or 0
    p95 = index.get("latency_p95_ms")
    if total and p95 is not None:
        if p95 > threshold_ms:
            print(f"[FAIL] 检索 p95={p95}ms 超过阈值 {threshold_ms}ms（共 {total} 次采样）")
            ok = False
        else:
            print(f"[ok] 检索 p95={p95}ms（{total} 次采样，阈值 {threshold_ms}ms）")
    else:
        print("[warn] 尚无检索历史（total_searches=0），跳过 p95 断言；--live 可生成采样")

    recent = index.get("recent") or []
    for item in recent[-5:]:
        print(f"      recent: {item.get('ts')} {item.get('route')} {item.get('backend')} {item.get('ms')}ms")
    return ok


def _wait_turn(base: str, turn_id: str) -> dict:
    deadline = time.time() + TURN_TIMEOUT_S
    while time.time() < deadline:
        try:
            data = _get_json(f"{base}/api/assistant/turn/{turn_id}")
            if data.get("status") in ("complete", "completed", "failed", "error"):
                return data
        except urllib.error.HTTPError:
            pass
        time.sleep(POLL_INTERVAL_S)
    return {"status": "timeout"}


def probe_live(base: str, threshold_ms: float) -> bool:
    """Level 2：真实 assistant turn 的 search_memories 时延断言。"""
    print("== Level 2: 5 题真实检索（需要主模型在线）==")
    ok = True
    for i, question in enumerate(PROBE_QUESTIONS, 1):
        try:
            started = time.time()
            turn = _post_json(f"{base}/api/assistant/turn", {"message": question})
            turn_id = turn.get("turn_id")
            if not turn_id:
                print(f"[FAIL] 题{i} 未返回 turn_id: {str(turn)[:200]}")
                ok = False
                continue
            result = _wait_turn(base, turn_id)
            status = result.get("status")
            wall = time.time() - started
            if status != "complete" and status != "completed":
                print(f"[FAIL] 题{i} turn 状态 {status}（墙钟 {wall:.1f}s）: {question}")
                ok = False
                continue
            payload = result.get("result") or {}
            trace = payload.get("tool_trace") or []
            latencies = [
                item.get("latency_s")
                for item in trace
                if isinstance(item, dict) and item.get("tool") == "search_memories"
                and item.get("latency_s") is not None
            ]
            if not latencies:
                print(f"[FAIL] 题{i} tool_trace 无 search_memories 记录: {question}")
                ok = False
                continue
            worst = max(latencies) * 1000
            verdict = "ok" if worst <= threshold_ms else "FAIL"
            print(
                f"[{verdict}] 题{i} search_memories ×{len(latencies)} 最慢 {worst:.0f}ms "
                f"（turn 墙钟 {wall:.1f}s）: {question}"
            )
            if worst > threshold_ms:
                ok = False
        except Exception as exc:
            print(f"[FAIL] 题{i} 异常: {type(exc).__name__}: {exc}")
            ok = False
    return ok


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--host", default="192.168.0.153:8091", help="Sentrix API host:port")
    parser.add_argument("--threshold-ms", type=float, default=3000, help="单次检索报警阈值")
    parser.add_argument("--live", action="store_true", help="追加 Level 2 真实检索探测")
    args = parser.parse_args()
    base = f"http://{args.host}"

    ok = probe_health(base, args.threshold_ms)
    if args.live:
        ok = probe_live(base, args.threshold_ms) and ok

    print("== 探测结果:", "PASS" if ok else "FAIL", "==")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
