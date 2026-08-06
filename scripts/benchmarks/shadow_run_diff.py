#!/usr/bin/env python3
"""Phase 8 shadow-run diff.

Runs the same query set through the old MemoryAgent path
(``SENTRIX_THIN_AGENT_V1=0``) and the new Thin Agent path
(``SENTRIX_THIN_AGENT_V1=1``) using a shared fixture; reports per-case
divergence in mode, evidence IDs, hard violations and latency so the user
can decide when to flip production.
"""

import argparse
import json
import os
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from backend.agent import MemoryAgent
from backend.db import MemoryStore


class ScriptedGamma:
    model = "shadow-scripted"

    def __init__(self, responses=None):
        self.calls = []
        self.responses = dict(responses or {})

    def chat(self, prompt, *args, **kwargs):
        self.calls.append(prompt)
        for marker, response in self.responses.items():
            if marker in prompt:
                return json.dumps(response, ensure_ascii=False) if isinstance(response, (dict, list)) else str(response)
        return "我在听。"

    def answer(self, query, context):
        return {"answer": "证据不足", "insufficient_evidence": True, "evidence": [], "confidence": 0.0}

    def embed_text(self, text):
        return []


CASES = (
    {"id": "kitchen_month", "query": "2024 年 5 月厨房的照片", "scope": "album1",
     "parser": {"mode": "evidence",
                "actions": [{"type": "return_assets", "coverage": "best"}],
                "time_expression": "2024 年 5 月",
                "media_expressions": ["照片"],
                "semantic_conditions": [{"dimension": "place", "value": "厨房"}]}},
    {"id": "person_intro", "query": "介绍一下明哥", "scope": "album3",
     "parser": {"mode": "evidence",
                "actions": [{"type": "summarize_person", "target": "person"}],
                "answer_target": "person", "entity_names": ["明哥"]}},
    {"id": "writing_prompt", "query": "帮我写一段生日祝福", "scope": "album1", "parser": None},
    {"id": "empty_result", "query": "贵阳夜晚步行街", "scope": None,
     "parser": {"mode": "evidence",
                "actions": [{"type": "return_assets", "coverage": "best"}],
                "semantic_conditions": [{"dimension": "place", "value": "贵阳夜晚步行街"}]}},
)


def _seed(store):
    store.create_asset("asset-may", "kitchen_may.jpg", "image", "/tmp/may.jpg",
                        metadata={"captured_at": "2024-05-12T10:00:00"}, scope_id="album1")
    store.add_observation("asset-may",
                          {"id": "obs-may", "scope_id": "album1", "captured_at": "2024-05-12T10:00:00",
                           "caption": "厨房拿碗", "place": "厨房", "activity": "拿碗", "people": ["明哥"],
                           "clothing": ["红色外套"], "confidence": 0.9}, scope_id="album1")
    store.create_asset("asset-july", "kitchen_july.jpg", "image", "/tmp/july.jpg",
                        metadata={"captured_at": "2024-07-12T10:00:00"}, scope_id="album1")
    store.add_observation("asset-july",
                          {"id": "obs-july", "scope_id": "album1", "captured_at": "2024-07-12T10:00:00",
                           "caption": "厨房做饭", "place": "厨房", "activity": "做饭", "people": ["明哥"],
                           "confidence": 0.9}, scope_id="album1")
    store.create_asset("asset-ming", "ming.jpg", "image", "/tmp/ming.jpg",
                        metadata={"captured_at": "2024-04-20T18:00:00"}, scope_id="album3")
    store.add_observation("asset-ming",
                          {"id": "obs-ming", "scope_id": "album3", "captured_at": "2024-04-20T18:00:00",
                           "caption": "明哥在厨房", "place": "厨房", "activity": "做饭", "people": ["明哥"],
                           "confidence": 0.9}, scope_id="album3")
    store.create_entity("明哥", "person", "confirmed", scope_id="album3")


def _run_once(store, case, thin_flag):
    original = os.environ.get("SENTRIX_THIN_AGENT_V1")
    if thin_flag is None:
        os.environ.pop("SENTRIX_THIN_AGENT_V1", None)
    else:
        os.environ["SENTRIX_THIN_AGENT_V1"] = thin_flag
    try:
        gamma = ScriptedGamma(responses={"查询解析器": case["parser"]} if case.get("parser") else {})
        agent = MemoryAgent(store, gamma=gamma)
        start = time.perf_counter()
        try:
            result = agent.answer_turn(case["query"], scope_id=case["scope"])
            error = None
        except Exception as exc:
            result = {"error": str(exc), "memory_used": False, "evidence": [], "image_results": []}
            error = str(exc)
        latency_ms = int((time.perf_counter() - start) * 1000)
    finally:
        if original is None:
            os.environ.pop("SENTRIX_THIN_AGENT_V1", None)
        else:
            os.environ["SENTRIX_THIN_AGENT_V1"] = original
    return {
        "memory_used": bool(result.get("memory_used")),
        "evidence_count": len(result.get("evidence") or []),
        "image_count": len(result.get("image_results") or []),
        "answer_snippet": (result.get("answer") or "")[:80],
        "error": error,
        "latency_ms": latency_ms,
    }


def evaluate():
    with tempfile.TemporaryDirectory(prefix="shadow-run-") as directory:
        store = MemoryStore(str(Path(directory) / "shadow.db"))
        try:
            _seed(store)
            results = []
            for case in CASES:
                old = _run_once(store, case, thin_flag=None)
                new = _run_once(store, case, thin_flag="1")
                diverges = (old["memory_used"] != new["memory_used"] or
                             old["evidence_count"] != new["evidence_count"] or
                             old["image_count"] != new["image_count"])
                results.append({"id": case["id"], "query": case["query"],
                                 "scope": case["scope"], "old_agent": old,
                                 "thin_agent": new, "diverges": diverges})
        finally:
            store.close()
    return {"generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
             "case_count": len(results),
             "divergent_count": sum(1 for item in results if item["diverges"]),
             "cases": results}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", default=None)
    args = parser.parse_args()
    report = evaluate()
    payload = json.dumps(report, ensure_ascii=False, indent=2)
    if args.report:
        Path(args.report).write_text(payload)
        print(f"wrote {args.report}")
    else:
        print(payload)


if __name__ == "__main__":
    main()
