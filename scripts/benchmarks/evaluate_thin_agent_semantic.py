#!/usr/bin/env python3
"""Phase 2R-8 semantic benchmark.

Runs paraphrase pools, contrast pairs and composite-task cases against the
Thin Agent runtime with a scripted gamma standing in for a real Ollama.
Records mode routing, actions/facets preservation and model-call counts so
regressions can be spotted between phases.
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
    model = "semantic-benchmark"

    def __init__(self, parser_response=None):
        self.calls = []
        self.parser_response = parser_response or {}

    def chat(self, prompt, *args, **kwargs):
        self.calls.append(prompt)
        if "查询解析器" in prompt and self.parser_response:
            return json.dumps(self.parser_response, ensure_ascii=False)
        return "我在听。"

    def answer(self, query, context):
        return {"answer": "证据不足", "insufficient_evidence": True}

    def embed_text(self, text):
        return []


PARAPHRASE_CASES = (
    # Writing prompts — mode=none, memory_used=False.
    {"kind": "writing", "prompt": "帮我写一段关于家庭照片的散文", "expected_mode": "none", "parser": None},
    {"kind": "writing", "prompt": "以相册为主题写一篇短文", "expected_mode": "none", "parser": None},
    {"kind": "writing", "prompt": "假设一家人在厨房做饭，写个故事", "expected_mode": "none", "parser": None},
    {"kind": "writing", "prompt": "为什么人们喜欢拍做饭照片", "expected_mode": "none",
     "parser": {"mode": "none", "actions": [{"type": "answer_question", "target": "general"}]}},
    {"kind": "writing", "prompt": "不用查我的照片，帮我写一句文案", "expected_mode": "none",
     "parser": {"mode": "none", "actions": [{"type": "answer_question", "target": "general"}]}},
    # Evidence paraphrases — mode=evidence, memory_used=True.
    {"kind": "evidence", "prompt": "聊聊那次做菜时她穿的什么", "expected_mode": "evidence",
     "parser": {"mode": "evidence",
                "actions": [{"type": "answer_question", "target": "clothing"}],
                "facets": [{"dimension": "activity", "surface_text": "做菜"},
                           {"dimension": "clothing", "surface_text": "穿的什么"}]}},
    {"kind": "evidence", "prompt": "我记得有人在灶台旁端着碗，是哪一次", "expected_mode": "evidence",
     "parser": {"mode": "evidence",
                "actions": [{"type": "summarize_event", "target": "event"}],
                "facets": [{"dimension": "place", "surface_text": "灶台"},
                           {"dimension": "object", "surface_text": "碗"}]}},
    {"kind": "evidence", "prompt": "找找之前下厨时拍的图", "expected_mode": "evidence",
     "parser": {"mode": "evidence",
                "actions": [{"type": "return_assets", "coverage": "best"}],
                "facets": [{"dimension": "activity", "surface_text": "下厨"}]}},
    {"kind": "evidence", "prompt": "哪些照片看起来是在准备晚餐", "expected_mode": "evidence",
     "parser": {"mode": "evidence",
                "actions": [{"type": "return_assets", "coverage": "all_relevant"}],
                "facets": [{"dimension": "activity", "surface_text": "准备晚餐"}]}},
    {"kind": "evidence", "prompt": "不要只是拿碗的，我想找真正烹饪的", "expected_mode": "evidence",
     "parser": {"mode": "evidence",
                "actions": [{"type": "return_assets", "coverage": "best"}],
                "facets": [{"dimension": "activity", "surface_text": "烹饪"}],
                "negative_conditions": [{"dimension": "activity", "value": "拿碗"}]}},
    # Contextual paraphrases — mode=contextual, memory_used=True but Observation reads=0.
    {"kind": "contextual", "prompt": "今晚回家时想起小黑了", "expected_mode": "contextual",
     "parser": {"mode": "contextual",
                "actions": [{"type": "answer_question", "target": "general"}],
                "facets": [{"dimension": "person", "surface_text": "小黑"}]}},
    {"kind": "contextual", "prompt": "突然有点怀念小黑", "expected_mode": "contextual",
     "parser": {"mode": "contextual",
                "actions": [{"type": "answer_question", "target": "general"}]}},
    {"kind": "contextual", "prompt": "今天看到一只猫让我想到小黑", "expected_mode": "contextual",
     "parser": {"mode": "contextual",
                "actions": [{"type": "answer_question", "target": "general"}]}},
    {"kind": "contextual", "prompt": "小黑啊，最近总会想到它", "expected_mode": "contextual",
     "parser": {"mode": "contextual",
                "actions": [{"type": "answer_question", "target": "general"}]}},
    {"kind": "contextual", "prompt": "刚才路过宠物店，想起家里的小黑", "expected_mode": "contextual",
     "parser": {"mode": "contextual",
                "actions": [{"type": "answer_question", "target": "general"}]}},
)


CONTRAST_PAIRS = (
    {"chat": ("帮我写关于厨房做饭的故事",
              None),
     "evidence": ("找厨房里真正做饭的照片",
                  {"mode": "evidence", "actions": [{"type": "return_assets", "coverage": "best"}]})},
    {"chat": ("为什么人喜欢拍照片",
              {"mode": "none", "actions": [{"type": "answer_question", "target": "general"}]}),
     "evidence": ("把去年拍的照片给我",
                  {"mode": "evidence", "actions": [{"type": "return_assets", "coverage": "best"}]})},
    {"chat": ("我想写一篇明哥的虚构故事", None),
     "evidence": ("我想问问明哥去年做了什么",
                  {"mode": "evidence", "actions": [{"type": "summarize_person", "target": "person"}],
                   "entity_names": ["明哥"]})},
    {"chat": ('介绍一下"家庭相册"这个产品概念',
              {"mode": "none", "actions": [{"type": "answer_question", "target": "general"}]}),
     "evidence": ("介绍一下明哥",
                  {"mode": "evidence", "actions": [{"type": "summarize_person", "target": "person"}],
                   "entity_names": ["明哥"]})},
)


COMPOSITE_CASES = (
    {"prompt": "说说去年春节妈妈穿了什么，再把最相关的照片给我",
     "expected_actions": {"answer_question", "return_assets"},
     "expected_facets": {"person", "time", "activity", "clothing"},
     "parser": {"mode": "evidence",
                "actions": [{"type": "answer_question", "target": "clothing"},
                             {"type": "return_assets", "coverage": "best"}],
                "facets": [{"dimension": "person", "surface_text": "妈妈"},
                            {"dimension": "time", "surface_text": "去年春节"},
                            {"dimension": "activity", "surface_text": "家庭聚餐"},
                            {"dimension": "clothing", "surface_text": "穿了什么"}]}},
)


def _seed_fixture(store):
    store.create_entity("明哥", "person", "confirmed", scope_id="home")
    store.create_entity("小黑", "person", "confirmed", scope_id="home")


def _run_single(store, prompt, parser_response):
    gamma = ScriptedGamma(parser_response=parser_response)
    agent = MemoryAgent(store, gamma=gamma)
    os.environ["SENTRIX_THIN_AGENT_V1"] = "1"
    start = time.perf_counter()
    try:
        result = agent.answer_turn(prompt, scope_id="home", viewer_id="owner")
    finally:
        pass
    latency_ms = int((time.perf_counter() - start) * 1000)
    return result, gamma, latency_ms


def _detected_mode(result):
    if not result.get("memory_used"):
        return "none"
    trace = result.get("retrieval_trace") or []
    for stage in trace:
        status = stage.get("status")
        if status in {"none", "contextual", "evidence"}:
            return status
    return "evidence" if result.get("evidence_required") else "none"


def _run_paraphrase(store):
    records = []
    for case in PARAPHRASE_CASES:
        result, gamma, latency_ms = _run_single(store, case["prompt"], case.get("parser"))
        detected = _detected_mode(result)
        parser_calls = sum(1 for call in gamma.calls if "查询解析器" in call)
        passed = detected == case["expected_mode"]
        records.append({
            "kind": case["kind"], "prompt": case["prompt"],
            "expected_mode": case["expected_mode"], "detected_mode": detected,
            "parser_calls": parser_calls, "memory_used": bool(result.get("memory_used")),
            "evidence": len(result.get("evidence") or []),
            "passed": passed, "latency_ms": latency_ms,
        })
    return records


def _run_contrast(store):
    records = []
    for pair in CONTRAST_PAIRS:
        for label, (prompt, parser) in (("chat", pair["chat"]), ("evidence", pair["evidence"])):
            result, gamma, latency_ms = _run_single(store, prompt, parser)
            detected = _detected_mode(result)
            expected = "none" if label == "chat" else "evidence"
            passed = detected == expected
            records.append({
                "kind": f"contrast_{label}", "prompt": prompt,
                "expected_mode": expected, "detected_mode": detected,
                "memory_used": bool(result.get("memory_used")),
                "evidence": len(result.get("evidence") or []),
                "passed": passed, "latency_ms": latency_ms,
            })
    return records


def _run_composite(store):
    records = []
    for case in COMPOSITE_CASES:
        result, gamma, latency_ms = _run_single(store, case["prompt"], case.get("parser"))
        actions = {item.get("type") for item in result.get("actions") or []}
        facets = {item.get("dimension") for item in result.get("facets") or []}
        passed = case["expected_actions"].issubset(actions) and case["expected_facets"].issubset(facets)
        records.append({
            "kind": "composite", "prompt": case["prompt"],
            "expected_actions": sorted(case["expected_actions"]),
            "expected_facets": sorted(case["expected_facets"]),
            "detected_actions": sorted(actions), "detected_facets": sorted(facets),
            "passed": passed, "latency_ms": latency_ms,
        })
    return records


def evaluate():
    with tempfile.TemporaryDirectory(prefix="phase2r-semantic-") as directory:
        store = MemoryStore(str(Path(directory) / "semantic.db"))
        try:
            _seed_fixture(store)
            paraphrase = _run_paraphrase(store)
            contrast = _run_contrast(store)
            composite = _run_composite(store)
        finally:
            store.close()
    all_records = paraphrase + contrast + composite
    return {
        "phase": "2R-8",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "totals": {
            "paraphrase_passed": sum(1 for item in paraphrase if item["passed"]),
            "paraphrase_total": len(paraphrase),
            "contrast_passed": sum(1 for item in contrast if item["passed"]),
            "contrast_total": len(contrast),
            "composite_passed": sum(1 for item in composite if item["passed"]),
            "composite_total": len(composite),
            "overall_passed": sum(1 for item in all_records if item["passed"]),
            "overall_total": len(all_records),
        },
        "paraphrase": paraphrase,
        "contrast": contrast,
        "composite": composite,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", default=None)
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()
    report = evaluate()
    payload = json.dumps(report, ensure_ascii=False, indent=2)
    if args.report:
        Path(args.report).write_text(payload)
        print(f"wrote {args.report}")
    else:
        print(payload)
    if args.strict:
        if report["totals"]["overall_passed"] < report["totals"]["overall_total"]:
            sys.exit(1)


if __name__ == "__main__":
    main()
