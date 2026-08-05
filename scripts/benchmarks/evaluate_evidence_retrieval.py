#!/usr/bin/env python3
"""Phase 1 Evidence Retrieval Kernel benchmark — 10 cases from plan §5.1.

Records per-case: mode, evidence Asset IDs, hard violations, result level,
model calls and latency.  Compares thin-agent path vs old-agent path.  Meant to
be run before Phase 2R fixes to establish the baseline, and again after each
subsequent phase to detect regressions.
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
    """Deterministic gamma with case-scoped QueryParser drafts.

    Simulates what a real Ollama call would return for the QueryParser prompt.
    Callers set ``parser_response`` before invoking the agent.
    """

    model = "phase1-benchmark-scripted"

    def __init__(self, parser_response=None):
        self.calls = []
        self.parser_response = parser_response or {}

    def chat(self, prompt, *args, **kwargs):
        self.calls.append(prompt)
        if "查询解析器" in prompt and self.parser_response:
            import json as _json
            return _json.dumps(self.parser_response, ensure_ascii=False)
        return '{"answer": "我在听。"}'

    def answer(self, query, context):
        return {"answer": "证据不足", "confidence": 0.0, "evidence": [], "insufficient_evidence": True}

    def embed_text(self, text):
        return []


# Backward-compat alias used by the unit-test wrapper.
DeterministicGamma = ScriptedGamma


CASES = (
    {
        "id": "B-01",
        "query": "2024 年 5 月厨房的照片",
        "scope": "album1",
        "expected_files": ["kitchen_may_bowl.jpg", "kitchen_may_cooking.jpg"],
        "forbidden_files": ["kitchen_july_dinner.jpg"],
        "expected_mode": "evidence",
        "note": "月份硬条件；7 月绝不进入结果",
        "parser": {"mode": "evidence",
                   "actions": [{"type": "return_assets", "coverage": "best"}],
                   "answer_target": "general",
                   "time_expression": "2024 年 5 月",
                   "media_expressions": ["照片"],
                   "semantic_conditions": [{"dimension": "place", "value": "厨房"}]},
    },
    {
        "id": "B-02",
        "query": "厨房里做晚饭的照片",
        "scope": "album1",
        "expected_files": ["kitchen_may_cooking.jpg", "kitchen_july_dinner.jpg"],
        "approximate_files": ["kitchen_may_bowl.jpg"],
        "expected_mode": "evidence",
        "note": "拿碗只能 approximate/unknown，不得声称确定做晚饭",
        "parser": {"mode": "evidence",
                   "actions": [{"type": "return_assets", "coverage": "best"}],
                   "media_expressions": ["照片"],
                   "semantic_conditions": [{"dimension": "place", "value": "厨房"},
                                              {"dimension": "activity", "value": "做晚饭"}]},
    },
    {
        "id": "B-03",
        "query": "浅黄色拼接毛绒睡衣自拍",
        "scope": "album2",
        "expected_files": ["pajamas_selfie.jpg"],
        "approximate_files": ["yellow_plush_shirt.jpg"],
        "expected_mode": "evidence",
        "note": "黄色毛绒可近似，不能确定睡衣/自拍",
        "parser": {"mode": "evidence",
                   "actions": [{"type": "return_assets", "coverage": "best"}],
                   "semantic_conditions": [{"dimension": "clothing", "value": "浅黄色拼接毛绒睡衣"},
                                              {"dimension": "activity", "value": "自拍"}]},
    },
    {
        "id": "B-04",
        "query": "贵阳夜晚步行街",
        "scope": None,
        "expected_files": [],
        "forbidden_files": ["shanghai_night.jpg"],
        "expected_mode": "evidence",
        "note": "无可靠证据时返回空，不返回相似高向量图",
        "tolerate_empty": True,
        "parser": {"mode": "evidence",
                   "actions": [{"type": "return_assets", "coverage": "best"}],
                   "semantic_conditions": [{"dimension": "place", "value": "贵阳夜晚步行街"}]},
    },
    {
        "id": "B-05",
        "query": "明哥的照片",
        "scope": "album3",
        "expected_files": ["ming_kitchen.jpg", "ming_outdoor.jpg"],
        "expected_mode": "evidence",
        "note": "只返回当前 scope 已确认明哥相关",
        "parser": {"mode": "evidence",
                   "actions": [{"type": "return_assets", "coverage": "best"}],
                   "entity_names": ["明哥"], "media_expressions": ["照片"]},
    },
    {
        "id": "B-06",
        "query": "不要妈妈和视频",
        "scope": None,
        "forbidden_files": ["mama_portrait.jpg", "family_video.mp4"],
        "expected_mode": "evidence",
        "note": "妈妈和 video 都不得进入",
        "tolerate_empty": True,
        "parser": {"mode": "evidence",
                   "actions": [{"type": "return_assets", "coverage": "all_relevant"}],
                   "negative_conditions": [{"dimension": "person", "value": "妈妈"},
                                             {"dimension": "media", "value": "video"}]},
    },
    {
        "id": "B-07",
        "query": "把厨房的所有相关照片都找出来",
        "scope": "album1",
        "expected_files": ["kitchen_may_bowl.jpg", "kitchen_may_cooking.jpg", "kitchen_july_dinner.jpg"],
        "expected_mode": "evidence",
        "note": "all_relevant 覆盖率，不只取第一张",
        "parser": {"mode": "evidence",
                   "actions": [{"type": "return_assets", "coverage": "all_relevant"}],
                   "media_expressions": ["照片"],
                   "result_requirement": {"mode": "all_relevant", "top_k": 50},
                   "semantic_conditions": [{"dimension": "place", "value": "厨房"}]},
    },
    {
        "id": "B-08",
        "query": "请直接给我明哥的相关原图",
        "scope": "album3",
        "expected_files": ["ming_kitchen.jpg", "ming_outdoor.jpg"],
        "expected_mode": "evidence",
        "requires_image_results": True,
        "forbid_visual_reinspection": True,
        "note": "授权 Asset，不视觉重读",
        "parser": {"mode": "evidence",
                   "actions": [{"type": "return_assets", "coverage": "best"}],
                   "entity_names": ["明哥"],
                   "media_expressions": ["原图"],
                   "result_requirement": {"return_original_assets": True}},
    },
    {
        "id": "B-09",
        "query": "帮我写生日祝福",
        "scope": None,
        "expected_files": [],
        "expected_mode": "none",
        "requires_zero_memory_reads": True,
        "note": "普通聊天：不检索、不访问原图",
        "tolerate_empty": True,
        # No parser response required — fast-path handles writing prompt.
        "parser": None,
    },
    {
        "id": "B-10",
        "query": "今天很累，突然有点想小黑",
        "scope": None,
        "expected_files": [],
        "expected_mode": "contextual",
        "requires_zero_observation_reads": True,
        "note": "只允许 Core Memory，不读 Observation/Asset",
        "tolerate_empty": True,
        "parser": {"mode": "contextual",
                   "actions": [{"type": "answer_question", "target": "general"}],
                   "facets": [{"dimension": "person", "surface_text": "小黑"}]},
    },
)


FIXTURE_ROWS = (
    # album1: kitchen scenes across months
    {
        "asset_id": "asset_kitchen_may_bowl", "file_name": "kitchen_may_bowl.jpg", "media_type": "image",
        "captured_at": "2024-05-12T10:00:00", "scope_id": "album1",
        "caption": "厨房里在拿碗", "place": "厨房", "activity": "拿碗", "people": ["妈妈"],
        "objects": ["碗"], "clothing": ["红色外套"],
    },
    {
        "asset_id": "asset_kitchen_may_cooking", "file_name": "kitchen_may_cooking.jpg", "media_type": "image",
        "captured_at": "2024-05-15T18:30:00", "scope_id": "album1",
        "caption": "厨房里做晚饭，锅里正在烹饪", "place": "厨房", "activity": "做晚饭", "people": ["爸爸"],
        "objects": ["锅"], "clothing": ["白色围裙"],
    },
    {
        "asset_id": "asset_kitchen_july_dinner", "file_name": "kitchen_july_dinner.jpg", "media_type": "image",
        "captured_at": "2024-07-10T19:00:00", "scope_id": "album1",
        "caption": "厨房里做晚饭，餐桌旁准备食物", "place": "厨房", "activity": "做晚饭", "people": [],
        "objects": ["炒锅"], "clothing": [],
    },
    # album2: pajamas + approximate + bracelet + mama portrait
    {
        "asset_id": "asset_pajamas_selfie", "file_name": "pajamas_selfie.jpg", "media_type": "image",
        "captured_at": "2024-06-01T22:00:00", "scope_id": "album2",
        "caption": "浅黄色拼接毛绒睡衣自拍", "place": "卧室", "activity": "自拍", "people": [],
        "objects": [], "clothing": ["浅黄色拼接毛绒睡衣"],
    },
    {
        "asset_id": "asset_yellow_plush_shirt", "file_name": "yellow_plush_shirt.jpg", "media_type": "image",
        "captured_at": "2024-06-02T14:00:00", "scope_id": "album2",
        "caption": "黄色毛绒毛衣近照", "place": "客厅", "activity": "拍照", "people": [],
        "objects": [], "clothing": ["黄色毛绒毛衣"],
    },
    {
        "asset_id": "asset_bracelet", "file_name": "bracelet.jpg", "media_type": "image",
        "captured_at": "2024-06-03T15:00:00", "scope_id": "album2",
        "caption": "银色心形手镯的近距离照片", "place": "室内", "activity": "拍摄", "people": [],
        "objects": ["手镯"], "clothing": [],
    },
    {
        "asset_id": "asset_mama_portrait", "file_name": "mama_portrait.jpg", "media_type": "image",
        "captured_at": "2024-06-05T09:00:00", "scope_id": "album2",
        "caption": "妈妈的近照", "place": "客厅", "activity": "拍照", "people": ["妈妈"],
        "objects": [], "clothing": ["深蓝色外套"],
    },
    # album3: confirmed 明哥
    {
        "asset_id": "asset_ming_kitchen", "file_name": "ming_kitchen.jpg", "media_type": "image",
        "captured_at": "2024-04-20T18:00:00", "scope_id": "album3",
        "caption": "明哥在厨房", "place": "厨房", "activity": "做饭", "people": ["明哥"],
        "objects": [], "clothing": ["灰色 T 恤"],
    },
    {
        "asset_id": "asset_ming_outdoor", "file_name": "ming_outdoor.jpg", "media_type": "image",
        "captured_at": "2024-04-25T15:00:00", "scope_id": "album3",
        "caption": "明哥户外", "place": "户外", "activity": "散步", "people": ["明哥"],
        "objects": [], "clothing": ["蓝色外套"],
    },
    # video (for B-06 negative)
    {
        "asset_id": "asset_family_video", "file_name": "family_video.mp4", "media_type": "video",
        "captured_at": "2024-05-20T20:00:00", "scope_id": "album1",
        "caption": "家庭聚会视频", "place": "客厅", "activity": "聊天", "people": ["妈妈"],
        "objects": [], "clothing": [],
    },
    # distractor for B-04
    {
        "asset_id": "asset_shanghai_night", "file_name": "shanghai_night.jpg", "media_type": "image",
        "captured_at": "2024-05-25T20:00:00", "scope_id": "album1",
        "caption": "上海夜晚街道散步", "place": "上海街道", "activity": "散步", "people": [],
        "objects": [], "clothing": [],
    },
)


def _seed_fixture(store):
    for row in FIXTURE_ROWS:
        store.create_asset(
            row["asset_id"], row["file_name"], row["media_type"], f"/tmp/{row['file_name']}",
            metadata={"captured_at": row["captured_at"], "scope_id": row["scope_id"]},
            scope_id=row["scope_id"],
        )
        observation = store.add_observation(
            row["asset_id"],
            {
                "id": f"obs_{row['asset_id']}", "scope_id": row["scope_id"],
                "captured_at": row["captured_at"], "caption": row["caption"], "place": row["place"],
                "activity": row["activity"], "people": row["people"], "objects": row["objects"],
                "clothing": row["clothing"], "confidence": 0.85, "source_type": row["media_type"],
            },
            scope_id=row["scope_id"],
        )
        store.merge_observation_into_event(observation)
    store.create_entity("明哥", "person", "confirmed", scope_id="album3")
    store.create_entity("妈妈", "person", "confirmed", scope_id="album2")


def _extract_file_names(result):
    names = set()
    for item in result.get("image_results", []) or []:
        if item.get("file_name"):
            names.add(item["file_name"])
    for item in result.get("evidence", []) or []:
        if item.get("file_name"):
            names.add(item["file_name"])
    return names


def _extract_mode(result):
    if not result.get("memory_used"):
        return "none"
    trace = result.get("retrieval_trace") or []
    for stage in trace:
        status = stage.get("status")
        if status in {"none", "contextual", "evidence"}:
            return status
    return "evidence" if result.get("evidence_required") else "none"


def _extract_hard_violations(case, files):
    return sorted(set(case.get("forbidden_files") or []).intersection(files))


def _extract_missing(case, files):
    return sorted(set(case.get("expected_files") or []).difference(files))


def _classify_case(case, result, gamma_calls_before, gamma_calls_after):
    files = _extract_file_names(result)
    mode = _extract_mode(result)
    violations = _extract_hard_violations(case, files)
    missing = _extract_missing(case, files)
    model_calls = gamma_calls_after - gamma_calls_before
    checks = {
        "hard_violations": violations,
        "missing_expected": missing,
        "mode_match": mode == case["expected_mode"],
        "detected_mode": mode,
        "expected_mode": case["expected_mode"],
        "model_calls": model_calls,
    }
    if case.get("requires_zero_memory_reads"):
        checks["memory_reads_zero"] = not bool(result.get("evidence")) and not result.get("image_results")
    if case.get("requires_zero_observation_reads"):
        checks["observation_reads_zero"] = not any(item.get("kind") == "observation" for item in result.get("evidence", []) or [])
    if case.get("requires_image_results"):
        checks["image_results_present"] = bool(result.get("image_results"))
    if case.get("forbid_visual_reinspection"):
        tools = result.get("tool_trace") or []
        checks["no_visual_reinspection"] = not any(item.get("tool") == "inspect_original_images" for item in tools)
    tolerate_empty = case.get("tolerate_empty")
    passed = (
        not violations
        and (tolerate_empty or not missing)
        and checks["mode_match"]
        and all(value is True for key, value in checks.items() if key.startswith("no_") or key.endswith("_zero") or key.endswith("_present"))
    )
    return passed, checks, sorted(files)


def _run_case(store, case):
    gamma = ScriptedGamma(parser_response=case.get("parser"))
    agent = MemoryAgent(store, gamma=gamma)
    calls_before = len(gamma.calls)
    start = time.perf_counter()
    try:
        result = agent.answer_turn(case["query"], scope_id=case["scope"])
        error = None
    except Exception as exc:
        result = {"error": str(exc), "memory_used": False, "evidence": [], "image_results": [], "retrieval_trace": []}
        error = str(exc)
    elapsed_ms = int((time.perf_counter() - start) * 1000)
    calls_after = len(gamma.calls)
    passed, checks, files = _classify_case(case, result, calls_before, calls_after)
    return {
        "id": case["id"],
        "query": case["query"],
        "scope": case["scope"],
        "note": case["note"],
        "passed": passed,
        "error": error,
        "checks": checks,
        "returned_files": files,
        "latency_ms": elapsed_ms,
    }


def _run_configuration(config_name, thin_flag_value):
    original = os.environ.get("SENTRIX_THIN_AGENT_V1")
    if thin_flag_value is None:
        os.environ.pop("SENTRIX_THIN_AGENT_V1", None)
    else:
        os.environ["SENTRIX_THIN_AGENT_V1"] = thin_flag_value
    try:
        with tempfile.TemporaryDirectory(prefix=f"phase1-{config_name}-") as directory:
            store = MemoryStore(str(Path(directory) / "phase1.db"))
            try:
                _seed_fixture(store)
                results = [_run_case(store, case) for case in CASES]
            finally:
                store.close()
    finally:
        if original is None:
            os.environ.pop("SENTRIX_THIN_AGENT_V1", None)
        else:
            os.environ["SENTRIX_THIN_AGENT_V1"] = original
    return {
        "configuration": config_name,
        "total_cases": len(results),
        "passed": sum(1 for item in results if item["passed"]),
        "cases": results,
    }


def evaluate():
    return {
        "phase": "1",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "configurations": [
            _run_configuration("thin_agent_v1_on", "1"),
            _run_configuration("thin_agent_v1_off", None),
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", default=None, help="write JSON report here")
    parser.add_argument("--strict", action="store_true", help="exit 1 if any case fails")
    args = parser.parse_args()
    report = evaluate()
    payload = json.dumps(report, ensure_ascii=False, indent=2)
    if args.report:
        Path(args.report).write_text(payload)
        print(f"wrote {args.report}")
    else:
        print(payload)
    if args.strict:
        for configuration in report["configurations"]:
            if configuration["passed"] < configuration["total_cases"]:
                sys.exit(1)


if __name__ == "__main__":
    main()
