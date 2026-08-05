#!/usr/bin/env python3
"""Run a small, deterministic end-to-end Agent image retrieval gate.

This benchmark deliberately starts from canonical Asset/Observation rows. It
does not run or alter the memory-generation pipeline, and it keeps the fixture
local to the process.
"""

import argparse
import json
import tempfile
from pathlib import Path

from backend.agent import MemoryAgent
from backend.db import MemoryStore


class DeterministicGamma:
    model = "mvp-deterministic"

    def answer(self, query, context):
        return {"answer": "证据不足", "confidence": 0.0, "evidence": [], "insufficient_evidence": True}

    def embed_text(self, text):
        return []


FIXTURE_CASES = (
    {
        "id": "pajamas",
        "query": "请找出浅黄色拼接毛绒睡衣自拍的照片",
        "ground_truth": ["IMG_4350.JPG"],
    },
    {
        "id": "kitchen",
        "query": "请找出厨房里做晚饭的照片",
        "ground_truth": ["IMG_0760.JPG", "IMG_0761.JPG"],
    },
    {
        "id": "bracelet",
        "query": "请找出银色心形手镯的照片",
        "ground_truth": ["IMG_3726.JPG"],
    },
    {
        "id": "empty_location",
        "query": "请找出贵阳夜晚步行街的照片",
        "ground_truth": [],
    },
)


def _seed_fixture(store):
    rows = (
        ("asset_pajamas", "IMG_4350.JPG", "浅黄色拼接毛绒睡衣自拍", "卧室"),
        ("asset_kitchen_1", "IMG_0760.JPG", "厨房里做晚饭，锅里正在烹饪", "厨房"),
        ("asset_kitchen_2", "IMG_0761.JPG", "厨房里做晚饭，餐桌旁准备食物", "厨房"),
        ("asset_bracelet", "IMG_3726.JPG", "银色心形手镯的近距离照片", "室内"),
        ("asset_distractor", "IMG_DISTRACTOR.JPG", "上海夜晚街道散步", "上海"),
    )
    for asset_id, file_name, caption, place in rows:
        asset = store.create_asset(asset_id, file_name, "image", f"/tmp/{file_name}")
        observation = store.add_observation(
            asset["id"], {"caption": caption, "place": place, "activity": caption},
        )
        store.merge_observation_into_event(observation)


def evaluate_fixture():
    """Return the machine-readable MVP accuracy report."""
    failures = []
    records = []
    false_positives = 0
    with tempfile.TemporaryDirectory(prefix="sentrix-agent-mvp-benchmark-") as directory:
        store = MemoryStore(str(Path(directory) / "fixture.db"))
        try:
            _seed_fixture(store)
            agent = MemoryAgent(store, gamma=DeterministicGamma())
            for case in FIXTURE_CASES:
                result = agent.answer_turn(case["query"], f"mvp-benchmark-{case['id']}")
                returned = [item.get("file_name") for item in result.get("image_results", []) if item.get("file_name")]
                expected = set(case["ground_truth"])
                returned_set = set(returned)
                hits = sorted(expected.intersection(returned_set))
                wrong = sorted(returned_set - expected)
                if not expected:
                    false_positives += len(wrong)
                    passed = not returned_set
                else:
                    passed = bool(hits) and not wrong
                if not passed:
                    failures.append({"case_id": case["id"], "expected": sorted(expected), "returned": returned, "wrong": wrong})
                records.append({"case_id": case["id"], "expected": sorted(expected), "returned": returned, "hits": hits, "wrong": wrong, "passed": passed})
        finally:
            store.close()
    return {
        "passed": not failures,
        "query_count": len(records),
        "failed_cases": failures,
        "empty_ground_truth_false_positive_count": false_positives,
        "records": records,
        "pipeline_rerun": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = evaluate_fixture()
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
