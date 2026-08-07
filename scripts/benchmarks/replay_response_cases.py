#!/usr/bin/env python3
"""RX offline replay — old (12B-FC) vs new (RX) answers, paired for scoring.

Reads the recorded 12B-FC cases and, where available, the RX E2E results, and
produces a per-case pair {message, old_answer, new_answer, images, leak}.  Cases
without an RX E2E result are replayed deterministically through the new
AnswerBrief -> safe-fallback pipeline so the pair list stays complete.

Output: docs/baseline/rx-replay-pairs.json (also the input for
score_human_experience.py).

Run locally:
  PYTHONPATH=. .venv-mac/bin/python scripts/benchmarks/replay_response_cases.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from backend.answer_brief import build_answer_brief
from backend.evidence_retrieval import EvidencePacket
from backend.query_contracts import QueryAction, QuerySpec
from backend.response_plan import plan_response
from backend.response_validator import finalize_answer, scan_internal_leak
from backend.response_writer import safe_fallback
from backend.visible_evidence import select_visible_assets

_DEFAULT_OLD = REPO_ROOT / "docs" / "baseline" / "sentrix-12b-full-chain-cases.json"
_DEFAULT_NEW = REPO_ROOT / "docs" / "baseline" / "rx-experience-cases.json"
_OUT = REPO_ROOT / "docs" / "baseline" / "rx-replay-pairs.json"


def _packet_from_case(case):
    assets = []
    exact, approx, strong = [], [], []
    for item in case.get("evidence") or []:
        asset = {
            "asset_id": item.get("asset_id"),
            "file_name": item.get("file_name"),
            "media_type": "image",
            "observation_ids": [item.get("id")] if item.get("id") else [],
            "evidence_ids": item.get("evidence_ids") or [],
            "condition_results": item.get("condition_results") or {},
            "level": item.get("level") or "approximate",
            "recall_strength": item.get("recall_strength"),
            "captured_at": item.get("captured_at"),
        }
        assets.append(asset)
        level = item.get("level")
        if level == "exact":
            exact.append(asset)
        elif level == "strong":
            strong.append(asset)
        else:
            approx.append(asset)
    return EvidencePacket("replay", case.get("scope") or "album2_e2b", "general",
                          assets=assets, exact_results=exact, strong_results=strong,
                          approximate_results=approx, gaps=case.get("gaps") or [])


def _spec_for_message(case):
    message = case.get("message", "")
    scope = case.get("scope")
    text = str(message or "")
    roles = case.get("expected_model_roles") or []
    answer_target = "person" if "writer" in roles or "claim" in roles else "general"
    actions = []
    if any(token in text for token in ("给我", "原图", "都给我", "全部", "还有哪些")):
        actions.append(QueryAction(type="return_assets"))
    return QuerySpec("replay", "single", [scope or "home-default"], "owner", "c",
                     "answer", answer_target, constraints=[], actions=actions)


def _replay_new(case):
    """Deterministic new-side proxy for cases without an RX E2E result."""
    packet = _packet_from_case(case)
    spec = _spec_for_message(case)
    visible = select_visible_assets(packet,
                                    all_relevant=spec.result_requirement.get("mode") == "all_relevant")
    brief = build_answer_brief(case.get("message", ""), spec, packet, visible_assets=visible)
    plan = plan_response(brief)
    answer, statements = safe_fallback(brief, plan)
    answer, statements, validation = finalize_answer(
        answer, statements, brief, plan, plan.image_count, lambda: safe_fallback(brief, plan))
    return {
        "response_mode": brief.response_mode,
        "image_count": len(brief.visible_assets),
        "answer": answer,
        "leak": scan_internal_leak(answer),
        "fallback": True,
    }


def main():
    old_path = Path(sys.argv[1]) if len(sys.argv) > 1 else _DEFAULT_OLD
    new_path = Path(sys.argv[2]) if len(sys.argv) > 2 else _DEFAULT_NEW
    old = json.loads(old_path.read_text(encoding="utf-8"))
    new = json.loads(new_path.read_text(encoding="utf-8")) if new_path.exists() else {"cases": []}

    new_by_message = {c.get("message"): c for c in new.get("cases") or []}
    pairs = []
    for case in old.get("cases") or []:
        message = case.get("message", "")
        new_case = new_by_message.get(message)
        new_side = {
            "response_mode": new_case.get("response_mode") if new_case else None,
            "image_count": (new_case.get("image_count") if new_case else None),
            "answer": (new_case.get("answer") if new_case else None),
            "leak": (new_case.get("internal_leak") if new_case else None),
            "fallback": bool(new_case.get("rx_fallback_used")) if new_case else None,
            "proxy": not bool(new_case),
        }
        if not new_case:
            new_side.update(_replay_new(case))
        pairs.append({
            "name": case.get("name"),
            "message": message,
            "scope": case.get("scope"),
            "expected_model_roles": case.get("expected_model_roles"),
            "old_verdict": case.get("verdict"),
            "old_answer": case.get("answer"),
            "old_evidence_count": case.get("evidence_count"),
            "old_image_count": len((case.get("image_results") or [])),
            "new_answer": new_side["answer"],
            "new_response_mode": new_side["response_mode"],
            "new_image_count": new_side["image_count"],
            "new_leak": new_side["leak"],
            "new_fallback": new_side["fallback"],
            "proxy": new_side.get("proxy", False),
            "scores": {},
        })

    _OUT.parent.mkdir(parents=True, exist_ok=True)
    _OUT.write_text(json.dumps({"count": len(pairs), "pairs": pairs}, ensure_ascii=False, indent=2),
                    encoding="utf-8")
    print(f"wrote {_OUT} with {len(pairs)} pairs "
          f"({sum(1 for p in pairs if not p['proxy'])} from RX E2E, "
          f"{sum(1 for p in pairs if p['proxy'])} deterministic proxy)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
