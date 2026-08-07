#!/usr/bin/env python3
"""RX human experience scoring — aggregate user scores over the replay pairs.

The user opens docs/baseline/rx-replay-pairs.json, fills each pair's "scores"
with 1-5 ratings for naturalness_new / naturalness_old / consistency / helpful,
then runs this script.  It reports the paired naturalness win rate and the
RX-7 acceptance gates (new >= old on naturalness >= 80%, contradiction=0, leak=0).

Usage:
  PYTHONPATH=. .venv-mac/bin/python scripts/benchmarks/score_human_experience.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
_IN = REPO_ROOT / "docs" / "baseline" / "rx-replay-pairs.json"


def main():
    data = json.loads(_IN.read_text(encoding="utf-8"))
    pairs = [p for p in data["pairs"] if p.get("scores")]
    if not pairs:
        print("no scored pairs yet — fill each pair's 'scores' object in "
              f"{_IN} first", file=sys.stderr)
        return 1
    wins = losses = ties = 0
    leak_hits = 0
    contradiction_hits = 0
    helpful_sum = 0
    for pair in pairs:
        s = pair["scores"]
        new_n = float(s.get("naturalness_new", 0) or 0)
        old_n = float(s.get("naturalness_old", 0) or 0)
        if new_n > old_n:
            wins += 1
        elif new_n < old_n:
            losses += 1
        else:
            ties += 1
        if pair.get("new_leak"):
            leak_hits += 1
        if pair.get("new_answer") and "无法提供" in str(pair["new_answer"]) and pair.get("new_image_count"):
            contradiction_hits += 1
        helpful_sum += float(s.get("helpful", 0) or 0)
    scored = len(pairs)
    # "新>=旧" includes ties: a tied pair is not worse, so it counts toward the
    # naturalness gate (plan §15.2: 盲测自然度新>=旧 >= 80%).
    win_rate = (wins + ties) / scored if scored else 0.0
    print(f"scored pairs: {scored}/{data['count']}")
    print(f"naturalness 新>旧: {wins}  新<旧: {losses}  持平: {ties}")
    print(f"naturalness win rate (新>=旧): {round(win_rate, 3) if scored else 0}")
    print(f"internal leak (new): {leak_hits}")
    print(f"text/image contradiction (new): {contradiction_hits}")
    print(f"avg helpful: {round(helpful_sum / scored, 2) if scored else 0}")
    ok = win_rate >= 0.8 and leak_hits == 0 and contradiction_hits == 0
    print("RX-7 human acceptance:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
