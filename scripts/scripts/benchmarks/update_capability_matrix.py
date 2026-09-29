#!/usr/bin/env python3
"""Phase H H8 — 用 L2 模型判分结果回写 Capability Matrix。

规则（capability.py.judge_status）：n<5 一律 experimental；n>=5 且支持率>=0.7
且误自信率<=0.3 才 ready；其余 limited。样本不足不凑数、不特判。
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from backend.agent_runtime.capability import judge_status  # noqa: E402

CAP_LABEL = {
    "people_count": "人数",
    "clothing_color": "衣着颜色",
    "clothing_type": "衣着类型",
    "large_object": "大物体",
    "small_object": "小物体",
    "visible_text_short": "短文字",
    "ocr_number": "数字/OCR",
    "scene": "场景",
    "activity": "活动",
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--l2-eval", required=True, help="inspect_cap_l2_eval.json")
    ap.add_argument("--matrix", default=str(ROOT / "configs" / "tool_capability_matrix.json"))
    args = ap.parse_args()

    l2 = json.loads(Path(args.l2_eval).read_text(encoding="utf-8"))
    caps_official = l2.get("capabilities") or {}
    rows = l2.get("rows") or []
    by_cap: dict[str, list] = defaultdict(list)
    for r in rows:
        by_cap[r.get("capability") or "?"].append(r)

    matrix = json.loads(Path(args.matrix).read_text(encoding="utf-8"))
    inspect = matrix.setdefault("inspect_photo", {})

    print(f"{'capability':<20} {'n':>3} {'support':>8} {'fconf':>6}  status")
    for cap in sorted(caps_official):
        rs = by_cap.get(cap) or []
        n = (caps_official[cap] or {}).get("tested", len(rs))
        support_rate = (caps_official[cap] or {}).get("support_rate")
        if support_rate is None and n:
            support_rate = (caps_official[cap] or {}).get("supported", 0) / n
        contradicted = (caps_official[cap] or {}).get("contradicted", 0)
        unknown = (caps_official[cap] or {}).get("unknown", 0)
        fconf = sum(1 for r in rs if r.get("state") == "false_confident")
        fconf_rate = round(fconf / n, 3) if n else None
        status = judge_status(n=n, support_rate=support_rate, false_confident_rate=fconf_rate)
        print(f"{cap:<20} {n:>3} {str(round(support_rate,3) if support_rate is not None else None):>8} {str(fconf_rate):>6}  {status}")
        inspect[cap] = {
            "status": status,
            "accuracy": round(support_rate, 3) if support_rate is not None else None,
            "support_rate": round(support_rate, 3) if support_rate is not None else None,
            "false_confident_rate": fconf_rate,
            "tested": n,
            "passed": (caps_official[cap] or {}).get("supported", 0),
            "failed": contradicted + unknown,
            "cases": sorted({r.get("file") or r.get("asset_id") or "" for r in rs}),
            "evidence": "inspect_capability_benchmark + L2 eval (gemma4-12b-it)",
        }

    # read_photo_text：A/B 实测前保持受限，n<5 的类别不允许 ready
    rpt = matrix.setdefault("read_photo_text", {})
    for cap in ("ocr_number", "visible_text_short"):
        entry = rpt.get(cap) or {}
        tested = entry.get("tested")
        acc = entry.get("accuracy")
        if isinstance(tested, int) and tested >= 5 and isinstance(acc, (int, float)) and acc >= 0.7:
            entry["status"] = "ready"
        elif isinstance(tested, int) and tested:
            entry["status"] = "limited"
        else:
            entry["status"] = "limited"
        rpt[cap] = entry

    Path(args.matrix).write_text(json.dumps(matrix, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"\nmatrix 已回写: {args.matrix}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
