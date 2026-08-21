"""Phase 4: offline post-hoc evidence judge on an existing run.

Reuses benchmark_orchestrator._judge_evidence (benchmark/offline only; no online
agent calls, no vLLM). Scores evidence for each run item from stored
predicted_images + answer, using the judge LLM from config/judge_providers.json.
"""
import sys, os, json, argparse
from pathlib import Path

ROOT = Path("/home/asus/Github/Sentrix-Home-Web")
sys.path.insert(0, str(ROOT / "services/photobench/backend"))
os.chdir(ROOT / "services/photobench")

import benchmark_orchestrator as bo
from benchmark_orchestrator import BenchmarkRun, resolve_judge_provider, DEFAULT_SENTRIX_URL

def build_assets_by_name(sentrix_url, scope_id=None):
    import requests
    url = f"{sentrix_url.rstrip('/')}/api/assets"
    params = {"limit": 2000}
    if scope_id:
        params["scope_id"] = scope_id
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    assets = (r.json() or {}).get("assets") or []
    by_name = {}
    for a in assets:
        by_name.setdefault(Path(a.get("file_name") or "").name, []).append(a)
    return by_name, assets

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--provider", default="volcengine-doubao-seed-2.0-lite")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    d = json.load(open(args.run))
    items = d["items"]
    _, url, model, key = resolve_judge_provider(args.provider)
    sentrix_url = DEFAULT_SENTRIX_URL

    by_name, _ = build_assets_by_name(sentrix_url)
    print(f"assets loaded: {len(by_name)} names")

    runner = object.__new__(BenchmarkRun)
    runner.judge_url = url
    runner.judge_model = model
    runner.judge_api_key = key

    out = []
    n = 0
    for it in items:
        q = it["question"]
        ans = it["answer"]
        pred = it.get("predicted_images") or []
        tj = (it.get("task_judge") or {}).get("actual_action")
        if tj == "clarify":
            res = {"qa": it["qa_id"], "score": None, "reason": "skipped_clarify"}
        elif not pred:
            res = {"qa": it["qa_id"], "score": None, "reason": "no_predicted_images"}
        else:
            res = runner._judge_evidence(q, ans, pred, by_name, sentrix_url)
            res = {"qa": it["qa_id"], "score": res.get("score"),
                   "applicable": res.get("applicable"),
                   "reason": (res.get("reason") or "")[:120]}
        out.append(res)
        n += 1
        print(json.dumps(res, ensure_ascii=False))
        if args.limit and n >= args.limit:
            break
    json.dump(out, open(args.out, "w"), ensure_ascii=False, indent=1)
    print(f"wrote {len(out)} to {args.out}")

if __name__ == "__main__":
    main()
