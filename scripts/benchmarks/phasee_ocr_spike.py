#!/usr/bin/env python3
"""Phase E — OCR Spike：crop/tile 放大 + 12B VLM，判断小字/数字读取是否可救。

目的：对 V 层失败的题，把 GT 图切成 2x2/3x3 tile 放大后做 OCR，
回答一个问题：需要上专用 OCR(PaddleOCR) 还是 crop/tile+VLM 就够。

用法：
  phasee_ocr_spike.py --qa-result <qa_result.json> [--ids q1,q2] [--tiles 2x2|3x3]
"""
from __future__ import annotations

import argparse
import base64
import json
import re
import sys
import urllib.request
from datetime import datetime
from io import BytesIO
from pathlib import Path

from PIL import Image

_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))
DEFAULT_BASE = "http://192.168.0.153:4174"
DEFAULT_VLM = "http://192.168.0.153:8100/v1"

OCR_PROMPT = """请读出这张照片中的全部文字（招牌/菜单/价格/数字/电话/年份）。只输出文字内容本身，不要描述图片。如果看不清就输出空。"""

# V 层失败题 + 关注实体
SPIKE_CASES = {
    "validation-album3-012-q03": {"ent": ["大头儿子", "小头爸爸"], "focus": "沙雕主题文字"},
    "validation-album3-024-q02": {"ent": ["大圣葱油拌面", "大圣"], "focus": "店铺招牌"},
    "validation-album3-024-q07": {"ent": ["22048084", "22048085"], "focus": "报警电话"},
    "validation-album3-026-q03": {"ent": ["34"], "focus": "汉堡套餐价格"},
    "validation-album3-026-q06": {"ent": ["10"], "focus": "台式奶茶价格"},
    "validation-album3-026-q01": {"ent": ["1974"], "focus": "创始年份"},
    "validation-album3-040-q01": {"ent": ["兔子"], "focus": "雕塑类型"},
}


def http_json(method, url, payload=None, timeout=90):
    body = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(url, data=body, method=method,
                                 headers={"Content-Type": "application/json"})
    with _OPENER.open(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode())


def fetch_bytes(url, timeout=90):
    req = urllib.request.Request(url, method="GET")
    with _OPENER.open(req, timeout=timeout) as resp:
        return resp.read()


def vlm_ocr(vlm_base, image_b64, timeout=120):
    payload = {
        "model": "gemma4-12b-it",
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": OCR_PROMPT},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}},
        ]}],
        "max_tokens": 200, "temperature": 0.0,
    }
    try:
        data = http_json("POST", f"{vlm_base}/chat/completions", payload, timeout=timeout)
        return (data.get("choices") or [{}])[0].get("message", {}).get("content", "")
    except Exception as exc:
        return f"__ERROR__ {exc}"


def tile_image(img, rows, cols, scale=2.0):
    """切 tile，每块放大 scale 倍后返回 (label, base64) 列表。"""
    w, h = img.size
    tiles = []
    tw, th = w // cols, h // rows
    for r in range(rows):
        for c in range(cols):
            box = (c * tw, r * th, (c + 1) * tw, (r + 1) * th)
            tile = img.crop(box)
            if scale != 1.0:
                tile = tile.resize((int(tile.width * scale), int(tile.height * scale)),
                                   Image.LANCZOS)
            buf = BytesIO()
            tile.convert("RGB").save(buf, "JPEG", quality=92)
            tiles.append((f"r{r}c{c}", base64.b64encode(buf.getvalue()).decode()))
    return tiles


def contains(text, ents):
    t = (text or "").lower()
    return [e for e in ents if e.lower() in t]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qa-result", required=True)
    ap.add_argument("--base", default=DEFAULT_BASE)
    ap.add_argument("--vlm", default=DEFAULT_VLM)
    ap.add_argument("--out", default="~/Downloads/sentrix_qa_report/phasee_ocr_spike.json")
    ap.add_argument("--cache", default="~/Downloads/sentrix_qa_report/cache/assets")
    ap.add_argument("--tiles", default="3x3", help="3x3 或 2x2")
    ap.add_argument("--ids", default="", help="逗号分隔，缺省跑全部 spike cases")
    args = ap.parse_args()

    rows = json.loads(Path(args.qa_result).expanduser().read_text())["rows"]
    scope_id = "album3-v2"
    cache = Path(args.cache).expanduser()
    cache.mkdir(parents=True, exist_ok=True)
    asset_map = {}
    for item in http_json("GET", f"{args.base}/api/assets?scope_id={scope_id}&limit=2000").get("assets", []):
        asset_map.setdefault(Path(item["file_name"] or "").name.lower(), item["id"])
    rows_by_id = {r["qa_id"]: r for r in rows}
    trows, tcols = (int(x) for x in args.tiles.lower().split("x"))
    ids = [x for x in args.ids.split(",") if x] if args.ids else list(SPIKE_CASES)

    results = []
    for qid in ids:
        case = SPIKE_CASES[qid]
        r = rows_by_id[qid]
        ents = case["ent"]
        gold_ev = r.get("gold_evidence_ids") or []
        tiles_all = []
        for gpath in gold_ev[:2]:
            fn = Path(gpath).name.lower()
            aid = asset_map.get(fn)
            if not aid:
                continue
            dest = cache / f"spike_{fn.replace('.', '_')}.jpg"
            if not dest.is_file():
                dest.write_bytes(fetch_bytes(f"{args.base}/api/assets/{aid}/file"))
            img = Image.open(dest)
            tiles_all.extend(tile_image(img, trows, tcols))
        hits_all, texts = [], []
        for label, b64 in tiles_all:
            out = vlm_ocr(args.vlm, b64)
            hit = contains(out, ents)
            texts.append(f"[{label}] {out.strip()[:60]}")
            if hit:
                hits_all.append((label, hit))
        results.append({
            "qa_id": qid, "focus": case["focus"], "question": r["question"],
            "gold": r.get("gold_answer"), "ents": ents,
            "tiles": f"{trows}x{tcols}", "tile_count": len(tiles_all),
            "hit_tiles": hits_all, "tile_texts": texts,
            "resolved": bool(hits_all),
        })
        print(f"{qid} [{case['focus']}] ents={ents} resolved={bool(hits_all)} hits={hits_all}")
        for t in texts:
            print(f"    {t}")

    out = Path(args.out).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "meta": {"created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                 "note": "OCR spike: crop/tile 放大 + 12B VLM"},
        "results": results,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n输出: {out}")
    n_resolved = sum(1 for x in results if x["resolved"])
    print(f"resolved {n_resolved}/{len(results)}")


if __name__ == "__main__":
    main()
