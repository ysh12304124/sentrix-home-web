#!/usr/bin/env python3
"""Phase H H5 — OCR Small vs VLM A/B Benchmark（同 GT 真实照片）。

对比 RapidOCR(small) 与 gemma4-12b VLM 在菜单/电话/招牌类照片上的：
Numeric/Price/Phone Exact Match、延迟、fallback 判定。
"""
from __future__ import annotations

import base64
import json
import re
import sys
import time
import urllib.request
from pathlib import Path

_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))
BASE = "http://127.0.0.1:8100/v1"
ROOT = Path("/home/asus/Github/Sentrix-Home-Web")
PHOTO_DIR = ROOT / "data/album3-v2-source/photos"


def small_ocr(path: Path):
    from rapidocr_onnxruntime import RapidOCR
    engine = RapidOCR()
    t0 = time.monotonic()
    result, _ = engine(str(path))
    lat = time.monotonic() - t0
    items = []
    for box, text, conf in (result or []):
        if text and text.strip():
            items.append((str(text).strip(), float(conf or 0)))
    return {"texts": [t for t, _ in items], "conf": [c for _, c in items],
            "latency_s": round(lat, 3), "n": len(items)}


def vlm_ocr(path: Path, question: str) -> dict:
    def chat(prompt, images):
        content = [{"type": "text", "text": prompt}]
        for b64 in images:
            content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})
        payload = {"model": "gemma4-12b-it", "messages": [{"role": "user", "content": content}],
                   "temperature": 0.0, "max_tokens": 800}
        req = urllib.request.Request(BASE + "/chat/completions", data=json.dumps(payload).encode(),
                                     headers={"Content-Type": "application/json"})
        with _OPENER.open(req, timeout=180) as resp:
            data = json.loads(resp.read().decode())
        return (data.get("choices") or [{}])[0].get("message", {}).get("content", "")

    from io import BytesIO
    from PIL import Image
    full_b64 = base64.b64encode(path.read_bytes()).decode()
    img = Image.open(path).convert("RGB")
    w, h = img.size
    tw, th = w // 2, h // 2
    canvas = Image.new("RGB", (w, h), "black")
    for r in range(2):
        for c in range(2):
            box = (c * tw, r * th, min(w, (c + 1) * tw), min(h, (r + 1) * th))
            tile = img.crop(box).resize((tw, th), Image.LANCZOS)
            canvas.paste(tile, (c * tw, r * th))
    buf = BytesIO()
    canvas.save(buf, "JPEG", quality=92)
    tiles_b64 = base64.b64encode(buf.getvalue()).decode()
    t0 = time.monotonic()
    raw = chat("请逐字读出这张照片里的所有文字，尤其是数字、价格、电话、年份。只输出读到的文字。",
               [full_b64, tiles_b64])
    lat = time.monotonic() - t0
    return {"text": raw.strip()[:1500], "latency_s": round(lat, 3)}


CASES = [
    {"id": "26-q01", "photo": "IMG_20220716_141621.jpg",
     "q": "顶呱呱品牌创始于哪一年？", "gold": ["1974"]},
    {"id": "26-q03", "photo": "IMG_20220716_141625.jpg",
     "q": "汉堡单人套餐价格是多少？", "gold": ["34", "38"]},
    {"id": "26-q06", "photo": "IMG_20220716_141625.jpg",
     "q": "台式奶茶售价多少？", "gold": ["10"]},
    {"id": "26-q07", "photo": "IMG_20220716_141625.jpg",
     "q": "可乐加多少钱换购玉米浓汤或美式咖啡？", "gold": ["8"]},
    {"id": "24-q07", "photo": "IMG_20220623_212642.jpg",
     "q": "墙上报警电话是多少？", "gold": ["22048084", "22048085"]},
    {"id": "24-q02", "photo": "IMG_20220623_212642.jpg",
     "q": "店铺招牌店名是什么？", "gold": ["大圣葱油拌面"]},
]


def exact_match(text: str, golds: list[str]) -> bool:
    return any(g in text.replace(" ", "").replace("\n", "") for g in golds)


def main():
    out = []
    for case in CASES:
        path = PHOTO_DIR / case["photo"]
        small = small_ocr(path)
        vlm = vlm_ocr(path, case["q"])
        sm_text = "".join(small["texts"])
        hit_small = exact_match(sm_text, case["gold"])
        hit_vlm = exact_match(vlm["text"], case["gold"])
        out.append({**case, "small": small, "vlm": vlm,
                    "small_exact": hit_small, "vlm_exact": hit_vlm})
        print(f"{case['id']:<8} small={'HIT' if hit_small else 'miss'} "
              f"({small['latency_s']}s, n={small['n']}) | "
              f"vlm={'HIT' if hit_vlm else 'miss'} ({vlm['latency_s']}s)")
        print(f"    gold={case['gold']} small_text={'/'.join(small['texts'])[:90]!r}")
        print(f"    vlm_text={vlm['text'][:90]!r}")
    Path("/tmp/phaseh_ocr_ab.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    n = len(out)
    print(f"\nSUMMARY small exact={sum(1 for o in out if o['small_exact'])}/{n} | "
          f"vlm exact={sum(1 for o in out if o['vlm_exact'])}/{n}")


if __name__ == "__main__":
    main()
