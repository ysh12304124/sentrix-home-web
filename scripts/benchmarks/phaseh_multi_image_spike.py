#!/usr/bin/env python3
"""Phase H H9 — Multi-image Resolution v2 Spike（真实 QA 场景）。

对比三种策略，回答"哪张/哪些照片拍到了 X"类问题：
  A. Montage/contact sheet：N 张拼一张带编号图，问 VLM
  B. Native multi-image：单请求多图（先探测 8100 是否支持）
  C. Per-image 观察 -> 12B 文本对比

结论只基于真实照片与真实问题，不构造刷分样本。
"""
from __future__ import annotations

import base64
import json
import sys
import time
import urllib.request
from io import BytesIO
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))
BASE = "http://127.0.0.1:8100/v1"
MODEL = "gemma4-12b-it"
ROOT = Path("/home/asus/Github/Sentrix-Home-Web")
PHOTO_DIR = ROOT / "data/album3-v2-source/photos"


def chat(prompt: str, images: list[str] | None = None, max_tokens: int = 600) -> str:
    content: list[dict] = [{"type": "text", "text": prompt}]
    for b64 in images or []:
        content.append({"type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})
    payload = {"model": MODEL, "messages": [{"role": "user", "content": content}],
               "temperature": 0.0, "max_tokens": max_tokens}
    req = urllib.request.Request(BASE + "/chat/completions",
                                 data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with _OPENER.open(req, timeout=180) as resp:
        data = json.loads(resp.read().decode())
    return (data.get("choices") or [{}])[0].get("message", {}).get("content", "")


def b64_of(name: str) -> str:
    p = PHOTO_DIR / name
    return base64.b64encode(p.read_bytes()).decode()


def montage(photos: list[str], cols: int = 2, label_font_size: int = 40) -> str:
    imgs = [Image.open(PHOTO_DIR / p).convert("RGB") for p in photos]
    cell_w = 512
    rows = (len(imgs) + cols - 1) // cols
    canvas = Image.new("RGB", (cols * cell_w, rows * cell_w), "white")
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
                                  label_font_size)
    except Exception:
        font = ImageFont.load_default()
    for i, im in enumerate(imgs):
        im.thumbnail((cell_w, cell_w))
        r, c = divmod(i, cols)
        x, y = c * cell_w, r * cell_w
        canvas.paste(im, (x, y))
        draw.rectangle([x, y, x + 90, y + label_font_size + 10], fill="red")
        draw.text((x + 8, y + 4), str(i + 1), fill="white", font=font)
    buf = BytesIO()
    canvas.save(buf, "JPEG", quality=88)
    return base64.b64encode(buf.getvalue()).decode()


CASES = [
    {
        "id": "47-q07",
        "question": "哪一张照片拍到了开场时持火把的表演？",
        "photos": ["2018-04-01 210440.jpg", "2018-04-01 210536.jpg",
                   "2018-04-01 210547.jpg", "2018-04-01 210550.jpg"],
        "gold": [1],  # 第 1 张（210440）
    },
    {
        "id": "47-q03",
        "question": "开场环节表演者使用了什么道具？",
        "photos": ["2018-04-01 210440.jpg", "2018-04-01 210536.jpg",
                   "2018-04-01 210547.jpg", "2018-04-01 210550.jpg"],
        "gold": "火把/大型扇形道具",
    },
    {
        "id": "40-q01",
        "question": "主要和什么样的雕塑互动合影？",
        "photos": ["IMG_20230806_203916.jpg", "IMG_20230806_204143.jpg",
                   "IMG_20230806_204342.jpg", "IMG_20230806_205755.jpg",
                   "IMG_20230806_205817.jpg"],
        "gold": "卡通兔子雕塑",
    },
]


def run() -> dict:
    out = {"base": BASE, "model": MODEL, "cases": []}
    # 探测 native multi-image 支持
    try:
        probe = chat("这两张图分别是什么？", [b64_of("2018-04-01 210440.jpg"),
                                           b64_of("2018-04-01 210536.jpg")])
        native_supported = bool(probe.strip())
        out["native_multi_image_supported"] = native_supported
        out["native_probe_answer"] = probe[:80]
    except Exception as exc:
        out["native_multi_image_supported"] = False
        out["native_probe_error"] = str(exc)[:200]

    for case in CASES:
        photos = case["photos"]
        res = {"id": case["id"], "question": case["question"], "photos": photos,
               "gold": case["gold"]}
        # C: per-image 观察
        per_obs, lat_c = [], 0.0
        for p in photos:
            t0 = time.monotonic()
            obs = chat(f"描述这张照片里与“{case['question']}”相关的内容，尽量具体。",
                       [b64_of(p)])
            lat_c += time.monotonic() - t0
            per_obs.append({"photo": p, "observation": obs.strip()[:300]})
        # C 第二步：12B 文本对比
        t0 = time.monotonic()
        text_block = "\n".join(f"[照片{i+1}] {o['observation']}" for i, o in enumerate(per_obs))
        verdict_c = chat(f"问题：{case['question']}\n\n以下是逐张照片的观察：\n{text_block}\n\n"
                         f"请回答：哪一张（或哪些）照片符合？用[照片N]格式回答。")
        res["latency_per_image_s"] = round(lat_c, 2)
        res["latency_text_compare_s"] = round(time.monotonic() - t0, 2)
        res["per_image_observations"] = per_obs
        res["c_verdict"] = verdict_c.strip()[:200]
        # A: montage
        t0 = time.monotonic()
        m = montage(photos)
        verdict_a = chat(f"这张图里有多张照片，每张左上角有红色数字编号。"
                         f"问题：{case['question']}。请回答：哪一张（哪些）照片符合？用编号回答。",
                         [m])
        res["latency_montage_s"] = round(time.monotonic() - t0, 2)
        res["a_verdict"] = verdict_a.strip()[:200]
        out["cases"].append(res)
    return out


if __name__ == "__main__":
    result = run()
    out_path = "/tmp/phaseh_multi_image_spike.json"
    Path(out_path).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"native": result.get("native_multi_image_supported"),
                      "native_probe": result.get("native_probe_answer", ""),
                      "cases": [{"id": c["id"], "A": c["a_verdict"][:80],
                                 "C": c["c_verdict"][:80],
                                 "latA": c["latency_montage_s"], "latC": c["latency_per_image_s"]}
                                for c in result["cases"]]}, ensure_ascii=False, indent=1))
