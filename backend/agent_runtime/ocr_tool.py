"""OCR Tool（Phase E/H）：read_photo_text 的内部实现。

只负责"读照片文字"：PaddleOCR 小模型优先，读不到回退 VLM；
不承担语义理解，语义正确性由 guard/judge 模型层负责。
"""

from __future__ import annotations

import base64
import json
import os
import re
import tempfile
import threading
import time
from pathlib import Path

_RUNTIME: dict = {}


def bind_ocr_runtime(runtime: dict) -> None:
    """由 tools.bind_runtime 同步共享运行时状态。"""
    global _RUNTIME
    _RUNTIME = runtime


def _handle_to_asset_id(handle: str) -> str | None:
    return (_RUNTIME.get("last_handles") or {}).get(handle)


# ---- Tool 4b: read_photo_text（Phase E：OCR 专用 Tool）----
_OCR_PROMPT = """请读出这张照片中的全部文字（招牌/菜单/价格/数字/电话/年份/小字）。只输出文字内容本身，不要描述图片、不要评价。如果完全看不清就输出空字符串。"""
_OCR_PROMPT_FULL = """观察这张照片，墙面上有哪些招牌、牌子或文字？请逐条列出文字内容本身（店名、电话、价格、年份、标语等），不要描述人物和场景。看不清的部分不要编造。"""
_OCR_CACHE: dict[str, tuple[tuple, dict]] = {}  # (asset_id, mtime, provider, tiles) -> result

# Phase H H6：OCR provider 遥测（dashboard 汇总 small/VLM 使用率、延迟、fallback）
_OCR_TELEMETRY_LOCK = threading.Lock()
_OCR_TELEMETRY: dict[str, dict] = {
    "small": {"calls": 0, "latency_sum_s": 0.0, "conf_sum": 0.0, "fallback": 0},
    "vlm": {"calls": 0, "latency_sum_s": 0.0, "conf_sum": 0.0, "fallback": 0},
    "errors": 0,
}


def record_ocr_telemetry(provider: str, latency_s: float, confidence: float | None = None,
                         fallback: bool = False) -> None:
    with _OCR_TELEMETRY_LOCK:
        bucket = _OCR_TELEMETRY.get(provider)
        if bucket is None:
            bucket = _OCR_TELEMETRY.setdefault(provider, {"calls": 0, "latency_sum_s": 0.0,
                                                          "conf_sum": 0.0, "fallback": 0})
        bucket["calls"] += 1
        bucket["latency_sum_s"] += latency_s
        if confidence is not None:
            bucket["conf_sum"] += confidence
        if fallback:
            bucket["fallback"] += 1
            _OCR_TELEMETRY["errors"] += 1


def ocr_telemetry_snapshot() -> dict:
    with _OCR_TELEMETRY_LOCK:
        out = {}
        for provider, b in _OCR_TELEMETRY.items():
            if not isinstance(b, dict):
                out[provider] = b
                continue
            calls = b["calls"]
            out[provider] = {
                "calls": calls,
                "latency_avg_s": round(b["latency_sum_s"] / calls, 3) if calls else None,
                "confidence_avg": round(b["conf_sum"] / calls, 3) if calls and b["conf_sum"] else None,
                "fallback": b["fallback"],
            }
        return out

# Phase F F5：OCR Provider 抽象（预留 lightweight 插槽，当前只有 VLM）
# SENTRIX_OCR_PROVIDER=vlm  |  SENTRIX_OCR_TILES=none|2x2|3x3（默认 2x2，提速用）
_OCR_PROVIDERS: dict[str, str] = {"vlm": "vlm", "small": "small"}

_small_ocr_available_cache: bool | None = None


def small_ocr_available() -> bool:
    """PaddleOCR (CPU) 是否可导入——零显存、进程内推理。

    结果缓存；首次调用会尝试 import（失败=不可用，不影响 read_photo_text 主路径）。
    """
    global _small_ocr_available_cache
    if _small_ocr_available_cache is None:
        try:
            from paddleocr import PaddleOCR  # noqa: F401
            _small_ocr_available_cache = True
        except Exception:
            _small_ocr_available_cache = False
    return _small_ocr_available_cache
_OCR_TILE_DEFAULT = "2x2"


def _ocr_provider() -> str:
    return os.getenv("SENTRIX_OCR_PROVIDER", "vlm").strip().lower() or "vlm"


def _ocr_tile_layout() -> str:
    layout = os.getenv("SENTRIX_OCR_TILES", _OCR_TILE_DEFAULT).strip().lower()
    return layout if layout in {"none", "2x2", "3x3"} else _OCR_TILE_DEFAULT


def _clean_ocr_text(raw) -> str:
    """清洗 12B OCR 输出：去掉 thought/code block/JSON 包装与重复行，只保留文字。"""
    if not raw:
        return ""
    text = str(raw).strip()
    text = re.sub(r"```(?:json|JSON)?", "", text)
    text = re.sub(r"^\s*thought\s*", "", text, flags=re.I)
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            for key in ("text", "ocr", "content", "result"):
                val = obj.get(key)
                if isinstance(val, str):
                    text = val.strip()
                    break
            else:
                return ""
    except (TypeError, ValueError):
        pass
    seen, lines = set(), []
    for line in text.splitlines():
        s = line.strip()
        if s and s not in seen:
            seen.add(s)
            lines.append(s)
    return "\n".join(lines)[:1200]


def _tile_images(path: str, rows: int = 3, cols: int = 3, scale: float = 2.0):
    """把图片切成 rows*cols 个 tile，每块放大 scale 倍，返回 [(label, base64), ...]。"""
    try:
        from io import BytesIO
        from PIL import Image
    except Exception:
        return []
    try:
        img = Image.open(path)
    except Exception:
        return []
    w, h = img.size
    tiles = []
    tw, th = max(1, w // cols), max(1, h // rows)
    for r in range(rows):
        for c in range(cols):
            box = (c * tw, r * th, min(w, (c + 1) * tw), min(h, (r + 1) * th))
            tile = img.crop(box)
            if scale != 1.0:
                tile = tile.resize((int(tile.width * scale), int(tile.height * scale)),
                                   Image.LANCZOS)
            buf = BytesIO()
            tile.convert("RGB").save(buf, "JPEG", quality=92)
            tiles.append((f"tile_r{r}c{c}", base64.b64encode(buf.getvalue()).decode()))
    return tiles


def _text_rows_montage(path: str, items, max_rows: int = 8) -> str:
    """用检测框裁剪文字行，拼成一张竖排 montage（放大 2x）给 VLM 读。

    替代固定 2x2 切块：文字区域更聚焦，小字/菜单更友好。
    返回 base64；无有效区域返回空串。
    """
    try:
        from io import BytesIO
        from PIL import Image
    except Exception:
        return ""
    if not items:
        return ""
    try:
        img = Image.open(path)
    except Exception:
        return ""
    w, h = img.size
    sorted_items = sorted(items, key=lambda t: (t[0], t[1]))
    rows = []
    for it in sorted_items:
        yc = (it[0] + it[2]) / 2
        placed = False
        for row in rows:
            if abs(row["yc"] - yc) <= 32:
                row["items"].append(it)
                placed = True
                break
        if not placed:
            rows.append({"yc": yc, "items": [it]})
    crops = []
    for row in rows[:max_rows]:
        xs = [p for it in row["items"] for p in (it[1], it[3])]
        ys = [p for it in row["items"] for p in (it[0], it[2])]
        x0, y0, x1, y1 = max(0, min(xs) - 12), max(0, min(ys) - 12), min(w, max(xs) + 12), min(h, max(ys) + 12)
        if x1 - x0 < 8 or y1 - y0 < 8:
            continue
        crop = img.crop((int(x0), int(y0), int(x1), int(y1)))
        crop = crop.resize((int(crop.width * 2), int(crop.height * 2)), Image.LANCZOS)
        crops.append(crop)
    if not crops:
        return ""
    gap = 8
    total_h = sum(c.height for c in crops) + gap * (len(crops) - 1)
    max_w = max(c.width for c in crops)
    canvas = Image.new("RGB", (max_w, total_h), "white")
    y = 0
    for c in crops:
        canvas.paste(c, (0, y))
        y += c.height + gap
    buf = BytesIO()
    canvas.save(buf, "JPEG", quality=90)
    return base64.b64encode(buf.getvalue()).decode()


# ============ Phase H H2/H4: Small OCR Provider（RapidOCR · CPU · 零显存） ============
_small_engine = None
_small_engine_lock = threading.Lock()


def _get_small_engine():
    global _small_engine
    if _small_engine is None:
        with _small_engine_lock:
            if _small_engine is None:
                from paddleocr import PaddleOCR
                _small_engine = PaddleOCR(
                    text_detection_model_name="PP-OCRv5_mobile_det",
                    text_recognition_model_name="PP-OCRv5_mobile_rec",
                    use_doc_orientation_classify=False,
                    use_doc_unwarping=False,
                    use_textline_orientation=False,
                    enable_mkldnn=False,
                    device="cpu",  # 强制 CPU：onnxruntime 默认试 CUDA，缺 libcudnn.so.9 会初始化失败
                )
    return _small_engine


def _paddle_items(result):
    """PaddleOCR result -> [(y0, x0, y1, x1, text, conf)]。"""
    items = []
    for page in (result or []):
        texts = page.get("rec_texts") or []
        scores = page.get("rec_scores") or []
        polys = page.get("rec_polys") or []
        for i, (text, score) in enumerate(zip(texts, scores)):
            if not text or not str(text).strip():
                continue
            poly = polys[i] if i < len(polys) else None
            if poly is not None and len(poly) > 0:
                xs = [p[0] for p in poly]
                ys = [p[1] for p in poly]
                y0, y1, x0, x1 = min(ys), max(ys), min(xs), max(xs)
            else:
                y0 = y1 = x0 = x1 = 0.0
            items.append((y0, x0, y1, x1, str(text).strip(), float(score or 0)))
    return items


def _small_row_cluster(items, y_tol=26):
    """按 y 聚类成行（菜单/招牌通常水平成行）。返回 [(y_center, [(x, text, conf)])]。"""
    rows = []
    for it in sorted(items, key=lambda t: (t[0], t[1])):
        yc = (it[0] + it[2]) / 2
        placed = False
        for row in rows:
            if abs(row[0] - yc) <= y_tol:
                row[1].append((it[1], it[4], it[5]))
                placed = True
                break
        if not placed:
            rows.append((yc, [(it[1], it[4], it[5])]))
    return rows


_PRICE_RE = re.compile(r"(?:¥|￥)\s*(\d+(?:\.\d+)?)|(\d+(?:\.\d+)?)\s*(?:元|块)")
_PHONE_RE = re.compile(r"(?<!\d)(\d{7,12})(?!\d)")
_YEAR_RE = re.compile(r"(?<!\d)((?:19|20)\d{2})(?!\d)")
# 问题明显在问硬值（价格/电话/年份/数字）时，small 没读到硬值就回退 VLM 再读一次
_HARD_VALUE_QUESTION_RE = re.compile(
    r"价格|多少钱|售价|单价|几块|几元|电话|号码|年份|哪一年|数字")
def _extract_simple_exact(text: str) -> list[dict]:
    """从 OCR 文本极简提取硬值（价格/电话/年份），不做 label 关联。

    这是"顺带产物"：帮助 Answer Nucleus 绑定硬值、避免 LLM 改写数字；
    语义理解仍由 agent 负责。
    """
    values = []
    for m in _PRICE_RE.finditer(text or ""):
        value = m.group(1) or m.group(2)
        values.append({"type": "price", "value": value, "unit": "元",
                       "text": m.group(0).strip()})
    for m in _YEAR_RE.finditer(text or ""):
        values.append({"type": "year", "value": m.group(1), "text": m.group(1)})
    for m in _PHONE_RE.finditer(text or ""):
        values.append({"type": "phone", "value": m.group(1), "text": m.group(1)})
    return values


def _small_ocr_rows(items):
    """RapidOCR items -> 行聚类文本 + 简单硬值 + 平均置信度（仅用于遥测）。"""
    rows = _small_row_cluster(items)
    lines = []
    for _, cells in sorted(rows, key=lambda r: r[0]):
        lines.append(" ".join(text for _, text, _ in sorted(cells, key=lambda c: c[0])))
    text = "\n".join(lines)
    confs = [it[5] for it in items]
    avg_conf = round(sum(confs) / len(confs), 3) if confs else 0.0
    return text, _extract_simple_exact(text), avg_conf


def _try_small_ocr(path: str, context: dict | None) -> tuple[dict | None, list]:
    """Small OCR 优先路径：读文字；读不到就返回 None 让 VLM 接管。

    不再做置信度阈值回退、放大重识别、价格 label 关联——这些把工具搞复杂了。
    返回 (observation, items)；observation 为 None 表示走 VLM 主路径，
    items 是检测框（供 VLM 做文字区域 montage）。
    """
    settings = (context or {}).get("ocr_settings") or {}
    # W3.4：PaddleOCR 可用时默认启用 small（app 的 DB 设置默认 "false"，不应禁用专用小模型；
    # 专用小模型能力优于 VLM，这是 OCR 主路径，不是可选项）。用户仍可显式 small_ocr_enabled=false 关闭。
    if settings.get("small_ocr_enabled", small_ocr_available()) is False:
        return None, []
    if not small_ocr_available():
        return None, []
    try:
        engine = _get_small_engine()
        t0 = time.monotonic()
        result = engine.predict(path)
        latency = round(time.monotonic() - t0, 3)
        items = _paddle_items(result)
        if not items:
            return None, []
        ocr_text, exact, avg_conf = _small_ocr_rows(items)
        if not ocr_text.strip():
            return None, items
        regions = [{"text": line[:150], "source": "small_ocr"}
                   for line in ocr_text.splitlines()[:6]]
        record_ocr_telemetry("small", latency, avg_conf)
        return {
            "summary": f"已读取 {len(regions)} 个文字区域。",
            "full_text": ocr_text[:1600],
            "text_regions": regions[:6],
            "source": "small_ocr",
            "provider": "small",
            "confidence": avg_conf,
            "exact_values": exact,
            "fallback_used": False,
            "certainty": "supported" if regions else "uncertain",
            "persisted": False,
            "cache_hit": False,
            "latency_s": latency,
            "_model_call_metrics": [],
        }, items
    except Exception:
        return None, []


def _crop_region_to_temp(path: str, box, pad: int = 8, scale: float = 2.0) -> str | None:
    """裁剪检测框区域 → 放大 + 对比度增强 → 写入临时文件，返回临时文件路径（调用方负责删除）。

    PaddleOCR predict 只支持 file path / numpy.ndarray，不支持 BytesIO，故写临时文件。
    """
    try:
        from PIL import Image, ImageOps, ImageEnhance
        y0, x0, y1, x1 = (int(v) for v in box)
        with Image.open(path) as img:
            w, h = img.size
            y0, x0 = max(0, y0 - pad), max(0, x0 - pad)
            y1, x1 = min(h, y1 + pad), min(w, x1 + pad)
            if y1 <= y0 or x1 <= x0:
                return None
            crop = img.crop((x0, y0, x1, y1))
            crop = ImageOps.autocontrast(crop)
            crop = ImageEnhance.Contrast(crop).enhance(1.6)
            crop = crop.convert("RGB")
            if scale > 1:
                crop = crop.resize((int(crop.width * scale), int(crop.height * scale)), Image.LANCZOS)
            fd, tmp = tempfile.mkstemp(suffix=".jpg")
            crop.save(tmp, "JPEG", quality=95)
            os.close(fd)
            return tmp
    except Exception:
        return None


def _adaptive_small_retry(path: str, items: list, context: dict | None,
                          deadline_s: float = 10.0) -> dict | None:
    """Level 2：small 检测出区域但文本不全/缺失时，对检测框 crop + upscale + 对比度增强，
    再用 small 重识别（PaddleOCR 专用模型，能力优于 VLM）。失败返回 None。"""
    if not items or not small_ocr_available():
        return None
    engine = _get_small_engine()
    t0 = time.monotonic()
    texts = []
    exact = []
    for it in items[:12]:
        if time.monotonic() - t0 > deadline_s:
            break
        tmp = _crop_region_to_temp(path, (it[0], it[1], it[2], it[3]))
        if not tmp:
            continue
        try:
            res = engine.predict(tmp)
            sub = _paddle_items(res)
            sub_text, sub_exact, _ = _small_ocr_rows(sub)
            if sub_text.strip():
                texts.append(sub_text)
                exact.extend(sub_exact)
        except Exception:
            continue
        finally:
            try:
                os.remove(tmp)
            except Exception:
                pass
    if not texts:
        return None
    merged = "\n".join(texts)[:1600]
    record_ocr_telemetry("small", round(time.monotonic() - t0, 3), None, fallback=True)
    return {
        "summary": f"已读取 {len(texts)} 个文字区域（放大重识别）。",
        "full_text": merged,
        "text_regions": [{"text": line[:150], "source": "small_retry"} for line in merged.splitlines()[:6]],
        "source": "small_ocr",
        "provider": "small_retry",
        "exact_values": exact,
        "fallback_used": True,
        "certainty": "supported" if texts else "uncertain",
        "persisted": False,
        "cache_hit": False,
        "latency_s": round(time.monotonic() - t0, 3),
        "_model_call_metrics": [],
    }


def _read_photo_text(arguments: dict, *, context: dict | None = None) -> dict:
    """OCR 专用：读取照片中的文字（菜单/价格/招牌/电话/年份/小字）。

    与 inspect_photo 的分工：inspect_photo 做视觉理解（颜色/物体/场景），
    read_photo_text 做文本读取。内部把照片切成 3x3 tile 放大后交给 VLM OCR，
    避免整图小字被压缩丢失。

    G6：任何未预期异常都降级为 natural partial（status=partial / reason=ocr_failed），
    绝不让 OCR 失败变成 tool_execution_error 或“这次处理没有完成”式工程错误。
    """
    try:
        return _read_photo_text_impl(arguments, context=context)
    except Exception as exc:
        try:
            import sys as _sys
            print(f"[read_photo_text] ocr_failed fallback: {type(exc).__name__}: {exc}",
                  file=_sys.stderr)
        except Exception:
            pass
        return {
            "summary": "这次没能可靠读出照片里的文字。",
            "full_text": "", "text_regions": [],
            "certainty": "uncertain",
            "status": "partial",
            "reason": "ocr_failed",
            "persisted": False,
            "cache_hit": False,
            "_model_call_metrics": [],
        }


def _ocr_single_asset(asset_handle: str, arguments: dict, context: dict | None) -> dict | None:
    """对单张 asset 做 Level1(small) + Level2(adaptive retry) OCR，返回 observation 或 None。

    不做 VLM 兜底（OCR 专用小模型能力优于 VLM，VLM 又慢又幻觉）。"""
    scope_id = (context or {}).get("scope_id") or ""
    task_state = (context or {}).get("task_state") or {}
    asset_id = None
    result_set_id = task_state.get("current_result_set")
    rs_store = _RUNTIME.get("result_sets")
    if result_set_id and rs_store is not None:
        asset_id = rs_store.resolve_handle(result_set_id, asset_handle)
    if not asset_id:
        asset_id = _handle_to_asset_id(asset_handle)
    store = _RUNTIME.get("store")
    if not asset_id or store is None:
        return None
    row = store.connection.execute("SELECT path FROM assets WHERE id = ?", (asset_id,)).fetchone()
    if not row or not row["path"] or not Path(row["path"]).is_file():
        return None
    try:
        _mtime = Path(row["path"]).stat().st_mtime
    except Exception:
        _mtime = 0.0
    provider = _ocr_provider()
    tile_layout = _ocr_tile_layout()
    cache_key = (asset_id, _mtime, provider, tile_layout)
    cached = _OCR_CACHE.get(cache_key)
    if cached is not None:
        hit = dict(cached)
        hit["cache_hit"] = True
        hit["_model_call_metrics"] = []
        return hit
    small, small_items = _try_small_ocr(row["path"], context)
    question = arguments.get("question") or ""
    # Level1 small 优先；硬值题（价格/电话/年份）且 small 没拿到 exact_value 时，
    # Level2 对检测框 crop+放大+对比度重识别补读。
    if small is not None:
        if _HARD_VALUE_QUESTION_RE.search(question) and not (small.get("exact_values") or []):
            retry = _adaptive_small_retry(row["path"], small_items, context)
            if retry is not None and (retry.get("exact_values") or retry.get("full_text")):
                _OCR_CACHE[cache_key] = dict(retry)
                return retry
        _OCR_CACHE[cache_key] = dict(small)
        return small
    # Level2：small 检测到区域但文本缺失 → crop+放大+对比度重识别
    if small_items:
        retry = _adaptive_small_retry(row["path"], small_items, context)
        if retry is not None:
            _OCR_CACHE[cache_key] = dict(retry)
            return retry
    return None


def _read_photo_text_impl(arguments: dict, *, context: dict | None = None) -> dict:
    """read_photo_text 实际实现（异常由 _read_photo_text 兜底为 natural partial）。

    无显式 asset_handle 时，OCR 整个 preview 的候选图（前 5 张）并合并结果——
    避免只读第一张漏掉目标图（事件召回通常多张，顺序由检索排序决定，不可依赖）。
    """
    asset_handle = arguments.get("asset_handle") or ""
    task_state = (context or {}).get("task_state") or {}
    if asset_handle:
        handles = [asset_handle]
    else:
        preview = (task_state.get("result_preview") or []) or []
        handles = [p.get("handle") for p in preview[:5] if p.get("handle")]
    if not handles:
        return {"summary": "无法定位照片。", "full_text": "", "text_regions": [],
                "certainty": "uncertain", "persisted": False}
    texts = []
    exact = []
    for h in handles:
        res = _ocr_single_asset(h, arguments, context)
        if res and (res.get("full_text") or res.get("exact_values")):
            if res.get("full_text"):
                texts.append(res["full_text"])
            exact.extend(res.get("exact_values") or [])
    if texts:
        merged = "\n".join(texts)[:1600]
        return {
            "summary": f"已读取 {len(texts)} 张照片的文字。",
            "full_text": merged,
            "text_regions": [{"text": line[:150], "source": "small_ocr"} for line in merged.splitlines()[:8]],
            "source": "small_ocr",
            "provider": "small_multi",
            "exact_values": exact,
            "fallback_used": True,
            "certainty": "supported" if texts else "uncertain",
            "persisted": False,
            "cache_hit": False,
            "_model_call_metrics": [],
        }
    # 全部候选都失败 → partial
    return {"summary": "这次没能可靠读出照片里的文字。", "full_text": "", "text_regions": [],
            "certainty": "uncertain", "status": "partial", "reason": "ocr_failed",
            "persisted": False, "cache_hit": False, "_model_call_metrics": []}
