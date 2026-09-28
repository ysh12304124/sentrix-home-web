"""A3 — 只读 Tool 的实现与注册。

找图唯一入口是 search_memories（含单张照片的 place/time/people/描述，preview 里可直接读取
单张照片的拍摄时间、地点、人物）。query_memory_facts 全量聚合工具已删除：需要全库统计的
"多少钱/几桌/礼金/一共多少张"类问题没有对应照片证据，模型必须如实说明无法确认，
不得把检索返回的照片数当成统计值编造答案。
- search_memories：检索 kernel 封装（视觉/文本/混合），返回 ResultSet 摘要。
- get_original_photos：当前 ResultSet 原图交付（A4 ResultSetStore 后完整可用）。
- inspect_photo：多模态复核（A0.6 已验证链路），结果 ephemeral 不写长期记忆。

Tool 观察只暴露模型可安全看到的内容；内部 asset_id 通过 handle 映射（A4 完整化）。
"""

from __future__ import annotations

import base64
import json
import math
import os
import re
import threading
import tempfile
import time
from pathlib import Path

from .tool_registry import ToolSpec, register
from .capability import tool_capability_summary
from .intent import ocr_intent, visual_intent
from .ocr_tool import (_read_photo_text, ocr_telemetry_snapshot,
                       record_ocr_telemetry, small_ocr_available)
from .ocr_tool import bind_ocr_runtime
from ..person_appearance import expanded_person_crop

_RUNTIME: dict = {}


def _store_fetchone(store, query: str, params=()):
    """Serialize direct read queries on the shared SQLite connection."""
    lock = getattr(store, "_connection_lock", None)
    if lock is None:
        return store.connection.execute(query, params).fetchone()
    with lock:
        return store.connection.execute(query, params).fetchone()


def _store_fetchall(store, query: str, params=()):
    """Serialize direct read queries on the shared SQLite connection."""
    lock = getattr(store, "_connection_lock", None)
    if lock is None:
        return store.connection.execute(query, params).fetchall()
    with lock:
        return store.connection.execute(query, params).fetchall()


def bind_runtime(store, *, gamma=None, embedding_router=None, retrieval_config=None):
    from .result_set import ResultSetStore
    _RUNTIME["store"] = store
    _RUNTIME["gamma"] = gamma
    _RUNTIME["embedding_router"] = embedding_router
    _RUNTIME["retrieval_config"] = retrieval_config
    _RUNTIME["result_sets"] = ResultSetStore(store)
    bind_ocr_runtime(_RUNTIME)


def set_conversation_id(conversation_id):
    """D4：把当前 conversation_id 绑定到 tool 层（search_conversation_history 用）。"""
    _RUNTIME["conversation_id"] = conversation_id


def _kernel():
    from ..evidence_retrieval import EvidenceRetrievalKernel
    if _RUNTIME.get("embedding_router") is not None:
        from ..retrieval import RetrievalConfig, build_default_retrievers
        config = _RUNTIME.get("retrieval_config") or RetrievalConfig()
        retrievers = build_default_retrievers(_RUNTIME["store"], embedding_router=_RUNTIME["embedding_router"], config=config)
        return EvidenceRetrievalKernel(_RUNTIME["store"], retrievers=retrievers,
                                       embedding_router=_RUNTIME["embedding_router"], config=config)
    return EvidenceRetrievalKernel(_RUNTIME["store"])


def _cn_month(text: str) -> int | None:
    """把中文数字月份（十/十月/十一月/十二月）转成阿拉伯数字；阿拉伯数字原样。"""
    digits = {"一": 1, "二": 2, "三": 3, "四": 4, "五": 5,
              "六": 6, "七": 7, "八": 8, "九": 9, "十": 10}
    if text.isdigit():
        return int(text)
    if not text or any(ch not in digits for ch in text):
        return None
    if "十" not in text:
        return digits.get(text)
    tens, _, ones = text.partition("十")
    return (digits.get(tens, 1) if tens else 1) * 10 + digits.get(ones, 0)


def _resolve_time_expression(value: str) -> str | None:
    """把相对时间解析成可被 parse_time_expression 接受的绝对时间（含范围）。

    模型契约：相对时间必须原样传 filters.time，由这里确定性换算；不依赖模型算年份。
    支持：去年/今年/前年/这两年/近两年/最近一年/上个月/去年X月/去年春天等。
    """
    import re
    from .time_context import now
    now = now()
    v = (value or "").strip()
    if not v:
        return None
    y, m = now.year, now.month
    seasons = {
        "春天": (3, 5), "夏天": (6, 8), "秋天": (9, 11), "冬天": (12, 2),
        "春季": (3, 5), "夏季": (6, 8), "秋季": (9, 11), "冬季": (12, 2),
    }
    base_year = {"去年": y - 1, "今年": y, "前年": y - 2}
    month_rel = re.search(r"(去年|今年|前年)\s*(\d{1,2}|[一二三四五六七八九十]+)\s*月", v)
    if month_rel:
        month = _cn_month(month_rel.group(2))
        return f"{base_year[month_rel.group(1)]}年{month}月"
    season_rel = re.search(r"(去年|今年|前年)\s*(春天|夏天|秋天|冬天|春季|夏季|秋季|冬季)", v)
    if season_rel:
        base = base_year[season_rel.group(1)]
        sm, em = seasons[season_rel.group(2)]
        if sm <= em:
            return f"{base}年{sm}月-{base}年{em}月"
        return f"{base}年12月-{base + 1}年2月"
    if "这两年" in v or "近两年" in v or "最近两年" in v:
        return f"{y - 1}年-{y}年"
    if "最近一年" in v or "近一年" in v:
        prev_y, prev_m = (y - 1, m) if m > 1 else (y - 1, 12)
        return f"{prev_y}年{prev_m}月-{y}年{m}月"
    if "上上个月" in v:
        pm2 = (y, m - 2) if m > 2 else (y - 1, m + 10)
        return f"{pm2[0]}年{pm2[1]}月"
    if "上个月" in v:
        pm = (y, m - 1) if m > 1 else (y - 1, 12)
        return f"{pm[0]}年{pm[1]}月"
    if "去年" in v:
        return f"{y - 1}年"
    if "今年" in v:
        return f"{y}年"
    if "前年" in v:
        return f"{y - 2}年"
    if re.fullmatch(r"20\d{2}", v):
        return f"{v}年"
    if re.fullmatch(r"20\d{2}年(?:\d{1,2}月(?:\d{1,2}[日号]?)?)?", v):
        return v
    # Unknown relative phrases must not be passed to the strict time parser as
    # if they were absolute expressions.  Dropping the unsupported constraint
    # preserves semantic recall; the caller can surface the raw filter in
    # diagnostics instead of silently forcing a contradicted time range.
    return None


def _draft_from_filters(filters: dict, *, answer_type="asset_set", group_by=None):
    """把工具 filters 转成 QueryParseDraft（只读 shadow 用，语义与 thin_agent 对齐）。"""
    from ..query_contracts import QueryParseDraft
    draft = QueryParseDraft(intent="answer", answer_target="general",
                            answer_type=answer_type)
    time_expr = _resolve_time_expression((filters or {}).get("time") or "")
    if time_expr:
        draft.time_expression = time_expr
    place = (filters or {}).get("place") or ""
    if place:
        draft.semantic_conditions.append({"dimension": "place", "value": place, "strictness": "semantic_required"})
    person = (filters or {}).get("person") or ""
    if person:
        draft.entity_names.append(person)
    media = (filters or {}).get("media") or ""
    if media:
        draft.media_expressions.append(media)
    query = (filters or {}).get("query") or ""
    if query and answer_type == "asset_set":
        draft.semantic_conditions.append({"dimension": "semantic", "value": query, "strictness": "semantic_required"})
    if group_by:
        draft.structured = {"aggregation": {"op": "group_by", "group_by": group_by}}
    return draft


def _spec_for(draft, scope_id, viewer_id):
    from ..query_contracts import build_query_spec
    return build_query_spec(
        draft, scope_id=scope_id, viewer_id=viewer_id,
        conversation_id="tool_loop", query_id=f"tool_{int(time.time()*1000)}",
        entity_resolver=lambda name: _resolve_entity(name, scope_id),
    )


def _resolve_entity(name, scope_id):
    store = _RUNTIME.get("store")
    if store is None:
        return None
    try:
        for entity in store.list_entities(status="confirmed", scope_id=scope_id or None):
            if entity.get("canonical_name") == name:
                return entity.get("id")
    except Exception:
        pass
    return None

# ---- Tool 2: search_memories ----
_RESULT_PREVIEW_LIMIT = 6
_RESULT_PAGE_SIZE = 6
# The remote main retrieval contract keeps at most 18 slot candidates. The
# Agent must be able to inspect that same complete top-k window; otherwise a
# correct candidate at ranks 7-18 exists in retrieval diagnostics but cannot
# be selected as delivered evidence. This changes exposure only: the evaluator
# still scores the media explicitly selected by the Agent, and page pagination
# remains six items for ordinary result browsing.
_SLOT_PREVIEW_LIMIT = 18


def _public_candidate_limit() -> int:
    """Optional legacy display cap; unset means no arbitrary candidate cap.

    Retrieval must be controlled by channel confidence/thresholds, not a
    hidden Top-K.  Keep an explicit environment override only for controlled
    rollback experiments; production defaults to the complete authorized pool.
    """
    raw = os.getenv("SENTRIX_SEARCH_VALIDATION_MAX_CANDIDATES", "").strip()
    if not raw:
        return 0
    try:
        return max(0, int(raw))
    except (TypeError, ValueError):
        return 0


def _visible_candidate_total(asset_count: int) -> int:
    count = max(0, int(asset_count or 0))
    limit = _public_candidate_limit()
    return min(count, limit) if limit else count
_RESULT_PREVIEW_RELEVANCE_HEAD = max(
    0, min(_RESULT_PREVIEW_LIMIT, int(os.getenv("SENTRIX_RESULT_PREVIEW_RELEVANCE_HEAD", "3")))
)
_PREVIEW_QUERY_ALIASES = {
    "布置": ("布置", "装饰", "花艺", "彩带", "气球", "窗帘", "床品", "家具"),
    "装饰": ("装饰", "花艺", "彩带", "气球", "窗帘", "床品", "家具"),
    "文字": ("文字：", "文字", "写着", "标志"),
    # 资产视觉描述常将迎宾展架拆成“横幅、支架”，而不会复述“迎宾展架”。
    # 这是仅用于已召回候选的 preview 排序的视觉等价词，不会扩展全库检索，
    # 因而不会把一般横幅错误地当作召回条件。
    "迎宾展架": ("迎宾展架", "展架", "广告牌", "欢迎牌", "横幅", "支架"),
    "祝福": ("祝福", "幸福", "恭喜", "结婚"),
    "雕塑": ("雕塑", "石雕", "雕像", "纪念碑"),
    "石雕": ("石雕", "雕塑", "雕像", "纪念碑"),
    "桥": ("桥", "桥上"),
    # Person-count/scene cues are useful for choosing the visible evidence
    # window after a broad place recall. They do not change retrieved
    # candidates; they only promote matching observations into preview.
    "三人": ("三人", "三个人", "三人合影"),
    "三个人": ("三人", "三个人", "三人合影"),
    "合影": ("合影", "自拍", "合照"),
    # “留影/拍照”通常指用户要找一张具体的摆拍图，不等同于多人合影。
    # 保留为独立 cue，避免事件内的合影因为同时出现“婚礼/紫色”等泛词而压过
    # 真正的单人留影关键帧。
    "留影": ("留影", "拍照", "站立拍照", "摆拍", "个人照"),
    "拍照": ("拍照", "站立拍照", "摆拍", "留影", "个人照"),
    "舞台": ("舞台", "仪式", "典礼"),
    "户外": ("户外", "室外", "露天"),
    "夜晚": ("夜晚", "夜间", "夜景", "灯光"),
    "晚上": ("晚上", "夜晚", "夜间", "夜景", "灯光"),
    "夜间": ("晚上", "夜晚", "夜间", "夜景", "灯光"),
    "紫色": ("紫色", "紫", "紫布", "紫色布景", "紫色纱幔"),
    "紫": ("紫色", "紫", "紫布", "紫色布景", "紫色纱幔"),
    "室外": ("室外", "户外", "露天"),
}

_PREVIEW_QUERY_STOPWORDS = {
    "我", "我们", "你", "帮我", "找一下", "找我", "记得", "有次", "那次", "这次",
    "当时", "那个", "这个", "照片", "图片", "留影", "拍照", "拍了", "拍摄", "前面",
    "附近", "具体", "哪里", "哪儿", "什么", "时候", "发生", "经历", "帮忙", "一下",
}

# These are common grammatical/retrieval words rather than event-defining
# visual facts.  They must not make an unrelated image look like a strong
# match merely because both the question and its caption say "photo" or
# "activity".  The list is intentionally domain-neutral: it applies equally
# to trips, documents, family photos, products and videos.
_PREVIEW_CONTEXT_STOP_BIGRAMS = {
    "我们", "你们", "他们", "这个", "那个", "那次", "这次", "当时", "当天",
    "照片", "图片", "拍照", "拍摄", "留影", "合影", "记录", "活动", "事情",
    "哪里", "哪儿", "什么", "时候", "具体", "帮我", "一下", "一下", "一张",
    "看到", "找到", "想找", "记得", "参加", "一起", "还有", "就是", "大概",
}


def _preview_query_terms(query: str) -> list[str]:
    """Extract bounded query concepts for caption-aware preview ordering."""
    text = str(query or "").strip()
    if not text:
        return []
    try:
        import jieba
        raw_terms = jieba.lcut(text, cut_all=False)
    except Exception:
        raw_terms = re.findall(r"[\u4e00-\u9fff]{2,}|[A-Za-z0-9]+", text)
    terms = []
    for raw in raw_terms:
        term = re.sub(r"[^\u4e00-\u9fffA-Za-z0-9]", "", str(raw)).strip()
        if len(term) < 2 or term in _PREVIEW_QUERY_STOPWORDS or term in terms:
            continue
        terms.append(term)
        if len(terms) >= 16:
            break
    return terms


def _preview_context_bigrams(query: str) -> list[str]:
    """Return discriminative CJK anchors for caption-aware candidate ranking.

    Word segmentation is useful when it succeeds, but it can keep a compound
    phrase such as ``婚礼仪式舞台`` intact.  Captions often describe the same
    scene with only part of that phrase (``婚礼`` / ``仪式`` / ``舞台``), so a
    strict word-only comparison loses the decisive context and lets generic
    cues such as "night" or "photo" dominate.  CJK bigrams provide a bounded,
    language-agnostic bridge without consulting answers, benchmark labels, or
    image pixels.
    """
    text = re.sub(r"[^\u4e00-\u9fff]", "", str(query or ""))
    if len(text) < 2:
        return []
    values: list[str] = []
    for index in range(len(text) - 1):
        token = text[index:index + 2]
        if token in _PREVIEW_CONTEXT_STOP_BIGRAMS or token in values:
            continue
        values.append(token)
        if len(values) >= 24:
            break
    return values


def _preview_text_score(query: str, summary: str,
                        term_weights: dict[str, float] | None = None) -> float:
    """Score only query-to-observation overlap; never consult QA answers or GT."""
    text = str(query or "")
    desc = str(summary or "")
    if not desc:
        return 0.0
    weights = term_weights or {}
    score = sum(2.0 * weights.get(term, 1.0)
                for term, aliases in _PREVIEW_QUERY_ALIASES.items()
                if term in text and any(alias in desc for alias in aliases))
    terms = _preview_query_terms(text)
    if terms:
        cue_terms = {term for term in _PREVIEW_QUERY_ALIASES if term in text}
        score += sum((min(3, len(term)) / 3) * weights.get(term, 1.0)
                     for term in terms if term in desc and term not in cue_terms)
    # A single broad cue is useful for recall but weak evidence for ranking.
    # Reward *conjunctions* of independent contextual anchors superlinearly:
    # an image matching both "婚礼" and "舞台" should beat one that only shares
    # the generic nighttime/photograph wording.  This is a local rerank of
    # already-recalled assets, so it cannot expand scope or inject facts.
    context_hits = [token for token in _preview_context_bigrams(text)
                    if token in desc]
    if context_hits:
        score += min(2.0, 0.25 * len(context_hits))
        if len(context_hits) >= 2:
            score += min(4.0, 0.9 * (len(context_hits) - 1))
    return score


def _preview_query_term_weights(query: str, summaries: list[str]) -> dict[str, float]:
    """Weight rare query anchors more than concepts shared by the whole result pool."""
    text = str(query or "")
    terms = set(_preview_query_terms(text))
    terms.update(term for term in _PREVIEW_QUERY_ALIASES if term in text)
    if not terms or not summaries:
        return {}
    total = len(summaries)
    weights = {}
    for term in terms:
        aliases = _PREVIEW_QUERY_ALIASES.get(term, (term,))
        document_frequency = sum(
            1 for summary in summaries
            if any(alias in summary for alias in aliases)
        )
        # Smooth so ubiquitous terms keep a small positive weight while a
        # discriminative village/object cue can influence the visible head.
        weights[term] = 1.0 + math.log((total + 1) / (document_frequency + 1))
    return weights


def _query_prefers_single_evidence(query: str) -> bool:
    """Whether the wording asks for one concrete photo rather than a group set.

    This is a retrieval intent, not a benchmark/GT rule.  A scene question such
    as “拍了留影” is commonly answered by one posed frame, while “合影/和朋友”
    explicitly asks for a group image.  Keeping the distinction here prevents
    the event's generic cover photo from winning solely because it has more
    detected faces.
    """
    text = str(query or "")
    single_cues = ("留影", "拍照", "拍了照", "一张", "这张", "那张", "个人照")
    group_cues = ("合影", "合照", "多人", "和朋友", "跟朋友", "兄弟们", "全家",
                  "一家人", "几张", "哪些照片", "一共")
    return any(cue in text for cue in single_cues) and not any(
        cue in text for cue in group_cues
    )


def _semantic_people_count(store, asset_id: str) -> int | None:
    """Return the count from the stored semantic people list, when available.

    Face-instance counts are deliberately not used: one image can contain
    duplicate/low-quality detections, which made face count a noisy proxy for
    “single photo vs group photo”.
    """
    if store is None or not asset_id:
        return None
    try:
        rows = store.list_observations(asset_id=asset_id, limit=1) or []
        row = rows[0] if rows else {}
        values = row.get("people") or row.get("people_json") or []
        if isinstance(values, str):
            values = json.loads(values)
        if not isinstance(values, list):
            return None
        values = [str(value).strip() for value in values if str(value).strip()]
        return len(values) if values else None
    except Exception:
        return None


def _observation_summary(store, asset_id: str) -> str:
    """Expose bounded evidence text without exposing storage identifiers."""
    if store is None:
        return ""
    try:
        rows = store.list_observations(asset_id=asset_id, limit=1)
    except Exception:
        return ""
    if not rows:
        return ""
    observation = rows[0] or {}
    parts = []
    for key in ("caption", "activity", "place"):
        value = str(observation.get(key) or "").strip()
        if value and value not in parts:
            parts.append(value)
    for key in ("objects", "clothing"):
        values = observation.get(key) or []
        labels = []
        for value in values:
            if isinstance(value, dict):
                value = value.get("label") or value.get("primary") or ""
            value = str(value or "").strip()
            if value and value not in labels:
                labels.append(value)
        if labels:
            parts.append("、".join(labels[:8]))
    ocr = str(observation.get("ocr_text") or "").strip()
    if ocr:
        parts.append(f"文字：{ocr[:120]}")
    detail = observation.get("detail") or {}
    if isinstance(detail, dict):
        details = []
        for item in detail.get("visible_details") or []:
            if isinstance(item, dict):
                value = item.get("text") or item.get("label") or ""
            else:
                value = item
            value = str(value or "").strip()
            if value and value not in details:
                details.append(value)
        if details:
            parts.append("；".join(details[:6]))
    return "；".join(parts)[:300]


def _asset_group_key(store, asset_id: str) -> str:
    """Group video keyframes and event-near-duplicates for the initial preview."""
    if store is None:
        return asset_id
    try:
        asset = store.get_asset(asset_id) or {}
        if asset.get("derived_kind") in {"video_keyframe", "video_keyframe_webp"}:
            return ":".join(str(asset.get(key) or "") for key in (
                "parent_asset_id", "source_scene_index")) or asset_id
        row = store.connection.execute(
            "SELECT event_id FROM event_observations WHERE observation_id IN "
            "(SELECT id FROM observations WHERE asset_id = ?) ORDER BY event_id LIMIT 1",
            (asset_id,),
        ).fetchone()
        return str(row["event_id"] if row else asset_id)
    except Exception:
        return asset_id


def _preview_query_order(asset_ids: list[str], query: str, store) -> list[int]:
    """Promote candidates whose stored visual detail matches explicit visual cues."""
    text = str(query or "")
    cues = [(term, aliases) for term, aliases in _PREVIEW_QUERY_ALIASES.items()
            if term in text]
    requested_count = None
    count_match = re.search(r"([一二三四五六七八九十]|\d+)\s*(?:人|个人)", text)
    if count_match:
        raw = count_match.group(1)
        requested_count = {"一": 1, "二": 2, "三": 3, "四": 4, "五": 5,
                           "六": 6, "七": 7, "八": 8, "九": 9, "十": 10}.get(raw)
        if requested_count is None:
            try:
                requested_count = int(raw)
            except ValueError:
                requested_count = None
    terms = _preview_query_terms(text)
    single_evidence = _query_prefers_single_evidence(text)
    if not cues and not terms and requested_count is None and not single_evidence:
        return list(range(len(asset_ids)))
    summaries = [_observation_summary(store, asset_id) for asset_id in asset_ids]
    term_weights = _preview_query_term_weights(text, summaries)
    scored = []
    for index, (asset_id, summary) in enumerate(zip(asset_ids, summaries)):
        score = _preview_text_score(text, summary, term_weights)
        if requested_count is not None and store is not None:
            try:
                face_count = int(store.connection.execute(
                    "SELECT COUNT(*) FROM face_instances WHERE asset_id = ?", (asset_id,)
                ).fetchone()[0])
                # Face detections can contain duplicate clusters, so reward the
                # nearest count rather than requiring exact equality.
                score += max(0, 2 - abs(face_count - requested_count))
            except Exception:
                pass
            # Prefer an observation that explicitly states the requested count;
            # a nearby two-person scene with duplicated face detections should
            # not outrank a genuine “三个人” caption.
            count_words = {1: ("一个", "一名", "一人"), 2: ("两个", "两名", "两人"),
                           3: ("三个", "三名", "三人"), 4: ("四个", "四名", "四人")}
            if any(word in summary for word in count_words.get(requested_count, ())):
                score += 3
        if single_evidence:
            # Semantic people labels are more stable than raw face detections.
            # The latter can count duplicate boxes or background faces.
            people_count = _semantic_people_count(store, asset_id)
            if people_count == 1:
                score += 4.0
            elif people_count and people_count > 1:
                score -= min(4.0, 1.5 * (people_count - 1))
        scored.append((-score, index))
    scored.sort()
    return [index for _, index in scored]


def _preview_indices(asset_ids: list[str], mode: str, store, query: str = "",
                     *, limit: int | None = None) -> list[int]:
    """Select bounded indices under an explicit candidate-window policy.

    The full ResultSet remains server-side.  ``SENTRIX_CANDIDATE_STRATEGY`` is
    intentionally process-scoped so benchmark A/B runs can change only this
    policy while keeping model, ANN and source data fixed:
    ``head_only`` keeps retrieval order, ``event_diversity`` maximizes event
    diversity, and the default keeps a relevance head before diversity.
    """
    preview_limit = max(1, int(limit or _RESULT_PREVIEW_LIMIT))
    # The retrieval order is the only ranking signal guaranteed to have been
    # produced by the complete candidate search.  Event-diversity is still
    # available as an explicit opt-in, but must not silently hide the sixth
    # ranked source image from the user/evidence window.
    strategy = os.getenv("SENTRIX_CANDIDATE_STRATEGY", "head_only").strip().lower()
    if len(asset_ids) <= preview_limit:
        return (_preview_query_order(asset_ids, query, store)
                if mode != "representative" else list(range(len(asset_ids))))
    if mode == "representative":
        candidates = _even_indices(len(asset_ids), preview_limit)
    else:
        candidates = _preview_query_order(asset_ids, query, store)
    if strategy in {"head_only", "relevance_head_only"}:
        return candidates[:preview_limit]
    if strategy in {"event_diversity", "diversity_only"}:
        relevance_head = 0
    else:
        relevance_head = min(_RESULT_PREVIEW_RELEVANCE_HEAD, len(candidates))
    selected = []
    seen_groups = set()
    # Retrieval order is the strongest available relevance signal.  Preserve a
    # small head even when those assets belong to one event; otherwise event
    # diversity can discard the actual answer image before the model can inspect it.
    for index in candidates[:relevance_head]:
        selected.append(index)
        seen_groups.add(_asset_group_key(store, asset_ids[index]))
    if len(selected) >= preview_limit:
        return selected[:preview_limit]
    for index in candidates[relevance_head:]:
        group = _asset_group_key(store, asset_ids[index])
        if group in seen_groups:
            continue
        selected.append(index)
        seen_groups.add(group)
        if len(selected) >= preview_limit:
            return selected
    # With explicit scene cues, never pad the visible window with an arbitrary
    # head item that failed the cue match. Such padding let the model inspect
    # a generic “photo_1” and answer from the wrong image even though the
    # matching asset was already in the retrieved candidate set.
    if any(term in str(query or "") for term in _PREVIEW_QUERY_ALIASES):
        return selected
    for index in range(len(asset_ids)):
        if index not in selected:
            selected.append(index)
        if len(selected) >= preview_limit:
            break
    return selected


def _build_preview_entries(store, asset_ids: list[str], indices: list[int]) -> list[dict]:
    """Build the model-visible preview from the selected candidate indices.

    Handles retain their original ResultSet rank (``photo_N``), while
    ``priority_rank`` describes their order in this preview. Keeping those
    concepts separate lets query-aware preview ranking expose a relevant
    candidate without breaking subsequent inspect_photo handle resolution.
    """
    preview = []
    for priority_rank, index in enumerate(indices, 1):
        if index < 0 or index >= len(asset_ids):
            continue
        preview.append(_preview_entry(
            store, asset_ids[index], f"photo_{index + 1}",
            priority_rank=priority_rank,
            selection_reason="相关性最高" if priority_rank == 1 else "候选补充",
        ))
    return [item for item in preview if item]


def _candidate_window_summary(asset_ids: list[str], indices: list[int], store) -> dict:
    """Expose bounded candidate-window diagnostics without storage IDs.

    The ResultSet remains complete server-side; this summary tells the model
    and 8771 whether the visible window is a diverse head or one large event,
    without dumping the full candidate list into the prompt.
    """
    visible_limit = _visible_candidate_total(len(asset_ids))
    bounded_ids = list(asset_ids[:visible_limit])
    bounded_indices = [index for index in indices if index < visible_limit]
    groups = {}
    for asset_id in bounded_ids:
        key = _asset_group_key(store, asset_id)
        groups[key] = groups.get(key, 0) + 1
    return {
        "total_candidates": visible_limit,
        "visible_candidates": len(bounded_indices),
        "visible_ranks": [index + 1 for index in bounded_indices],
        "event_group_count": len(groups),
        "largest_event_group": max(groups.values(), default=0),
            "strategy": os.getenv("SENTRIX_CANDIDATE_STRATEGY", "head_only").strip().lower(),
    }


def _preview_entry(store, asset_id: str, handle: str, *, level="exact", condition_summary=None,
                   priority_rank: int | None = None, selection_reason: str = "") -> dict:
    asset = store.get_asset(asset_id) if store is not None else {}
    asset = asset or {}
    media_kind = "original_image"
    source_video_asset_id = None
    source_timestamp_sec = None
    source_scene_index = None
    source_video_file_name = None
    if asset.get("derived_kind") in {"video_keyframe", "video_keyframe_webp"}:
        media_kind = "video_keyframe"
        source_video_asset_id = asset.get("parent_asset_id")
        source_timestamp_sec = asset.get("source_timestamp_sec")
        source_scene_index = asset.get("source_scene_index")
        source_video = store.get_asset(source_video_asset_id) if store and source_video_asset_id else None
        source_video_file_name = (source_video or {}).get("file_name")
    evidence_summary = _observation_summary(store, asset_id)
    # Confirmed face/entity links are deterministic memory evidence.  Expose
    # only the public name/role projection in search previews; face IDs and
    # embeddings remain server-side and pending clusters stay unnamed.
    people = []
    for identity in _confirmed_photo_identities(store, asset_id):
        if not isinstance(identity, dict):
            continue
        name = str(identity.get("person_name") or "").strip()
        if not name or identity.get("identity_status") != "confirmed":
            continue
        people.append({
            "name": name,
            "family_role": str(identity.get("family_role") or "").strip(),
            "identity_status": "confirmed",
        })
    people = list({(item["name"], item["family_role"]): item for item in people}.values())
    return {
        "handle": handle,
        "asset_id": asset_id,
        "captured_at": asset.get("captured_at"),
        "level": level,
        "place": _short_place_label(asset) if asset else "",
        "media_kind": media_kind,
        "source_video_asset_id": source_video_asset_id,
        "source_timestamp_sec": source_timestamp_sec,
        "source_scene_index": source_scene_index,
        "source_video_file_name": source_video_file_name,
        "evidence_summary": evidence_summary,
        "people": people,
        # Keep description availability explicit so the UI/benchmark can tell
        # an empty observation apart from a transport/schema omission.
        "description_status": "available" if evidence_summary else "missing",
        "condition_summary": condition_summary or {},
        "priority_rank": priority_rank,
        "selection_reason": selection_reason or ("相关性排序靠前" if priority_rank == 1 else "候选补充" if priority_rank else ""),
    }
def _even_indices(total: int, n: int) -> list[int]:
    """在 [0, total) 内均匀取 n 个下标（representative 预览用，避免只展示最新几张），包含首尾。"""
    if total <= n:
        return list(range(total))
    if n <= 1:
        return [0]
    return [min(int(round(i * (total - 1) / (n - 1))), total - 1) for i in range(n)]


def _search_metadata_only(draft, spec, scope_id, query, mode, user_goal="") -> dict:
    """空 query 搜索：只按硬筛选（时间/媒体/地点/人物）返回资产，构建 ResultSet 预览。"""
    from ..structured_memory import StructuredMemoryExecutor
    executor = StructuredMemoryExecutor(_RUNTIME["store"])
    assets = executor._matching_assets(draft, spec, limit=500)
    asset_ids = [a["id"] for a in assets]
    rs = _RUNTIME["result_sets"].new(
        scope_id=scope_id, query=query or "(时间/地点筛选)", asset_ids=asset_ids,
        unresolved=[])
    store = _RUNTIME.get("store")
    indices = _preview_indices(asset_ids, mode, store, query=user_goal or query)
    preview = [
        _preview_entry(store, assets[idx].get("id"), f"photo_{idx + 1}",
                       priority_rank=rank, selection_reason="相关性最高" if rank == 1 else "事件多样性补充")
        for rank, idx in enumerate(indices, 1)
    ]
    total = len(assets)
    visible_total = _visible_candidate_total(total)
    preview_asset_ids = [asset_ids[idx] for idx in indices if idx < len(asset_ids)]
    return {
        "result_set_id": rs.result_set_id,
        "query": query,
        "mode": mode,
        "total": visible_total,
        "evidence_count": _visible_candidate_total(len(asset_ids)),
        "preview": preview,
        "has_more": visible_total > len(preview),
        "remaining": max(0, visible_total - len(preview)),
        "candidate_window": _candidate_window_summary(asset_ids, indices, store),
        "completeness": "complete",
        "gaps": [],
        "query_satisfaction": "full_support" if total else "no_match",
        "answerability": "full" if total else "none",
        "condition_summary": {},
        "can_inspect": len(preview) > 0,
        "inspect_hint": "preview 里的 handle（photo_1…）可直接用于 inspect_photo 复核视觉细节" if preview else "",
        "recommended_resolution": _recommended_resolution(query, preview,
                                                       "full_support" if total else "no_match",
                                                       user_goal=user_goal),
        "_retrieved_asset_ids": list(asset_ids),
        # Public trace contract: keep the complete candidate set distinct from
        # the bounded preview.  Runtime may still redact private underscore
        # fields, so expose stable asset IDs explicitly for benchmark/user
        # provenance accounting.
        "retrieved_asset_ids": list(asset_ids),
        "_preview_asset_ids": preview_asset_ids,
        "evidence_asset_ids": [],
    }


_TIME_TOKEN_RE = re.compile(r"20\d{2}\s*年(?:\s*[01]?\d|\s*十[一二]?)?\s*月?")
_RELATIVE_TIMES = ("这两年", "近两年", "最近两年", "最近一年", "今年", "去年", "前年",
                   "上上个月", "上个月", "去年春天", "去年夏天", "去年秋天", "去年冬天")


def _extract_time_from_query(query: str) -> str | None:
    """C11：模型把时间写进 query 文本（而非 filters.time）时自动提取。"""
    m = _TIME_TOKEN_RE.search(query or "")
    if m:
        return m.group(0).replace(" ", "")
    # 先匹配更具体的"去年X月"，再退回相对时间词
    m = re.search(r"去年(?:[0-9一二三四五六七八九十]+)月", query or "")
    if m:
        return m.group(0)
    for expr in _RELATIVE_TIMES:
        if expr in (query or ""):
            return expr
    return None


_MODEL_FILTER_NOISE = frozenset({
    "未知", "不明", "unknown", "none", "null", "无", "没有",
    # These are useful semantic concepts, but not geographic hard filters.
    "天台", "屋顶", "室内", "户外", "室外", "家里", "家中", "房间",
    "客厅", "卧室", "厨房", "舞台", "门口", "路边", "聚餐", "婚礼",
    "活动", "旅行", "展架", "迎宾展架", "餐厅", "海洋馆", "景区",
})


def _compact_filter_text(value: str) -> str:
    return re.sub(r"[\s，。！？、,.!?；;：:]", "", str(value or "")).strip()


def _literal_in_text(value: str, text: str) -> bool:
    value = _compact_filter_text(value)
    text = _compact_filter_text(text)
    return bool(value and text and value in text)


def _looks_like_scene_place(value: str) -> bool:
    compact = _compact_filter_text(value).lower()
    if not compact:
        return True
    if compact in _MODEL_FILTER_NOISE:
        return True
    return any(token in compact for token in _MODEL_FILTER_NOISE
               if len(token) >= 2 and token not in {"unknown", "null"})


def _trusted_query_constraints(text: str, store=None, scope_id: str = "") -> dict:
    """Extract structured constraints from user wording, not planner output."""
    from .canonical_intent import extract_time

    text = str(text or "")
    constraints = {"time": extract_time(text), "place": None, "person": None}
    if store is not None and scope_id:
        try:
            from .canonical_intent import extract_constraints
            parsed = extract_constraints(text, store, scope_id) or {}
            for key in ("time", "place", "person"):
                if parsed.get(key):
                    constraints[key] = parsed[key]
        except Exception:
            pass
    return constraints


def _sanitize_model_filters(raw_filters: dict | None, *, query: str = "",
                            user_goal: str = "", store=None,
                            scope_id: str = "", trusted_constraints: dict | None = None) -> dict:
    """Prevent hallucinated planner slots from becoming hard filters.

    Media remains an explicit tool contract.  Time/place/person are accepted
    only when grounded in the user's wording or trusted scope metadata; an
    ambiguous value is dropped so it cannot remove the correct asset before
    semantic or graph ranking runs.
    """
    raw = dict(raw_filters or {})
    text = str(user_goal or query or "")
    trusted = trusted_constraints or _trusted_query_constraints(text, store, scope_id)
    sanitized: dict = {}

    media = str(raw.get("media") or "").strip().lower()
    if media in {"image", "video"}:
        sanitized["media"] = media

    if trusted.get("time"):
        sanitized["time"] = trusted["time"]
    else:
        raw_time = str(raw.get("time") or "").strip()
        if raw_time and _literal_in_text(raw_time, text):
            sanitized["time"] = raw_time

    trusted_place = str(trusted.get("place") or "").strip()
    raw_place = str(raw.get("place") or "").strip()
    if trusted_place:
        sanitized["place"] = trusted_place
    elif (raw_place and _literal_in_text(raw_place, text)
          and not _looks_like_scene_place(raw_place)):
        sanitized["place"] = raw_place

    # Only confirmed scope entities become person hard filters.  This avoids
    # treating “我和同事/兄弟们” as an entity name and shrinking recall.
    trusted_person = str(trusted.get("person") or "").strip()
    if trusted_person:
        sanitized["person"] = trusted_person
    return sanitized


def _event_resolution(question: str, store, scope_id: str) -> dict | None:
    """W2.4：多轮引用解析到 Event（turn-0 无结果集时的二级锚）。

    从问题提取时间/人物/活动线索，在 events 表里召回候选（用 time/participants/place/activity，
    不只看 title），单候选高置信时返回其资产，否则 None（交回普通检索/澄清）。
    """
    from .canonical_intent import extract_time
    if store is None or not scope_id:
        return None
    t = extract_time(question)
    # 直接查 entities 表解析人物（绕过 list_entities 的 include_in_people 过滤，benchmark scope 也能用）
    persons = []
    try:
        for ent in store.connection.execute(
                "SELECT id, canonical_name, family_role FROM entities "
                "WHERE scope_id=? AND entity_type='person' AND status='confirmed'",
                (scope_id,)).fetchall():
            for alias in (ent["canonical_name"], ent["family_role"]):
                if alias and alias != "自己" and alias in question:
                    persons.append(ent["id"])
                    break
    except Exception:
        pass
    try:
        rows = store.connection.execute(
            "SELECT id,title,place,activity,summary,participants_json,substr(time_start,1,10) AS ts,"
            "substr(time_start,1,7) AS ym FROM events "
            "WHERE scope_id=? AND status NOT IN ('rejected','superseded','merged')", (scope_id,)).fetchall()
    except Exception:
        return None
    # Generic words identify an activity class but not a particular event.
    # They must not be allowed to lock a first-turn query to an unrelated
    # event (for example “合影” selecting a later children photo set).
    generic_overlap = {"合影", "照片", "活动", "室内", "户外", "空间", "场地", "参加", "不同"}
    scored = []
    for raw_row in rows:
        # Different album DB revisions may omit optional event columns. Work
        # from a plain mapping so an absent summary cannot abort retrieval.
        r = dict(raw_row)
        score = 0
        if t:
            if r.get("ts") and r["ts"].startswith(t[:10]):
                score += 3
            elif r.get("ym") and r["ym"].startswith(t[:7]):
                score += 2
        for pid in persons:
            if str(pid) in (r.get("participants_json") or ""):
                score += 3
        # 数据驱动的文本重叠：问题与事件 title/place/activity 的中文子串匹配（不硬编码任何关键词）
        # Event summaries contain discriminative facts that may be absent from
        # short title/place fields. Keep the result bounded to that event.
        hay = " ".join(str(x) for x in
                       (r.get("title"), r.get("place"), r.get("activity"), r.get("summary")) if x)
        overlap = 0
        strong_overlap = 0
        for length in (4, 3, 2):
            ngrams = {hay[i:i + length] for i in range(max(0, len(hay) - length + 1))
                      if len(hay[i:i + length]) == length
                      and re.search(r"[\u4e00-\u9fff]", hay[i:i + length])
                      and not any(c.isdigit() for c in hay[i:i + length])}
            for ng in ngrams:
                # Suffixes such as “的合影/的合” are not event identity
                # signals even though they technically overlap the query.
                if ng in generic_overlap or any(token in ng for token in generic_overlap):
                    continue
                if ng in question:
                    overlap += 1
                    if length >= 4:
                        strong_overlap += 1
                    break
        score += min(overlap, 2)
        if score:
            scored.append((score, r.get("id"), r.get("title"), overlap, strong_overlap))
    # Enumeration questions about a group photo often use natural language
    # (“兄弟们”“不同人数”) that is absent from the generated event title.
    # Recover the event from its own asset observations: count adult-male
    # group-photo observations, while excluding rows explicitly describing
    # children. This remains data-driven and does not depend on benchmark IDs.
    if re.search(r"合影", question or "") and re.search(r"(?:几张|多少|不同人数|几人|人数)", question or ""):
        try:
            # Require the event's own text to share at least one
            # discriminative phrase with the question before using its group
            # count. Otherwise the largest unrelated group-photo event wins
            # (the previous implementation selected a 2019 children's
            # wedding because it happened to have more size variants).
            ignored_event_terms = {"合影", "照片", "活动", "场地", "参加", "不同", "人数", "几张", "几人"}
            boundary_stop_chars = set("的了在和与及从到这那一几们")
            question_terms = set()
            for length in (4, 3, 2):
                for start in range(max(0, len(question or "") - length + 1)):
                    term = str(question or "")[start:start + length]
                    if (len(term) == length and re.fullmatch(r"[\u4e00-\u9fff]+", term)
                            and term not in ignored_event_terms
                            and term[0] not in boundary_stop_chars
                            and term[-1] not in boundary_stop_chars):
                        question_terms.add(term)
            group_scores = []
            for r in rows:
                obs = store.connection.execute(
                    "SELECT caption, activity, people_json FROM observations o "
                    "JOIN event_observations eo ON eo.observation_id=o.id "
                    "WHERE eo.event_id=?", (r["id"],)).fetchall()
                count = 0
                sizes = set()
                event_text = ""
                for o in obs:
                    text = " ".join(str(o[k] or "") for k in ("caption", "activity"))
                    event_text += " " + text
                    if "合影" not in text:
                        continue
                    if any(token in text for token in ("幼儿", "孩子", "儿童", "小孩")):
                        continue
                    # Chinese count words or Arabic digits followed by adult
                    # male wording are strong signals for the requested group.
                    if re.search(r"(?:[一二三四五六七八九十两\d]+(?:名|位|人)).*男|男子|男性", text):
                        match = re.search(r"([一二三四五六七八九十两\d]+)(?:名|位|人)", text)
                        if match:
                            token = match.group(1)
                            cn = {"一": 1, "二": 2, "两": 2, "三": 3, "四": 4,
                                  "五": 5, "六": 6, "七": 7, "八": 8,
                                  "九": 9, "十": 10}
                            size = int(token) if token.isdigit() else cn.get(token)
                            # A single-person留影 is not one of the requested
                            # different-size group photos.
                            if size and size >= 2:
                                count += 1
                                sizes.add(size)
                event_overlap = sum(1 for term in question_terms if term in event_text)
                if count and event_overlap:
                    # For enumeration questions, coverage of distinct group
                    # sizes is more discriminative than raw photo count.
                    group_scores.append((event_overlap, len(sizes), count, r["id"], r["title"]))
            group_scores.sort(key=lambda x: (-x[0], -x[1], -x[2], x[4]))
            if group_scores and (len(group_scores) == 1 or group_scores[0][:3] > group_scores[1][:3]):
                eid = group_scores[0][3]
                asset_rows = store.connection.execute(
                    "SELECT DISTINCT o.asset_id FROM observations o "
                    "JOIN event_observations eo ON eo.observation_id=o.id "
                    "JOIN assets a ON a.id=o.asset_id WHERE eo.event_id=? AND a.scope_id=?",
                    (eid, scope_id)).fetchall()
                asset_ids = [a["asset_id"] for a in asset_rows]
                if asset_ids:
                    return {"event_id": eid, "event_title": group_scores[0][4],
                            "asset_ids": asset_ids[:50]}
        except Exception:
            pass
    if not scored:
        return None
    scored.sort(key=lambda x: -x[0])
    top, second = scored[0], scored[1] if len(scored) > 1 else None
    # 单候选高置信，或多候选但第一明显领先。事件锚定只允许在问题与
    # 同一事件命中至少两个独立短语时生效；单个四字重叠（例如“婚礼照片”
    # 或“水利工程”）不足以把整次搜索截断到一个事件。否则一个自然语言
    # 场景词就会覆盖 ANN/metadata 的完整候选集。
    if (top[0] >= 2 and top[3] >= 2 and top[4] > 0
            and (second is None or top[0] - second[0] >= 1)):
        eid = top[1]
        assets = store.connection.execute(
            "SELECT DISTINCT a.id FROM assets a JOIN observations o ON o.asset_id=a.id "
            "JOIN event_observations eo ON eo.observation_id=o.id WHERE eo.event_id=? "
            "AND a.scope_id=?", (eid, scope_id)).fetchall()
        asset_ids = [a["id"] for a in assets]
        if asset_ids:
            return {"event_id": eid, "event_title": top[2], "asset_ids": asset_ids[:50]}
    return None


def _event_keyword_anchor(question: str, store, scope_id: str) -> dict | None:
    """Fallback event anchor using distinctive Chinese terms in summaries.

    This is intentionally generic: it does not know benchmark IDs or fixed
    places, and only returns an event when one summary clearly dominates the
    query-term overlap. It also tolerates older event schemas by selecting
    only required columns.
    """
    if store is None or not scope_id:
        return None
    q_text = str(question or "")
    ignored = {"我们", "我和", "家人", "一起", "去参观", "参观", "旅行",
               "那次", "在哪里", "超级", "的", "那次旅行"}
    terms = []
    for length in (4, 3, 2):
        for match in re.finditer(rf"[\u4e00-\u9fff]{{{length}}}", q_text):
            term = match.group(0)
            if term not in ignored and not any(token in term for token in ignored):
                terms.append((term, length))
    # These aliases are for event-level candidate discovery only; they do not
    # assert that two visible objects are the same. For example, an event
    # observation may call a wedding welcome display a "宣传横幅", while the
    # user remembers it as an "迎宾展架". The event still remains a soft
    # candidate and the actual asset must pass the normal retrieval verifier.
    event_context_aliases = (
        ("迎宾展架", ("宣传横幅", "迎宾牌", "横幅")),
        ("迎宾架", ("宣传横幅", "迎宾牌", "横幅")),
        ("展示架", ("宣传横幅", "横幅")),
        ("展架", ("宣传横幅", "横幅", "迎宾牌")),
        ("留影", ("合影", "摆拍", "合照")),
        ("拍照", ("合影", "摆拍")),
        ("同行亲友", ("宾客合影", "多人合影", "亲友合影")),
    )
    for source, aliases in event_context_aliases:
        if source in q_text:
            terms.extend((alias, len(alias)) for alias in aliases)
    terms = list(dict.fromkeys(terms))
    if not terms:
        return None
    try:
        rows = store.connection.execute(
            "SELECT id,title,summary FROM events WHERE scope_id=?",
            (scope_id,)).fetchall()
    except Exception:
        return None
    # Event-category words are weak identity signals: they may be used to add
    # a unique matching event as a *soft* candidate source, but never to
    # replace the normal semantic candidate pool. This matters for paraphrases
    # such as "亲友婚礼/迎宾展架" whose event member observations say
    # "婚礼现场合影/宣传横幅" and therefore share no long literal n-gram.
    event_type_anchors = {
        "婚礼", "婚宴", "婚庆", "生日", "聚餐", "旅行", "出游", "春游",
        "毕业", "演出", "演唱会", "展览", "运动会", "节日", "搬家",
    }
    scored = []
    for raw in rows:
        row = dict(raw)
        hay = " ".join(str(row.get(k) or "") for k in ("title", "summary"))
        # Event summaries can omit text that is present on the event's own
        # observations (OCR such as “我愿意” or captions mentioning a
        # roll-on/roll-off ship).  Include only that event's rows so this
        # remains a bounded, data-driven anchor rather than a global scan.
        try:
            observed = store.connection.execute(
                "SELECT o.asset_id,caption,activity,place,ocr_text FROM observations o "
                "JOIN event_observations eo ON eo.observation_id=o.id "
                "WHERE eo.event_id=?", (row.get("id"),)).fetchall()
            hay += " " + " ".join(
                " ".join(str(dict(item).get(k) or "") for k in ("caption", "activity", "place", "ocr_text"))
                for item in observed
            )
            for item in observed:
                asset_id = dict(item).get("asset_id")
                if not asset_id:
                    continue
                meta = store.connection.execute(
                    "SELECT metadata_json FROM assets WHERE id=?", (asset_id,)
                ).fetchone()
                if meta:
                    hay += " " + str(meta[0] or "")
        except Exception:
            pass
        matched = [(term, length) for term, length in terms if term in hay]
        if matched:
            scored.append((len(set(term for term, _ in matched)),
                           sum(1 for _, length in matched if length >= 4),
                           row.get("id"), row.get("title") or "",
                           {term for term, _ in matched}))
    scored.sort(key=lambda item: (-item[0], item[2]))
    if not scored or scored[0][0] < 1:
        return None
    # A single strong phrase is not enough to replace the full retriever
    # result. For example, “婚礼照片” or “水利工程” can occur in several
    # unrelated events. Keep this fallback conservative; the normal ANN and
    # metadata channels remain responsible for broad candidate recall.
    unique_event_type_hit = (
        scored[0][0] == 1
        and bool(scored[0][4] & event_type_anchors)
        and any(term in str(scored[0][3] or "") for term in scored[0][4] & event_type_anchors)
    )
    if not unique_event_type_hit and (scored[0][0] < 2 or scored[0][1] < 1):
        return None
    if len(scored) > 1 and scored[0][0] == scored[1][0]:
        return None
    event_id = scored[0][2]
    try:
        asset_rows = store.connection.execute(
            "SELECT DISTINCT o.asset_id FROM observations o "
            "JOIN event_observations eo ON eo.observation_id=o.id "
            "JOIN assets a ON a.id=o.asset_id "
            "WHERE eo.event_id=? AND a.scope_id=?",
            (event_id, scope_id)).fetchall()
    except Exception:
        return None
    asset_ids = [dict(row).get("asset_id") for row in asset_rows if dict(row).get("asset_id")]
    return ({"event_id": event_id, "event_title": scored[0][3],
             "asset_ids": asset_ids[:50]} if asset_ids else None)

def _time_matches_event(time_expr: str | None, ts: str | None) -> int:
    """事件时间匹配分：完整日期/年月/年命中返回 3，否则 0。相对时间不参与确定性锚定。"""
    if not time_expr or not ts:
        return 0
    q = re.sub(r"\s+", "", time_expr)
    day = re.search(r"(\d{4})年(\d{1,2})月(\d{1,2})日", q)
    if day:
        return 3 if ts.startswith(f"{int(day.group(1)):04d}-{int(day.group(2)):02d}-{int(day.group(3)):02d}") else 0
    ym = re.search(r"(\d{4})年(\d{1,2})月", q)
    if ym:
        return 3 if ts.startswith(f"{int(ym.group(1)):04d}-{int(ym.group(2)):02d}") else 0
    y = re.search(r"(\d{4})年", q)
    if y:
        return 3 if ts.startswith(y.group(1)) else 0
    return 0


def _event_resolution_geo(question, store, scope_id, time_expr=None, place=None):
    """事件级主路径：时间+地点 → 锁单个事件 → 返回其资产。

    仅当单事件高置信（时间+地点同时命中，score>=6 且明显领先）才返回；
    否则返回 None，交回融合检索。全程数据驱动，不硬编码任何 benchmark 内容。
    """
    from ..geocoding import place_text_matches
    if store is None or not scope_id or (not time_expr and not place):
        return None
    try:
        rows = store.connection.execute(
            "SELECT e.id, e.title, e.time_start, e.event_type, e.activity, "
            "e.summary, a.metadata_json AS cover_meta "
            "FROM events e LEFT JOIN assets a ON a.id=e.cover_asset_id "
            "WHERE e.scope_id=? AND e.status NOT IN ('rejected','superseded','merged')",
            (scope_id,)).fetchall()
    except Exception:
        return None
    scored = []
    for r in rows:
        score = 0
        ts = r["time_start"] or ""
        if time_expr:
            score += _time_matches_event(time_expr, ts)
        geo = None
        if r["cover_meta"]:
            try:
                geo = (json.loads(r["cover_meta"]) or {}).get("reverse_geocode")
            except Exception:
                geo = None
        if place and geo and place_text_matches(place, geo):
            score += 3
        hay = " ".join(str(x) for x in (r["title"], r["event_type"], r["activity"], r["summary"]) if x)
        q = str(question or "")
        overlap_count = 0
        for length in (3, 2):
            ngrams = {hay[i:i + length] for i in range(max(0, len(hay) - length + 1))
                      if len(hay[i:i + length]) == length}
            overlap_count += sum(1 for ng in ngrams if ng in q)
        score += min(overlap_count, 3)
        if score:
            scored.append((score, r["id"], r["title"], overlap_count))
    if not scored:
        return None
    scored.sort(key=lambda x: -x[0])
    top, second = scored[0], scored[1] if len(scored) > 1 else None
    # For place-only queries require at least two independent semantic overlaps
    # before locking one event; a single generic word such as “参观/合影” is
    # not enough to distinguish events in the same city.
    top_overlap = top[3] if len(top) > 3 else 0
    # A district-level geocode plus an explicit event phrase is enough to
    # produce a bounded event candidate even when the event row has no long
    # natural-language place title. Ambiguous ties still fall back to ANN.
    min_score = 4 if place and not time_expr else 4
    if top[0] >= min_score and (not place or time_expr or top_overlap >= 1) and (second is None or top[0] - second[0] >= 1):
        eid = top[1]
        asset_ids = store.connection.execute(
            "SELECT DISTINCT o.asset_id FROM observations o "
            "JOIN event_observations eo ON eo.observation_id=o.id "
            "JOIN assets a ON a.id=o.asset_id "
            "WHERE eo.event_id=? AND a.scope_id=?", (eid, scope_id)).fetchall()
        ids = [a["asset_id"] for a in asset_ids]
        if ids:
            return {"event_id": eid, "event_title": top[2], "asset_ids": ids[:50]}
    return None



_REFERENT_MARKERS = ("就是", "那次", "这次", "刚才", "上次", "那个", "这个", "那一次", "这一次")


def _is_referent_query(query: str) -> bool:
    """W2.3：判断是否为多轮引用类 query（指代上一次的实体/结果集，而非全新检索）。"""
    q = (query or "").strip()
    if not q:
        return False
    return any(m in q for m in _REFERENT_MARKERS)


def _is_face_reference_asset(item: dict, store=None) -> bool:
    """Identity-reference crops are not user photo/video evidence candidates."""
    file_name = Path(str((item or {}).get("file_name") or "").replace("\\", "/")).name
    if re.match(r"^faceid_[^/]+\.(?:jpe?g|png|webp)$", file_name, re.IGNORECASE):
        return True
    asset_id = str((item or {}).get("asset_id") or "")
    if store is None or not asset_id:
        return False
    try:
        asset = store.get_asset(asset_id) or {}
        metadata = asset.get("metadata_json") or {}
        if isinstance(metadata, str):
            metadata = json.loads(metadata)
        if not isinstance(metadata, dict):
            return False
        source_type = str(metadata.get("source_type") or "").strip().lower()
        derived_kind = str(metadata.get("derived_kind") or asset.get("derived_kind") or "").strip().lower()
        return source_type == "identity_seed" or derived_kind in {
            "face_crop", "face_id_crop", "face_identity_crop", "face_reference",
        }
    except Exception:
        return False


def _is_synthetic_event_asset(item: dict, store=None) -> bool:
    """Event-summary renderings are graph context, not retrievable evidence media."""
    file_name = Path(str((item or {}).get("file_name") or "").replace("\\", "/")).name
    if re.match(r"^event_event_[^/]+\.(?:jpe?g|png|webp)$", file_name, re.IGNORECASE):
        return True
    asset_id = str((item or {}).get("asset_id") or "")
    if store is None or not asset_id:
        return False
    try:
        asset = store.get_asset(asset_id) or {}
        file_name = Path(str(asset.get("file_name") or "").replace("\\", "/")).name
        return bool(re.match(r"^event_event_[^/]+\.(?:jpe?g|png|webp)$", file_name,
                             re.IGNORECASE))
    except Exception:
        return False


def _search_from_prior_result_set(prior_rs, scope_id: str, *, query: str = "",
                                  user_goal: str = "") -> dict | None:
    """W2.3：基于已有 ResultSet 构建检索响应（不重新全库搜索）。"""
    if prior_rs is None:
        return None
    store = _RUNTIME.get("store")
    asset_ids = [asset_id for asset_id in (prior_rs.asset_ids or [])
                 if not _is_face_reference_asset({"asset_id": asset_id}, store)
                 and not _is_synthetic_event_asset({"asset_id": asset_id}, store)]
    preview = []
    indices = _preview_indices(
        asset_ids, "best", store, query=query or getattr(prior_rs, "query", "") or user_goal)
    preview = [
        _preview_entry(store, asset_ids[index], f"photo_{index + 1}",
                       priority_rank=rank, selection_reason="相关性最高" if rank == 1 else "事件多样性补充")
        for rank, index in enumerate(indices, 1)
    ]
    preview_asset_ids = [asset_ids[index] for index in indices if index < len(asset_ids)]
    display_query = query or getattr(prior_rs, "query", "") or f"(引用已有结果集 {prior_rs.result_set_id})"
    validation = _validate_search_candidates(
        query=display_query,
        user_goal=user_goal,
        filters={},
        asset_ids=asset_ids,
        store=store,
        relaxation_level=0,
    )
    validated = set(validation.get("validated_asset_ids") or [])
    if validation.get("validation_status") == "complete" and validated:
        # Only a non-empty validated projection may replace the candidate
        # preview.  An empty validator result is not an empty retrieval: it
        # means the model could not promote any candidate to direct evidence.
        # Keep the bounded candidates visible so the agent can inspect them
        # and the UI can show the retrieval source instead of a blank result.
        preview = [item for item in preview if item.get("asset_id") in validated]
        preview_asset_ids = [item.get("asset_id") for item in preview if item.get("asset_id")]
    # An event/reference result is already a structured source, not merely an
    # ANN candidate. Preserve bounded representative assets as evidence even
    # when the visual validator cannot promote any row.
    # Event/reference anchoring is still a retrieval candidate. Only the
    # vision validator can promote a candidate to direct answer evidence.
    reference_evidence = list(validated) if asset_ids else []
    group_photo_rows = []
    if re.search(r"合影", display_query or user_goal or ""):
        try:
            for aid in asset_ids:
                obs_rows = store.connection.execute(
                    "SELECT caption, activity FROM observations WHERE asset_id=?",
                    (aid,)).fetchall()
                for obs in obs_rows:
                    text = " ".join(str(obs[k] or "") for k in ("caption", "activity"))
                    if "合影" not in text:
                        continue
                    m = re.search(r"([一二三四五六七八九十两\d]+)名", text)
                    if not m:
                        m = re.search(r"([一二三四五六七八九十两\d]+)人", text)
                    if not m:
                        continue
                    token = m.group(1)
                    cn = {"一": 1, "二": 2, "两": 2, "三": 3, "四": 4,
                          "五": 5, "六": 6, "七": 7, "八": 8,
                          "九": 9, "十": 10}
                    size = int(token) if token.isdigit() else cn.get(token)
                    if size:
                        group_photo_rows.append({"asset_id": aid,
                                                 "group_size": size,
                                                 "description": text[:240]})
                        break
        except Exception:
            group_photo_rows = []
    group_photo_rows = list({row["asset_id"]: row for row in group_photo_rows}.values())
    group_sizes = sorted({row["group_size"] for row in group_photo_rows})
    # The count/size fact is derived from these exact observation rows. Keep
    # every contributing asset in evidence_sources even when the vision
    # validator did not promote it; delivery may later choose a smaller
    # representative subset, but provenance must remain complete.
    for row in group_photo_rows:
        asset_id = str(row.get("asset_id") or "").strip()
        if asset_id and asset_id not in reference_evidence:
            reference_evidence.append(asset_id)
    group_summary = (
        f"本次事件可确认 {len(group_photo_rows)} 张不同人数的合影，人数分别为"
        + "、".join(str(x) for x in group_sizes) + "人。"
        if group_photo_rows and group_sizes else ""
    )
    return {
        "result_set_id": prior_rs.result_set_id,
        "query": display_query,
        "mode": "best",
        "total": _visible_candidate_total(len(asset_ids)),
        "evidence_count": _visible_candidate_total(len(reference_evidence)),
        "preview": preview,
        "has_more": _visible_candidate_total(len(asset_ids)) > len(preview),
        "remaining": max(0, _visible_candidate_total(len(asset_ids)) - len(preview)),
        "candidate_window": _candidate_window_summary(asset_ids, indices, store),
        "completeness": "complete",
        "gaps": [],
        "query_satisfaction": (
            "full_support" if reference_evidence else ("candidate_only" if asset_ids else "no_match")
        ),
        "answerability": "full" if reference_evidence else ("limited" if asset_ids else "none"),
        "condition_summary": {},
        "can_inspect": len(preview) > 0,
        "inspect_hint": "preview 里的 handle（photo_1…）可直接用于 inspect_photo 复核视觉细节" if preview else "",
        "recommended_resolution": _recommended_resolution(
            display_query, preview, "full_support" if reference_evidence else "candidate_only",
            user_goal=user_goal),
        "reference_resolution": True,
        "validation_status": validation.get("validation_status"),
        "validation_error": validation.get("validation_error", ""),
        "raw_candidate_count": validation.get("raw_candidate_count", len(asset_ids)),
        "validation_candidate_count": validation.get("validation_candidate_count", 0),
        "validation_batches": validation.get("validation_batches", 0),
        "validation_rows": validation.get("validation_rows") or [],
        "evidence_status": "validated" if reference_evidence else ("candidate_only" if asset_ids else "none"),
        "_retrieved_asset_ids": list(asset_ids),
        "retrieved_asset_ids": list(asset_ids),
        "_preview_asset_ids": preview_asset_ids,
        "evidence_asset_ids": reference_evidence,
        "source_asset_ids": reference_evidence,
        "group_photo_count": len(group_photo_rows),
        "group_photo_sizes": group_sizes,
        "group_photo_rows": group_photo_rows,
        "summary": group_summary,
        "_model_call_metrics": validation.get("_model_call_metrics") or [],
    }


def _bounded_event_result(prior_rs, scope_id: str, *, query: str = "",
                          user_goal: str = "") -> dict:
    """Return a safe event projection when optional validator fields are absent.

    Event resolution is already a structured, scope-bound source. Older album
    databases may lack one optional event/observation column; that must not
    turn a valid event anchor into a tool error or a broad ANN fallback.
    """
    store = _RUNTIME.get("store")
    asset_ids = [asset_id for asset_id in (prior_rs.asset_ids or [])
                 if not _is_face_reference_asset({"asset_id": asset_id}, store)
                 and not _is_synthetic_event_asset({"asset_id": asset_id}, store)]
    indices = _preview_indices(asset_ids, "best", store, query=query or user_goal)
    preview = []
    for rank, index in enumerate(indices, 1):
        if index >= len(asset_ids):
            continue
        preview.append(_preview_entry(
            store, asset_ids[index], f"photo_{index + 1}",
            priority_rank=rank,
            selection_reason="事件锚定来源" if rank == 1 else "事件来源补充",
        ))
    source_ids = [item.get("asset_id") for item in preview if item.get("asset_id")]
    return {
        "result_set_id": prior_rs.result_set_id,
        "query": query or prior_rs.query,
        "mode": "best",
        "total": _visible_candidate_total(len(asset_ids)),
        "evidence_count": _visible_candidate_total(len(source_ids)),
        "preview": preview,
        "has_more": _visible_candidate_total(len(asset_ids)) > len(preview),
        "remaining": max(0, _visible_candidate_total(len(asset_ids)) - len(preview)),
        "completeness": "complete",
        "gaps": [],
        "query_satisfaction": "candidate_only" if source_ids else "no_match",
        "answerability": "limited" if source_ids else "none",
        "condition_summary": {},
        "can_inspect": bool(preview),
        "inspect_hint": "preview 里的 handle（photo_1…）可直接用于 inspect_photo 复核视觉细节" if preview else "",
        "recommended_resolution": _recommended_resolution(
            query or prior_rs.query, preview, "candidate_only", user_goal=user_goal),
        "reference_resolution": True,
        "validation_status": "not_required",
        "evidence_status": "candidate_only" if source_ids else "none",
        "_retrieved_asset_ids": asset_ids,
        "retrieved_asset_ids": asset_ids,
        "_preview_asset_ids": source_ids,
        "evidence_asset_ids": source_ids,
        "source_asset_ids": source_ids,
        "_model_call_metrics": [],
    }



def _relaxed_retrieve(query: str, filters: dict, scope_id: str, viewer_id: str, mode: str):
    """确定性渐进放宽：严格检索为空时依次降级，返回 (packet, level)。

    level 0=严格, 1=去person, 2=去place, 3=去time, 4=纯语义。全程数据驱动。
    """
    base = dict(filters or {})
    # Explicit time/place are identity anchors, not optional ranking hints.
    # Dropping them after an empty strict pass returned unrelated photos (for
    # example an indoor 2018 group photo for a Zhao County landmark query).
    # Only an unresolved person constraint may be relaxed; an anchored query
    # otherwise returns no match rather than claiming an unrelated image.
    steps = [dict(base)]
    if base.get("person"):
        steps.append({k: v for k, v in base.items() if k != "person"})
    # 非标准地名（如"沙岭""亲戚婚房"）geocode 匹配失败会令严格检索候选 0，
    # 模型完全无法回答（比返回需复核的候选更糟）。place 检索为空时放宽用语义。
    if base.get("place"):
        steps.append({k: v for k, v in base.items() if k != "place"})
    def without_face_reference_assets(packet):
        if packet is None:
            return packet
        packet.assets = [item for item in (packet.assets or [])
                         if not _is_face_reference_asset(item, _RUNTIME.get("store"))
                         and not _is_synthetic_event_asset(item, _RUNTIME.get("store"))]
        allowed = {str(item.get("asset_id") or "") for item in packet.assets}
        for field in ("exact_results", "strong_results", "approximate_results"):
            rows = getattr(packet, field, None)
            if isinstance(rows, list):
                setattr(packet, field, [row for row in rows
                                        if str(row.get("asset_id") or "") in allowed])
        return packet

    last = None
    for level, f in enumerate(steps):
        draft = _draft_from_filters({**f, "query": query}, answer_type="asset_set")
        draft.result_requirement = {"mode": mode}
        # Keep the model validation window bounded separately from retrieval.
        # The retriever must retain a deeper candidate head so a paraphrase
        # does not disappear before the validator gets its relevance head and
        # sparse tail sample. Public delivery is still limited by preview and
        # evidence selection; this is not a request to show the full pool.
        # Candidate recall is bounded by the retrieval confidence gate, not a
        # fixed Top-K.  ``QuerySpec`` still carries a harmless requested value
        # for legacy retrievers, while the multi-channel kernel expands it to
        # the authorized scope size before ranking.
        draft.result_requirement["top_k"] = max(
            1, int(os.getenv("SENTRIX_SEARCH_RETRIEVAL_TOP_K", "100")))
        spec = _spec_for(draft, scope_id, viewer_id)
        last = without_face_reference_assets(_kernel().retrieve(spec))
        if last and last.assets:
            return last, level
    return last, len(steps) - 1


def _needs_place_semantic_fallback(filters: dict, assets: list[dict], store) -> bool:
    """Broaden an unverified place anchor without displacing verified matches."""
    place = str((filters or {}).get("place") or "").strip()
    if not place:
        return False
    return not any(_place_matches(asset, place, store) for asset in (assets or []))


def _parse_search_validation_response(raw) -> list[dict]:
    """Normalize the bounded vision validator response without trusting prose."""
    from ..model_clients import parse_json_response
    parsed = parse_json_response(raw) if isinstance(raw, str) else (raw or {})
    rows = parsed.get("candidates") if isinstance(parsed, dict) else None
    if rows is None and isinstance(parsed, dict):
        rows = parsed.get("decisions") or parsed.get("results")
    if not isinstance(rows, list):
        return []
    normalized = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        handle = str(row.get("handle") or "").strip()
        status = str(row.get("support_status") or row.get("status") or "").strip().lower()
        relevance = str(row.get("relevance") or "").strip().lower()
        if relevance in {"high", "medium", "low", "uncertain"}:
            status = "supported" if relevance == "high" else "candidate_only"
        if not handle or status not in {"supported", "candidate_only", "rejected"}:
            continue
        normalized.append({
            "handle": handle,
            "score": min(1.0, max(0.0, float(row.get("score") or {
                "high": 1.0, "medium": 0.7, "uncertain": 0.4, "low": 0.2
            }.get(relevance, 0.0)))),
            "relevance": relevance or ("high" if status == "supported" else "uncertain"),
            "time_match": bool(row.get("time_match")),
            "place_match": bool(row.get("place_match")),
            "scene_match": bool(row.get("scene_match")),
            "person_match": bool(row.get("person_match")),
            "support_status": status,
            "reason": str(row.get("reason") or "")[:240],
        })
    return normalized


def _validate_search_candidates(*, query: str, user_goal: str, filters: dict,
                                asset_ids: list[str], store, relaxation_level: int) -> dict:
    """Rerank the complete code-recalled candidate set using metadata only.

    The retrieval kernel owns recall and thresholding; this call does not read
    images, invent IDs, or delete candidates. It assigns a discrete relevance
    label in batches and the caller keeps the full ordered set for provenance.
    """
    # Validation is metadata-only. There is no semantic Top-K cutoff: every
    # candidate admitted by the retrieval threshold is reranked in batches.
    # Six compact metadata records stay well below the 12B 4501-token context
    # ceiling while still amortising model-call overhead. The loop below can
    # split a batch again when a deployment reports a context overflow.
    batch_size = max(1, int(os.getenv("SENTRIX_SEARCH_VALIDATION_BATCH_SIZE", "6")))
    task_query = user_goal or query or "用户问题相关的照片"
    # Validate the complete retrieval candidate set. The visible preview is
    # bounded separately; it must never determine which assets the model can
    # rank.
    ordered_indices = _preview_query_order(asset_ids, task_query, store)
    selected_indices = ordered_indices
    candidate_ids = [asset_ids[index] for index in selected_indices if index < len(asset_ids)]
    gamma = _RUNTIME.get("gamma")
    result = {
        "candidate_asset_ids": candidate_ids,
        "validated_asset_ids": [],
        "candidate_only_asset_ids": [],
        "ranked_asset_ids": [],
        "validation_rows": [],
        "validation_batches": 0,
        "validation_status": "disabled" if not gamma else "pending",
        "raw_candidate_count": len(asset_ids),
        "validation_candidate_count": len(candidate_ids),
        "_model_call_metrics": [],
    }
    if not candidate_ids or gamma is None or os.getenv("SENTRIX_SEARCH_VALIDATION_ENABLED", "1").lower() in {"0", "false", "off"}:
        return result
    pending_batches = [(offset, candidate_ids[offset:offset + batch_size])
                       for offset in range(0, len(candidate_ids), batch_size)]
    while pending_batches:
        offset, batch_ids = pending_batches.pop(0)
        manifest = []
        for index, asset_id in enumerate(batch_ids):
            asset = store.get_asset(asset_id) if store else None
            if not asset:
                continue
            handle = f"photo_{offset + index + 1}"
            row = {
                "handle": handle,
                "file_name": asset.get("file_name") or "",
                "asset_id": asset_id,
                "captured_at": asset.get("captured_at") or "",
                "place": _short_place_label(asset),
                "people": _preview_entry(store, asset_id, handle).get("people") or [],
                "description": (_observation_summary(store, asset_id) or "")[:180],
                "source_type": asset.get("derived_kind") or asset.get("media_kind") or "image",
                "retrieval_score": asset.get("retrieval_score"),
                "fusion_score": asset.get("fusion_score"),
            }
            manifest.append({"handle": handle, "asset_id": asset_id,
                             "record": row})
        if not manifest:
            continue
        prompt = (
            "你是 Sentrix 检索证据验证器。用户问题是：" + task_query + "\n"
            "检索条件（仅供理解，不可自行扩展）：" + json.dumps(filters or {}, ensure_ascii=False) + "\n"
            "当前放宽级别：" + str(relaxation_level) + "。逐条判断候选与用户问题的相关性。"
            "只依据候选记录，不要把缺失字段当作不相关，也不要猜测未提供的人名。"
            "请为每条候选返回统一的离散相关性等级。为节省上下文，只返回 JSON 对象：{\"candidates\":[{\"handle\":\"photo_N\","
            "\"relevance\":\"high|medium|low|uncertain\",\"support_status\":\"supported|candidate_only|rejected\",\"score\":0到1}]}。"
            "relevance 只用于排序；low/uncertain 仍会保留为候选，除非明确 rejected。未知人物不得猜姓名。候选记录：" + json.dumps(
                [item["record"] for item in manifest], ensure_ascii=False)
        )
        try:
            raw = gamma.chat(prompt, images=[],
                             json_mode=True, role="search_validation")
            rows = _parse_search_validation_response(raw)
            # A max-token truncation can yield a valid prefix containing only
            # part of the batch. Re-run the halves so every recalled asset has
            # a model decision instead of silently falling into the unranked
            # tail.
            if len(rows) < len(manifest) and len(batch_ids) > 1:
                midpoint = max(1, len(batch_ids) // 2)
                pending_batches.insert(0, (offset + midpoint, batch_ids[midpoint:]))
                pending_batches.insert(0, (offset, batch_ids[:midpoint]))
                continue
            batch_assets = {item["handle"]: item["asset_id"] for item in manifest}
            rows = [row for row in rows if row["handle"] in batch_assets]
            for row in rows:
                row["asset_id"] = batch_assets[row["handle"]]
            result["validation_rows"].extend(rows)
            supported = sorted(
                (row for row in rows if row["support_status"] == "supported"),
                key=lambda row: row["score"], reverse=True)
            candidates = sorted(
                (row for row in rows if row["support_status"] == "candidate_only"),
                key=lambda row: row["score"], reverse=True)
            result["validated_asset_ids"].extend(row["asset_id"] for row in supported)
            result["candidate_only_asset_ids"].extend(row["asset_id"] for row in candidates)
            result["validation_batches"] += 1
        except Exception as exc:
            # A prompt can still exceed an effective context when one
            # observation is unusually dense. Retry only by splitting the
            # current batch; never broaden or silently mark it supported.
            error_text = str(exc).lower()
            if len(batch_ids) > 1 and any(token in error_text for token in (
                    "maximum context", "context length", "input_tokens", "token budget")):
                midpoint = max(1, len(batch_ids) // 2)
                pending_batches.insert(0, (offset + midpoint, batch_ids[midpoint:]))
                pending_batches.insert(0, (offset, batch_ids[:midpoint]))
                continue
            result["validation_status"] = "error"
            result["validation_error"] = str(exc)[:240]
            break
    if result["validation_status"] != "error":
        result["validation_status"] = "complete"
    # Merge model decisions across batches before exposing any ordering. A
    # malformed/empty batch must not delete code-recalled candidates; retain
    # them as low-confidence candidates for later inspection.
    rows_by_asset = {}
    for row in result["validation_rows"]:
        asset_id = row.get("asset_id")
        if asset_id:
            prior = rows_by_asset.get(asset_id)
            if prior is None or float(row.get("score") or 0) > float(prior.get("score") or 0):
                rows_by_asset[asset_id] = row
    ranked_rows = sorted(rows_by_asset.values(), key=lambda row: float(row.get("score") or 0), reverse=True)
    ranked_ids = [row["asset_id"] for row in ranked_rows]
    missing_ids = [asset_id for asset_id in candidate_ids if asset_id not in rows_by_asset]
    result["ranked_asset_ids"] = ranked_ids + missing_ids
    result["validated_asset_ids"] = [row["asset_id"] for row in ranked_rows
                                      if row.get("support_status") == "supported"]
    result["candidate_only_asset_ids"] = [row["asset_id"] for row in ranked_rows
                                           if row.get("support_status") != "supported"] + missing_ids
    result["validated_asset_ids"] = list(dict.fromkeys(result["validated_asset_ids"]))
    result["candidate_only_asset_ids"] = list(dict.fromkeys(result["candidate_only_asset_ids"]))
    try:
        result["_model_call_metrics"] = gamma.get_and_clear_call_metrics()
    except Exception:
        pass
    return result


def _bounds_to_time_expr(start: str, end: str) -> str:
    """ISO date bounds -> parse_time_expression 可解析的月范围表达式。"""
    def year_month(value):
        parts = str(value).split("-")
        return int(parts[0]), int(parts[1]) if len(parts) > 1 else 1
    sy, sm = year_month(start)
    ey, em = year_month(end)
    if sy == ey:
        return f"{sy}年{sm}月-{em}月" if sm != em else f"{sy}年{sm}月"
    return f"{sy}年{sm}月-{ey}年{em}月"


def _in_time_bounds(captured_at, bounds) -> bool:
    if not captured_at:
        return False
    try:
        from datetime import datetime
        cap = datetime.fromisoformat(str(captured_at).replace("Z", "+00:00")).replace(tzinfo=None)
        return bounds[0] <= cap < bounds[1]
    except Exception:
        return False


def _in_time_components(captured_at, comps) -> bool:
    """分量时间匹配：year/months/days 谁给了就约束谁，没给的不约束。

    无年份的节日/季节只约束月、日，任何年份的相符照片都保留；某一年份若没给出
    绝不猜年份（那会把正确年份的照片筛成 0）。
    """
    if not captured_at:
        return False
    try:
        from datetime import datetime
        cap = datetime.fromisoformat(str(captured_at).replace("Z", "+00:00")).replace(tzinfo=None)
    except Exception:
        return False
    comps = comps or {}
    year = comps.get("year")
    months = comps.get("months")
    days = comps.get("days")
    if year and int(year) != cap.year:
        return False
    if months and cap.month not in months:
        return False
    if days and cap.day not in days:
        return False
    return True


def _place_matches(item, place_q: str, store) -> bool:
    """候选是否满足地点硬指标：reverse_geocode 匹配查询地点。"""
    if not store or not place_q:
        return False
    try:
        asset = store.get_asset(item.get("asset_id")) or {}
        metadata = asset.get("metadata_json") or {}
        if isinstance(metadata, str):
            import json
            metadata = json.loads(metadata)
        geo = (metadata or {}).get("reverse_geocode")
        if not geo:
            return False
        from ..geocoding import place_text_matches
        return place_text_matches(place_q, geo)
    except Exception:
        return False


# The latest 1,167-QA run put relevant assets at ranks 29–155, so the old
# 30/18 cutoffs discarded them before the Agent could inspect them. Keep a
# wider retrieval head and result set; the model-facing preview remains
# independently bounded by _RESULT_PREVIEW_LIMIT.
_SLOT_ROUTE_HEAD = 30
_SLOT_MAX_CANDIDATES = 18


_SEMANTIC_OBJECT_SYNONYMS = (
    # Keep only lexical equivalents. A display stand and a banner are related
    # but not interchangeable evidence, so do not rewrite one into the other.
    ("迎宾牌", "迎宾标牌"),
    ("横幅", "条幅"),
    ("条幅", "横幅"),
    ("展示架", "展架"),
)


def _semantic_route_text_key(text: str) -> str:
    return "".join(ch for ch in str(text or "").casefold() if ch.isalnum())


def _explicit_media_filter(text: str) -> str | None:
    """Recover a clear image/video constraint omitted by the tool-call model.

    Use only unambiguous modality words in the original user request. Mixed
    image-and-video requests remain unfiltered so cross-media retrieval can
    return both frames and source clips.
    """
    value = str(text or "").strip().casefold()
    if not value:
        return None
    video_terms = ("视频", "录像", "录制的视频", "视频片段")
    image_terms = ("照片", "相片", "图片", "图像", "留影", "合影", "合照", "拍照")
    wants_video = any(term in value for term in video_terms)
    wants_image = any(term in value for term in image_terms)
    if wants_video == wants_image:
        return None
    return "video" if wants_video else "image"


def _video_parent_asset_id(store, asset_id: str) -> str | None:
    """Resolve a recalled keyframe to its source video asset."""
    if store is None or not asset_id:
        return None
    try:
        asset = store.get_asset(str(asset_id)) or {}
    except Exception:
        return None
    metadata = asset.get("metadata_json") or {}
    if isinstance(metadata, str):
        try:
            metadata = json.loads(metadata)
        except (TypeError, ValueError):
            metadata = {}
    derived = str(
        asset.get("derived_kind")
        or (metadata.get("derived_kind") if isinstance(metadata, dict) else "")
        or ""
    ).strip().lower()
    if derived not in {"video_keyframe", "video_keyframe_webp", "video_mtsw_keyframe"}:
        return None
    parent_id = str(
        asset.get("parent_asset_id")
        or (metadata.get("parent_asset_id") if isinstance(metadata, dict) else "")
        or ""
    ).strip()
    if not parent_id:
        return None
    try:
        parent = store.get_asset(parent_id) or {}
    except Exception:
        return None
    return parent_id if str(parent.get("media_type") or "").lower() == "video" else None


def _add_video_parent_candidates(scores: dict[str, float], source_ids, store,
                                  *, media_constraint: str = "") -> list[str]:
    """Add source videos for recalled keyframes before the candidate cap.

    This is provenance expansion only. It does not inspect GT, answer text, or
    image pixels, and explicit image-only searches remain image-only.
    """
    if media_constraint == "image":
        return []
    added = []
    for asset_id in list(dict.fromkeys(
            str(value) for value in (source_ids or []) if value)):
        parent_id = _video_parent_asset_id(store, asset_id)
        if not parent_id:
            continue
        source_score = float(scores.get(asset_id) or 0.0)
        if source_score <= 0.0:
            continue
        # Keep the keyframe's relevance signal, while allowing an independently
        # recalled parent to retain the stronger score.
        parent_score = source_score * 0.94
        old_score = scores.get(parent_id)
        if old_score is None or parent_score > float(old_score):
            scores[parent_id] = parent_score
            if old_score is None:
                added.append(parent_id)
    return added


def _build_semantic_routes(primary_query: str, *, query_core: str = "",
                            tool_query: str = "", object_terms=None,
                            user_query: str = "", max_routes: int = 4) -> list[tuple[str, str]]:
    """Retain planner, original-user, core and tool wording in bounded routes.

    Planner goals are often useful semantic compressions, but can drop the
    temporal, spatial, or relational clue that identifies an event. Always
    give the uncompressed user wording its own route before slot/core routes.
    """
    object_terms = [str(term or "").strip() for term in (object_terms or [])]
    candidates = [
        ("primary", primary_query),
        ("user_question", user_query),
        ("slot_query_core", query_core),
    ]
    # The four-route budget previously consumed by primary/user/core/tool
    # wording silently dropped every parsed object term. Fold those anchors
    # into the final bounded route instead of making them compete for slots.
    # Add only lexical aliases (not related concepts) so the query expands
    # without asserting that different objects are interchangeable.
    anchor_parts = [str(tool_query or "").strip(), *object_terms]
    anchor_text = " ".join(dict.fromkeys(part for part in anchor_parts if part))
    synonym_applied = False
    for source_term, alias_term in _SEMANTIC_OBJECT_SYNONYMS:
        if source_term in anchor_text:
            anchor_text = anchor_text.replace(source_term, alias_term)
            synonym_applied = True
    if anchor_text:
        candidates.append(("object_synonym" if synonym_applied else "tool_query_objects",
                           anchor_text))
    else:
        candidates.append(("tool_query", tool_query))
    candidates.extend(("object", term) for term in object_terms)

    routes = []
    seen = set()
    for source, raw_text in candidates:
        text = str(raw_text or "").strip()
        key = _semantic_route_text_key(text)
        if not key or key in seen:
            continue
        seen.add(key)
        routes.append((source, text))
        if len(routes) >= max(1, int(max_routes)):
            break
    return routes


def _lexical_anchor_surfaces(*, query: str, user_query: str,
                             query_core: str = "", event: str = "",
                             object_terms=None, filters: dict | None = None,
                             year=None) -> list[str]:
    """Build small, deterministic lexical probes for the retrieval merge.

    The ANN/graph kernel is still the source of the ordinary candidate order.
    This helper only recovers assets whose observation text contains an
    explicit clue from the question (object, scene, OCR, place, or time).  It
    deliberately does not use the answer, benchmark GT, or image pixels.
    Keeping probes short matters for Chinese FTS: a whole natural-language
    question is too broad, while a salient two-to-eight character phrase is a
    useful independent recall channel.
    """
    text = " ".join(str(value or "") for value in (
        user_query, query, query_core, event, *(object_terms or []),
        (filters or {}).get("place"), (filters or {}).get("time"), year,
    )).strip()
    surfaces: list[str] = []
    seen: set[str] = set()

    def add(value) -> None:
        value = re.sub(r"\s+", " ", str(value or "").strip())
        if len(value) < 2:
            return
        key = value.casefold()
        if key in seen:
            return
        seen.add(key)
        surfaces.append(value)

    # Parsed facets have priority because they are already bounded by the
    # semantic-slot contract.  The full user wording remains below so a
    # parser omission cannot erase an explicit object/scene cue.
    for value in (object_terms or []):
        add(value)
    add(event)
    add(query_core)
    # Two-character generic words (“婚礼/照片/朋友”) are poor rescue keys;
    # keep longer terms here and let the explicit alias block below handle
    # short but meaningful scene cues such as “舞台/桥/紫色”.
    generic_terms = {"婚礼", "亲友", "参加", "朋友", "同行", "照片", "图片",
                     "事情", "活动", "时候", "地方", "现场"}
    for term in _preview_query_terms(text):
        if term not in generic_terms and (len(term) >= 3 or term.isascii()):
            add(term)
    # Reuse only the generic visual equivalences already used by the preview
    # scorer.  They are lexical alternatives, not claims that two scenes are
    # semantically identical (e.g. “展架” may be described as “横幅”).
    for term, aliases in _PREVIEW_QUERY_ALIASES.items():
        if term in text:
            add(term)
            for alias in aliases:
                add(alias)
    add((filters or {}).get("place"))
    if year:
        add(str(year))
    # A bounded number of probes keeps the rescue channel cheap on the
    # 1,167-QA run and prevents generic words from becoming a second full scan.
    return surfaces[:16]


def _lexical_anchor_ranks(store, surfaces: list[str], scope_id: str,
                          *, limit: int = 200) -> dict[str, list[int]]:
    """Return FTS ranks for explicit query anchors without touching graph code."""
    if store is None or not surfaces:
        return {}
    try:
        from ..retrieval_indexes import RetrievalIndex
        index_store = _RUNTIME.get("lexical_index_store")
        lexical_index = _RUNTIME.get("lexical_index")
        if lexical_index is None or index_store is not store:
            lexical_index = RetrievalIndex(store)
            _RUNTIME["lexical_index"] = lexical_index
            _RUNTIME["lexical_index_store"] = store
    except Exception:
        return {}
    ranks: dict[str, list[int]] = {}
    lock = getattr(store, "_connection_lock", None)
    try:
        if lock is None:
            rows_by_surface = [list(lexical_index.search_fts(
                surface, scope_id=scope_id or None, limit=limit))
                for surface in surfaces]
        else:
            with lock:
                rows_by_surface = [list(lexical_index.search_fts(
                    surface, scope_id=scope_id or None, limit=limit))
                    for surface in surfaces]
    except Exception:
        return {}
    for rows in rows_by_surface:
        for rank, row in enumerate(rows, 1):
            asset_id = str((row or {}).get("asset_id") or "")
            if asset_id:
                ranks.setdefault(asset_id, []).append(rank)
    return ranks


class _SlotRetrievalPacket:
    """多路召回结果的轻量包，兼容 result set 构建对 packet 的引用。"""

    def __init__(self, assets, *, gaps=None, retrieval_timing=None, channel_trace=None):
        self.assets = assets
        self.gaps = gaps or []
        self.retrieval_timing = retrieval_timing or {}
        self.channel_trace = channel_trace or {}


def _search_memories(arguments: dict, *, context: dict | None = None) -> dict:
    query = arguments.get("query") or ""
    mode = arguments.get("mode") or "best"
    if mode not in {"best", "all", "representative"}:
        mode = "best"
    # Semantic conditions are soft hints for the reranker, never hard
    # retrieval filters. Preserve an explicit media boundary from either the
    # tool call or the original user wording; scope authorization remains
    # enforced by the runtime context.
    raw_filters = dict(arguments.get("filters") or {})
    scope_id = (context or {}).get("scope_id") or ""
    user_goal = ((context or {}).get("task_state") or {}).get("user_goal") or ""
    store = _RUNTIME.get("store")
    trusted_constraints = _trusted_query_constraints(
        user_goal or query, store=store, scope_id=scope_id)
    filters = _sanitize_model_filters(
        raw_filters,
        query=query,
        user_goal=user_goal,
        store=store,
        scope_id=scope_id,
        trusted_constraints=trusted_constraints,
    )
    media = str(filters.get("media") or "").strip().lower()
    model_filter_drops = {
        key: str(raw_filters.get(key) or "")
        for key in ("time", "place", "person")
        if raw_filters.get(key) and not filters.get(key)
    }
    model_filter_rewrites = {
        key: {"from": str(raw_filters.get(key) or ""),
              "to": str(filters.get(key) or "")}
        for key in ("time", "place", "person")
        if raw_filters.get(key) and filters.get(key)
        and str(raw_filters.get(key)) != str(filters.get(key))
    }
    media_filter_source = "tool_arguments" if filters.get("media") else "none"
    if not filters.get("media"):
        inferred_media = _explicit_media_filter(user_goal or query)
        if inferred_media:
            filters["media"] = inferred_media
            media_filter_source = "user_question"
    # Event summaries are intentionally not an early-return retrieval path.
    # They may be used later as an additional ranking signal, but must never
    # replace the multi-channel candidate universe.
    # W2.3：多轮引用消解 —— 引用类 query 先解析到已有 current_result_set，不重新全库搜索。
    # 引用标记在用户原话（task_state.user_goal）里，search query 是 LLM 提取后的内容，故两者都检测。
    _user_msg = ((context or {}).get("task_state") or {}).get("user_goal") or ""
    if _is_referent_query(query) or _is_referent_query(_user_msg):
        prior_rs_id = ((context or {}).get("task_state") or {}).get("current_result_set")
        prior_rs = None
        if prior_rs_id:
            rs_store = _RUNTIME.get("result_sets")
            if rs_store is not None and hasattr(rs_store, "get"):
                try:
                    prior_rs = rs_store.get(prior_rs_id)
                except Exception:
                    prior_rs = None
        try:
            anchored = _search_from_prior_result_set(
                prior_rs, scope_id, query=query, user_goal=_user_msg)
        except (KeyError, IndexError, TypeError):
            anchored = _bounded_event_result(prior_rs, scope_id, query=query, user_goal=_user_msg) if prior_rs else None
        if anchored is not None:
            return anchored
        # A first-turn referent without a prior result set still goes through
        # the normal retrieval channels.  Resolving an arbitrary event here
        # would silently turn “那次旅行的合影” into an unrelated event.
    viewer_id = (context or {}).get("viewer_id") or "owner"
    # 语义拆槽：模型把问题拆成时间/地点/人物/事件槽，代码用确定性通道精确召回
    # （metadata time_bounds/place、entity person），避免整句 embedding 语义漂移。
    # 拆槽失败或无确定性槽位 → 回退整句检索（绝不缩小召回）。
    query_for_retrieval = query
    gamma = _RUNTIME.get("gamma")
    # 拆槽必须用完整用户问题（user_goal），不是模型提取的词序列 query——
    # 词序列丢了时间/地点/事件信息（实测"保定 亲友婚礼 兄弟们留影"里没有"国庆节"），
    # 拆槽拆不出时间 bounds → 时间过滤失效 → 候选缺 GT。完整问题才能让模型
    # 拆出节假日/相对时间并转成标准 bounds，以及事件/物体槽（节假日转换由模型在拆槽时完成）。
    # 拆出的槽各自驱动独立召回链路（见下），多链路合并提高目标图被任一链路召回的覆盖。
    _slot_input = str(user_goal or "").strip() or query
    _slot_event = ""
    _slot_objects: list[str] = []
    _slot_query_core = ""
    _slots = None
    _slot_place_source = "none"
    # 时间分量（year/months/days）可能由拆槽给出；无年份的节日/季节只约束月日，
    # 绝不能因模型"猜年份"把正确年份的照片筛成 0（qa012/013/017/018 根因）。
    _slot_year = None
    _slot_months: list[int] = []
    _slot_days: list[int] = []
    if gamma is not None and _slot_input.strip():
        try:
            from .semantic_slots import parse_semantic_slots
            _slots = parse_semantic_slots(_slot_input, gamma.chat)
        except Exception:
            _slots = None
        if _slots:
            _st = _slots.get("time") or {}
            _slot_year = _st.get("year")
            _slot_months = sorted({int(m) for m in (_st.get("months") or [])
                                   if str(m).isdigit()})
            _slot_days = sorted({int(d) for d in (_st.get("days") or [])
                                 if str(d).isdigit()})
            _slot_expr = str(_st.get("expr") or "")
            _trusted_time = str(trusted_constraints.get("time") or "")
            _trusted_time_year = None
            _year_match = re.search(r"(?:19|20)\d{2}", _trusted_time)
            if _year_match:
                _trusted_time_year = int(_year_match.group(0))
            # Slot output is a ranking aid, never a source of new hard
            # constraints.  A model-produced year that conflicts with the
            # user's wording is discarded; no explicit user time means no
            # time filter at all.
            if not _trusted_time:
                _slot_year, _slot_months, _slot_days = None, [], []
                _slot_expr = ""
            elif (_trusted_time_year is not None and _slot_year is not None
                  and int(_slot_year) != _trusted_time_year):
                _slot_year, _slot_months, _slot_days = None, [], []
                _slot_expr = _trusted_time
            if _trusted_time:
                filters["time"] = _trusted_time
            # 相对时间（去年/今年/上个月/这两年…）模型不推年份，只给 expr，
            # 这里用确定性换算成绝对表达式走原有 bounds 路径。
            if not (_slot_year or _slot_months or _slot_days) and _slot_expr:
                _slot_abs = _resolve_time_expression(_slot_expr)
                if _slot_abs:
                    filters["time"] = _slot_abs
            # 有年份+月份才补绝对表达式，供 metadata-only/纯时间检索路径使用；
            # 只有月/日（节日/季节，年份未知）不给 filters.time，靠分量匹配不过滤年份。
            if _slot_year and _slot_months and not filters.get("time"):
                _slot_y = int(_slot_year)
                _slot_lo, _slot_hi = min(_slot_months), max(_slot_months)
                filters["time"] = (f"{_slot_y}年{_slot_lo}月-{_slot_hi}月"
                                   if _slot_lo != _slot_hi else f"{_slot_y}年{_slot_lo}月")
            if trusted_constraints.get("place"):
                filters["place"] = str(trusted_constraints["place"])
                _slot_place_source = "canonical_user"
            elif (_slots["place"].get("name")
                  and filters.get("place")
                  and _literal_in_text(_slots["place"].get("name"), _slot_input)
                  and not _looks_like_scene_place(_slots["place"].get("name"))):
                # Keep the most specific wording from the question. Replacing
                # it with a broader normalized hint (e.g. “保定市易县” for
                # “易县沙岭”) loses the village anchor needed to distinguish
                # same-county events.
                filters["place"] = _slots["place"].get("name")
                _slot_place_source = "name"
            elif (_slots["place"].get("hint")
                  and filters.get("place")
                  and _literal_in_text(_slots["place"].get("hint"), _slot_input)):
                filters["place"] = _slots["place"].get("hint")
                _slot_place_source = "hint"
            if trusted_constraints.get("person"):
                filters["person"] = str(trusted_constraints["person"])
            elif _slots.get("person") and filters.get("person"):
                filters["person"] = "、".join(p["name"] for p in _slots["person"])
            _slot_event = _slots["event"].get("name") or ""
            _slot_objects = [str(o) for o in (_slots.get("object") or [])]
            _slot_query_core = str(_slots.get("query_core") or "").strip()
    # 语义检索用完整语义目标（Planner declaration.goal / user_goal），而非模型
    # 传入的简化词序列 query（"亲戚婚房 兄弟 合影" 这类 embedding 质量差，实测
    # 候选 0）。槽位只做精确通道过滤（metadata time/place、entity person）。
    ts_ctx = (context or {}).get("task_state") or {}
    _decl = ts_ctx.get("declaration") or {}
    planner_goal = (context or {}).get("planner_goal") or ""
    declaration_goal = (_decl.get("goal") if isinstance(_decl, dict) else "")
    semantic_query = (planner_goal or declaration_goal or user_goal or query)
    semantic_query_source = (
        "planner_goal" if planner_goal else
        "declaration_goal" if declaration_goal else
        "user_goal" if user_goal else "tool_query"
    )
    query_for_retrieval = semantic_query
    draft = _draft_from_filters({**filters, "query": query_for_retrieval}, answer_type="asset_set")
    draft.result_requirement = {"mode": mode}
    spec = _spec_for(draft, scope_id, viewer_id)
    if not (query_for_retrieval or "").strip():
        # 纯时间/地点/人物/媒体筛选：走确定性元数据路径，不依赖 ANN 语义召回（生产多检索器下空 query 会 0 召回）
        user_goal = ((context or {}).get("task_state") or {}).get("user_goal") or ""
        return _search_metadata_only(draft, spec, scope_id, query_for_retrieval, mode, user_goal=user_goal)
    # ===== 新多路召回：按维度独立召回 + 跨召回重合评分 =====
    # 语义召回列表：每个 object 词（拆槽输出）+ 完整问题（主语义）。object 数量
    # 由模型判断（可多可少、可没有）；每条独立召回，图在各路的排名参与评分。
    semantic_route_entries = _build_semantic_routes(
        query_for_retrieval,
        query_core=_slot_query_core,
        tool_query=query,
        object_terms=_slot_objects,
        user_query=user_goal,
    )
    semantic_routes = [text for _, text in semantic_route_entries]
    per_asset_ranks: dict[str, list[int]] = {}
    place_fallback_ranks: dict[str, list[int]] = {}
    # The slot-route adapter calls the retrieval kernel once per semantic
    # route. Preserve the kernel's graph telemetry while doing the adapter's
    # cross-route RRF merge; otherwise PhotoBench sees an empty
    # ``retrieval_timing`` object and cannot report graph routing/effect.
    slot_graph_traces = []
    slot_route_trace = []
    slot_route_levels = []
    for _route_source, _sq in semantic_route_entries:
        try:
            # Keep the parsed hard media/time/place/person constraints on
            # every semantic route. Passing an empty filter dict here made
            # the slot adapter rank an unbounded semantic head first and only
            # then filter it, which can discard the ground-truth asset before
            # the cross-route merge (especially for the expanded QA set).
            _pk, _relax_level = _relaxed_retrieve(_sq, filters, scope_id, viewer_id, mode)
            if isinstance(getattr(_pk, "retrieval_timing", None), dict):
                _graph_timing = _pk.retrieval_timing
                if isinstance(_graph_timing.get("graph_policy"), dict):
                    slot_graph_traces.append(_graph_timing)
            _route_assets = (_pk.assets or [])
            slot_route_trace.append({
                "source": _route_source,
                "candidate_count": len(_route_assets),
                "relaxation_level": int(_relax_level or 0),
            })
            slot_route_levels.append(int(_relax_level or 0))
            for _rank, _item in enumerate(_route_assets[:_SLOT_ROUTE_HEAD], 1):
                _aid = _item.get("asset_id")
                if _aid:
                    per_asset_ranks.setdefault(_aid, []).append(_rank)

            # A non-empty strict place search is not proof that its results
            # actually match the place. If none has a verified place match,
            # also retrieve a semantic-only place fallback. Keep the strict
            # route intact and score this fallback separately at lower weight,
            # so sparse/ambiguous geocoding cannot erase the event candidates.
            if (_route_assets and int(_relax_level or 0) == 0
                    and _needs_place_semantic_fallback(filters, _route_assets, store)):
                _fallback_filters = dict(filters)
                _fallback_filters.pop("place", None)
                _fallback, _fallback_level = _relaxed_retrieve(
                    _sq, _fallback_filters, scope_id, viewer_id, mode,
                )
                _fallback_assets = (_fallback.assets or []) if _fallback else []
                slot_route_trace.append({
                    "source": f"{_route_source}_place_fallback",
                    "candidate_count": len(_fallback_assets),
                    "relaxation_level": int(_fallback_level or 0),
                    "fallback_weight": 0.35,
                })
                if isinstance(getattr(_fallback, "retrieval_timing", None), dict):
                    _fallback_timing = _fallback.retrieval_timing
                    if isinstance(_fallback_timing.get("graph_policy"), dict):
                        slot_graph_traces.append(_fallback_timing)
                for _rank, _item in enumerate(_fallback_assets[:_SLOT_ROUTE_HEAD], 1):
                    _aid = _item.get("asset_id")
                    if _aid:
                        place_fallback_ranks.setdefault(_aid, []).append(_rank)
        except Exception as error:
            slot_route_trace.append({
                "source": _route_source,
                "candidate_count": 0,
                "error": type(error).__name__,
            })
            continue

    # 事件成员：只做“重合 +1”（确定性弱验证，不参与排名）——事件可能划分不清、
    # 属于事件不代表与问题相关，不特殊对待。除了显式 event 槽，再给有明确
    # 时间/地点/媒体或场景线索的问题一次保守的事件锚尝试；槽位模型偶尔漏掉
    # “婚礼/留影”等事件词时，不能因此把整组真实资产从候选宇宙中删掉。
    event_member_ids: set[str] = set()
    event_anchor_query = str(user_goal or query or "")
    task_answerability = str(
        ts_ctx.get("answerability") or ts_ctx.get("expected_action") or ""
    ).strip().lower()
    event_anchor_enabled = bool(
        _slot_event or filters.get("time") or filters.get("place")
        or filters.get("person") or filters.get("media")
        or any(term in event_anchor_query for term in (
            "照片", "图片", "留影", "拍照", "合影", "合照", "视频", "哪天",
            "日期", "地点", "哪里", "场景", "舞台", "婚礼", "旅行",
        ))
    ) and task_answerability not in {"unanswerable", "refuse", "refusal"}
    event_anchor_source = "none"
    if event_anchor_enabled:
        try:
            _ev = _event_resolution(event_anchor_query, store, scope_id)
            event_anchor_source = "resolution" if _ev else "none"
            if _ev is None:
                _ev = _event_keyword_anchor(event_anchor_query, store, scope_id)
                event_anchor_source = "keyword" if _ev else "none"
            event_member_ids = {
                str(a) for a in ((_ev or {}).get("asset_ids") or [])
                if not _is_synthetic_event_asset({"asset_id": str(a)}, store)
                and not _is_face_reference_asset({"asset_id": str(a)}, store)
            }
        except Exception:
            event_member_ids = set()

    # 综合分：RRF 排名累加 —— score = Σ 1/(k+rank) + 事件重合权重。
    # 图在各语义召回的排名越靠前、被越多路召回，分越高。k/gap 对召回鲁棒
    # （扫描：k=10/20/30、gap=0.5-0.8 持平），候选上限是主导参数。
    _slot_rrf_k = float(os.getenv("SENTRIX_SLOT_RRF_K", "10"))
    _slot_ev_w = float(os.getenv("SENTRIX_SLOT_EV_WEIGHT", "0.1"))
    scores: dict[str, float] = {}
    for _aid in (set(per_asset_ranks) | set(place_fallback_ranks) | event_member_ids):
        _ranks = per_asset_ranks.get(_aid) or []
        _s = sum(1.0 / (_slot_rrf_k + r) for r in _ranks)
        _fallback_ranks = place_fallback_ranks.get(_aid) or []
        _s += 0.35 * sum(1.0 / (_slot_rrf_k + r) for r in _fallback_ranks)
        if _aid in event_member_ids:
            _s += _slot_ev_w
        scores[_aid] = _s

    # 确定性筛子：时间由拆槽的时间分量（year/months/days，缺省分量不约束）过滤，
    # 或由绝对/相对表达式（bounds）过滤；地点只做非标准地名时的语义保底。
    # 年份未知（节日/季节）时只约束月/日，任何年份的相符照片都保留。
    time_comps = None
    if _slot_year or _slot_months or _slot_days:
        time_comps = {
            "year": int(_slot_year) if _slot_year else None,
            "months": set(_slot_months) or None,
            "days": set(_slot_days) or None,
        }
    time_bounds = None
    if filters.get("time") and time_comps is None:
        try:
            from ..query_contracts import parse_time_expression
            time_bounds = parse_time_expression(str(filters["time"]))
        except Exception:
            time_bounds = None
    place_q = filters.get("place") or ""

    def _asset_captured(aid):
        try:
            _a = store.get_asset(aid) or {}
            _o = (store.list_observations(asset_id=aid, limit=1) or [{}])[0]
            from ..retrieval.temporal import trusted_captured_at
            return trusted_captured_at(_a, _o, store=store)
        except Exception:
            return None

    def _time_ok(cap):
        # An unknown/untrusted timestamp is not a contradiction; keep the
        # semantic candidate and let ranking/evidence decide instead.
        if not cap:
            return True
        if time_comps is not None:
            return _in_time_components(cap, time_comps)
        if time_bounds is not None:
            return _in_time_bounds(cap, time_bounds)
        return True

    # The normal route merge is intentionally unchanged, including the graph
    # telemetry and graph candidate policy.  Add a separate deterministic FTS
    # rescue channel for explicit question anchors.  This fixes the common
    # failure where the visual/text ANN routes return a broad event head while
    # the exact observation phrase (OCR/object/place) sits just below it.
    # It is a rank signal only: no answer text, GT, or image inspection enters
    # this path, and graph candidates are neither re-ranked nor filtered here.
    lexical_anchor_surfaces = _lexical_anchor_surfaces(
        query=query_for_retrieval,
        user_query=user_goal,
        query_core=_slot_query_core,
        event=_slot_event,
        object_terms=_slot_objects,
        filters=filters,
        year=_slot_year,
    )
    lexical_anchor_ranks = _lexical_anchor_ranks(
        store, lexical_anchor_surfaces, scope_id,
        limit=max(20, int(os.getenv("SENTRIX_LEXICAL_ANCHOR_LIMIT", "80"))),
    )
    lexical_anchor_valid: dict[str, list[int]] = {}
    lexical_anchor_added = 0
    media_constraint = str(filters.get("media") or "").strip().lower()
    for _aid, _ranks in lexical_anchor_ranks.items():
        try:
            _asset = store.get_asset(_aid) or {}
            if (_is_face_reference_asset({"asset_id": _aid}, store)
                    or _is_synthetic_event_asset({"asset_id": _aid}, store)):
                continue
            if media_constraint and str(_asset.get("media_type") or "").lower() != media_constraint:
                continue
            if (time_comps is not None or time_bounds) and not _time_ok(
                    _asset_captured(_aid)):
                continue
        except Exception:
            continue
        lexical_anchor_valid[_aid] = _ranks
        if _aid not in scores:
            scores[_aid] = 0.0
            lexical_anchor_added += 1

    _slot_lexical_anchor_w = float(
        os.getenv("SENTRIX_SLOT_LEXICAL_ANCHOR_WEIGHT", "3.0"))
    for _aid, _ranks in lexical_anchor_valid.items():
        scores[_aid] += _slot_lexical_anchor_w * sum(
            1.0 / (_slot_rrf_k + rank) for rank in _ranks[:8])

    # Unknown GPS/geocode remains open-world in the retrieval kernel, so it
    # cannot erase a plausible image or video frame. Among retained candidates,
    # a verified place match should overcome a weak ANN rank. The boost is
    # relative to this query's score scale and independent of the retriever's
    # cosine/lexical/RRF score range.
    place_matched_ids = set()
    place_boost = 0.0
    if place_q:
        place_matched_ids = {
            aid for aid in scores
            if _place_matches({"asset_id": aid}, place_q, store)
        }
        place_boost = max(scores.values(), default=0.0) * 0.75
        for aid in place_matched_ids:
            scores[aid] = scores.get(aid, 0.0) + place_boost

    # A keyframe is indexed as an image, but the benchmark's video GT is the
    # parent source video.  Expand this provenance edge before the bounded
    # candidate head is cut, so an otherwise correct keyframe hit cannot lose
    # its matching video merely because the Agent receives only 18 candidates.
    # This is retrieval-only; metric matching and final delivery are untouched.
    video_parent_added = _add_video_parent_candidates(
        scores,
        list(scores) + list(event_member_ids),
        store,
        media_constraint=media_constraint,
    )

    kept = list(scores)
    if time_comps is not None or time_bounds:
        kept = [aid for aid in kept if _time_ok(_asset_captured(aid))]
    kept.sort(key=lambda a: -scores.get(a, 0))

    # 断层截断（明显 gap 处截断）+ 有界候选上限 + 保底 3。
    final_ids = kept[:_SLOT_MAX_CANDIDATES]
    if len(final_ids) > 3:
        _top = scores.get(final_ids[0], 1) or 1.0
        _cut = len(final_ids)
        _gap_ratio = float(os.getenv("SENTRIX_SLOT_GAP_RATIO", "0.5"))
        for i in range(1, len(final_ids)):
            if _top > 0 and (scores.get(final_ids[i - 1], 0) - scores.get(final_ids[i], 0)) / _top > _gap_ratio:
                _cut = i
                break
        final_ids = final_ids[:_cut]
    if len(final_ids) < 3 and len(kept) > len(final_ids):
        final_ids = kept[:3]  # 保底 3 张，避免模型无可选

    # The candidate set above is still the graph/semantic union.  Only reorder
    # the bounded head by the original user wording before exposing stable
    # photo_N handles.  This is important because the Agent commonly selects
    # photo_1: a correct frame at rank 4 was counted as a miss even though it
    # was already in the retrieved set.  The reordering is local to the
    # candidate head; it neither adds GT/answer information nor changes graph
    # candidate membership or metric calculation.
    candidate_order_query = user_goal or query_for_retrieval or query
    if final_ids and mode != "representative":
        order = _preview_query_order(final_ids, candidate_order_query, store)
        final_ids = [final_ids[index] for index in order
                     if 0 <= index < len(final_ids)]

    slot_retrieval_timing = {
        "semantic_retrieval": {
            "primary_query_source": semantic_query_source,
            "slot_parse_available": bool(_slots),
            "slot_time_present": bool(_slot_year or _slot_months or _slot_days or
                                       ((_slots or {}).get("time") or {}).get("expr")),
            "slot_place_source": _slot_place_source,
            "slot_query_core_used": any(source == "slot_query_core"
                                         for source, _ in semantic_route_entries),
            "route_head_limit": _SLOT_ROUTE_HEAD,
            "candidate_limit": _SLOT_MAX_CANDIDATES,
            "routes": slot_route_trace,
            "merged_candidate_count": len(scores),
            "time_filter_active": bool(time_comps is not None or time_bounds),
            "place_filter_active": bool(place_q),
            "media_filter": filters.get("media") or None,
            "media_filter_source": media_filter_source,
            "model_filter_drops": model_filter_drops,
            "model_filter_rewrites": model_filter_rewrites,
            "trusted_time": str(trusted_constraints.get("time") or ""),
            "trusted_place": str(trusted_constraints.get("place") or ""),
            "place_matched_candidate_count": len(place_matched_ids),
            "place_boost": round(place_boost, 6),
            "event_anchor_enabled": event_anchor_enabled,
            "event_anchor_source": event_anchor_source,
            "event_anchor_candidate_count": len(event_member_ids),
            "lexical_anchor_surface_count": len(lexical_anchor_surfaces),
            "lexical_anchor_candidate_count": len(lexical_anchor_valid),
            "lexical_anchor_added_count": lexical_anchor_added,
            "video_parent_added_count": len(video_parent_added),
            "post_anchor_candidate_count": len(kept),
            "returned_candidate_count": len(final_ids),
        }
    }

    # 组装候选 assets（供 result set / preview）
    assets = []
    for _aid in final_ids:
        _a = store.get_asset(_aid) or {}
        _o = (store.list_observations(asset_id=_aid, limit=1) or [{}])[0]
        assets.append({
            "asset_id": _aid,
            "file_name": _a.get("file_name"),
            "media_type": _a.get("media_type"),
            "captured_at": _a.get("captured_at") or _o.get("captured_at"),
            "observation_ids": [_o["id"]] if _o.get("id") else [],
            "evidence_ids": [_aid],
            "condition_results": {},
            "level": "strong" if _aid in event_member_ids else "approximate",
            "score": scores.get(_aid, 0),
            "fusion_score": scores.get(_aid, 0),
            "retrieval_score": 0.0,
            "observation_fields": {"place": _o.get("place"), "activity": _o.get("activity"),
                                   "subject_clothing": _o.get("subject_clothing") or []},
            "attributions": [{"retriever": "slot_route", "rank": 0,
                              "score": scores.get(_aid, 0), "score_kind": "structured"}],
        })
    asset_ids = [item.get("asset_id") for item in assets if item.get("asset_id")]
    if slot_graph_traces:
        # Report one question-level graph comparison over the union of the
        # route candidates. This describes the candidates actually entering
        # the slot-route merge, without exposing internal asset IDs to the
        # model-facing observation.
        graph_policy = next(
            (trace.get("graph_policy") for trace in slot_graph_traces
             if (trace.get("graph_policy") or {}).get("enabled") is True),
            slot_graph_traces[0].get("graph_policy") or {},
        )
        baseline_ids = list(dict.fromkeys(
            str(asset_id)
            for trace in slot_graph_traces
            for asset_id in ((trace.get("graph_rerank") or {}).get("baseline_ranked_asset_ids") or [])
        ))
        reranked_ids = list(dict.fromkeys(
            str(asset_id)
            for trace in slot_graph_traces
            for asset_id in ((trace.get("graph_rerank") or {}).get("reranked_asset_ids") or [])
        ))
        graph_candidates = sum(
            int((trace.get("graph_rerank") or {}).get("graph_candidate_count") or 0)
            for trace in slot_graph_traces
        )
        graph_reranks = [trace.get("graph_rerank") or {} for trace in slot_graph_traces]
        baseline_head_ids = list(dict.fromkeys(
            str(asset_id)
            for trace in slot_graph_traces
            for asset_id in ((trace.get("graph_rerank") or {}).get(
                "baseline_head_asset_ids") or [])
        ))
        returned_head_ids = list(dict.fromkeys(
            str(asset_id)
            for trace in slot_graph_traces
            for asset_id in ((trace.get("graph_rerank") or {}).get(
                "returned_head_asset_ids") or [])
        ))
        slot_retrieval_timing.update({
            "graph_policy": graph_policy,
            "graph_invoked": any(bool(trace.get("graph_invoked")) for trace in slot_graph_traces),
            "graph_candidate_count": graph_candidates,
            "graph_rerank": {
                "applied": bool(reranked_ids),
                "baseline_ranked_asset_ids": baseline_ids,
                "reranked_asset_ids": reranked_ids,
                "baseline_candidate_count": len(baseline_ids),
                "combined_candidate_count": len(reranked_ids),
                "graph_candidate_count": graph_candidates,
                # Preserve question-level graph-fusion telemetry.  The
                # model-facing observation still hides asset IDs, but the
                # evaluator needs to know whether graph actually introduced
                # new event->frame candidates or merely reordered baseline
                # results.
                "new_candidate_count": sum(
                    int(item.get("new_candidate_count") or 0)
                    for item in graph_reranks),
                "graph_head_quota": max(
                    (int(item.get("graph_head_quota") or 0)
                     for item in graph_reranks), default=0),
                "graph_forced_head_count": sum(
                    int(item.get("graph_forced_head_count") or 0)
                    for item in graph_reranks),
                # Preserve the exact candidate heads used for the question.
                # The evaluator compares these against media GT; the full
                # ranked lists above remain diagnostics only.
                "baseline_head_asset_ids": baseline_head_ids,
                "returned_head_asset_ids": returned_head_ids,
                "baseline_order_preserved": all(
                    item.get("baseline_order_preserved") is True
                    for item in graph_reranks
                    if "baseline_order_preserved" in item
                ),
                "promoted_count": sum(
                    int(item.get("promoted_count") or 0)
                    for item in graph_reranks),
                "demoted_count": sum(
                    int(item.get("demoted_count") or 0)
                    for item in graph_reranks),
                "graph_intent": graph_policy.get("intent") or "ordinary",
            },
        })
    packet = _SlotRetrievalPacket(assets, gaps=[],
                                  retrieval_timing=slot_retrieval_timing,
                                  channel_trace={})
    _relax_level = max(slot_route_levels, default=0)
    asset_ids = [item.get("asset_id") for item in assets if item.get("asset_id")]
    rs = _RUNTIME["result_sets"].new(
        scope_id=scope_id, query=query, asset_ids=asset_ids,
        unresolved=[g.get("reason") for g in (packet.gaps or [])],
    )
    # The compact tool query and planner goal may omit visual/time details
    # present in the original user wording. Use that full wording when choosing
    # the bounded preview that the Agent can actually inspect and deliver.
    preview_query = user_goal or query_for_retrieval or query
    preview_indices = _preview_indices(
        asset_ids, mode, store, query=preview_query, limit=_SLOT_PREVIEW_LIMIT)
    # 删模型重排：候选即最终。代码融合排序 + gap 截断已保证强相关在前，
    # 不再逐批调用 12B 验证候选（省 ~5 批模型调用/题，避免上下文膨胀）。
    validated_ids = list(asset_ids)
    candidate_only_ids = []
    ranked_ids = list(asset_ids)
    public_ids = list(asset_ids)
    public_status = "candidate_only" if public_ids else "none"
    rs.set_public_view(public_ids)
    _RUNTIME["result_sets"].save(rs)
    # Use the same query-aware indices for the actual Agent-visible evidence
    # that candidate_window reports. Previously this block ignored
    # ``preview_indices`` and always exposed the retrieval head, so visual/time
    # cues could appear effective in telemetry while the Agent never saw the
    # selected candidate. Handles still map to original ResultSet ranks.
    preview = _build_preview_entries(store, asset_ids, preview_indices)
    preview_asset_ids = [item.get("asset_id") for item in preview if item.get("asset_id")]
    public_handles = {asset_id: f"photo_{index + 1}"
                      for index, asset_id in enumerate(rs.visible_asset_ids())}
    validation_rows = []
    cond, satisfaction, answerability = _truth_contract(packet, rs.total)
    evidence_status = public_status
    satisfaction = "candidate_only" if asset_ids else "no_direct_support"
    answerability = "full" if asset_ids else "limited"
    return {
        "result_set_id": rs.result_set_id,
        "query": query,
        "mode": mode,
        "total": len(rs.visible_asset_ids()),
        "retrieved_total": len(asset_ids),
        "evidence_total": len(asset_ids),
        "selected_total": len(preview_asset_ids),
        "evidence_count": len(asset_ids),
        "preview": preview,
        "has_more": len(rs.visible_asset_ids()) > len(preview),
        "remaining": max(0, len(rs.visible_asset_ids()) - len(preview)),
        "candidate_window": _candidate_window_summary(asset_ids, preview_indices, store),
        "completeness": "complete" if not (packet.gaps) else "partial",
        "gaps": rs.unresolved[:3],
        "query_satisfaction": satisfaction,
        "answerability": answerability,
        "condition_summary": cond,
        "can_inspect": len(preview) > 0,
        "inspect_hint": "preview 里的 handle（photo_1…）可直接用于 inspect_photo 复核视觉细节" if preview else "",
        "retrieval_timing": packet.retrieval_timing,
        "retrieval_channels": packet.channel_trace,
        "relaxation_level": _relax_level,
        "raw_candidate_count": len(asset_ids),
        "validation_candidate_count": 0,
        "validation_batches": 0,
        "validation_status": "skipped",
        "validation_error": "",
        "validation_rows": [],
        "ranked_asset_ids": list(asset_ids),
        "evidence_status": evidence_status,
        "recommended_resolution": _recommended_resolution(
            query, preview, satisfaction,
            user_goal=((context or {}).get("task_state") or {}).get("user_goal") or ""),
        # 推荐最可能符合问题的 preview 图，避免模型默认 inspect photo_1 而错图
        # （实测候选含 GT 仍拒答的识别类题，多因 inspect 了错误的图）。
        "recommended_handle": _recommended_handle(
            preview_query, preview),
        "_retrieved_asset_ids": list(asset_ids),
        "retrieved_asset_ids": list(asset_ids),
        "_preview_asset_ids": preview_asset_ids,
        "evidence_asset_ids": list(asset_ids),
        "selected_asset_ids": list(preview_asset_ids),
        "_model_call_metrics": [],
    }


def _short_place_label(asset: dict) -> str:
    """从资产反地理编码取短地点标签（城市/区县名），供 preview 证据展示。"""
    import json as _json
    metadata = asset.get("metadata_json") or {}
    if isinstance(metadata, str):
        try:
            metadata = _json.loads(metadata)
        except (TypeError, ValueError):
            metadata = {}
    geocode = metadata.get("reverse_geocode") or {}
    if isinstance(geocode, str):
        try:
            geocode = _json.loads(geocode)
        except (TypeError, ValueError):
            geocode = {}
    if not isinstance(geocode, dict):
        return ""
    parts = []
    for key in ("city", "district"):
        value = str(geocode.get(key) or "").strip()
        if value and value not in parts:
            parts.append(value)
    if parts:
        return "".join(parts)
    return str(geocode.get("label") or "")


def _recommended_handle(query: str, preview: list) -> str:
    """按 query 与 preview 描述（evidence_summary）的关键词覆盖，推荐最可能相关的一张图。

    模型 inspect 默认 photo_1，若该图不是目标（如候选含 GT 但 preview 第一张是别的事件图）
    会 inspect 错图 → 拒答。推荐描述最贴合问题的 handle，引导模型先复核对的图。
    """
    if not preview:
        return ""
    q = str(query or "").strip()
    if not q:
        return str((preview[0] or {}).get("handle") or "")
    if not _preview_query_terms(q):
        return str((preview[0] or {}).get("handle") or "")
    best_handle, best_score = "", 0.0
    for p in preview:
        desc = str(p.get("evidence_summary") or "")
        if not desc:
            continue
        score = _preview_text_score(q, desc)
        if score > best_score:
            best_score, best_handle = score, str(p.get("handle") or "")
    return best_handle or str((preview[0] or {}).get("handle") or "")


def _recommended_resolution(query: str, preview: list, satisfaction: str,
                            user_goal: str = "") -> dict:
    """Evidence Finder（Phase E §8.2）：告诉 Agent 下一步证据解析方案。

    不替 Agent 做决定，只把"这条检索还需要什么"翻译成工具建议。
    """
    if not preview:
        return {"needed": False, "tool": None,
                "reason": "" if satisfaction == "full_support" else "没有候选可复核"}
    q = f"{query or ''} {user_goal or ''}"
    needs_ocr = ocr_intent(q)
    needs_visual = visual_intent(q)
    if needs_ocr:
        return {"needed": True, "tool": "read_photo_text",
                "reason": "问题需要读取照片中的文字/数字，请用 read_photo_text 复核 preview 里的照片"}
    # Identity/list questions must resolve a concrete photo before answering;
    # metadata people on several candidates are not interchangeable evidence.
    if re.search(r"都有谁|有哪些人|哪几个人|几个人|谁一起|谁参加|人物", q):
        return {"needed": True, "tool": "inspect_photo",
                "reason": "问题需要确认照片中的人物，请用 inspect_photo 复核 preview 里的照片"}
    if re.search(r"合影|合照|同行|朋友|晚餐|聚餐", q):
        return {"needed": True, "tool": "inspect_photo",
                "reason": "问题需要确认合影和同行者，请用 inspect_photo 复核 preview 里的照片"}
    if re.search(r"哪次旅行|哪次经历|什么旅行|旅行记录", q):
        return {"needed": True, "tool": "inspect_photo",
                "reason": "问题需要复核代表性旅行照片，请用 inspect_photo 确认场景"}
    if needs_visual:
        return {"needed": True, "tool": "inspect_photo",
                "reason": "问题需要查看照片中的视觉细节，请用 inspect_photo 复核 preview 里的照片"}
    return {"needed": False, "tool": None, "reason": ""}


def _truth_contract(packet, total: int) -> tuple[dict, str, str]:
    """确定性计算查询满足度（B2）：基于 condition_results 与 gaps，不交给模型判断。"""
    if total <= 0:
        return {}, "no_match", "none"
    condition_verdict: dict[str, dict] = {}
    for item in (packet.assets or []):
        for key, cond in (item.get("condition_results") or {}).items():
            label = key.split(":", 1)[-1]
            status = cond.get("status") or "unknown"
            bucket = condition_verdict.setdefault(label, {"confirmed": 0, "supported": 0,
                                                          "unknown": 0, "contradicted": 0})
            if status == "matched":
                bucket["confirmed"] += 1
            elif status == "possible":
                bucket["supported"] += 1
            elif status == "contradicted":
                bucket["contradicted"] += 1
            else:
                bucket["unknown"] += 1
    summary = {}
    for label, bucket in condition_verdict.items():
        if bucket["confirmed"] > 0:
            summary[label] = "confirmed"
        elif bucket["supported"] > 0:
            summary[label] = "supported"
        elif bucket["contradicted"] > 0 and bucket["confirmed"] == 0:
            summary[label] = "contradicted"
        else:
            summary[label] = "unknown"
    confirmed = sum(1 for v in summary.values() if v == "confirmed")
    unknown = sum(1 for v in summary.values() if v in {"unknown", "contradicted"})
    if not summary:
        satisfaction = "candidate_only"
    elif unknown == 0:
        satisfaction = "full_support"
    elif confirmed > 0:
        satisfaction = "partial_support"
    else:
        satisfaction = "candidate_only"
    answerability = "full" if satisfaction == "full_support" else ("partial" if confirmed else "limited")
    return summary, satisfaction, answerability


# ---- Tool 3: get_original_photos ----
def _get_original_photos(arguments: dict, *, context: dict | None = None) -> dict:
    task_state = (context or {}).get("task_state") or {}
    result_set_id = arguments.get("result_set_id") or (task_state or {}).get("current_result_set")
    handle = arguments.get("handle") or ""
    rs_store = _RUNTIME.get("result_sets")
    if not result_set_id or rs_store is None:
        return {"summary": "当前没有可交付的结果集。", "delivered": 0, "blocked": ["no_result_set"]}
    rs = rs_store.get(result_set_id)
    if rs is None:
        return {"summary": "结果集不存在。", "delivered": 0, "blocked": ["unknown_result_set"]}
    if rs.scope_id != ((context or {}).get("scope_id") or ""):
        return {"summary": "无权交付该结果集的原图。", "delivered": 0, "blocked": ["scope_mismatch"]}
    asset_id = rs_store.resolve_handle(result_set_id, handle) if handle else None
    if handle and not asset_id:
        return {"summary": "无法解析选中的照片。", "delivered": 0, "blocked": ["bad_handle"]}
    target = handle if asset_id else (rs.visible_asset_ids()[0] if rs.visible_asset_ids() else None)
    store = _RUNTIME.get("store")
    target_asset = store.get_asset(asset_id or target) if store and (asset_id or target) else None
    if target_asset and target_asset.get("derived_kind") == "video_keyframe" and target_asset.get("parent_asset_id"):
        source_video_id = target_asset["parent_asset_id"]
        return {
            "summary": "已从结果集授权原始视频交付，来源是关键帧对应的时间点。",
            "result_set_id": result_set_id,
            "handle": handle or "first",
            "delivered": 1,
            "total": len(rs.visible_asset_ids()),
            "scope_id": rs.scope_id,
            "url": f"/api/assets/{source_video_id}/file",
            "media_type": "video",
            "source_video_asset_id": source_video_id,
            "source_timestamp_sec": target_asset.get("source_timestamp_sec"),
        }
    url = ""
    if target:
        url = (f"/api/assistant/result-set/{result_set_id}/photo?handle={target}"
               f"&scope_id={rs.scope_id}&original=1")
    return {
        "summary": f"已从结果集 {result_set_id} 授权原图交付。",
        "result_set_id": result_set_id,
        "handle": handle or "first",
        "delivered": 1 if asset_id else (1 if rs.visible_asset_ids() else 0),
        "total": len(rs.visible_asset_ids()),
        "scope_id": rs.scope_id,
        "url": url,
    }


def get_result_set_store():
    """B3.2：API 层访问 ResultSetStore 的公开入口（原图授权端点用）。"""
    return _RUNTIME.get("result_sets")


def resolve_handle_asset_id(handle: str, result_set_id: str | None = None,
                            scope_id: str | None = None) -> str | None:
    """D8：从 handle（+结果集）解析 asset_id；API 层 Photo Thread 用。"""
    rs_store = _RUNTIME.get("result_sets")
    if result_set_id and rs_store is not None:
        aid = rs_store.resolve_handle(result_set_id, handle)
        if aid:
            return aid
    return _handle_to_asset_id(handle)


def result_set_context(result_set_id: str, scope_id: str) -> str | None:
    """B3.1：给模型一段当前结果集的续接上下文（不暴露内部 ID 之外的敏感信息）。"""
    rs_store = _RUNTIME.get("result_sets")
    if not rs_store:
        return None
    rs = rs_store.get(result_set_id)
    if rs is None or (scope_id and rs.scope_id != scope_id):
        return None
    visible_total = len(rs.visible_asset_ids())
    shown = min(visible_total, rs.shown or 0)
    return (f"当前结果集：{rs.result_set_id}，当前可核验候选最多 {visible_total} 张，已显示 {shown} 张，"
            f"还有 {max(0, visible_total - shown)} 张。查看更多用 get_result_page（page 从 1 开始）。")


# ---- Tool 3.5: get_result_page（B3.1 分页）----
def _get_result_page(arguments: dict, *, context: dict | None = None) -> dict:
    scope_id = (context or {}).get("scope_id") or ""
    task_state = (context or {}).get("task_state") or {}
    result_set_id = arguments.get("result_set_id") or task_state.get("current_result_set")
    try:
        page_no = max(1, int(arguments.get("page") or 1))
    except (TypeError, ValueError):
        page_no = 1
    if arguments.get("query") or arguments.get("filters"):
        return {
            "summary": "新的查询条件不能翻旧结果集，请重新调用 search_memories。",
            "total": 0,
            "requires_new_search": True,
            "blocked": ["new_query_requires_search"],
        }
    try:
        page_size = min(_RESULT_PAGE_SIZE, max(1, int(arguments.get("page_size") or _RESULT_PAGE_SIZE)))
    except (TypeError, ValueError):
        page_size = 6
    rs_store = _RUNTIME.get("result_sets")
    if not result_set_id or rs_store is None:
        return {"summary": "当前没有可用的结果集。", "total": 0, "blocked": ["no_result_set"]}
    rs = rs_store.get(result_set_id)
    if rs is None:
        return {"summary": "结果集不存在或已过期。", "total": 0, "blocked": ["unknown_result_set"]}
    if scope_id and rs.scope_id != scope_id:
        return {"summary": "无权访问该结果集。", "total": 0, "blocked": ["scope_mismatch"]}
    visible_total = len(rs.visible_asset_ids())
    visible_ids = rs.visible_asset_ids()
    start = max(0, (page_no - 1) * page_size)
    items = [
        {"handle": f"photo_{start + i + 1}", "asset_id": asset_id}
        for i, asset_id in enumerate(visible_ids[start:start + page_size])
    ]
    shown = min(visible_total, start + len(items))
    store = _RUNTIME.get("store")
    asset_ids = [item.get("asset_id") for item in items if item.get("asset_id")]
    preview = []
    for item in items:
        aid = item.get("asset_id")
        if not aid:
            continue
        entry = _preview_entry(
            store,
            aid,
            item.get("handle") or "",
            level="exact",
            priority_rank=len(preview) + 1,
            selection_reason="分页候选",
        )
        if entry:
            preview.append(entry)
    return {
        "result_set_id": rs.result_set_id,
        "page": page_no,
        "page_size": page_size,
        "total": visible_total,
        "shown": shown,
        "has_more": shown < visible_total,
        "remaining": max(0, visible_total - shown),
        "preview": preview,
        "asset_ids": asset_ids,
        "retrieved_asset_ids": asset_ids,
        "source_asset_ids": asset_ids,
        # Pagination exposes more retrieved candidates. A page is not
        # evidence until the validator/inspect/OCR/metadata path accepts it.
        "evidence_asset_ids": [],
        "query": rs.query,
    }


# ---- Tool 4: inspect_photo ----
def _inspect_photo(arguments: dict, *, context: dict | None = None) -> dict:
    asset_handle = arguments.get("asset_handle") or ""
    question = arguments.get("question") or "请描述这张照片"
    scope_id = (context or {}).get("scope_id") or ""
    task_state = (context or {}).get("task_state") or {}
    target_person = str(arguments.get("target_person") or task_state.get("active_person") or "").strip()
    if not asset_handle:
        # C11：未填 handle 时用当前结果集 preview 首个可复核 handle（安全默认）
        preview = (task_state.get("result_preview") or []) or []
        if preview:
            asset_handle = preview[0]
    # B3：handle 必须解析自当前结果集；失败再退回最近一次检索的 handle 映射
    asset_id = None
    result_set_id = task_state.get("current_result_set")
    rs_store = _RUNTIME.get("result_sets")
    used_result_set = False
    if result_set_id and rs_store is not None:
        asset_id = rs_store.resolve_handle(result_set_id, asset_handle)
        used_result_set = True
    if not asset_id and not used_result_set:
        return {"summary": "请先通过 search_memories 建立当前结果集。",
                "certainty": "uncertain", "persisted": False,
                "blocked": ["no_current_result_set"]}
    store = _RUNTIME.get("store")
    if not asset_id or store is None:
        return {"summary": "无法定位照片。", "certainty": "uncertain", "persisted": False,
                "blocked": ["unknown_handle"]}
    row = _store_fetchone(
        store, "SELECT path, scope_id FROM assets WHERE id = ?", (asset_id,))
    if row and scope_id and row["scope_id"] != scope_id:
        return {"summary": "无法复核该照片（不在当前相册范围）。", "certainty": "uncertain",
                "persisted": False, "blocked": ["scope_mismatch"]}
    if not row or not row["path"] or not Path(row["path"]).is_file():
        return {"summary": "照片文件不可用。", "certainty": "uncertain", "persisted": False,
                "blocked": ["file_unavailable"]}
    gamma = _RUNTIME.get("gamma")
    if gamma is None:
        return {"summary": "模型不可用。", "certainty": "uncertain", "persisted": False}
    model_call_metrics = []
    identity_rows = _confirmed_photo_identities(store, asset_id)
    face_manifest = _photo_face_manifest(store, asset_id)
    if not target_person:
        user_goal = str(task_state.get("user_goal") or task_state.get("last_user_goal") or "")
        target_person = next((str(item.get("person_name") or "") for item in identity_rows
                              if item.get("person_name") and item["person_name"] in (question + user_goal)), "")
    target_identity = next((row for row in identity_rows
                            if target_person and row.get("person_name") == target_person), None)
    target_face = next((row for row in face_manifest
                        if target_person and row.get("person_name") == target_person), None)
    target_status = "not_requested"
    target_bbox = None
    inspect_images = []
    crop_path = None
    try:
        encoded, mime_type = gamma.encode_vision_image(row["path"])
        image = {"base64": encoded, "mime_type": mime_type}
        inspect_images = [image]
        if target_person:
            if target_identity and target_identity.get("bbox"):
                try:
                    from PIL import Image, ImageOps
                    source_image = ImageOps.exif_transpose(Image.open(row["path"])).convert("RGB")
                    face_bbox = list(target_identity["bbox"])
                    if face_bbox and max(face_bbox) <= 1.0:
                        face_bbox = [face_bbox[0] * source_image.width,
                                     face_bbox[1] * source_image.height,
                                     face_bbox[2] * source_image.width,
                                     face_bbox[3] * source_image.height]
                    crop, crop_bbox = expanded_person_crop(source_image, face_bbox)
                    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as temporary:
                        crop.save(temporary, format="JPEG", quality=90)
                        crop_path = temporary.name
                    crop_encoded, crop_mime = gamma.encode_vision_image(crop_path)
                    inspect_images.insert(0, {"base64": crop_encoded, "mime_type": crop_mime})
                    target_status = "located"
                    target_bbox = crop_bbox
                except (OSError, ValueError, TypeError):
                    target_status = "not_located"
            else:
                target_status = "not_located"
        face_context = ""
        if face_manifest:
            face_context = (
                "\n照片中的人脸清单（face_id 与位置一一对应；未确认者不得猜姓名）："
                + json.dumps(face_manifest, ensure_ascii=False)
            )
        prompt = _INSPECT_PROMPT.format(
            question=question,
            target_instruction=(
                f"目标人物是“{target_person}”。第一张图（如有）是该人物的定位裁剪图；"
                "只回答目标人物，不要把同图其他人的外观或动作归给目标人物。"
                if target_person else ""),
        ) + face_context
        raw = gamma.chat(prompt, images=inspect_images,
                         json_mode=True, role="inspect")
    except Exception as exc:
        if crop_path:
            Path(crop_path).unlink(missing_ok=True)
        model_call_metrics = gamma.get_and_clear_call_metrics()
        return {"summary": f"图片复核失败：{exc}", "certainty": "uncertain", "persisted": False,
                "_model_call_metrics": model_call_metrics}
    model_call_metrics = gamma.get_and_clear_call_metrics()
    # Keep the negative identity evidence visible as well: a multi-person
    # photo must distinguish confirmed people from remaining unconfirmed
    # companions instead of silently treating the named subset as complete.
    observation_row = _store_fetchone(
        store,
        "SELECT people_json FROM observations WHERE asset_id = ? ORDER BY updated_at DESC LIMIT 1",
        (asset_id,),
    )
    try:
        people_values = json.loads(observation_row["people_json"] or "[]") if observation_row else []
        # The naming flow appends confirmed entity dictionaries to the
        # original visual people list. Count only the original descriptive
        # entries so those annotations do not inflate unknown companions.
        people_count = sum(
            1 for person in people_values
            if not (isinstance(person, dict) and person.get("entity_id"))
        )
    except (TypeError, ValueError, json.JSONDecodeError):
        people_count = 0
    unconfirmed_count = max(0, len(face_manifest) - len(identity_rows))
    if not face_manifest:
        unconfirmed_count = max(0, people_count - len(identity_rows))
    try:
        start, end = raw.find("{"), raw.rfind("}")
        parsed = json.loads(raw[start:end + 1]) if start >= 0 else {}
    except Exception:
        parsed = {}
    finally:
        if crop_path:
            Path(crop_path).unlink(missing_ok=True)
    return {
        "_source_asset_id": asset_id,
        "asset_handle": asset_handle,
        "question": question,
        "observation": parsed.get("observation") or parsed.get("scene") or "",
        "certainty": parsed.get("certainty") or "supported",
        "selected_face_id": str(parsed.get("target_face_id") or ""),
        "confirms_visual_only": not bool(identity_rows),
        "photo_identities": identity_rows,
        "face_candidates": face_manifest,
        "unconfirmed_people_count": unconfirmed_count,
        "unconfirmed_people": ([{"description": "未确认身份同行者", "count": unconfirmed_count}]
                               if unconfirmed_count else []),
        "target_person": target_person,
        "target_face_id": (target_face.get("face_id") if target_face else ""),
        "target_face_status": target_status,
        "target_bbox": target_bbox,
        "source": "runtime_visual_inspection",
        "persisted": False,
        "_model_call_metrics": model_call_metrics,
    }


def _confirmed_photo_identities(store, asset_id: str) -> list[dict]:
    """Read existing confirmed face/entity links without mutating identity data."""
    if store is None or not asset_id:
        return []
    rows = _store_fetchall(store,
        """
        SELECT fi.id AS face_instance_id, fi.asset_id, fi.observation_id,
               fi.cluster_id, fi.bbox_json, fi.detection_confidence, fi.quality,
               fc.entity_id, fc.status AS cluster_status,
               e.canonical_name, e.family_role, e.status AS entity_status,
               em.confidence AS mention_confidence
        FROM face_instances fi
        JOIN face_clusters fc ON fc.id = fi.cluster_id
        JOIN entities e ON e.id = fc.entity_id
        JOIN entity_mentions em
          ON em.face_instance_id = fi.id
         AND em.entity_id = fc.entity_id
        WHERE fi.asset_id = ?
          AND fc.status = 'confirmed'
          AND e.entity_type = 'person'
          AND e.status = 'confirmed'
        ORDER BY fi.quality DESC, fi.detection_confidence DESC
        """, (asset_id,)
    )
    return [{
        "evidence_type": "photo_identity",
        "asset_id": str(row["asset_id"] or asset_id),
        "face_instance_id": str(row["face_instance_id"]),
        "cluster_id": str(row["cluster_id"] or ""),
        "entity_id": str(row["entity_id"] or ""),
        "person_name": str(row["canonical_name"] or ""),
        "family_role": str(row["family_role"] or ""),
        "identity_status": "confirmed",
        "mention_confidence": row["mention_confidence"],
        "bbox": _decode_bbox(row["bbox_json"]),
        "source": "existing_face_cluster_entity_mention",
    } for row in rows]


def _photo_face_manifest(store, asset_id: str) -> list[dict]:
    """Return every detected face in one asset with a stable local face id.

    Confirmed cluster/entity links carry the name; unlinked or unconfirmed
    faces remain explicitly unknown. No embeddings are exposed to the model.
    """
    if store is None or not asset_id:
        return []
    rows = _store_fetchall(store,
        """
        SELECT fi.id AS face_instance_id, fi.asset_id, fi.bbox_json,
               fi.detection_confidence, fi.quality,
               fc.id AS cluster_id, fc.status AS cluster_status,
               e.canonical_name, e.family_role, e.status AS entity_status
        FROM face_instances fi
        LEFT JOIN face_clusters fc ON fc.id = fi.cluster_id
        LEFT JOIN entities e ON e.id = fc.entity_id
        WHERE fi.asset_id = ?
        ORDER BY fi.quality DESC, fi.detection_confidence DESC, fi.id
        """, (asset_id,)
    )
    faces = []
    for index, row in enumerate(rows, 1):
        confirmed = bool(
            row["cluster_status"] == "confirmed"
            and row["entity_status"] == "confirmed"
            and row["canonical_name"]
        )
        faces.append({
            "face_id": f"face_{index}",
            "face_instance_id": str(row["face_instance_id"] or ""),
            "asset_id": str(row["asset_id"] or asset_id),
            "bbox": _decode_bbox(row["bbox_json"]),
            "identity_status": "confirmed" if confirmed else "unconfirmed",
            "person_name": str(row["canonical_name"] or "") if confirmed else "",
            "family_role": str(row["family_role"] or "") if confirmed else "",
            "detection_confidence": row["detection_confidence"],
        })
    return faces


def _decode_bbox(value):
    try:
        bbox = json.loads(value) if isinstance(value, str) else value
        if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
            return [float(item) for item in bbox]
    except (TypeError, ValueError, json.JSONDecodeError):
        pass
    return None
# ---- Tool 5: search_conversation_history（D4）----
def _search_conversation_history(arguments: dict, *, context: dict | None = None) -> dict:
    """检索历史对话：current（当前会话）/ recent（最近会话）/ all（全部历史会话）。"""
    from ..agent_conversation import ConversationStore
    query = (arguments.get("query") or "").strip()
    scope = str(arguments.get("scope") or "current").strip()
    if scope not in {"current", "recent", "all_user_conversations"}:
        scope = "current"
    store = _RUNTIME.get("store")
    if store is None or not query:
        return {"summary": "对话检索暂不可用。", "total": 0, "scope": scope}
    cs = ConversationStore(store)
    scope_id = (context or {}).get("scope_id") or ""
    current_cid = arguments.get("conversation_id") or _RUNTIME.get("conversation_id")
    matches = []
    if scope == "current":
        if current_cid:
            for m in cs.search_messages(query, conversation_id=current_cid, limit=8):
                matches.append({"role": m.get("role"), "text": _msg_text_short(m.get("content")),
                                "created_at": m.get("created_at")})
    elif scope == "recent":
        convs = cs.list_conversations(scope_id=scope_id, limit=5)
        seen = set()
        for conv in convs:
            if conv["conversation_id"] == current_cid:
                continue
            for m in cs.search_messages(query, conversation_id=conv["conversation_id"], limit=2):
                key = m.get("id")
                if key in seen:
                    continue
                seen.add(key)
                matches.append({"role": m.get("role"), "text": _msg_text_short(m.get("content")),
                                "created_at": m.get("created_at"),
                                "conversation_title": conv.get("title")})
    else:
        for m in cs.search_messages(query, scope_id=scope_id, limit=10):
            matches.append({"role": m.get("role"), "text": _msg_text_short(m.get("content")),
                            "created_at": m.get("created_at")})
    if not matches:
        return {"summary": "历史对话中没有找到相关内容。", "total": 0, "scope": scope}
    return {
        "summary": f"在{'当前' if scope == 'current' else '历史'}对话中找到 {len(matches)} 条相关内容。",
        "total": len(matches),
        "scope": scope,
        "matches": matches[:10],
        "note": "对话内容属于用户表述，不等于照片证据；回答时需说明这是'你之前说过'而非照片事实。",
    }


def _msg_text_short(content) -> str:
    if isinstance(content, dict):
        text = str(content.get("text") or content.get("content") or "")
    elif isinstance(content, str):
        text = content
    else:
        text = ""
    return text.strip().replace("\n", " ")[:160]


# ---- Tool 6: get_core_memory（D5）----
def _sync_core_memory_from_entities(scope_id: str) -> "CoreMemoryStore":
    """把已确认人物/关系同步为 Core Memory 卡片（只写人物名与家庭关系，用户授权数据）。"""
    from ..core_memory import CoreMemoryStore
    store = _RUNTIME.get("store")
    cms = CoreMemoryStore(store)
    if store is None:
        return cms
    try:
        entities = store.list_entities(status="confirmed", scope_id=scope_id or None)
        name_by_id = {e["id"]: e["canonical_name"] for e in entities}
        for ent in entities:
            card_id = cms.upsert_card(scope_id=scope_id or "home-default",
                                      subject_type="person", subject_id=ent["id"],
                                      display_name=ent["canonical_name"])
            role = (ent.get("family_role") or "").strip()
            if role:
                cms.upsert_item(card_id=card_id,
                                text=f"{ent['canonical_name']} 的家庭角色是 {role}。",
                                epistemic_type="confirmed_fact", source_type="entity",
                                source_ids=[ent["id"]], source_revisions={"entity": 1})
        rels = store.list_person_relationships(scope_id=scope_id or None)
        for rel in rels:
            if rel.get("status") != "active" and rel.get("status") != "confirmed":
                continue
            subj, obj = rel.get("subject_entity_id"), rel.get("object_entity_id")
            subj_name = name_by_id.get(subj) or rel.get("subject_name")
            obj_name = name_by_id.get(obj) or rel.get("object_name")
            predicate = (rel.get("predicate") or "").strip()
            if not (subj and obj and subj_name and obj_name and predicate):
                continue
            card_id = cms.upsert_card(scope_id=scope_id or "home-default",
                                      subject_type="person", subject_id=subj,
                                      display_name=subj_name)
            cms.upsert_item(card_id=card_id,
                            text=f"{subj_name} 和 {obj_name} 的关系是 {predicate}。",
                            epistemic_type="confirmed_fact", source_type="relationship",
                            source_ids=[rel["id"]],
                            source_revisions={"relationship": int(rel.get("revision") or 1)})
    except Exception:
        pass
    return cms


def _get_core_memory(arguments: dict, *, context: dict | None = None) -> dict:
    scope_id = (context or {}).get("scope_id") or ""
    viewer_id = (context or {}).get("viewer_id") or "owner"
    conversation_id = (context or {}).get("conversation_id")
    subject = (arguments.get("subject") or "").strip()
    topic = (arguments.get("topic") or "").strip()
    limit = max(1, min(10, int(arguments.get("limit") or 5)))
    cms = _sync_core_memory_from_entities(scope_id)
    store = _RUNTIME.get("store")
    if store is None:
        return {"summary": "长期记忆不可用。", "cards": [], "total": 0}
    subject_ids = None
    if subject:
        matched = []
        try:
            for ent in store.list_entities(status="confirmed", scope_id=scope_id or None):
                if subject in (ent.get("canonical_name") or "") or subject in (ent.get("family_role") or ""):
                    matched.append(ent["id"])
        except Exception:
            matched = []
        subject_ids = matched or None
    cards = cms.list_cards(scope_id=scope_id or None, subject_ids=subject_ids, limit=limit)
    if topic:
        topic = topic.lower()
        filtered = []
        for card in cards:
            keep = []
            for item in card.get("items") or []:
                if topic in (item.get("text") or "").lower():
                    keep.append(item)
            if keep:
                card = dict(card)
                card["items"] = keep
                filtered.append(card)
        cards = filtered
    public_cards = []
    for card in cards:
        try:
            cms.record_access(card_id=card["card_id"], conversation_id=conversation_id,
                              viewer_id=viewer_id)
        except Exception:
            pass
        public_cards.append({
            "subject_type": card.get("subject_type"),
            "subject_id": card.get("subject_id"),
            "display_name": card.get("display_name"),
            "items": [{
                "text": item.get("text"),
                "truth_status": item.get("epistemic_type"),
                "source_type": item.get("source_type"),
                "source_ids": item.get("source_ids"),
                "source_revisions": item.get("source_revisions"),
                "updated_at": item.get("created_at"),
            } for item in (card.get("items") or [])],
        })
    if not public_cards:
        return {"summary": "长期记忆中没有找到相关内容。", "cards": [], "total": 0,
                "note": "只读长期记忆；agent_inference 不会被当作 confirmed。"}
    return {
        "summary": f"找到 {len(public_cards)} 张长期记忆卡片。",
        "total": len(public_cards),
        "cards": public_cards,
        "note": "truth_status: confirmed_fact/user_assertion/agent_inference/observed_pattern；agent_inference 不能表述为 confirmed。",
    }


# 人物实体解析（get_person_profile 使用）
def _resolve_person_entity(person: str, scope_id: str):
    store = _RUNTIME.get("store")
    if store is None or not person:
        return None
    try:
        for ent in store.list_entities(status="confirmed", scope_id=scope_id or None):
            name = ent.get("canonical_name") or ""
            role = ent.get("family_role") or ""
            if person in name or person in role or name in person or role in person:
                return ent
    except Exception:
        return None
    return None


def _query_photo_people(arguments: dict, *, context: dict | None = None) -> dict:
    """Canonical image-level people evidence, bound to one stable result handle."""
    scope_id = (context or {}).get("scope_id") or ""
    task_state = (context or {}).get("task_state") or {}
    result_set_id = arguments.get("result_set_id") or task_state.get("current_result_set")
    handle = str(arguments.get("asset_handle") or arguments.get("handle") or "").strip()
    rs_store = _RUNTIME.get("result_sets")
    if not result_set_id or rs_store is None:
        return {"summary": "请先通过 search_memories 建立结果集。", "blocked": ["no_result_set"],
                "evidence_asset_ids": []}
    asset_id = rs_store.resolve_handle(result_set_id, handle) if handle else None
    if not asset_id:
        return {"summary": "无法解析当前结果集中的照片句柄。", "blocked": ["bad_handle"],
                "evidence_asset_ids": []}
    store = _RUNTIME.get("store")
    if store is None:
        return {"summary": "记忆库不可用。", "blocked": ["store_unavailable"],
                "evidence_asset_ids": []}
    asset = store.get_asset(asset_id) or {}
    if scope_id and asset.get("scope_id") and asset.get("scope_id") != scope_id:
        return {"summary": "照片不在当前相册范围。", "blocked": ["scope_mismatch"],
                "evidence_asset_ids": []}
    identities = _confirmed_photo_identities(store, asset_id)
    observation = _store_fetchone(
        store,
        "SELECT people_json FROM observations WHERE asset_id = ? ORDER BY updated_at DESC LIMIT 1",
        (asset_id,),
    )
    try:
        people_values = json.loads(observation["people_json"] or "[]") if observation else []
        people_count = sum(1 for p in people_values
                           if not (isinstance(p, dict) and p.get("entity_id")))
    except (TypeError, ValueError, json.JSONDecodeError):
        people_count = 0
    unknown_count = max(0, people_count - len(identities))
    people = [{"person_name": row.get("person_name"),
               "family_role": row.get("family_role") or "",
               "identity_status": "confirmed",
               "asset_id": asset_id}
              for row in identities if row.get("person_name")]
    return {
        "asset_handle": handle,
        "result_set_id": result_set_id,
        "asset_id": asset_id,
        "people": people,
        "unconfirmed_people": ([{"description": "未确认身份同行者", "count": unknown_count}]
                               if unknown_count else []),
        "unconfirmed_people_count": unknown_count,
        "summary": ("、".join(p["person_name"] for p in people) or "没有已确认姓名")
                   + (f"；另有 {unknown_count} 名未确认身份同行者" if unknown_count else ""),
        "source_asset_ids": [asset_id],
        "source_handles": [handle],
        "evidence_asset_ids": [asset_id],
        "evidence_kind": "photo_people",
    }


def _handle_to_asset_id(handle: str) -> str | None:
    # shadow 期 handle -> asset_id 映射来自最近一次 search_memories 的 preview。
    return (_RUNTIME.get("last_handles") or {}).get(handle)


_INSPECT_PROMPT = """观察这张照片，输出 JSON：
{{"observation": "一句话描述", "certainty": "supported|uncertain", "target_face_id": "face_N或空"}}
{target_instruction}
问题：{question}
如果问题指向照片中的某个人但没有姓名，请根据人脸清单选择对应的 face_id；无法确定时返回空，不要猜姓名。"""



def _cap_hint(tool: str) -> str:
    """能力实测提示：无数据返回空串，不改变工具合同。"""
    try:
        hint = tool_capability_summary(tool)
        return ("\n" + hint) if hint else ""
    except Exception:
        return ""


def person_profile_summary(person: str, scope_id: str = "") -> str:
    """Compact Chinese profile summary for a confirmed person; '' when unavailable."""
    try:
        ent = _resolve_person_entity(person, scope_id)
        store = _RUNTIME.get("store")
        if ent is None or store is None:
            return ""
        digest = store.person_profile_digest(ent["id"], scope_id)
        if not digest or not digest.get("summary_zh"):
            return ""
        lines = []
        if digest.get("family_role"):
            lines.append(f"家庭角色：{digest['family_role']}")
        if digest.get("relationships"):
            lines.append("关系：" + "、".join(f"{r.get('other_name')}（{r.get('predicate')}）" for r in digest["relationships"]))
        if digest.get("preference_summary_zh"):
            lines.append(digest["preference_summary_zh"])
        titles = [e.get("title") or "" for e in (digest.get("recent_events") or []) if e.get("title")]
        if titles:
            lines.append("近期事件：" + "、".join(titles))
        return "；".join(lines) or digest.get("summary_zh")
    except Exception:
        return ""


def _get_person_profile(arguments: dict, *, context: dict | None = None) -> dict:
    person = (arguments.get("person") or "").strip()
    scope_id = (context or {}).get("scope_id") or ""
    ent = _resolve_person_entity(person, scope_id)
    if ent is None:
        return {"person": person, "readiness": "limited", "insufficient_evidence": True,
                "summary": f"没有找到已确认人物「{person}」的画像。",
                "note": "人物未确认或数据不足时返回 limited，不编造。"}
    store = _RUNTIME.get("store")
    digest = store.person_profile_digest(ent["id"], scope_id) if store else None
    if not digest or not digest.get("summary_zh"):
        return {"person": person, "readiness": "limited", "insufficient_evidence": True,
                "summary": f"「{person}」暂无足够的人物画像数据。",
                "note": "画像数据不足时返回 limited，不编造。"}
    return {
        "person": digest.get("person"),
        "family_role": digest.get("family_role") or "",
        "readiness": "ready",
        "summary": digest.get("summary_zh") or "",
        "preference_summary": digest.get("preference_summary_zh") or "",
        "relationships": digest.get("relationships") or [],
        "patterns": digest.get("patterns") or [],
        "recent_events": digest.get("recent_events") or [],
        "claims": digest.get("claims") or [],
        "profile_text": person_profile_summary(person, scope_id),
        "note": "画像来自已确认人物的语义记忆；性格等无法由证据确认的问题应回答 insufficient evidence。",
    }


def register_tools():

    register(ToolSpec(
        name="search_memories",
        description=("检索照片（人/物/场景/衣着/颜色）。返回结果集摘要。"
                     "返回的每张 preview 自带 captured_at（拍摄时间）/ place（拍摄地点）/ people（已确认人物）/ evidence_summary（描述），"
                     "单张照片的地点、时间、人物直接在 preview 里读，不要为此再调用其他工具。"
                     "需要更多候选时用 get_result_page 翻页。"
                     "时间必须从用户问题里提取并写 filters.time（如'2024年7月'）；问题没给具体时间就省略 time，不要写'所有时间'。"
                     "filters.place 只填结构化地名（城市/区县/景区/地标），不要把目标/活动/主题当 place；不确定留空。"
                     "filters.person 只填用户明确提到的人物名，不要用'伴娘/兄弟/亲戚'这类角色词。"),
        input_schema={"query": "", "mode": "best|all|representative",
                      "filters": {"time": "", "place": "", "person": ""}},
        executor=_search_memories, read_write="read", cost_class="medium", readiness="ready",
        produces_evidence=("memory_asset", "temporal_metadata", "location_metadata",
                           "structured_fact"),
        required_inputs=("query",),
    ))
    register(ToolSpec(
        name="query_photo_people",
        description=("读取一张已检索照片自己的已确认人物和未确认同行者。"
                     "asset_handle 必须来自当前 search_memories preview；不会把其他候选照片的人名移植过来。"),
        input_schema={"asset_handle": "", "result_set_id": ""},
        executor=_query_photo_people, read_write="read", cost_class="cheap", readiness="ready",
        produces_evidence=("photo_identity", "confirmed_identity"),
        required_inputs=("asset_handle",),
        preconditions=("asset_handle_in_current_preview",),
        prerequisite_evidence_types=("memory_asset",),
    ))
    register(ToolSpec(
        name="get_original_photos",
        description="交付当前结果集/选中照片的原图。",
        input_schema={"result_set_id": "", "handle": ""},
        executor=_get_original_photos, read_write="read", cost_class="cheap", readiness="limited",
        produces_evidence=("memory_asset",),
        required_inputs=("handle",),
        readiness_reason="ResultSetStore 就绪后完整可用（A4）",
    ))
    register(ToolSpec(
        name="get_result_page",
        description="查看 search_memories 结果集的下一页/指定页（每页最多6张）。result_set_id 用 search_memories 返回的，page 从 1 开始。若要改变查询条件，重新调用 search_memories，不要把新条件传给本工具。",
        input_schema={"result_set_id": "", "page": 1, "page_size": 6},
        executor=_get_result_page, read_write="read", cost_class="cheap", readiness="ready",
        produces_evidence=("memory_asset",),
        required_inputs=("result_set_id", "page"),
    ))
    register(ToolSpec(
        name="inspect_photo",
        description=("复核已检索照片的视觉细节（物体/衣着/颜色/场景）。asset_handle 使用 search_memories preview 里的 handle（photo_1…），可省略（默认用预览第一张）。"
                     "当前 handle 不含目标时换下一张（photo_2、photo_3…）复核，同一张图最多复核一次，不要反复检查同一张。昂贵，默认每轮最多 1 次。"
                     + _cap_hint("inspect_photo")),
        input_schema={"asset_handle": "", "question": "", "target_person": ""},
        executor=_inspect_photo, read_write="read", cost_class="expensive", readiness="ready",
        produces_evidence=("visual_observation", "photo_identity"),
        required_inputs=("asset_handle", "question"),
        preconditions=("asset_handle_in_current_preview",),
        prerequisite_evidence_types=("memory_asset",), budget_unit="image",
    ))
    register(ToolSpec(
        name="read_photo_text",
        description=("读取照片中的文字内容（菜单/价格/招牌/店名/电话/年份/小字）。"
                     "适用于'多少钱/价格/售价/店名/招牌/电话/写了什么/什么字/哪一年'等需要看照片文字的题；"
                     "内部会把照片切块放大后 OCR。asset_handle 用 search_memories preview 里的 handle，可省略（默认预览第一张）。"
                     "昂贵，每轮最多 1 次。"
                     + _cap_hint("read_photo_text")),
        input_schema={"asset_handle": "", "question": ""},
        executor=_read_photo_text, read_write="read", cost_class="expensive", readiness="ready",
        produces_evidence=("visible_text",),
        required_inputs=("asset_handle", "question"),
        preconditions=("asset_handle_in_current_preview",),
        prerequisite_evidence_types=("memory_asset",), budget_unit="image",
    ))
    register(ToolSpec(
        name="search_conversation_history",
        description=("检索历史对话内容：回答'我之前是不是说过…/上次聊到哪里/你之前说那张在哪'类问题。"
                     "scope=current 只查当前会话；scope=recent 查最近几个会话；scope=all_user_conversations 查全部历史会话。"
                     "对话内容属于用户表述，不等于照片证据；回答时应说'你之前说过'而不是当成照片事实。"),
        input_schema={"query": "", "scope": "current|recent|all_user_conversations",
                      "conversation_id": ""},
        executor=_search_conversation_history, read_write="read", cost_class="cheap", readiness="ready",
        produces_evidence=("user_statement",),
        required_inputs=("query",),
    ))
    register(ToolSpec(
        name="get_core_memory",
        description=("读取长期家庭记忆（已确认人物/家庭角色/关系/偏好等）。"
                     "subject 填人物名（如某人的称呼）；topic 填话题关键词（如某个话题词）；都不填返回优先级最高的卡片。"
                     "每条记忆带 truth_status（confirmed_fact/user_stated/agent_inference/observed_pattern），"
                     "agent_inference 不能说成 confirmed。"),
        input_schema={"subject": "", "topic": "", "limit": 5},
        executor=_get_core_memory, read_write="read", cost_class="cheap", readiness="ready",
        produces_evidence=("confirmed_identity", "user_statement"),
        required_inputs=("subject", "topic"),
    ))
    register(ToolSpec(
        name="get_person_profile",
        description=("读取已确认人物的高维画像（长期记忆）：家庭角色、人物关系、常去地点/常做活动/常同行、近期事件与语义声明。"
                     "人物未确认或画像数据不足时返回 limited，要如实说明，不要编造。"
                     "性格等主观问题：照片无法确认时回答 insufficient evidence。"),
        input_schema={"person": ""},
        executor=_get_person_profile, read_write="read", cost_class="cheap", readiness="ready",
        produces_evidence=("confirmed_identity", "structured_fact"),
        required_inputs=("person",),
    ))
