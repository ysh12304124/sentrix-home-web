# -*- coding: utf-8 -*-
"""
Semantic Label Matcher

Replaces translation + exact-string matching with direct cross-lingual
semantic matching between user questions and database labels.

Uses paraphrase-multilingual-MiniLM-L12-v2 (470MB, cached locally) to encode
both the question and all object_labels / relation_labels into a shared
embedding space. Cosine similarity finds the best matching labels regardless
of input language.

Build cost (one-time at load): encode ~100 labels, ~0.1s.
Query cost: encode 1 question (~5ms) + matmul (~0.1ms).
"""

import hashlib
import logging
import os
import re
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import numpy as np

logger = logging.getLogger(__name__)

_DEFAULT_MODEL = os.getenv("GRAPH_SEMANTIC_MODEL", "").strip()
if not _DEFAULT_MODEL:
    _MODEL_CANDIDATES = [
        Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) / "models" / "multilingual-minilm",
        Path(r"D:\MAGMA-V3\models\multilingual-minilm"),
    ]
    _DEFAULT_MODEL = str(next((item for item in _MODEL_CANDIDATES if item.is_dir()), _MODEL_CANDIDATES[0]))

# Predicate labels that map to structured_index predicates
_KNOWN_PREDICATES = {
    "holding", "carrying", "standing on", "leaning on", "looking at",
    "sitting on", "walking on", "driving on", "parked on", "hanging from",
    "wearing", "touching", "enclosing", "on back of", "in front of",
    "attached to", "on", "over", "in", "beside",
}

# Visual attribute keywords: questions about these CANNOT be answered
# from text labels alone - they require looking at the actual frame image.
_VISUAL_ATTR_KEYWORDS = {
    # colors (Chinese + English)
    "黑", "白", "红", "绿", "蓝", "黄", "紫", "灰", "棕", "粉",
    "color", "colour", "black", "white", "red", "green", "blue",
    "yellow", "purple", "gray", "grey", "brown", "pink", "orange",
    # materials / texture
    "木", "金属", "玻璃", "塑料", "布料", "皮质", "皮革", "纹理",
    "材质", "材料", "质地",
    "wood", "metal", "glass", "plastic", "fabric", "leather", "texture",
    "material",
    # spatial position (frame-local, not covered by labels)
    "左边", "右边", "上面", "下面", "左侧", "右侧", "上方", "下方",
    "远处", "近处", "位置", "角落", "中间",
    "left", "right", "top", "bottom", "above", "below", "corner",
    # appearance / clothing / worn items
    "穿", "戴", "眼镜", "帽子", "围巾", "手套", "大小", "形状",
    "wear", "wearing", "glasses", "hat", "scarf", "glove", "size", "shape",
    "look like", "appearance", "looks like",
    # facial expression / emotion
    "表情", "笑", "哭", "开心", "难过", "生气", "惊讶", "疲惫",
    "expression", "smile", "crying", "happy", "sad", "angry", "surprised",
    "tired", "emotion",
    # person action described from the picture itself
    "在干嘛", "在干吗", "干什么", "在干什么", "干嘛呢", "干啥",
    "在做什么", "做什么呢", "在忙什么",
    # text on screen / signs
    "文字", "写着", "屏幕", "招牌", "标语", "什么字",
    "text", "written", "screen", "sign", "label", "logo",
    # counting / visual quantity
    "几个", "多少个", "多少人", "数量", "几件",
    "how many", "how much", "count",
}

# Seed phrases describing visual-only questions, used for semantic matching.
_VISUAL_ATTR_SEEDS = [
    "what color is the object",
    "what is the person wearing",
    "what does the person look like",
    "what is the facial expression",
    "what is on the left side of the image",
    "what is the object made of",
    "what does the text on screen say",
    "how many people are in the scene",
    "what is the shape of the object",
    "is the object big or small",
]

# Questions answerable from structured labels/relations/graph edges.
_TEXT_ATTR_SEEDS = [
    "what is the person holding",
    "what is the person doing",
    "where is the person standing",
    "what is the person looking at",
    "what is the person leaning on",
    "what is the scene",
    "is this indoor or outdoor",
    "what time is this",
    "what objects are in the frame",
    "what happened before or after this",
    "is there an object in the frame",
]

# Identity/metadata questions that images and labels cannot answer.
_METADATA_ATTR_SEEDS = [
    "what is the person's name",
    "who is the person",
    "what is the person's identity",
    "what country was this filmed in",
    "what is the person's job",
    "what company does the person work for",
    "what is the person's phone number",
    "how old is the person",
]


def _contains_visual_keyword(q_low: str) -> bool:
    """Match rule keywords as words, not as accidental substrings.

    English keywords use word boundaries so "hat" does not match "what";
    Chinese category words are multi-character, so "字" does not match "名字".
    """
    for kw in _VISUAL_ATTR_KEYWORDS:
        if re.fullmatch(r"[\u4e00-\u9fff]+", kw):
            if kw in q_low:
                return True
        elif re.search(r"(?<![a-z0-9])" + re.escape(kw) + r"(?![a-z0-9])", q_low):
            return True
    return False



class SemanticLabelMatcher:
    """Match user questions to database labels via cross-lingual embeddings."""

    def __init__(self, model_name=None):
        self._model_name = model_name or _DEFAULT_MODEL
        self._model = None
        self._visual_seed_embs = None
        self._text_seed_embs = None
        self._meta_seed_embs = None
        self._labels: List[str] = []
        self._label_embs: Optional[np.ndarray] = None
        self._predicates: set = set()

    def _ensure_model(self):
        if self._model is not None:
            return
        os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
        from sentence_transformers import SentenceTransformer
        logger.info("Loading %s ...", self._model_name)
        self._model = SentenceTransformer(self._model_name)

    def build_from_graph(self, graph_db, structured_index=None):
        """Collect all unique labels from EVENT nodes and encode them."""
        self._ensure_model()

        label_set = set()
        for nid, node in graph_db.nodes.items():
            if node.node_type.value != "EVENT":
                continue
            attrs = getattr(node, "attributes", {}) or {}
            for label in (attrs.get("object_labels") or []):
                label_set.add(label.lower())
            for label in (attrs.get("relation_labels") or []):
                label_set.add(label.lower())

        # add predicates from structured index if available
        if structured_index:
            for pred in structured_index.predicate_index:
                label_set.add(pred.lower())
                self._predicates.add(pred.lower())

        # also mark known predicates in the label set
        for label in label_set:
            if label in _KNOWN_PREDICATES:
                self._predicates.add(label)

        self._labels = sorted(label_set)
        if not self._labels:
            logger.warning("No labels found in graph, matcher is empty")
            return

        # Label embeddings are deterministic for a given model + label set.
        # Cache them so server restarts do not re-encode thousands of labels.
        cache_key = hashlib.sha256(
            (self._model_name + "\n" + "\n".join(self._labels)).encode("utf-8")
        ).hexdigest()[:20]
        cache_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "model_cache",
        )
        cache_path = os.path.join(cache_dir, "semantic_labels_%s.npz" % cache_key)
        if os.path.isfile(cache_path):
            try:
                cached = np.load(cache_path, allow_pickle=False)
                cached_labels = cached["labels"].tolist()
                if cached_labels == self._labels:
                    self._label_embs = np.asarray(cached["embeddings"], dtype=np.float32)
                    logger.info("SemanticLabelMatcher cache hit: %d labels (%d predicates)",
                                len(self._labels), len(self._predicates))
                    return
            except Exception as e:
                logger.warning("Semantic label cache load failed, rebuilding: %s", e)

        self._ensure_model()
        self._label_embs = self._model.encode(
            self._labels, normalize_embeddings=True,
            show_progress_bar=False,
        )
        try:
            os.makedirs(cache_dir, exist_ok=True)
            tmp_path = cache_path + ".tmp.npz"
            np.savez_compressed(
                tmp_path,
                labels=np.asarray(self._labels),
                embeddings=np.asarray(self._label_embs, dtype=np.float32),
            )
            os.replace(tmp_path, cache_path)
        except Exception as e:
            logger.warning("Semantic label cache save failed: %s", e)
        logger.info("SemanticLabelMatcher built: %d labels (%d predicates)",
                    len(self._labels), len(self._predicates))

    def needs_visual(self, question: str, threshold: float = 0.55) -> bool:
        """Decide whether answering this question requires looking at images.

        Three layers:
        1. Rule layer: explicit visual attribute keywords (color, material,
           spatial position, clothing, expression, text, count).
        2. Semantic layer: compare the question against visual, text-answerable
           and metadata seed groups; visual only wins when it clearly beats
           the other interpretations.

        Returns True when text labels are unlikely to cover the answer.
        """
        if not question:
            return False

        q_low = question.lower().strip()

        # Layer 1: explicit visual attribute categories
        if _contains_visual_keyword(q_low):
            return True

        # Layer 2: semantic similarity against competing seed groups
        if self._label_embs is not None:
            try:
                # Model can remain lazy when embeddings came from disk.
                self._ensure_model()
                q_emb = self._model.encode(
                    [question], normalize_embeddings=True,
                    show_progress_bar=False,
                )[0]
                if self._visual_seed_embs is None:
                    self._visual_seed_embs = self._model.encode(
                        _VISUAL_ATTR_SEEDS, normalize_embeddings=True,
                        show_progress_bar=False,
                    )
                if self._text_seed_embs is None:
                    self._text_seed_embs = self._model.encode(
                        _TEXT_ATTR_SEEDS, normalize_embeddings=True,
                        show_progress_bar=False,
                    )
                if self._meta_seed_embs is None:
                    self._meta_seed_embs = self._model.encode(
                        _METADATA_ATTR_SEEDS, normalize_embeddings=True,
                        show_progress_bar=False,
                    )
                seed_embs = self._visual_seed_embs
                text_embs = self._text_seed_embs
                meta_embs = self._meta_seed_embs
                visual_score = float((seed_embs @ q_emb).max())
                text_score = float((text_embs @ q_emb).max())
                meta_score = float((meta_embs @ q_emb).max())
                if (visual_score >= threshold
                        and visual_score >= text_score + 0.03
                        and visual_score >= meta_score + 0.03):
                    return True
            except Exception as e:
                logger.warning("visual attribute semantic check failed: %s", e)

        return False

    def match(self, question: str, threshold: float = 0.40, top_n: int = 8) -> Dict:
        """Match a question (any language) to database labels.

        Returns dict with:
            objects: list of matched object labels (score > threshold)
            predicates: list of matched predicate labels
            all: list of (label, score) sorted by score
        """
        if self._label_embs is None or not question:
            return {"objects": [], "predicates": [], "all": []}

        self._ensure_model()
        q_emb = self._model.encode(
            [question], normalize_embeddings=True, show_progress_bar=False,
        )[0]
        scores = self._label_embs @ q_emb

        # collect top matches above threshold
        ranked = [(self._labels[i], float(scores[i]))
                  for i in np.argsort(-scores, kind="stable")[:top_n]]
        ranked.sort(key=lambda x: (-x[1], x[0]))
        matched = [(lbl, sc) for lbl, sc in ranked if sc >= threshold]

        objects = []
        predicates = []
        for lbl, sc in matched:
            if lbl in self._predicates:
                predicates.append(lbl)
            else:
                objects.append(lbl)

        return {"objects": objects, "predicates": predicates, "all": matched}
