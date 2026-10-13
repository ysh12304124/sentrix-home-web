"""
Keyframe Query Engine

Adapts MAGMA's multi-stage retrieval to keyframe-based memory:
  parallel local indexed recall + Qdrant payload recall (+ optional scan)
  -> RRF fusion -> adjacency-list graph traversal -> rerank.

Designed for scale: all vector operations are delegated to Qdrant
(never local N*N), neighbour lookup uses the NetworkX adjacency list,
and the full-scan channel can be disabled for large graphs.

The text->visual vector gap is handled pragmatically: in the default
configuration the "vector" channel queries Qdrant by payload filter
(object_labels / relation_labels match). A text_encoder hook is
reserved so a future CLIP-style encoder can issue true vector searches.
"""

import logging
import os
import time
import math
import re
from bisect import bisect_left
from collections import deque, defaultdict
from typing import List, Tuple, Set, Optional, Dict

import numpy as np

from .graph_db import LinkType, TraversalConstraints
from .keyword_enrichment import tokenize_text, contains_cjk
from .structured_index import extract_query_intent
from .semantic_label_matcher import SemanticLabelMatcher
from .label_translations import OBJECT_ZH

logger = logging.getLogger(__name__)


class KeyframeQueryEngine:
    """Retrieval engine for keyframe memory graphs."""

    # Query-type adaptive parameters (mirrors MAGMA's structure)
    ADAPTIVE_PARAMS = {
        "object": {
            "max_depth": 4,
            "prefer_link_types": [LinkType.ENTITY, LinkType.SEMANTIC],
            "similarity_threshold": 0.2,
            "weights": {"keyword": 5.0, "object": 6.0, "relation": 2.0, "temporal": 1.0},
        },
        "action": {
            "max_depth": 4,
            "prefer_link_types": [LinkType.CAUSAL, LinkType.TEMPORAL],
            "similarity_threshold": 0.2,
            "weights": {"keyword": 3.0, "object": 2.0, "relation": 6.0, "temporal": 3.0, "causal": 5.0},
        },
        "temporal": {
            "max_depth": 5,
            "prefer_link_types": [LinkType.TEMPORAL],
            "similarity_threshold": 0.2,
            "weights": {"keyword": 2.0, "object": 1.0, "relation": 1.0, "temporal": 8.0},
        },
        "scene": {
            "max_depth": 3,
            "prefer_link_types": [LinkType.SEMANTIC],
            "similarity_threshold": 0.15,
            "weights": {"keyword": 3.0, "object": 4.0, "relation": 3.0, "temporal": 1.0},
        },
        "multi_hop": {
            "max_depth": 8,
            "prefer_link_types": [LinkType.ENTITY, LinkType.SEMANTIC, LinkType.CAUSAL, LinkType.TEMPORAL],
            "similarity_threshold": 0.10,
            "weights": {"keyword": 4.0, "object": 5.0, "relation": 5.0,
                        "temporal": 3.0, "causal": 4.0},
        },
        "general": {
            "max_depth": 4,
            "prefer_link_types": None,
            "similarity_threshold": 0.25,
            "weights": {"keyword": 4.0, "object": 3.0, "relation": 3.0, "temporal": 1.5},
        },
    }

    # Edge sub-types consumed by traversal/rerank (mirrors original MAGMA).
    TEMPORAL_FORWARD = "TIME_PRECEDES"
    CAUSAL_FORWARD_SUBTYPES = ("LEADS_TO", "ENABLES")
    OBJECT_SUBTYPES = ("SAME_ENTITY", "MENTIONS_OBJECT")
    SCENE_SUBTYPES = ("VISUAL_SIMILAR", "SHARES_RELATION",
                      "CLIP_CONTAINS", "VIDEO_CONTAINS")
    ACTION_SUBTYPES = ("LEADS_TO", "ENABLES")
    STRONG_CROSS_VIDEO_SUBTYPES = (
        "VISUAL_SIMILAR", "SHARES_RELATION", "SAME_ENTITY")
    MAX_HUB_EXPANSION = 160
    # Graph edges are not equally reliable.  Deterministic timeline and
    # containment links may always be traversed; similarity/LLM-derived links
    # must carry a minimum score or they turn common objects into noisy hubs.
    GRAPH_EDGE_MIN_CONFIDENCE = {
        "VISUAL_SIMILAR": 0.80,
        "SHARES_RELATION": 0.65,
        "SAME_ENTITY": 0.80,
        "LEADS_TO": 0.60,
        "ENABLES": 0.60,
    }
    GRAPH_DETERMINISTIC_EDGES = {
        "TIME_PRECEDES", "TIME_SUCCEEDS", "SCENE_NEXT", "CLIP_CONTAINS",
        "PART_OF_CLIP", "VIDEO_CONTAINS", "PART_OF_VIDEO", "OCCURRED_AT",
        "CAPTURED_ON", "MENTIONS_PERSON", "PERSON_IN_FRAME",
    }

    def __init__(
        self,
        graph_db,
        node_index: dict,
        qdrant_adapter=None,
        text_encoder=None,
        enable_full_scan: bool = True,
        max_scan_nodes: int = 500,
        structured_index=None,
        label_matcher=None,
        enable_multi_hop_expansion: bool = False,
    ):
        """
        Args:
            graph_db: a NetworkXGraphDB (or SQLiteGraphDB) with the keyframe graph
            node_index: keyword -> set(node_id) inverted index (from the builder)
            qdrant_adapter: QdrantAdapter for payload-filter / vector recall
            text_encoder: optional callable(text)->np.ndarray for future
                CLIP-style text->visual encoding. When None, the vector
                channel falls back to Qdrant payload filtering.
            enable_full_scan: set False to skip the O(N) scan channel on large graphs
            max_scan_nodes: cap on nodes examined by the full-scan channel
            enable_multi_hop_expansion: opt-in synthetic-benchmark path expansion;
                keep False for production and real-GT comparison
        """
        self.graph_db = graph_db
        self.node_index = node_index
        self.qa = qdrant_adapter
        self.text_encoder = text_encoder
        self.enable_full_scan = enable_full_scan
        self.max_scan_nodes = max_scan_nodes
        self.structured_index = structured_index
        self.label_matcher = label_matcher
        # Synthetic multi-hop benchmarks need aggressive backward path recovery.
        # Production retrieval keeps the original conservative graph traversal so
        # benchmark tuning does not change normal user-facing answers.
        self.enable_multi_hop_expansion = enable_multi_hop_expansion
        self._event_time_index = None

    @staticmethod
    def _scope_match(node, scope: Optional[Dict]) -> bool:
        """Return whether an EVENT node belongs to the requested eval scope.

        Non-EVENT hubs are allowed as traversal conduits. Their own attributes
        do not carry clip/video identity, while their neighbours are filtered
        against the scope.
        """
        if not scope:
            return True
        attrs = getattr(node, "attributes", {}) or {}
        scope_id = scope.get("scope_id")
        if scope_id and node.node_type.value == "EVENT":
            if str(attrs.get("scope_id") or "home-default") != str(scope_id):
                return False
        if node.node_type.value == "EVENT":
            clip_uid = scope.get("clip_uid")
            video_uid = scope.get("video_uid")
            if clip_uid and attrs.get("clip_uid") != clip_uid:
                return False
            if video_uid and attrs.get("video_uid") != video_uid:
                return False

        tasks = [str(x) for x in (scope.get("tasks") or []) if str(x)]
        if tasks:
            node_tasks = set(attrs.get("tasks") or [])
            single_task = attrs.get("task")
            if single_task:
                node_tasks.add(single_task)
            # Old graph snapshots may not carry task lists. Keep them usable;
            # a graph built with a task filter is already restricted at build time.
            if node_tasks and not (node_tasks & set(tasks)):
                return False
        return True

    @staticmethod
    def _qdrant_scope_filter(scope: Optional[Dict]):
        if not scope:
            return None
        must = []
        if scope.get("clip_uid"):
            must.append({"key": "clip_uid", "match": {"value": scope["clip_uid"]}})
        if scope.get("video_uid"):
            must.append({"key": "video_uid", "match": {"value": scope["video_uid"]}})
        tasks = [str(x) for x in (scope.get("tasks") or []) if str(x)]
        if len(tasks) == 1:
            must.append({"key": "task", "match": {"value": tasks[0]}})
        elif tasks:
            must.append({"key": "task", "match": {"any": tasks}})
        return {"must": must} if must else None

    # ------------------------------------------------------------------ #
    #  Query-type detection (video-oriented)
    # ------------------------------------------------------------------ #

    def detect_query_type(self, question: str, matched=None) -> str:
        """Detect query type from original question + semantic label matches.

        Works on ANY language: Chinese patterns, English patterns, and
        semantic label matches are all checked. No translation needed.
        """
        q = question.lower()
        matched = matched or {}

        # temporal: explicit time anchors / durations (both languages)
        if re.search(r"\d+\s*(s|sec|second|seconds|\u79d2|\u5206\u949f|\u5c0f\u65f6)", q):
            return "temporal"
        if re.search(r"\u7b2c.*?(\u79d2|\u5206\u949f|\u5e27)", q):
            return "temporal"
        temporal_kw = [
            "when", "what time", "what second", "what minute", "what hour",
            "how long", "at what time", "which moment", "timestamp",
            "\u4ec0\u4e48\u65f6\u5019", "\u4f55\u65f6", "\u51e0\u70b9",
            "\u51e0\u79d2", "\u7b2c\u51e0\u79d2", "\u7b2c\u51e0\u5206\u949f",
            "\u7b2c\u51e0\u5e27", "\u54ea\u4e2a\u65f6\u95f4", "\u4ec0\u4e48\u65f6\u95f4",
            "\u591a\u4e45", "\u591a\u957f\u65f6\u95f4", "\u65f6\u95f4\u70b9",
            "\u65f6\u95f4\u7ebf",
        ]
        for k in temporal_kw:
            if k in q:
                return "temporal"

        # causal / sequence / cross-video reasoning
        mh_kw = [
            "after", "before", "then", "next", "later", "first", "sequence",
            "order", "timeline", "chronological", "process", "steps",
            "both", "together", "same time", "simultaneously",
            "state change", "change from", "change to", "transition",
            "became", "becomes", "changed", "different videos", "across videos",
            "two videos", "multiple videos", "other video", "another video",
            "which videos", "videos contain", "interact with",
            "what happened before", "what happened after",
            "what happened next", "what led to", "what caused", "how did",
            "compare", "relationship between", "cause", "because", "why",
            "result of", "result in", "reason",
            "\u540c\u65f6", "\u4e4b\u540e", "\u4e4b\u524d", "\u7136\u540e",
            "\u63a5\u7740", "\u5148", "\u540e\u6765", "\u524d\u540e",
            "\u987a\u5e8f", "\u8fc7\u7a0b", "\u6b65\u9aa4",
            "\u4e0d\u540c\u89c6\u9891", "\u5404\u4e2a\u89c6\u9891",
            "\u4e24\u4e2a\u89c6\u9891", "\u591a\u4e2a\u89c6\u9891",
            "\u4e24\u4e2a\u7247\u6bb5", "\u591a\u4e2a\u7247\u6bb5",
            "\u8de8\u89c6\u9891",
            "\u72b6\u6001\u53d8\u5316", "\u53d8\u4e3a", "\u53d8\u6210",
            "\u72b6\u6001", "\u7ecf\u5386\u4e86\u54ea\u4e9b\u72b6\u6001",
            "\u53d1\u751f\u4e86\u4ea4\u4e92", "\u53d1\u751f\u4ea4\u4e92",
            "\u4ea4\u4e92", "\u4e3a\u4ec0\u4e48", "\u539f\u56e0", "\u5bfc\u81f4",
            "\u56e0\u4e3a", "\u4e3a\u4f55", "\u6240\u4ee5", "\u7ed3\u679c",
            "\u56e0\u679c\u5173\u7cfb", "\u600e\u4e48\u53d1\u751f",
            "\u5982\u4f55\u53d1\u751f", "\u600e\u4e48\u56de\u4e8b",
            "\u6bd4\u8f83", "\u5bf9\u6bd4", "\u5173\u7cfb", "\u8054\u7cfb",
        ]
        if re.search(r"\u4ece.*?\u5230", q):
            return "multi_hop"
        if any(k in q for k in mh_kw):
            return "multi_hop"

        # action: verbs in question OR matched predicate labels
        action_kw = ["doing", "hold", "holding", "carry", "carrying",
                     "pick", "picked", "drop", "dropped", "put", "leave",
                     "left", "take", "took", "bring", "brought", "move",
                     "moved", "place", "placed", "last see", "last saw",
                     "use", "using", "cook", "cut", "open", "wash", "brush",
                     "stand", "standing", "look", "looking", "lean", "leaning",
                     "wear", "wearing", "walk", "walking", "sit", "sitting",
                     "drive", "driving", "hang", "hanging", "touch", "touching",
                     "talk", "talking", "interact", "interacting", "converse",
                     "conversing", "turn", "turned",
                     "drink", "eating", "eat", "run", "running", "jump",
                     "push", "pull", "lift", "pick up", "put down", "close",
                     "type", "play", "clean", "repair", "fix", "write", "read",
                     "teach", "learn", "kick", "lie", "lying", "kneel",
                     "squat", "climb", "grab", "watch", "stare", "remove",
                     "take off", "put on", "doing what",
                     "what is the person doing", "what are the people doing",
                     "what was the person doing", "what were the people doing",
                     "做", "拿", "放", "用", "切", "洗", "站", "看", "靠",
                     "穿", "走", "坐", "开", "拿取", "放下", "放置", "交谈",
                     "喝", "吃", "跑", "跳", "抱", "举", "推", "拉", "搬",
                     "捡", "扔", "拖", "刷", "擦", "整理", "收拾", "打扫",
                     "修理", "维修", "安装", "拆卸", "写字", "画画", "打字",
                     "点击", "滑动", "玩", "聊天", "说话", "打电话", "拍照",
                     "录像", "浇花", "做饭", "煮", "炒", "倒", "装", "脱",
                     "戴", "系", "背", "提", "抬", "扶", "抓", "握", "摸",
                     "碰", "踢", "踩", "蹲", "跪", "躺", "趴", "站起", "坐下",
                     "起身", "离开", "走进", "走出", "进入", "拿起", "打开",
                     "关上", "使用", "操作", "演示", "教", "学", "读", "写",
                     "在干嘛", "在干吗", "干什么", "在干什么", "干嘛呢", "干吗",
                     "干啥", "在做什么", "做什么呢", "在忙什么", "在干嘛呢",
                     "正在做什么", "正在干嘛", "做什么工作"]
        if any(k in q for k in action_kw):
            return "action"
        if matched.get("predicates"):
            return "action"

        # Explicit scene questions describe a place. Generic "where" is not
        # treated as scene: NLQ "where did I put/pick/drop X" is action/state
        # localization, while "where is X" should stay an object query.
        scene_kw = [
            "scene", "room", "kitchen", "bedroom", "living room", "bathroom",
            "office", "classroom", "restaurant", "corridor", "hallway",
            "garage", "supermarket", "store", "mall", "street", "park",
            "outdoor", "indoor", "what place", "which room", "what room",
            "where was this filmed", "environment", "location",
            "场景", "房间", "厨房", "客厅", "卧室", "卫生间", "浴室", "洗手间",
            "阳台", "办公室", "教室", "餐厅", "走廊", "车库", "超市", "商店",
            "商场", "街道", "公园", "室外", "户外", "室内", "车里", "车上",
            "是什么地方", "什么地方", "在哪里拍的", "在哪拍的", "什么地点",
            "拍摄地点", "什么环境", "哪个房间",
        ]
        if any(k in q for k in scene_kw):
            return "scene"

        # General overview/summary questions
        general_kw = [
            "summarize", "summary", "describe", "overview", "what happened",
            "what is going on", "what's going on", "main idea", "topic",
            "总结", "概括", "介绍", "概要", "内容是什么", "讲什么",
            "讲了什么", "怎么回事", "发生了什么", "什么情况", "整体",
            "大致", "主题", "主旨", "描述一下", "说说",
        ]
        if any(k in q for k in general_kw):
            return "general"

        return "object"

    def get_adaptive_params(self, query_type: str) -> dict:
        return self.ADAPTIVE_PARAMS.get(query_type, self.ADAPTIVE_PARAMS["general"])

    # ------------------------------------------------------------------ #
    #  Channel 1: Qdrant recall (payload filter or text-vector search)
    # ------------------------------------------------------------------ #

    def _extract_object_terms(self, question: str) -> List[str]:
        """Extract candidate object/relation keywords from the question."""
        terms = [t.strip(".,!?;:\"'") for t in tokenize_text(question)]
        terms = [t for t in terms if t and (len(t) >= 2 or contains_cjk(t))]
        stop = {"the", "a", "an", "is", "was", "are", "were", "what", "when",
                "where", "who", "how", "did", "do", "does", "in", "on", "at",
                "with", "of", "to", "and", "or", "not", "no", "yes", "first",
                "same", "later", "also", "appear", "appears", "appeared", "moved",
                "move", "moves", "moving", "after", "before", "then", "next",
                "one", "within", "another", "other", "video", "videos", "which",
                "common", "action", "relation", "connects", "contain", "contains",
                "happened", "seen", "was", "were",
                "这个", "那个", "哪个", "什么", "同一个", "另一个", "其他",
                "后来", "然后", "出现", "出现过", "视频", "里面", "哪里",
                "在哪", "移动", "到了", "之后", "最开始", "一起", "动作",
                "关系", "连接", "之间", "有哪些", "有"}
        return [t.lower() for t in terms if t.lower() not in stop]

    def _qdrant_payload_recall(self, question: str, top_k: int = 20, keywords=None,
                               scope: Optional[Dict] = None) -> List[str]:
        """Recall frame node_ids via Qdrant payload match on object/relation labels."""
        if not self.qa or not self.qa.is_available():
            return []
        terms = keywords or self._extract_object_terms(question)
        if not terms:
            return []

        seen = set()
        ordered = []  # preserve rank order for RRF
        per_term_limit = top_k  # cap per-term scroll to avoid scanning full collection
        for term in terms:
            label_filter = [
                {"key": "object_labels", "match": {"value": term}},
                {"key": "relation_labels", "match": {"any": [term]}},
            ]
            filt = self._qdrant_scope_filter(scope) or {}
            filt = dict(filt)
            filt["should"] = label_filter
            try:
                points = list(self.qa._scroll(
                    filt, with_vector=False,
                    page_size=min(per_term_limit, 50),
                    max_points=per_term_limit,
                ))
            except Exception as e:
                logger.debug("payload recall failed for '%s': %s", term, e)
                continue
            for p in points:
                pid = str(p.get("id"))
                if pid not in seen:
                    seen.add(pid)
                    ordered.append(pid)
                    if len(ordered) >= top_k:
                        break
            if len(ordered) >= top_k:
                break
        return ordered

    def _qdrant_vector_recall(self, question: str, top_k: int = 20, vector_name="visual",
                              scope: Optional[Dict] = None) -> List[str]:
        """Recall via true text->visual vector search (requires text_encoder)."""
        if not self.text_encoder or not self.qa or not self.qa.is_available():
            return []
        try:
            vec = self.text_encoder(question)
            hits = self.qa.search(
                vec,
                vector_name=vector_name,
                top_k=top_k,
                filter_obj=self._qdrant_scope_filter(scope),
            )
            return [str(h.get("id")) for h in hits if h.get("id")]
        except Exception as e:
            logger.warning("vector recall failed: %s", e)
            return []

    def _vector_channel(self, question: str, top_k: int, keywords=None,
                        scope: Optional[Dict] = None,
                        vector_name="image") -> List:
        """Unified vector channel: true vector search when an encoder exists.

        Payload scrolling is only a fallback for tools without a text encoder.
        The chat server already has the same labels in its local inverted and
        structured indexes, so repeating those filters over Qdrant HTTP adds
        latency without adding a distinct recall signal.
        """
        if not self.text_encoder:
            if os.getenv("QDRANT_PAYLOAD_FALLBACK", "0").lower() not in {
                "1", "true", "yes", "on"
            }:
                return []
            node_ids = self._qdrant_payload_recall(
                question, top_k=top_k, keywords=keywords, scope=scope)
        else:
            node_ids = self._qdrant_vector_recall(
                question, top_k=top_k, scope=scope, vector_name=vector_name)

        nodes = []
        for nid in node_ids:
            node = self.graph_db.get_node(nid)
            if node:
                nodes.append(node)
        return nodes

    # ------------------------------------------------------------------ #
    #  Channel 2: keyword index (reused from MAGMA, data source swapped)
    # ------------------------------------------------------------------ #

    def _keyword_search(self, question: str, top_k: int = 25, keywords=None,
                        scope: Optional[Dict] = None) -> List:
        if not self.node_index:
            return []
        terms = keywords or self._extract_object_terms(question)
        total_nodes = max(1, len(self.graph_db.nodes))

        node_scores = defaultdict(float)
        # direct term lookup
        for term in terms:
            if term in self.node_index:
                df = len(self.node_index[term])
                idf = 1.0 + math.log((total_nodes + 1) / (df + 1))
                for nid in self.node_index[term]:
                    node_scores[nid] += 5.0 * idf
            # partial match for ascii terms
            if len(term) >= 4 and not contains_cjk(term):
                for key in self.node_index:
                    if term in key or key in term:
                        df = len(self.node_index[key])
                        idf = 1.0 + math.log((total_nodes + 1) / (df + 1))
                        for nid in self.node_index[key]:
                            node_scores[nid] += 2.0 * idf
        # bigram
        for i in range(len(terms) - 1):
            bigram = "%s %s" % (terms[i], terms[i + 1])
            if bigram in self.node_index:
                for nid in self.node_index[bigram]:
                    node_scores[nid] += 4.0

        ranked = sorted(node_scores.items(), key=lambda x: (-x[1], x[0]))
        result = []
        for nid, score in ranked:
            node = self.graph_db.get_node(nid)
            if node and self._scope_match(node, scope):
                node.keyword_score = score
                result.append(node)
            if len(result) >= top_k:
                break
        return result

    # ------------------------------------------------------------------ #
    #  Channel 3: full scan (optional, disableable for scale)
    # ------------------------------------------------------------------ #

    def _full_scan(self, question: str, top_k: int = 20, keywords=None,
                   scope: Optional[Dict] = None) -> List:
        if not self.enable_full_scan:
            return []
        terms = keywords or self._extract_object_terms(question)
        if not terms:
            return []
        scored = []
        count = 0
        for node in self.graph_db.nodes.values():
            if node.node_type.value != "EVENT":
                continue
            count += 1
            if count > self.max_scan_nodes:
                break
            text = (getattr(node, "content_narrative", "") or "").lower()
            score = sum(1 for t in terms if t in text)
            if score > 0 and self._scope_match(node, scope):
                scored.append((score, node))
        scored.sort(key=lambda x: (-x[0], x[1].node_id))
        return [n for _, n in scored[:top_k]]

    @staticmethod
    def _metadata_query_anchors(question: str) -> tuple[list[str], list[str]]:
        """Return generic calendar and place anchors from a user query.

        This deliberately does not use benchmark answers or a fixed place
        vocabulary.  Calendar expressions are normalized and CJK runs are
        matched against the persisted ``place`` field at query time.
        """
        text = str(question or "").lower()
        dates = []
        for match in re.finditer(
                r"(\d{4})\s*(?:[-/.年]\s*)(\d{1,2})\s*(?:[-/.月]\s*)(\d{1,2})\s*(?:日)?",
                text):
            dates.append("%04d-%02d-%02d" % tuple(map(int, match.groups())))
        # Keep meaningful CJK runs; generic interrogative/time words cannot
        # identify a place and would otherwise match most nodes.
        stop = {
            "什么", "哪个", "哪里", "地点", "场景", "时间", "时候", "发生",
            "拍摄", "视频", "照片", "画面", "事情", "事件", "内容", "当时",
            "之后", "之前", "然后", "现在", "那里", "这里", "有没有",
        }
        places = []
        for run in re.findall(r"[\u4e00-\u9fff]{2,}", text):
            if run in stop:
                continue
            # Long natural-language runs contain verbs and question words;
            # retain their useful n-grams and let node metadata validate them.
            if len(run) <= 12:
                places.append(run)
            else:
                places.extend(run[i:i + 2] for i in range(len(run) - 1))
                places.extend(run[i:i + 3] for i in range(len(run) - 2))
        return dates, list(dict.fromkeys(places))

    def _metadata_recall(self, question: str, top_k: int = 20,
                         scope: Optional[Dict] = None) -> List:
        """Recall nodes by exact persisted date/place context.

        Vector/semantic channels are intentionally fuzzy.  This channel is
        deterministic and only adds candidates when the query contains an
        explicit date or a phrase that occurs in a node's stored location.
        It is therefore useful for dates and GPS-derived locations without
        changing ordinary visual retrieval behavior.
        """
        dates, place_terms = self._metadata_query_anchors(question)
        if not dates and not place_terms:
            return []
        scored = []
        for node in self.graph_db.nodes.values():
            if node.node_type.value != "EVENT" or not self._scope_match(node, scope):
                continue
            attrs = getattr(node, "attributes", {}) or {}
            captured = str(attrs.get("captured_at") or "").lower()
            place = str(attrs.get("place") or "").lower()
            event_text = " ".join(str(attrs.get(k) or "").lower()
                                  for k in ("event_title", "event_summary",
                                            "sentrix_event_title", "sentrix_event_summary",
                                            "sentrix_event_place", "event_text"))
            score = 0.0
            if any(date in captured[:10].replace("/", "-") for date in dates):
                score += 100.0
            for term in place_terms:
                if len(term) >= 2 and term in place:
                    score += 120.0 if len(term) >= 3 else 70.0
                elif len(term) >= 3 and term in event_text:
                    # Event summaries are identity-bearing evidence, not a
                    # weak visual hint.  Let them compete with exact place
                    # fields while still requiring the term to be present.
                    score += 90.0 if len(term) >= 4 else 55.0
            if score > 0:
                node.metadata_score = score
                scored.append((score, node))
        scored.sort(key=lambda item: (-item[0], item[1].node_id))
        return [node for _, node in scored[:top_k]]

    # ------------------------------------------------------------------ #
    #  RRF fusion (reused verbatim from MAGMA)
    # ------------------------------------------------------------------ #


    def _structured_recall(self, search_q, top_k=20, force_predicates=None,
                           scope: Optional[Dict] = None):
        """Predicate-precise and spatial recall using structured index."""
        if not self.structured_index:
            return []
        predicates = force_predicates
        if predicates:
            intent = {"predicates": predicates}
        else:
            intent = extract_query_intent(search_q)
        node_ids = set()

        if intent.get("predicates"):
            preds = intent["predicates"]
            matched = self.structured_index.query_predicate(preds)
            for nid in matched:
                node = self.graph_db.get_node(nid)
                if node and self._scope_match(node, scope):
                    node.structured_score = 3.0
                    node_ids.add(nid)

        region = intent.get("spatial_region")
        if region:
            terms = self._extract_object_terms(search_q)
            for term in terms:
                if len(term) >= 3 and term not in ("the","left","right","top","bottom","center"):
                    matched = self.structured_index.query_spatial(term, region)
                    for nid in matched:
                        node = self.graph_db.get_node(nid)
                        if node and self._scope_match(node, scope):
                            node.structured_score = getattr(node, "structured_score", 0) + 4.0
                            node_ids.add(nid)

        nodes = []
        for nid in sorted(node_ids):
            node = self.graph_db.get_node(nid)
            if node and self._scope_match(node, scope):
                if not hasattr(node, "structured_score"):
                    node.structured_score = 1.0
                nodes.append(node)
        nodes.sort(key=lambda n: (-getattr(n, "structured_score", 0), n.node_id))
        return nodes[:top_k]

    def _rrf_fusion(self, ranked_lists: List[List], k: int = 60) -> List[Tuple]:
        if not ranked_lists or all(not lst for lst in ranked_lists):
            return []
        if len(ranked_lists) == 1 and ranked_lists[0]:
            return [(n, 1.0 / (k + r + 1)) for r, n in enumerate(ranked_lists[0])]
        scores = {}
        for rl in ranked_lists:
            for rank, node in enumerate(rl):
                nid = node.node_id
                contrib = 1.0 / (k + rank + 1)
                if nid in scores:
                    scores[nid] = (scores[nid][0], scores[nid][1] + contrib)
                else:
                    scores[nid] = (node, contrib)
        return sorted(scores.values(), key=lambda x: (-x[1], x[0].node_id))

    # ------------------------------------------------------------------ #
    #  Graph traversal (optimised: adjacency-list neighbours)
    # ------------------------------------------------------------------ #

    def _question_direction(self, question: str) -> Optional[str]:
        """Return "forward", "backward" or None from temporal reasoning words."""
        q = (question or "").lower()
        if re.search(r"(之后|然后|接着|后来|\bafter\b|\bthen\b|\bnext\b|\bfollowing\b)", q):
            return "forward"
        if re.search(r"(之前|以前|先前|\bbefore\b|\bearlier\b|\bprevious\b|\bpreceding\b)", q):
            return "backward"
        return None

    def _link_allowed(self, link, is_outgoing: bool,
                      follow_link_types=None, direction: str = None) -> bool:
        """Filter edges by link type plus directional temporal/causal sub-types."""
        if follow_link_types and link.link_type not in follow_link_types:
            return False
        st = (link.properties or {}).get("sub_type", "")

        if link.link_type == LinkType.TEMPORAL and direction in ("forward", "backward"):
            # TIME_PRECEDES is the only directional temporal edge;
            # forward follows outgoing, backward follows incoming
            if st != self.TEMPORAL_FORWARD:
                return False
            return is_outgoing if direction == "forward" else not is_outgoing

        if link.link_type == LinkType.CAUSAL:
            if st not in self.CAUSAL_FORWARD_SUBTYPES:
                return False
            if direction == "forward":
                return is_outgoing

        return True

    @classmethod
    def _trusted_graph_edge(cls, link) -> bool:
        """Reject weak/inferred edges before they contaminate graph recall.

        Older graph files only stored ``confidence`` while newer files also
        store ``confidence_score``.  Read both formats so this remains
        backwards compatible and does not require changing the database.
        """
        props = link.properties or {}
        subtype = str(props.get("sub_type") or "").upper()
        if subtype in cls.GRAPH_DETERMINISTIC_EDGES:
            return True
        minimum = cls.GRAPH_EDGE_MIN_CONFIDENCE.get(subtype)
        if minimum is None:
            # Unknown semantic edges are useful only when explicitly marked
            # as confirmed/supported; inferred edges are not safe expansion
            # bridges for unrelated frames.
            return str(props.get("evidence_tier") or "").lower() in {
                "confirmed", "supported"
            }
        raw = props.get("confidence_score", props.get("confidence",
                       props.get("similarity", 0.0)))
        try:
            score = float(raw)
        except (TypeError, ValueError):
            return False
        return score >= minimum

    def _get_neighbor_links(self, node, follow_link_types=None,
                            direction: str = None,
                            max_links: Optional[int] = None) -> List[Tuple]:
        """Neighbour nodes plus the sub_type of the edge used to reach them.

        ``max_links`` enables a bounded, batched SQLite path used by graph
        traversal: only a limited slice of each node's edges is materialized,
        and Link objects are batch-loaded instead of one query per edge.
        """
        g = self.graph_db.graph
        if node.node_id not in g:
            return []
        result = []
        seen = set()
        bounded = (
            max_links is not None
            and hasattr(g, "out_edges_bounded")
            and hasattr(self.graph_db.links, "get_many")
        )
        if bounded:
            out_rows = g.out_edges_bounded(node.node_id, max_links)
            in_rows = g.in_edges_bounded(node.node_id, max_links)
            out_links = {lk.link_id: lk for lk in self.graph_db.links.get_many(
                [k for _, k in out_rows])}
            in_links = {lk.link_id: lk for lk in self.graph_db.links.get_many(
                [k for _, k in in_rows])}
            out_iter = [(target_id, key, out_links.get(key))
                        for target_id, key in out_rows]
            in_iter = [(source_id, key, in_links.get(key))
                       for source_id, key in in_rows]
        else:
            out_iter = [(target_id, key, self.graph_db.links.get(key))
                        for _, target_id, key in g.out_edges(node.node_id, keys=True)]
            in_iter = [(source_id, key, self.graph_db.links.get(key))
                       for source_id, _, key in g.in_edges(node.node_id, keys=True)]
        for target_id, key, link in out_iter:
            if target_id == node.node_id or target_id in seen:
                continue
            if link is None:
                continue
            if not self._trusted_graph_edge(link):
                continue
            if not self._link_allowed(link, True, follow_link_types, direction):
                continue
            t = self.graph_db.get_node(target_id)
            if t:
                result.append((t, (link.properties or {}).get("sub_type", "")))
                seen.add(target_id)
        for source_id, key, link in in_iter:
            if source_id == node.node_id or source_id in seen:
                continue
            if link is None:
                continue
            if not self._trusted_graph_edge(link):
                continue
            if not self._link_allowed(link, False, follow_link_types, direction):
                continue
            t = self.graph_db.get_node(source_id)
            if t:
                result.append((t, (link.properties or {}).get("sub_type", "")))
                seen.add(source_id)
        return result

    def _get_neighbors(self, node, follow_link_types=None,
                       direction: str = None) -> List:
        """Neighbours via NetworkX adjacency list -> O(degree), not O(total edges)."""
        return [n for n, _ in self._get_neighbor_links(
            node, follow_link_types=follow_link_types, direction=direction)]

    def _incident_subtypes(self, node) -> Set[str]:
        """All edge sub-types touching a node (outgoing + incoming)."""
        sts = set()
        g = self.graph_db.graph
        nid = node.node_id
        if nid not in g:
            return sts
        for _, _, key in g.out_edges(nid, keys=True):
            link = self.graph_db.links.get(key)
            if link:
                st = (link.properties or {}).get("sub_type")
                if st:
                    sts.add(st)
        for _, _, key in g.in_edges(nid, keys=True):
            link = self.graph_db.links.get(key)
            if link:
                st = (link.properties or {}).get("sub_type")
                if st:
                    sts.add(st)
        return sts

    def incident_subtypes(self, node) -> List[str]:
        """Public: sorted edge sub-types touching a node (for evidence/debug)."""
        return sorted(self._incident_subtypes(node))

    def _visual_generic_fallback(self, top_k: int = 8,
                                 scope: Optional[Dict] = None) -> List:
        """When label/keyword retrieval finds nothing for a visual question,
        return a diverse set of frames so the VLM still has images to inspect."""
        buckets = {}
        for node in self.graph_db.nodes.values():
            if node.node_type.value != "EVENT":
                continue
            if not self._scope_match(node, scope):
                continue
            attrs = getattr(node, "attributes", {}) or {}
            clip = attrs.get("clip_uid") or "?"
            try:
                t = float(attrs.get("clip_time_sec") or 0.0)
            except (TypeError, ValueError):
                t = 0.0
            key = (clip, int(t // 60))
            buckets.setdefault(key, []).append(node)

        for bucket in buckets.values():
            bucket.sort(key=lambda n: (getattr(n, "attributes", {}).get("clip_time_sec", 0) or 0,
                                       n.node_id))

        picked = []
        while buckets and len(picked) < top_k:
            for key in sorted(buckets.keys()):
                if len(picked) >= top_k:
                    break
                node = buckets[key].pop(0)
                node.similarity_score = 0.1
                picked.append(node)
                if not buckets[key]:
                    del buckets[key]
        return picked

    def _lightweight_filter(self, node, keywords: List[str]) -> bool:
        if not keywords:
            return True
        text = (getattr(node, "content_narrative", "") or "").lower()
        attrs = getattr(node, "attributes", {}) or {}
        obj_labels = " ".join(attrs.get("object_labels", []) or []).lower()
        rel_labels = " ".join(attrs.get("relation_labels", []) or []).lower()
        place = str(attrs.get("place") or "").lower()
        captured_at = str(attrs.get("captured_at") or "").lower()
        person_labels = " ".join(attrs.get("person_labels", []) or []).lower()
        event_text = str(attrs.get("event_text") or "").lower()
        combined = " ".join((text, obj_labels, rel_labels, place,
                              captured_at, person_labels, event_text))
        matches = sum(1 for kw in keywords if kw in combined)
        return matches >= min(1, len(keywords))

    def _graph_traversal(self, anchor_nodes, keywords, params,
                         question: str = None, max_nodes=600,
                         scope: Optional[Dict] = None,
                         bidirectional: bool = False,
                         allow_unmatched_events: bool = False):
        """BFS from anchors, filtering by keyword relevance, link types and
        temporal direction (sub-type consumption).

        ``allow_unmatched_events`` is used only by temporal/multi-hop routes:
        a path's next event often does not repeat the anchor's object words,
        so applying the ordinary keyword gate to every neighbour would erase
        the very relation the graph is meant to recover.

        ENTITY nodes (object/relation hubs) are conduits: all of their
        neighbours are expanded (no hard cap), while the total result count
        is still bounded by max_nodes. Hub neighbours are ordered by temporal
        proximity to the anchor so the expanded set is narrative-coherent and
        deterministic.
        """
        threshold = params["similarity_threshold"]
        max_depth = params["max_depth"]
        prefer = params.get("prefer_link_types")
        follow = set(prefer) if prefer else None
        direction = self._question_direction(question)
        # Multi-hop QA needs the complete path.  Following only the asked-for
        # direction cannot recover the earlier event when a later event is the
        # first anchor (e.g. phone -> umbrella is found, but spoon is missed).
        if bidirectional:
            direction = None

        visited = set()
        results = []
        paths = []
        queue = deque()

        # seed scores: anchor similarity is its RRF-derived score
        for node in anchor_nodes:
            if not self._scope_match(node, scope):
                continue
            sim = getattr(node, "similarity_score", 0.5)
            queue.append((node, sim, 0, node, None, None, None))

        while queue and len(results) < max_nodes:
            current, base_sim, depth, anchor, from_id, link_id, edge_subtype = queue.popleft()
            if current.node_id in visited:
                continue
            visited.add(current.node_id)
            # ENTITY nodes (object/relation hubs) are traversal conduits
            # only; they must not appear in retrieval results.
            if current.node_type.value == "EVENT":
                results.append((current, base_sim))
            if depth >= max_depth:
                continue
            is_hub = current.node_type.value != "EVENT"
            neighbors = self._get_neighbor_links(
                current, follow_link_types=follow, direction=direction,
                max_links=(self.MAX_HUB_EXPANSION * 8 if is_hub else 200))
            if is_hub and len(neighbors) > self.MAX_HUB_EXPANSION:
                # High-frequency hubs can connect to a large slice of the graph.
                neighbors = neighbors[:self.MAX_HUB_EXPANSION]
            promising = [(n, edge_subtype) for n, edge_subtype in neighbors
                         if n.node_id not in visited
                         and (n.node_type.value != "EVENT"
                              or (self._scope_match(n, scope)
                                  and (allow_unmatched_events
                                       or self._lightweight_filter(n, keywords))))]
            if is_hub:
                # Hub neighbours keep narrative locality: same clip as the
                # anchor first, then temporal distance, then node id.
                anchor_attrs = getattr(anchor, "attributes", {}) or {}
                anchor_clip = anchor_attrs.get("clip_uid")
                try:
                    anchor_t = float(anchor_attrs.get("clip_time_sec") or 0.0)
                except (TypeError, ValueError):
                    anchor_t = 0.0

                def hub_key(item):
                    n, _ = item
                    attrs = getattr(n, "attributes", {}) or {}
                    same_clip = 0 if attrs.get("clip_uid") == anchor_clip else 1
                    try:
                        t = float(attrs.get("clip_time_sec") or 0.0)
                    except (TypeError, ValueError):
                        t = 0.0
                    return (same_clip, abs(t - anchor_t), n.node_id)

                promising.sort(key=hub_key)
                expand_limit = len(promising)
            else:
                promising.sort(key=lambda item: item[0].node_id)
                expand_limit = 10
            for nb, edge_subtype in promising[:expand_limit]:
                # Hops through a hub are cheap (conduit), so decay less than
                # for event-to-event hops to keep 2-hop hub paths competitive.
                decay = 0.95 if is_hub else 0.85
                nb_sim = base_sim * decay
                if nb_sim >= threshold:
                    paths.append({
                        "from_node_id": current.node_id,
                        "to_node_id": nb.node_id,
                        "link_id": None,
                        "edge_subtype": edge_subtype,
                        "entity": None,
                        "reason": None,
                    })
                    queue.append((nb, nb_sim, depth + 1, anchor,
                                  current.node_id, None, edge_subtype))

        results.sort(key=lambda x: (-x[1], x[0].node_id))
        return results, paths

    # ------------------------------------------------------------------ #
    #  Rerank
    # ------------------------------------------------------------------ #

    def _build_event_time_index(self):
        if self._event_time_index is not None:
            return self._event_time_index
        index = defaultdict(list)
        for nid, node in self.graph_db.nodes.items():
            if node.node_type.value != "EVENT":
                continue
            attrs = getattr(node, "attributes", {}) or {}
            clip_uid = attrs.get("clip_uid")
            clip_time = attrs.get("clip_time_sec")
            if clip_uid and clip_time is not None:
                index[clip_uid].append((float(clip_time), nid))
        for bucket in index.values():
            bucket.sort(key=lambda item: (item[0], item[1]))
        self._event_time_index = index
        return self._event_time_index

    @staticmethod
    def _parse_time_anchor(question: str) -> Optional[Dict]:
        """Extract a relative video-time anchor from a temporal question.

        Frames carry only in-video time (clip_time_sec / video_time_sec and
        frame_seq), never wall-clock time. Returns {"time_sec": float} or
        {"frame_seq": int}, or None when no explicit anchor exists.
        """
        q = question.lower()
        m = re.search(r"第\s*(\d+(?:\.\d+)?)\s*(?:帧|f(?:rame)?s?\b)", q)
        if m:
            return {"frame_seq": int(float(m.group(1)))}
        m = re.search(r"(\d+)\s*[:：]\s*(\d{2})\s*秒?", q)
        if m:
            return {"time_sec": int(m.group(1)) * 60.0 + int(m.group(2))}
        m = re.search(r"(?:第\s*)?(\d+(?:\.\d+)?)\s*分钟", q)
        if m:
            return {"time_sec": float(m.group(1)) * 60.0}
        m = re.search(r"(?:第\s*)?(\d+(?:\.\d+)?)\s*(?:s|sec|second|seconds|秒)", q)
        if m:
            return {"time_sec": float(m.group(1))}
        return None

    def _temporal_recall(self, question: str, top_k: int = 15,
                         scope: Optional[Dict] = None) -> List:
        """Recall keyframes by the relative time/frame anchor in the question.

        No real-world time exists in this dataset, so explicit anchors such as
        "第5秒" / "5s" / "第10帧" are answered by nearest-time recall against
        clip_time_sec / frame_seq. Questions without an anchor still receive a
        diverse frame sample so the LLM can explain the relative-time evidence
        instead of the server returning "no relevant keyframes".
        """
        index = self._build_event_time_index()
        if not index:
            return []
        anchor = self._parse_time_anchor(question)
        clip_filter = None
        uid_match = re.search(
            r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}",
            question.lower())
        if uid_match:
            clip_filter = uid_match.group(0)

        nodes = []
        seen = set()
        if anchor is None:
            for clip, bucket in index.items():
                if clip_filter and clip_filter not in clip:
                    continue
                if not bucket:
                    continue
                _, nid = bucket[0]
                node = self.graph_db.get_node(nid)
                if node and nid not in seen and self._scope_match(node, scope):
                    nodes.append(node)
                    seen.add(nid)
                if len(nodes) >= top_k:
                    break
            return nodes

        scored = []
        frame_seq = anchor.get("frame_seq")
        for clip, bucket in index.items():
            if clip_filter and clip_filter not in clip:
                continue
            if frame_seq is not None:
                for t, nid in bucket:
                    node = self.graph_db.get_node(nid)
                    if not node or not self._scope_match(node, scope):
                        continue
                    attrs = getattr(node, "attributes", {}) or {}
                    if str(attrs.get("frame_seq")) == str(frame_seq):
                        scored.append((0.0, node))
            else:
                times = [row[0] for row in bucket]
                pos = bisect_left(times, anchor["time_sec"])
                for off in (-2, -1, 0, 1, 2):
                    idx = pos + off
                    if 0 <= idx < len(bucket):
                        t, nid = bucket[idx]
                        node = self.graph_db.get_node(nid)
                        if node and self._scope_match(node, scope):
                            scored.append((abs(t - anchor["time_sec"]), node))
        scored.sort(key=lambda x: (x[0], x[1].node_id))
        for _, node in scored:
            if node.node_id not in seen:
                seen.add(node.node_id)
                nodes.append(node)
            if len(nodes) >= top_k:
                break
        return nodes

    def _multi_hop_target_labels(self, matched: Dict, question: str = "",
                                  primary_only: bool = False) -> Set[str]:
        labels = {
            str(x).lower() for x in list(matched.get("objects", []))
            + list(matched.get("predicates", [])) if str(x).strip()
        }
        if primary_only and question:
            explicit = {
                raw for raw, zh in OBJECT_ZH.items()
                if zh and len(zh) > 1 and zh in str(question).lower()
                and raw not in {"person", "people"}
            }
            selected = {
                label for label in labels
                if label in explicit or any(raw in label for raw in explicit)
            }
            holding = {label for label in selected if label.startswith("person holding ")}
            if holding:
                return holding
            if selected:
                return selected
        action_labels = {x for x in labels if x.startswith("person ")}
        return action_labels or labels

    def _covered_target_labels(self, node, target_labels: Set[str]) -> Set[str]:
        attrs = getattr(node, "attributes", {}) or {}
        node_labels = {
            str(x).lower() for x in list(attrs.get("object_labels", []))
            + list(attrs.get("relation_labels", []))
        }
        return target_labels & node_labels

    def _expand_multi_hop_paths(self, nodes: List, matched: Dict,
                                max_visited: int = 600):
        """Expand matched nodes over temporal/SAME_ENTITY edges.

        This recovers the complete path when only a later event is present in
        the initial recall pool (for example phone -> umbrella is found, while
        the earlier spoon event must be followed backwards).
        """
        target_labels = self._multi_hop_target_labels(matched)
        if not target_labels:
            return [], set()

        path_ids = set()
        expanded = []
        visited = set()
        queue = deque()
        for node in nodes:
            if node.node_id in visited:
                continue
            visited.add(node.node_id)
            if self._covered_target_labels(node, target_labels):
                path_ids.add(node.node_id)
                queue.append(node)
                if node not in expanded:
                    expanded.append(node)

        link_types = [LinkType.TEMPORAL, LinkType.ENTITY]
        while queue and len(visited) < max_visited:
            current = queue.popleft()
            for neighbor, subtype in self._get_neighbor_links(
                    current, follow_link_types=link_types, direction=None):
                if subtype not in (self.TEMPORAL_FORWARD, "SAME_ENTITY"):
                    continue
                if neighbor.node_id in visited:
                    continue
                visited.add(neighbor.node_id)
                if neighbor.node_type.value != "EVENT":
                    queue.append(neighbor)
                    continue
                if self._covered_target_labels(neighbor, target_labels):
                    path_ids.add(neighbor.node_id)
                    if neighbor not in expanded:
                        expanded.append(neighbor)
                    queue.append(neighbor)
        return expanded, path_ids

    def _multi_hop_cover_select(self, nodes: List, matched: Dict,
                                question: str, top_k: int) -> List:
        """Greedy coverage selection for multi-hop evidence."""
        target_labels = self._multi_hop_target_labels(
            matched, question=question, primary_only=True)
        if not target_labels:
            return nodes[:top_k]

        remaining = set(target_labels)
        selected = []
        selected_ids = set()
        while len(selected) < top_k and remaining:
            best = None
            best_cover = set()
            for node in nodes:
                if node.node_id in selected_ids:
                    continue
                cover = self._covered_target_labels(node, target_labels) & remaining
                score = float(getattr(node, "ranking_score", 0) or 0)
                key = (len(cover), score, node.node_id)
                if best is None or key > (len(best_cover), float(
                        getattr(best, "ranking_score", 0) or 0), best.node_id):
                    best = node
                    best_cover = cover
            if best is None or not best_cover:
                break
            selected.append(best)
            selected_ids.add(best.node_id)
            remaining -= best_cover

        seen = {n.node_id for n in selected}
        for node in sorted(
                nodes,
                key=lambda n: (-float(getattr(n, "ranking_score", 0) or 0), n.node_id)):
            if len(selected) >= top_k:
                break
            if node.node_id not in seen:
                selected.append(node)
                seen.add(node.node_id)
        return selected[:top_k]

    def _temporal_event_select(self, nodes, top_k, query_type,
                               scope: Optional[Dict] = None) -> List:
        """Return a time-localized TopK around the strongest event anchors."""
        if query_type not in ("action", "multi_hop", "temporal"):
            return nodes[:top_k]
        # Multi-hop evidence is selected by graph paths and cross-video links.
        # Time-neighbor selection would replace ranked path nodes with nearby
        # but unranked frames from one anchor clip, destroying the graph result.
        if query_type == "multi_hop":
            return nodes[:top_k]
        if not nodes:
            return []

        time_index = self._build_event_time_index()
        selected = []
        seen = set()
        anchor_count = max(2, min(4, top_k // 2))

        def add_node(node):
            if node.node_id in seen:
                return
            if not self._scope_match(node, scope):
                return
            seen.add(node.node_id)
            selected.append(node)

        for anchor in nodes[:anchor_count]:
            if len(selected) >= top_k:
                break
            add_node(anchor)
            attrs = getattr(anchor, "attributes", {}) or {}
            clip_uid = attrs.get("clip_uid")
            clip_time = attrs.get("clip_time_sec")
            if not clip_uid or clip_time is None:
                continue
            bucket = time_index.get(clip_uid) or []
            if not bucket:
                continue
            pos = bisect_left(bucket, (float(clip_time), anchor.node_id))
            for offset in (-2, -1, 1, 2):
                idx = pos + offset
                if 0 <= idx < len(bucket):
                    _, nid = bucket[idx]
                    neighbor = self.graph_db.get_node(nid)
                    if neighbor:
                        add_node(neighbor)
                if len(selected) >= top_k:
                    break

        for node in nodes:
            if len(selected) >= top_k:
                break
            add_node(node)
        return selected[:top_k]

    def _coherent_select(self, nodes, top_k, graph_enabled=True, query_type=None):
        """Select top-k evidence using graph relationships, not just video ids.

        The same keyword can match frames in many clips/videos (e.g. "person"
        is everywhere), which previously made answers mix unrelated scenes.
        Cross-video frames are merged only when the graph connects them with
        strong evidence (visual similarity, shared relation, shared rare
        object). Otherwise evidence stays within one clip/video.
        """
        if not nodes:
            return []

        def node_score(node):
            return getattr(node, "ranking_score",
                           getattr(node, "similarity_score", 0.0))

        ranked = sorted(nodes, key=lambda n: (-node_score(n), n.node_id))
        if not graph_enabled:
            return ranked[:top_k]

        # Multi-hop questions intentionally require evidence from multiple
        # scenes/videos.  Collapsing them to one "coherent" component defeats
        # the purpose, so preserve the graph-aware reranker's order.
        if query_type == "multi_hop":
            return ranked[:top_k]

        parent = {n.node_id: n.node_id for n in ranked}

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a, b):
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[rb] = ra

        def strong_link(key):
            link = self.graph_db.links.get(key) if self.graph_db else None
            return bool(link and (link.properties or {}).get("sub_type")
                        in self.STRONG_CROSS_VIDEO_SUBTYPES)

        def relation_hub(node_id):
            node = self.graph_db.get_node(node_id) if self.graph_db else None
            return bool(node and node.node_type.value == "ENTITY"
                        and (getattr(node, "attributes", {}) or {}).get("entity_type") == "relation")

        relation_hub_members = defaultdict(list)
        if self.graph_db is not None:
            g = self.graph_db.graph
            for nid in list(parent):
                for _, nb_id, key in g.out_edges(nid, keys=True):
                    if nb_id in parent and strong_link(key):
                        union(nid, nb_id)
                    elif relation_hub(nb_id) and strong_link(key):
                        relation_hub_members[nb_id].append(nid)
                for src_id, _, key in g.in_edges(nid, keys=True):
                    if src_id in parent and strong_link(key):
                        union(nid, src_id)
                    elif relation_hub(src_id) and strong_link(key):
                        relation_hub_members[src_id].append(nid)

            for members in relation_hub_members.values():
                if len(members) > 1:
                    first = members[0]
                    for member in members[1:]:
                        union(first, member)

        components = defaultdict(list)
        for node in ranked:
            components[find(node.node_id)].append(node)

        def bucket_score(bucket):
            top = bucket[:top_k]
            return sum(node_score(n) for n in top) / max(1, len(top))

        best_component = max(components.values(), key=bucket_score) if components else []
        if len(best_component) >= top_k:
            return best_component[:top_k]

        video_buckets = defaultdict(list)
        for node in ranked:
            attrs = getattr(node, "attributes", {}) or {}
            video_buckets[attrs.get("video_uid") or "unknown"].append(node)
        for bucket in video_buckets.values():
            bucket.sort(key=lambda n: (-node_score(n), n.node_id))
        best_video = max(video_buckets.values(), key=bucket_score) if video_buckets else []
        if len(best_video) >= top_k:
            return best_video[:top_k]

        selected = best_component[:top_k] if best_component else best_video[:top_k]
        seen = {n.node_id for n in selected}
        for node in nodes:
            if len(selected) >= top_k:
                break
            if node.node_id not in seen:
                selected.append(node)
                seen.add(node.node_id)
        return selected

    def _rerank(self, nodes, question, query_type, top_k, weights,
                graph_enabled=True) -> List:
        terms = self._extract_object_terms(question)
        # Rare query labels distinguish a true path node from high-frequency
        # frames containing only generic labels such as person/phone/table.
        rare_terms = {
            term for term in terms
            if term and 0 < len(self.node_index.get(term, ())) <= 10000
        }
        scored = []
        for node in nodes:
            attrs = getattr(node, "attributes", {}) or {}
            obj_labels = set(o.lower() for o in (attrs.get("object_labels") or []))
            rel_labels = set(r.lower() for r in (attrs.get("relation_labels") or []))
            text = (getattr(node, "content_narrative", "") or "").lower()

            kw_score = sum(2.0 for t in terms if t in text)
            obj_score = sum(1.0 for t in terms if t in obj_labels) * 3.0
            rel_score = sum(1.0 for t in terms
                            if any(t in r for r in rel_labels)) * 2.0
            # A node reached over a graph edge is only a useful path node when
            # it also matches the queried objects/relations.  This bonus keeps
            # true temporal/SAME_ENTITY paths competitive with dense but
            # unrelated high-frequency frames.
            rare_hit_count = sum(
                1 for term in rare_terms
                if term in obj_labels or any(term in rel for rel in rel_labels)
            )
            graph_path_score = 0.0
            if (graph_enabled and query_type == "multi_hop"
                    and self.enable_multi_hop_expansion
                    and bool(getattr(node, "graph_path_hit", False))
                    and (obj_score > 0 or rel_score > 0)):
                graph_path_score = 2000.0 + 500.0 * rare_hit_count

            # temporal: reward nodes near the queried time
            time_score = 0.0
            t_match = re.search(r"\b(\d+)\s*(?:s|sec|秒)?\b", question.lower())
            if t_match and query_type == "temporal":
                target_t = float(t_match.group(1))
                node_t = attrs.get("clip_time_sec", 0) or 0
                diff = abs(node_t - target_t)
                time_score = max(0, 30.0 - diff)  # closer = higher

            # Real-world location/date anchors are identity evidence, not
            # fuzzy visual concepts.  Semantic label matching can mistake a
            # place token (e.g. 沙岭) for a visually similar object (沙发);
            # exact hits in the node's persisted context fields must win.
            place_text = str(attrs.get("place") or "").lower()
            captured_text = str(attrs.get("captured_at") or "").lower()
            event_context = " ".join(str(attrs.get(key) or "").lower() for key in (
                "event_title", "event_summary", "sentrix_event_title",
                "sentrix_event_summary", "sentrix_event_place",
                "sentrix_event_time_start", "sentrix_event_time_end"))
            context_anchor_score = 0.0
            anchor_dates, anchor_places = self._metadata_query_anchors(question)
            for term in set(terms) | set(anchor_places):
                if len(term) >= 2 and term in place_text:
                    context_anchor_score += 120.0
                elif len(term) >= 3 and term in event_context:
                    context_anchor_score += 65.0
            for date in anchor_dates:
                if date in captured_text[:10].replace("/", "-"):
                    context_anchor_score += 100.0
                elif date in event_context:
                    context_anchor_score += 90.0

            sim = getattr(node, "similarity_score", 0)
            try:
                quality = float(attrs.get("worldmm_information_gain") or 0.0)
            except (TypeError, ValueError):
                quality = 0.0

            # causal-edge bonus: nodes reachable via LEADS_TO/ENABLES
            causal_score = 0.0
            if graph_enabled:
                g = self.graph_db.graph
                if node.node_id in g and query_type in ("action", "multi_hop"):
                    edge_keys = (list(g.out_edges(node.node_id, keys=True))[:25]
                                 + list(g.in_edges(node.node_id, keys=True))[:25])
                    for _, _, key in edge_keys:
                        link = self.graph_db.links.get(key)
                        if link and link.link_type == LinkType.CAUSAL:
                            st = (link.properties or {}).get("sub_type", "")
                            if st in self.CAUSAL_FORWARD_SUBTYPES:
                                causal_score += 1.0

            # sub-type bonus: reward frames connected through the edge
            # families that matter most for this query type
            subtype_score = 0.0
            if graph_enabled:
                subtypes = self._incident_subtypes(node)
                if query_type == "object" and subtypes & set(self.OBJECT_SUBTYPES):
                    subtype_score = 3.0
                elif query_type == "scene" and subtypes & set(self.SCENE_SUBTYPES):
                    subtype_score = 3.0
                elif query_type == "multi_hop" and subtypes & set(self.STRONG_CROSS_VIDEO_SUBTYPES):
                    subtype_score = 3.0
                elif query_type in ("action", "multi_hop") and subtypes & set(self.ACTION_SUBTYPES):
                    subtype_score = 3.0

            total = (kw_score * weights.get("keyword", 4.0)
                     + obj_score * weights.get("object", 3.0)
                     + rel_score * weights.get("relation", 3.0)
                     + time_score * weights.get("temporal", 1.5)
                     + causal_score * weights.get("causal", 0.0)
                     + subtype_score
                     + graph_path_score
                     + context_anchor_score
                     + quality * weights.get("quality", 0.5)
                     + sim * 10.0)
            node.ranking_score = total
            scored.append((total, node))

        scored.sort(key=lambda x: (-x[0], x[1].node_id))
        return [n for _, n in scored[:top_k]]

    # ------------------------------------------------------------------ #
    #  Context formatting
    # ------------------------------------------------------------------ #

    def _format_context(self, nodes, question) -> str:
        if not nodes:
            return ""
        parts = ["=== Relevant Keyframes ==="]
        for i, node in enumerate(nodes[:15]):
            attrs = getattr(node, "attributes", {}) or {}
            objs = attrs.get("object_labels") or []
            rels = attrs.get("relation_labels") or []
            t = attrs.get("clip_time_sec", "?")
            video = (attrs.get("video_uid") or "")[:12]
            clip = (attrs.get("clip_uid") or "")[:12]
            captured = str(attrs.get("captured_at") or "").strip()
            place = str(attrs.get("place") or "").strip()
            title = str(attrs.get("event_title") or attrs.get("sentrix_event_title") or "").strip()
            summary = str(attrs.get("event_summary") or attrs.get("sentrix_event_summary") or "").strip()
            narration = str(attrs.get("narration_text") or "").strip()
            action = str(attrs.get("action_text") or "").strip()
            source = str(attrs.get("source_video_file_name") or attrs.get("clip_path") or "").strip()
            metadata = " | ".join(item for item in (
                f"captured={captured}" if captured else "",
                f"place={place}" if place else "",
                f"event={title}" if title else "",
                f"summary={summary[:240]}" if summary else "",
                f"source={source}" if source else "",
            ) if item)
            visual = " | ".join(item for item in (
                f"objects={objs[:8]}" if objs else "",
                f"relations={rels[:6]}" if rels else "",
                f"narration={narration[:160]}" if narration else "",
                f"action={action[:120]}" if action else "",
            ) if item)
            parts.append("[%d] t=%ss video=%s clip=%s | %s%s" % (
                i + 1, t, video, clip, metadata,
                (" | " + visual) if visual else ""))
        return "\n".join(parts)

    # ------------------------------------------------------------------ #
    #  Main query entry
    # ------------------------------------------------------------------ #

    def _collect_causal_context(self, nodes):
        """Attach LEADS_TO causal edge reasons to each result node.

        For every node in the result set, look at its CAUSAL edges and
        find neighbours that are also in the result set. Store a readable
        description list in ``node.causal_context`` so the QA layer can
        include causal relationships in LLM evidence.
        """
        if not self.graph_db or not nodes:
            for n in nodes:
                if not hasattr(n, "causal_context"):
                    n.causal_context = []
            return

        g = self.graph_db.graph
        result_ids = {n.node_id for n in nodes}
        id_to_index = {n.node_id: i + 1 for i, n in enumerate(nodes)}

        for node in nodes:
            descs = []
            nid = node.node_id
            if nid not in g:
                node.causal_context = descs
                continue

            # outgoing LEADS_TO: this frame causes a later frame
            for _, target_id, key in g.out_edges(nid, keys=True):
                if target_id not in result_ids:
                    continue
                link = self.graph_db.links.get(key)
                if not link or link.link_type != LinkType.CAUSAL:
                    continue
                st = (link.properties or {}).get("sub_type", "")
                if st not in self.CAUSAL_FORWARD_SUBTYPES:
                    continue
                reason = (link.properties or {}).get("reason", "")
                target = self.graph_db.get_node(target_id)
                if not target:
                    continue
                t_attrs = getattr(target, "attributes", {}) or {}
                try:
                    t_sec = float(t_attrs.get("clip_time_sec", 0))
                except (TypeError, ValueError):
                    t_sec = 0.0
                idx = id_to_index.get(target_id, "?")
                descs.append("->[#%d %.0fs] %s" % (idx, t_sec, reason))

            # incoming LEADS_TO: an earlier frame caused this one
            for source_id, _, key in g.in_edges(nid, keys=True):
                if source_id not in result_ids:
                    continue
                link = self.graph_db.links.get(key)
                if not link or link.link_type != LinkType.CAUSAL:
                    continue
                st = (link.properties or {}).get("sub_type", "")
                if st not in self.CAUSAL_FORWARD_SUBTYPES:
                    continue
                reason = (link.properties or {}).get("reason", "")
                source = self.graph_db.get_node(source_id)
                if not source:
                    continue
                s_attrs = getattr(source, "attributes", {}) or {}
                try:
                    s_sec = float(s_attrs.get("clip_time_sec", 0))
                except (TypeError, ValueError):
                    s_sec = 0.0
                idx = id_to_index.get(source_id, "?")
                descs.append("<-[#%d %.0fs] %s" % (idx, s_sec, reason))

            node.causal_context = descs

    def query(self, question: str, top_k: int = 15, recall_k: int | None = None,
              scope: Optional[Dict] = None, graph_enabled: bool = True) -> dict:
        """Run the full retrieval pipeline using semantic label matching.

        The question (any language) is matched directly to database labels
        via cross-lingual embeddings. No translation needed for retrieval.
        """
        # 1. Semantic label matching (works for Chinese, English, any language)
        matched = {"objects": [], "predicates": [], "all": []}
        needs_visual = False
        if self.label_matcher:
            matched = self.label_matcher.match(question)
            needs_visual = self.label_matcher.needs_visual(question)
        keywords = matched["objects"] + matched["predicates"]

        # 2. Fallback: if no semantic matches, try keyword extraction
        if not keywords:
            keywords = self._extract_object_terms(question)

        # 3. Query type detection (uses original question + matched labels)
        query_type = self.detect_query_type(question, matched)
        params = self.get_adaptive_params(query_type)

        # 4. Channel sizes. Evaluation can enlarge the anchor pool without
        # changing the final top_k used by rerank/coherent selection.
        recall_budget = max(top_k, min(int(recall_k or top_k), 30))
        multi_hop_expansion = (
            graph_enabled
            and query_type == "multi_hop"
            and self.enable_multi_hop_expansion
        )
        if multi_hop_expansion:
            vec_n, kw_n, scan_n = 30, 30, 0
        elif query_type == "temporal":
            vec_n, kw_n, scan_n = 15, 15, 0
        else:
            vec_n, kw_n, scan_n = 20, 20, 20
        vec_n = min(60, max(vec_n, recall_budget))
        kw_n = min(60, max(kw_n, recall_budget))

        # 5. Run independent recall channels in parallel, then let RRF decide
        # which channel is useful for this query. Qdrant is skipped only in
        # offline/no-adapter mode. Full scan remains an emergency fallback.
        ranked_lists = []
        recall_timings = {}

        t_channel = time.perf_counter()
        kw_nodes = self._keyword_search(
            question, top_k=kw_n, keywords=keywords, scope=scope)
        recall_timings["keyword_ms"] = round((time.perf_counter() - t_channel) * 1000, 1)
        if kw_nodes:
            ranked_lists.append(kw_nodes)

        t_channel = time.perf_counter()
        struct_nodes = self._structured_recall(
            question, top_k=recall_budget, force_predicates=matched.get("predicates"),
            scope=scope)
        recall_timings["structured_ms"] = round((time.perf_counter() - t_channel) * 1000, 1)
        if struct_nodes:
            ranked_lists.append(struct_nodes)

        # Exact date/GPS-derived place context is a separate recall signal.
        # Keeping it independent from semantic labels prevents a place name
        # from being confused with a visually similar object (e.g. 沙岭/沙发).
        t_channel = time.perf_counter()
        metadata_nodes = self._metadata_recall(
            question, top_k=recall_budget, scope=scope)
        recall_timings["metadata_ms"] = round((time.perf_counter() - t_channel) * 1000, 1)
        if metadata_nodes:
            ranked_lists.append(metadata_nodes)

        t_channel = time.perf_counter()
        vec_nodes = self._vector_channel(
            question, top_k=vec_n, keywords=keywords, scope=scope,
            vector_name="image")
        recall_timings["vector_ms"] = round((time.perf_counter() - t_channel) * 1000, 1)
        if vec_nodes:
            ranked_lists.append(vec_nodes)

        t_nodes = []
        if query_type == "temporal":
            t_channel = time.perf_counter()
            t_nodes = self._temporal_recall(
                question, top_k=recall_budget, scope=scope)
            recall_timings["temporal_recall_ms"] = round(
                (time.perf_counter() - t_channel) * 1000, 1)
            if t_nodes:
                ranked_lists.append(t_nodes)

        scan_nodes = []
        if not ranked_lists:
            t_channel = time.perf_counter()
            scan_nodes = self._full_scan(
                question, top_k=scan_n, keywords=keywords, scope=scope)
            recall_timings["scan_ms"] = round((time.perf_counter() - t_channel) * 1000, 1)
            if scan_nodes:
                ranked_lists.append(scan_nodes)
        logger.info("semantic match: objects=%s predicates=%s -> keywords=%s",
                     matched.get("objects", []), matched.get("predicates", []), keywords[:6])

        if not ranked_lists:
            if needs_visual:
                fallback_nodes = self._visual_generic_fallback(top_k, scope=scope)
                if fallback_nodes:
                    fallback_nodes = self._coherent_select(fallback_nodes, top_k)
                    return {
                        "question": question,
                        "query_type": query_type,
                        "nodes": fallback_nodes,
                        "context": self._format_context(fallback_nodes, question),
                        "matched_labels": matched,
                        "needs_visual": True,
                        "stats": {"recall": {}, "candidates": len(fallback_nodes),
                                  "returned": len(fallback_nodes),
                                  "fallback": "visual_generic"},
                    }
            return {"question": question, "query_type": query_type, "nodes": [], "context": "",
                    "matched_labels": matched, "needs_visual": needs_visual}

        fused = self._rrf_fusion(ranked_lists, k=60)
        for node, rrf_score in fused:
            node.similarity_score = min(1.0, rrf_score * 20)
        candidates = [n for n, _ in fused]
        candidates = [c for c in candidates if c.node_type.value == "EVENT"]

        phase_ms = {}
        # graph traversal
        base_candidate_ids = [node.node_id for node in candidates]
        # Multi-hop answers often have their first golden event outside the
        # final Top-10 but inside the recall pool.  Seeding traversal from only
        # Top-10 anchors prevents the graph from ever reaching the true path.
        anchor_pool = (
            recall_budget if multi_hop_expansion
            else min(top_k, recall_budget)
        )
        initial = sorted(candidates, key=lambda n: (-getattr(n, "similarity_score", 0), n.node_id))[:anchor_pool]
        traversed = []
        graph_paths = []
        if graph_enabled:
            t_phase = time.perf_counter()
            traversed, graph_paths = self._graph_traversal(
                initial, keywords, params, question=question, max_nodes=200,
                scope=scope, bidirectional=multi_hop_expansion,
                allow_unmatched_events=query_type in {"multi_hop", "temporal"})
            phase_ms["traversal_ms"] = round((time.perf_counter() - t_phase) * 1000, 1)
        traversed_node_ids = {node.node_id for node, _ in traversed}
        # Persist reachability for graph-aware reranking.  Reset on every node
        # so stale state from a previous query cannot leak into this query.
        existing = {c.node_id for c in candidates}
        for node, sim in traversed:
            if node.node_id not in existing:
                node.similarity_score = sim * 0.8
                candidates.append(node)
                existing.add(node.node_id)

        explicit_path_nodes = []
        explicit_path_ids = set()
        primary_path_nodes = []
        if multi_hop_expansion:
            explicit_path_nodes, explicit_path_ids = self._expand_multi_hop_paths(
                candidates, matched)
            for node in explicit_path_nodes:
                if node.node_id not in existing:
                    node.similarity_score = 0.5
                    candidates.append(node)
                    existing.add(node.node_id)

        # Mark reachability after traversal-expanded nodes are appended; doing
        # it earlier leaves newly appended path nodes without the bonus.
        all_path_ids = traversed_node_ids | explicit_path_ids
        for node in candidates:
            node.graph_path_hit = node.node_id in all_path_ids

        # rerank
        # Keep a wider rerank pool for multi-hop so temporally connected graph
        # neighbours are not discarded before path-aware selection.
        rerank_pool = (
            max(top_k, min(recall_budget, top_k * 3))
            if multi_hop_expansion else top_k
        )
        t_phase = time.perf_counter()
        top_nodes = self._rerank(
            candidates, question, query_type, rerank_pool, params["weights"],
            graph_enabled=graph_enabled)
        phase_ms["rerank_ms"] = round((time.perf_counter() - t_phase) * 1000, 1)
        t_phase = time.perf_counter()
        top_nodes = self._coherent_select(
            top_nodes, rerank_pool, graph_enabled=graph_enabled, query_type=query_type)
        phase_ms["coherent_ms"] = round((time.perf_counter() - t_phase) * 1000, 1)
        if multi_hop_expansion:
            # Graph path nodes that explicitly match the question must survive
            # the wide rerank pool even when dense unrelated frames have higher
            # aggregate scores.  Coverage selection below then chooses them.
            primary_labels = self._multi_hop_target_labels(
                matched, question=question, primary_only=True)
            primary_path_nodes = [
                node for node in candidates
                if bool(getattr(node, "graph_path_hit", False))
                and self._covered_target_labels(node, primary_labels)
            ]
            if primary_path_nodes:
                seen_nodes = set()
                merged = []
                for node in primary_path_nodes + top_nodes:
                    if node.node_id not in seen_nodes:
                        seen_nodes.add(node.node_id)
                        merged.append(node)
                top_nodes = merged[:rerank_pool]
            top_nodes = self._multi_hop_cover_select(
                top_nodes, matched, question, top_k)
        elif graph_enabled:
            t_phase = time.perf_counter()
            top_nodes = self._temporal_event_select(
                top_nodes, top_k, query_type, scope=scope)
            phase_ms["temporal_ms"] = round((time.perf_counter() - t_phase) * 1000, 1)

        # attach LEADS_TO causal descriptions to result nodes for LLM evidence
        if graph_enabled:
            t_phase = time.perf_counter()
            self._collect_causal_context(top_nodes)
            phase_ms["causal_ms"] = round((time.perf_counter() - t_phase) * 1000, 1)

        t_phase = time.perf_counter()
        context = self._format_context(top_nodes, question)
        phase_ms["format_ms"] = round((time.perf_counter() - t_phase) * 1000, 1)
        for node in top_nodes:
            node._query_stats = {
                "query_type": query_type,
                "path_ids": sorted(all_path_ids),
                "primary_path_ids": [n.node_id for n in primary_path_nodes],
                "traversed": sorted(traversed_node_ids),
            }
        return {
            "question": question,
            "query_type": query_type,
            "nodes": top_nodes,
            "context": context,
            "matched_labels": matched,
            "needs_visual": needs_visual,
            "graph_enabled": graph_enabled,
            "graph_paths": graph_paths,
            "stats": {
                "recall": {
                    "qdrant": len(vec_nodes),
                    "keyword": len(kw_nodes),
                    "structured": len(struct_nodes),
                    "metadata": len(metadata_nodes),
                    "scan": len(scan_nodes),
                    "temporal": len(t_nodes),
                    "timings": recall_timings,
                },
                "phase_ms": phase_ms,
                "candidates": len(candidates),
                "returned": len(top_nodes),
                "base_candidate_ids": base_candidate_ids,
                "traversed_node_ids": sorted(traversed_node_ids),
                "path_ids": sorted(all_path_ids),
                "primary_path_ids": [n.node_id for n in primary_path_nodes],
            },
        }
