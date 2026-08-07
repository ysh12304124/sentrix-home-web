"""Single source of structural routing rules (Phase R9).

These are narrow *structural* patterns only: writing-operation prefixes,
explicit "don't look up memory", and generic person/time/place anchors used
for strict-empty gating and weak-parser rescue.  They are NOT a semantic
classifier — no topic/object vocabulary lives here.

Classification (R9 §4):
- ``_WRITING_PREFIX_RE`` / ``_NO_LOOKUP_RE``  -> protocol fast paths
- ``_ANCHOR_*`` / ``message_anchored``        -> deterministic normalization
- ``has_general_verb``                        -> narrowed general-intent predicate
                                              (never the sole reason for "none")
"""

from __future__ import annotations

import re

_WRITING_PREFIX_RE = re.compile(r"^\s*(帮我写|请写|写一段|写一篇|生成一段|翻译|帮我起草|拟一份|写个|写篇)")
_NO_LOOKUP_RE = re.compile(r"^\s*(不用查|别查|别找|不用找|不需要查|不用看|不用搜|不看我的)")

# Generic person/date/geo/relation anchors.  Specific family names are never
# hard-coded here (they are benchmark/family data and the runtime guard forbids
# it).  "家人/全家" are deliberately absent: they false-triggered the
# writing-prefix rescue on "假设一家人在厨房做饭，写个故事" in R8.
_ANCHOR_GEO_RE = re.compile(r"(市|区|省|县|镇|湾|湖|山|路|街|城|岛)")
_ANCHOR_DATE_RE = re.compile(r"(年|月|日|节|跨年|元旦|春节)")
_ANCHOR_RELATION_RE = re.compile(r"(搂着|抱着|牵着|靠着|合影|一起|全家福)")
_ANCHOR_PERSON_TOKENS = ("自己", "我们", "合照")

# R9: general-intent verbs.  "介绍一下" IS included — but the Router only lets
# it decide "none" AFTER confirmed-entity and household-signal checks fail, so
# "介绍一下明哥" (confirmed person) still routes to evidence.  On its own it
# never overrides a household signal.
_CONCEPT_VERB_RE = re.compile(r"^(请)?(介绍一下|介绍下|解释|解释一下|为什么|什么是|什么叫|假设|假如|编个|编一段|讲讲你怎么看|说明一下|说说|讲讲)")

# Composition verbs ("写一篇 / 起草 / 拟一份 / 编个故事") mark a text-writing
# request.  Deliberately excludes "写着/写下" so "照片里写着什么？" and
# "帮我写下那次明哥穿的衣服" are NOT writing.
_WRITING_COMPOSE_RE = re.compile(
    r"(写一篇|写一段|写个|写篇|写一首|写一封|写一个|写两句|写作文|写个故事|写一封信|写一段话|"
    r"起草|拟一份|拟个|生成一段|生成一篇|编个|编一段|编一个)"
)

# Follow-up markers reused from the legacy dialogue state (agent.py) so the thin
# path shares the same "current conversation focus" semantics.
_FOLLOW_UP_TOKENS = (
    "然后", "后来", "接着", "继续", "为什么", "具体", "详细", "还有呢", "那呢",
    "他呢", "她呢", "它呢", "这里呢", "那里呢", "这段呢", "那个呢", "那次", "那段", "这次",
)

HOUSEHOLD_DIMENSIONS = {"person", "place", "activity", "clothing", "object",
                        "visual", "time", "relationship", "ocr"}


def message_anchored(message) -> bool:
    """True when a raw message carries a concrete person / date / geo /
    relationship anchor (used by strict-empty gating and weak-parser rescue)."""
    value = str(message or "")
    return bool(_ANCHOR_GEO_RE.search(value) or _ANCHOR_DATE_RE.search(value)
                or _ANCHOR_RELATION_RE.search(value)
                or any(token in value for token in _ANCHOR_PERSON_TOKENS))


def is_writing_request(message) -> bool:
    """High-precision writing/translation prefix (protocol fast path)."""
    return bool(_WRITING_PREFIX_RE.search(str(message or "")))


def is_writing_compose(message) -> bool:
    """A mid-sentence composition verb marks a text-writing request."""
    return bool(_WRITING_COMPOSE_RE.search(str(message or "")))


def has_general_verb(message) -> bool:
    """Concept-question / general-intro verb set (Router applies it only after
    confirmed-entity and household-signal checks fail)."""
    return bool(_CONCEPT_VERB_RE.match(str(message or "")))


def is_no_lookup(message) -> bool:
    return bool(_NO_LOOKUP_RE.search(str(message or "")))


def is_contextual_follow_up(message) -> bool:
    value = str(message or "").strip()
    return any(token in value for token in _FOLLOW_UP_TOKENS)


def has_household_signal(draft) -> bool:
    """Any household structure survived the parser (facets / conditions /
    media / time / negation / entity names)."""
    if not draft:
        return False
    if any(facet.dimension in HOUSEHOLD_DIMENSIONS for facet in getattr(draft, "facets", [])):
        return True
    if getattr(draft, "semantic_conditions", None):
        return True
    if getattr(draft, "negative_conditions", None):
        return True
    if getattr(draft, "media_expressions", None):
        return True
    if getattr(draft, "time_expression", None):
        return True
    if getattr(draft, "entity_names", None):
        return True
    return False


# RX-0 fix (D7): casual self-inquiry / greetings must never be dragged into the
# evidence path by the parser's generic semantic condition.  These are general
# self-referential questions — not household lookups.  The Router only applies
# this AFTER confirming there is no strong household anchor (time / media /
# negation / entity / explicit evidence action), so "去年拍的合影感觉怎么样"
# (media/time anchor) is never misclassified.
_CASUAL_CHAT_RE = re.compile(
    r"(感觉怎么样|心情如何|心情怎么样|过得怎么样|最近怎么样|你还好吗|你好|在吗|"
    r"你是谁|你叫什么|介绍一下你自己|你在干嘛|你现在感觉|状态如何|今天过得如何|"
    r"今天感觉怎么样|你今天心情|陪我聊聊|陪我说话|聊聊天|想聊聊|有点累|陪我说说话)"
)


def is_casual_chat(message) -> bool:
    value = str(message or "").strip()
    if not value or len(value) > 40:
        return False
    return bool(_CASUAL_CHAT_RE.search(value))
