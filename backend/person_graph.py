"""Constrained person graph inference for one memory space.

The model only produces anonymized hypotheses (P01..P10). Role vocabulary and
inverse relationships are fixed here; attribute language (age, gender,
generation) is stripped before anything reaches the memory store. Confirmed
facts are never derived from hypotheses.
"""

import re

from .person_moments import SENSITIVE_TERMS

PERSON_GRAPH_PROMPT = (
    "你是家庭相册的人物关系推断器。输入是匿名编号 P01 至 P10 的人物代表照片、"
    "事件覆盖、人物共现矩阵、人物瞬间摘要和设备线索。\n"
    "你的目标是从长期证据推断相册主人、人物角色和人物关系候选。\n"
    "可以基于外观年龄、代际或性别呈现做弱评分，但输出中不得包含年龄、性别、"
    "代际、健康、收入、宗教、民族、政治、性取向等属性文字。\n"
    "角色词表仅限：本人、父亲、母亲、配偶、孩子、祖父母、兄弟姐妹、其他亲属、"
    "朋友、同事、同学、邻居、照护者、老师、亲友、访客、一次性人物、无法判断。\n"
    "每个角色的候选必须包含至少两个选项并补充‘无法判断’。\n"
    "关系只给出有证据支持的候选；证据不足时保持‘无法判断’，不要强行家庭化。\n"
    "严格返回 JSON：\n"
    '{"album_owner_candidates":[{"person_ref":"P01","confidence":0.0,"reason":""}],\n'
    '"roles":[{"person_ref":"P02","relative_to":"P01","candidates":[{"role":"母亲","confidence":0.0,"reason":""}]}],\n'
    '"relationships":[{"subject_ref":"P02","predicate":"母亲","object_ref":"P01","inverse_predicate":"孩子","confidence":0.0,"reason":""}]}'
)

ROLE_OPTIONS = {
    "本人", "父亲", "母亲", "配偶", "孩子", "祖父母", "兄弟姐妹",
    "其他亲属", "朋友", "同事", "同学", "邻居", "照护者", "老师",
    "亲友", "访客", "一次性人物", "无法判断",
}

INVERSE_RELATION = {
    "配偶": "配偶", "朋友": "朋友", "同事": "同事", "同学": "同学",
    "邻居": "邻居", "兄弟姐妹": "兄弟姐妹",
    "父亲": "孩子", "母亲": "孩子", "孩子": "父母",
    "祖父母": "孙辈", "孙辈": "祖父母",
    "老师": "学生", "学生": "老师",
    "照护者": "被照护者", "被照护者": "照护者",
    "其他亲属": "其他亲属", "亲友": "亲友", "访客": "主人",
}

SYMMETRIC_RELATIONS = {"配偶", "朋友", "同事", "同学", "邻居", "兄弟姐妹"}

GENERATION_TERMS = {
    "长辈", "晚辈", "老人", "老年", "中年", "青年", "几岁", "岁数", "四十岁", "三十岁",
}

_AGE_PATTERN = re.compile(r"[一二三四五六七八九十百千\d]+岁")


def sanitize_reason(text):
    if not text:
        return ""
    result = str(text)
    for term in SENSITIVE_TERMS | GENERATION_TERMS:
        result = result.replace(term, "")
    result = _AGE_PATTERN.sub("", result)
    result = re.sub(r"\s+", " ", result)
    return result.strip("，。、 　")


def ensure_unknown_fallback(candidates):
    candidates = list(candidates or [])
    roles = {str(candidate.get("role") or "").strip() for candidate in candidates}
    if "无法判断" not in roles:
        candidates.append({"role": "无法判断", "confidence": 0.0, "reason": "保留不确定性"})
    return candidates


def normalize_relationship(rel):
    rel = dict(rel)
    predicate = str(rel.get("predicate") or "").strip()
    if predicate in SYMMETRIC_RELATIONS:
        subject, obj = rel["subject_ref"], rel["object_ref"]
        if str(subject) > str(obj):
            rel["subject_ref"], rel["object_ref"] = obj, subject
        rel["inverse_predicate"] = predicate
    else:
        rel["inverse_predicate"] = rel.get("inverse_predicate") or \
            INVERSE_RELATION.get(predicate, "无法判断")
    return rel


def find_graph_violations(relationships):
    violations = []
    directed = {}
    for rel in relationships:
        key = (rel["subject_ref"], rel["object_ref"])
        directed.setdefault(key, []).append(rel["predicate"])
    for (subject, obj), predicates in directed.items():
        has_parent = any(predicate in ("父亲", "母亲") for predicate in predicates)
        has_child = any(predicate == "孩子" for predicate in predicates)
        if has_parent and has_child:
            violations.append({
                "subject_ref": subject, "object_ref": obj,
                "kind": "parent_and_child",
            })
    return violations


def apply_relationship_threshold(relationships, min_events=2, min_moments_with_event=2):
    result = []
    for rel in relationships:
        event_count = len(rel.get("evidence_event_ids") or [])
        moment_count = len(rel.get("evidence_moment_ids") or [])
        if event_count >= min_events or (event_count >= 1 and moment_count >= min_moments_with_event):
            result.append(rel)
        else:
            downgraded = dict(rel)
            downgraded["predicate"] = "无法判断"
            downgraded["inverse_predicate"] = "无法判断"
            result.append(downgraded)
    return result


def _clean_candidate(candidate):
    role = str(candidate.get("role") or "").strip()
    if role not in ROLE_OPTIONS:
        return None
    return {
        "role": role,
        "confidence": float(candidate.get("confidence") or 0),
        "reason": sanitize_reason(candidate.get("reason") or ""),
    }


def normalize_person_graph(parsed, people):
    people = set(people or [])
    owner_candidates = []
    for item in parsed.get("album_owner_candidates") or []:
        ref = str(item.get("person_ref") or "").strip()
        if ref not in people:
            continue
        owner_candidates.append({
            "person_ref": ref,
            "confidence": float(item.get("confidence") or 0),
            "reason": sanitize_reason(item.get("reason") or ""),
        })

    roles = []
    for item in parsed.get("roles") or []:
        ref = str(item.get("person_ref") or "").strip()
        relative_to = str(item.get("relative_to") or "").strip()
        if ref not in people or (relative_to and relative_to not in people):
            continue
        candidates = [
            cleaned for cleaned in (
                _clean_candidate(candidate) for candidate in (item.get("candidates") or [])
            )
            if cleaned
        ]
        candidates = ensure_unknown_fallback(candidates)
        candidates = sorted(candidates, key=lambda candidate: -candidate["confidence"])
        roles.append({
            "person_ref": ref,
            "relative_to": relative_to,
            "candidates": candidates,
        })

    relationships = []
    for item in parsed.get("relationships") or []:
        subject = str(item.get("subject_ref") or "").strip()
        obj = str(item.get("object_ref") or "").strip()
        predicate = str(item.get("predicate") or "").strip()
        if subject not in people or obj not in people:
            continue
        if predicate not in INVERSE_RELATION:
            continue
        relationships.append(normalize_relationship({
            "subject_ref": subject,
            "predicate": predicate,
            "object_ref": obj,
            "inverse_predicate": str(item.get("inverse_predicate") or "").strip(),
            "confidence": float(item.get("confidence") or 0),
            "reason": sanitize_reason(item.get("reason") or ""),
            "evidence_event_ids": item.get("evidence_event_ids") or [],
            "evidence_moment_ids": item.get("evidence_moment_ids") or [],
        }))

    return {
        "album_owner_candidates": owner_candidates,
        "roles": roles,
        "relationships": relationships,
        "graph_violations": find_graph_violations(relationships),
    }
