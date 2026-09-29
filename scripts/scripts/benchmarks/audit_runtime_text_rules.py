#!/usr/bin/env python3
"""Phase R9-0 — runtime text rule audit.

Extract every routing-relevant string / regex / word-list from ``backend/`` and
classify it per the R9 contract:

  A. prompt             only in model prompts (schema / open semantics / examples)
  B. protocol           explicit product operations (feedback / selected entity /
                        "don't look up memory" / high-precision writing prefix)
  C. normalization      deterministic format & hard constraints (date / media /
                        negation / scope / schema whitelists)
  D. semantic_routing   open-vocabulary words deciding mode directly  -> forbidden
  E. semantic_extraction  hard-coded open-semantic vocabulary          -> forbidden

Extraction is AST-based (module-level ``re.compile`` / word-list constants) plus
a curated ``MANUAL_RULES`` table for rules AST cannot see (method-local regexes,
prompt content, legacy path word-lists).  Unknown rules are emitted as
category "review" so the agent finalizes them; the final inventory must contain
zero ``review`` and zero runtime ``semantic_routing`` / ``semantic_extraction``
entries (asserted by backend/tests/test_runtime_text_rule_audit.py).

Output: docs/baseline/runtime_text_rule_inventory.json
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
BACKEND = REPO_ROOT / "backend"

# Routing-relevant files for inline membership-check scanning.  Module-level
# regex/word-list extraction covers all of backend/.
ROUTING_FILES = {
    "memory_gate.py", "routing_rules.py", "thin_agent.py", "query_parser.py",
    "query_contracts.py", "model_routing.py", "answer_composer.py",
    "evidence_retrieval.py", "agent.py", "retrieval/probes.py",
}

# (file_basename, symbol) -> classification.
KNOWLEDGE = {
    ("memory_gate.py", "_WRITING_PREFIX_RE"): ("protocol", "keep", ["test_router_decision", "test_semantic_routing"]),
    ("memory_gate.py", "_WRITING_ANYWHERE_RE"): ("semantic_routing", "remove_or_narrow", ["test_router_decision"]),
    ("memory_gate.py", "_NO_LOOKUP_RE"): ("protocol", "keep", ["test_router_decision"]),
    ("memory_gate.py", "_ANCHOR_GEO_RE"): ("normalization", "keep", ["test_gate_probe"]),
    ("memory_gate.py", "_ANCHOR_DATE_RE"): ("normalization", "keep", ["test_gate_probe"]),
    ("memory_gate.py", "_ANCHOR_RELATION_RE"): ("normalization", "keep", ["test_gate_probe"]),
    ("memory_gate.py", "_ANCHOR_PERSON_TOKENS"): ("normalization", "keep", ["test_gate_probe"]),
    ("routing_rules.py", "_WRITING_PREFIX_RE"): ("protocol", "keep", ["test_router_decision", "test_semantic_routing"]),
    ("routing_rules.py", "_NO_LOOKUP_RE"): ("protocol", "keep", ["test_router_decision"]),
    ("routing_rules.py", "_ANCHOR_GEO_RE"): ("normalization", "keep", ["test_gate_probe"]),
    ("routing_rules.py", "_ANCHOR_DATE_RE"): ("normalization", "keep", ["test_gate_probe"]),
    ("routing_rules.py", "_ANCHOR_RELATION_RE"): ("normalization", "keep", ["test_gate_probe"]),
    ("routing_rules.py", "_ANCHOR_PERSON_TOKENS"): ("normalization", "keep", ["test_gate_probe"]),
    ("routing_rules.py", "HOUSEHOLD_DIMENSIONS"): ("normalization", "keep", ["test_gate_probe"]),
    ("routing_rules.py", "_CONCEPT_VERB_RE"): ("protocol", "keep", ["test_router_decision"]),
    ("routing_rules.py", "_WRITING_COMPOSE_RE"): ("protocol", "keep", ["test_router_decision"]),
    ("routing_rules.py", "_FOLLOW_UP_TOKENS"): ("normalization", "keep", ["test_router_decision"]),
    ("router.py", "_EVIDENCE_ACTIONS"): ("normalization", "keep", ["test_router_decision"]),
    ("router.py", "_STRONG_TARGETS"): ("normalization", "keep", ["test_router_decision"]),
    ("query_parser.py", "_DATE_RE"): ("normalization", "keep", ["test_query_parser"]),
    ("retrieval_indexes.py", "_FIELD_TYPES"): ("normalization", "keep", ["test_retrieval_indexes"]),
}

# Curated rules AST cannot extract (method-local regexes, dict mappings, prompt
# content, legacy-path word-lists).  Fields: file, symbol, kind, content, scope,
# category, decision, tests, note.
MANUAL_RULES = [
    {
        "file": "memory_gate.py", "symbol": "household_dimensions",
        "kind": "schema_whitelist", "scope": "runtime",
        "content": "{person, place, activity, clothing, object, visual, time, relationship, ocr}",
        "category": "normalization", "decision": "keep",
        "tests": ["test_gate_probe"],
        "note": "Facet dimension schema whitelist, not topic vocabulary.",
    },
    {
        "file": "query_parser.py", "symbol": "_apply_deterministic_overlay negation/media tokens",
        "kind": "inline_tokens", "scope": "runtime",
        "content": "(\"不要\",\"排除\",\"不是\") + media window (\"视频\",\"照片\")",
        "category": "normalization", "decision": "keep",
        "tests": ["test_query_parser"],
        "note": "Explicit negation structure recovery; window is positional, not topic-based.",
    },
    {
        "file": "query_contracts.py", "symbol": "_media_type",
        "kind": "enum_map", "scope": "runtime",
        "content": "视频/video/录像->video; 音频/audio/录音->audio; 文本/text/文字->text; 照片/图片/原图/image->image",
        "category": "normalization", "decision": "keep",
        "tests": ["test_query_parser"],
        "note": "Media type -> enum normalization.",
    },
    {
        "file": "query_contracts.py", "symbol": "semantic_dimensions",
        "kind": "schema_whitelist", "scope": "runtime",
        "content": "{place, activity, object, clothing, visual, ocr, person, relationship, time, other, semantic}",
        "category": "normalization", "decision": "keep",
        "tests": ["test_query_parser"],
        "note": "Semantic condition dimension whitelist.",
    },
    {
        "file": "retrieval/probes.py", "symbol": "_per_space_minimum mapping",
        "kind": "dict_map", "scope": "runtime",
        "content": "{\"visual_ann\":\"visual\",\"text_ann\":\"text\",\"lexical\":\"lexical\"}",
        "category": "normalization", "decision": "keep",
        "tests": ["test_gate_probe"],
        "note": "Retriever name -> probe-space threshold key.",
    },
    {
        "file": "query_parser.py", "symbol": "_PARSER_PROMPT",
        "kind": "prompt", "scope": "runtime",
        "content": "Rules/examples include open semantic categories (做饭/晚饭/自拍/颜色/材质) and schema",
        "category": "prompt", "decision": "keep",
        "tests": ["test_query_parser"],
        "note": "Prompt Instruction; examples must never contain benchmark sentences (guard test enforces).",
    },
    {
        "file": "agent.py", "symbol": "_is_entity_introduction_query",
        "kind": "method_tokens", "scope": "legacy",
        "content": "any(token in query for token in (\"介绍\",\"是谁\",\"了解\",\"档案\",\"画像\"))",
        "category": "semantic_routing", "decision": "remove_or_retire",
        "tests": [],
        "note": "Legacy non-Thin dialogue path (SENTRIX_THIN_AGENT_V1 off). Thin path must not adopt; flag for cutover removal.",
    },
    {
        "file": "agent.py", "symbol": "_is_comparison_query",
        "kind": "method_tokens", "scope": "legacy",
        "content": "len(focused_people)>=2 and any(token in query for token in (\"比较\",\"区别\",\"不同\",\"共同\",\"对比\"))",
        "category": "semantic_routing", "decision": "remove_or_retire",
        "tests": [],
        "note": "Legacy comparison-intent word-list; thin path does not use it.",
    },
]


def _module_rules(path):
    """Extract module-level re.compile(...) and string-tuple/list word lists."""
    try:
        tree = ast.parse(path.read_text(encoding="utf-8", errors="ignore"))
    except SyntaxError:
        return []
    found = []
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if not (isinstance(target, ast.Name) and target.id.startswith("_")
                    and not target.id.startswith("__")):
                continue
            value = node.value
            func = value.func if isinstance(value, ast.Call) else None
            if isinstance(func, ast.Attribute):
                is_compile = func.attr == "compile"
            elif isinstance(func, ast.Name):
                is_compile = func.id == "compile"
            else:
                is_compile = False
            if is_compile and value.args and isinstance(value.args[0], ast.Constant) \
                    and isinstance(value.args[0].value, str):
                found.append({"symbol": target.id, "kind": "regex",
                              "content": value.args[0].value, "line": node.lineno})
            elif isinstance(value, (ast.Tuple, ast.List, ast.Set)) \
                    and all(isinstance(elt, ast.Constant) and isinstance(elt.value, str) for elt in value.elts):
                found.append({"symbol": target.id, "kind": "wordlist",
                              "content": list(elt.value for elt in value.elts), "line": node.lineno})
    return found


# Tokens already curated under MANUAL_RULES — inline hits that are substrings of
# a manual rule's content are normalization and must not be re-flagged.
_MANUAL_CONTENT_TEXT = " ".join(str(rule.get("content")) for rule in MANUAL_RULES)


def _inline_checks(path, basename):
    """Find multi-char Chinese literals used as membership tests within routing
    files.  Single-char tokens (pronouns etc.) are noise and skipped; tokens
    already covered by a curated manual rule are skipped."""
    if basename not in ROUTING_FILES:
        return []
    text = path.read_text(encoding="utf-8", errors="ignore")
    found = []
    pat = re.compile(r"['\"]([一-鿿]{2,8})['\"]\s+in\s+(\w+)")
    for m in pat.finditer(text):
        token, haystack = m.group(1), m.group(2)
        if token in _MANUAL_CONTENT_TEXT:
            continue
        found.append({"symbol": f"inline@{haystack}", "kind": "inline_token",
                      "content": token, "line": text.count("\n", 0, m.start()) + 1})
    return found


def _classify(basename, symbol):
    cat, decision, tests = KNOWLEDGE.get((basename, symbol), (None, None, None))
    if cat is None:
        return "review", "review", []
    return cat, decision, tests


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", default=str(REPO_ROOT / "docs" / "baseline" / "runtime_text_rule_inventory.json"))
    args = parser.parse_args()

    entries = []
    for path in sorted(BACKEND.rglob("*.py")):
        if "tests" in path.parts:
            continue
        basename = str(path.relative_to(BACKEND))
        for rule in _module_rules(path):
            cat, decision, tests = _classify(basename, rule["symbol"])
            entries.append({
                "file": str(path.relative_to(REPO_ROOT)), "symbol": rule["symbol"],
                "kind": rule["kind"], "content": rule["content"], "line": rule["line"],
                "scope": "runtime", "category": cat, "decision": decision,
                "tests": tests, "note": "",
            })
        for rule in _inline_checks(path, basename):
            entries.append({
                "file": str(path.relative_to(REPO_ROOT)), "symbol": rule["symbol"],
                "kind": rule["kind"], "content": rule["content"], "line": rule["line"],
                "scope": "runtime", "category": "review", "decision": "review",
                "tests": [], "note": "inline membership check - classify",
            })

    for rule in MANUAL_RULES:
        entries.append(rule)

    # Deduplicate by (file, symbol, content).
    seen = set()
    uniq = []
    for e in entries:
        key = (e["file"], e["symbol"], str(e["content"]))
        if key in seen:
            continue
        seen.add(key)
        uniq.append(e)

    uniq.sort(key=lambda e: (e["file"], e.get("line", 0) or 0, e["symbol"]))
    summary = {
        "total": len(uniq),
        "by_category": {},
        "runtime_semantic_routing": sum(1 for e in uniq if e["scope"] == "runtime" and e["category"] == "semantic_routing"),
        "runtime_semantic_extraction": sum(1 for e in uniq if e["scope"] == "runtime" and e["category"] == "semantic_extraction"),
        "review_pending": sum(1 for e in uniq if e["category"] == "review"),
    }
    for e in uniq:
        summary["by_category"][e["category"]] = summary["by_category"].get(e["category"], 0) + 1

    out = {"generated_at": "2026-08-06", "summary": summary, "rules": uniq}
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {args.report}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
