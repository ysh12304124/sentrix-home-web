#!/usr/bin/env python3
"""B4 — 生成 Phase B 独立 JSON 工件（§4.3/§17/§19）：

- tool_readiness_matrix.json    工具/通道就绪矩阵
- agent_profile_manifest.json   profile 配置 + runbook
- structured_memory_coverage.json  结构化记忆覆盖率（按 scope）

用法:
  python emit_phaseb_artifacts.py --db data/sentrix.db --out docs/phaseb
"""
import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path.cwd()
sys.path.insert(0, str(ROOT))


def emit_tool_readiness() -> dict:
    from backend.agent_runtime import tools as runtime_tools
    from backend.agent_runtime.tool_registry import tool_readiness_matrix
    runtime_tools.bind_runtime(None)
    runtime_tools.register_tools()
    matrix = tool_readiness_matrix()
    return {
        "schema_version": 1,
        "generated_at": __import__("datetime").datetime.now().isoformat(timespec="seconds"),
        "visual_backbone": "chinese-clip-768d",
        "text_backbone": "ViT-B-32-512d",
        "ann_status": {"visual": "ready", "text": "limited"},
        "tools": {
            "query_memory_facts": {
                "count": "ready", "media": "ready", "date": "ready",
                "first": "ready", "last": "ready", "group": "ready",
                "place": "limited", "person": "limited",
            },
            "search_memories": {
                "visual": "ready", "text": "limited",
                "semantic_truth": "limited",
                "known_limits": ["condition=unknown 时不能视为确认", "place/person 语义覆盖不足"],
            },
            "inspect_photo": {
                "status": "ready",
                "known_limits": ["低光/小物体计数可能不准", "观察为临时性，不写入长期记忆"],
            },
            "get_original_photos": {"status": "ready", "auth": "scope+handle 双校验"},
            "get_result_page": {"status": "ready", "auth": "scope 校验", "ttl_s": 1800},
        },
        "registry": matrix,
    }


def emit_profile_manifest() -> dict:
    from backend.agent_runtime.profile import PROFILES
    profiles = {}
    for name, cfg in PROFILES.items():
        profiles[name] = {
            "model": "gemma4-12b-it@8105(4bit)",
            "tools": list(cfg.tools),
            "budgets": {
                "max_model_steps": cfg.max_model_steps,
                "max_tool_calls": cfg.max_tool_calls,
                "max_inspections": cfg.max_inspections,
                "wall_time_s": cfg.wall_time_s,
                "final_reserve_s": cfg.final_reserve_s,
            },
            "features": cfg.features,
            "guard": "FinalGuard(L1 rule) + LLM judge(L2)" if name != "pipeline" else "pipeline-internal",
            "runbook": {
                "start": f"SENTRIX_AGENT_PROFILE={name} SENTRIX_API_PORT=<port> .venv/bin/python -m uvicorn backend.app:app",
                "health_check": "curl -s http://127.0.0.1:<port>/api/health",
                "verify": "python scripts/benchmarks/evaluate_search_inspect_e2e.py --base http://127.0.0.1:8105/v1",
                "rollback": "SENTRIX_AGENT_PROFILE=pipeline 重启即回退；canary 独立端口不影响 8091",
            },
        }
    return {"schema_version": 1, "profiles": profiles,
            "default": "pipeline", "canary_candidates": ["tool_loop_shadow"]}


def emit_coverage(store) -> dict:
    def count(sql, params=()):
        try:
            return store.connection.execute(sql, params).fetchone()[0]
        except Exception:
            return None
    return {
        "schema_version": 1,
        "scopes": {
            "home-default": {
                "assets": count("SELECT COUNT(*) FROM assets WHERE scope_id='home-default'"),
                "observations": count("SELECT COUNT(*) FROM observations WHERE scope_id='home-default'"),
                "events": count("SELECT COUNT(*) FROM events WHERE scope_id='home-default'"),
                "entities": count("SELECT COUNT(*) FROM entities WHERE scope_id='home-default'"),
                "facts": count("SELECT COUNT(*) FROM facts"),
                "memory_vectors": count("SELECT COUNT(*) FROM memory_vectors"),
                "query_gaps": count("SELECT COUNT(*) FROM query_gaps"),
            },
            "album2_e2b": {
                "assets": count("SELECT COUNT(*) FROM assets WHERE scope_id='album2_e2b'"),
                "observations": count("SELECT COUNT(*) FROM observations WHERE scope_id='album2_e2b'"),
                "events": count("SELECT COUNT(*) FROM events WHERE scope_id='album2_e2b'"),
                "entities": count("SELECT COUNT(*) FROM entities WHERE scope_id='album2_e2b'"),
                "memory_vectors": count("SELECT COUNT(*) FROM memory_vectors WHERE scope_id='album2_e2b'"),
            },
        },
        "structured_query_types": {
            "count": "ready", "exists": "ready", "first": "ready", "last": "ready",
            "date": "ready", "media": "ready", "group_by_month": "ready",
            "place": "limited", "person": "limited",
        },
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="data/sentrix.db")
    ap.add_argument("--out", default="docs/phaseb")
    args = ap.parse_args()

    from backend.db import MemoryStore
    store = MemoryStore(args.db)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    files = {
        "tool_readiness_matrix.json": emit_tool_readiness(),
        "agent_profile_manifest.json": emit_profile_manifest(),
        "structured_memory_coverage.json": emit_coverage(store),
    }
    for name, payload in files.items():
        path = out / name
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"wrote {path} ({len(json.dumps(payload))} bytes)")


if __name__ == "__main__":
    main()
