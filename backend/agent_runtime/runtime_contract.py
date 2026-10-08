from __future__ import annotations

import json
from pathlib import Path
from typing import Any

REGISTRY_PATH = Path(__file__).resolve().parents[2] / "configs" / "sentrix_runtime_tool_registry_aligned_v1.json"


def _registry() -> dict[str, dict[str, Any]]:
    try:
        payload = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return {str(item.get("name")): item for item in payload.get("tools") or [] if item.get("name")}


def tool_contract(name: str, fallback: Any = None) -> dict[str, Any]:
    item = _registry().get(str(name))
    if item:
        return item
    return {
        "name": str(name),
        "description": str(getattr(fallback, "description", "") or ""),
        "parameters": getattr(fallback, "input_schema", None) or {"type": "object"},
    }


def serialize_runtime_action(action: dict[str, Any]) -> str:
    kind = str(action.get("action") or "")
    if kind == "tool_call":
        value: dict[str, Any] = {
            "action": "tool_call",
            "tool": str(action.get("tool") or ""),
            "arguments": dict(action.get("arguments") or {}),
        }
        if action.get("public_status") not in (None, ""):
            value["public_status"] = str(action.get("public_status"))
    elif kind == "final":
        refs = action.get("evidence_refs")
        value = {
            "action": "final",
            "answer": str(action.get("answer") or ""),
            "evidence_refs": refs if isinstance(refs, list) else [],
        }
    elif kind == "declare":
        value = {"action": "declare", "declaration": action.get("declaration") or {}}
    else:
        value = dict(action)
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def registry_metadata() -> dict[str, Any]:
    try:
        payload = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return {key: payload.get(key) for key in (
        "contract_version", "protocol_mode", "target_serialization",
        "tool_registry_version", "tool_registry_sha256", "source_data_sha256",
    )}
