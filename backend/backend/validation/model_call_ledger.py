"""ModelCallLedger — per-call proof that the configured 12B actually ran.

Thread-local, request-scoped.  ``begin_turn`` / ``end_turn`` bracket one E2E
request; ``new_call`` / ``finish_call`` bracket one model call.  The HTTP layer
(GammaClient) writes the actual_model / endpoint / sizes / JSON legality /
response hash into the active record so the ledger proves which model served the
call — not just what the config says.

Only active when full_chain_profile.validation_active(); otherwise it is a no-op
(no overhead on production).
"""

from __future__ import annotations

import hashlib
import threading
import time

_local = threading.local()


def begin_turn():
    _local.records = []
    _local.active = None


def end_turn():
    records = list(getattr(_local, "records", []) or [])
    _local.records = []
    _local.active = None
    return records


def new_call(role: str, expected_model: str, endpoint: str = "") -> dict:
    records = getattr(_local, "records", None)
    if records is None:
        _local.records = records = []
        _local.active = None
    record = {
        "call_id": f"call_{len(records)}",
        "role": role,
        "expected_model": expected_model,
        "actual_model": None,
        "endpoint": endpoint,
        "started_at": time.time(),
        "completed_at": None,
        "latency_ms": None,
        "input_size": 0,
        "output_size": 0,
        "json_valid": None,
        "fallback_used": False,
        "cache_hit": False,
        "circuit_breaker_state": "closed",
        "error": None,
        "response_sha256": None,
    }
    records.append(record)
    _local.active = record
    return record


def active_record():
    return getattr(_local, "active", None)


def finish_call():
    record = getattr(_local, "active", None)
    if record is not None:
        record["completed_at"] = time.time()
        record["latency_ms"] = round((record["completed_at"] - record["started_at"]) * 1000, 1)
    _local.active = None
    return record


def record_response(text: str, *, actual_model: str, endpoint: str, json_mode: bool,
                    fallback_used: bool = False, cache_hit: bool = False,
                    breaker_state: str = "closed", error: str | None = None):
    record = active_record()
    if record is None:
        return None
    record["actual_model"] = actual_model
    record["endpoint"] = endpoint or record["endpoint"]
    record["input_size"] = len(str(record.get("input_size") or ""))
    record["output_size"] = len(text or "")
    record["json_valid"] = _json_ok(text) if json_mode else None
    record["response_sha256"] = hashlib.sha256(str(text or "").encode()).hexdigest()[:16]
    record["fallback_used"] = bool(fallback_used)
    record["cache_hit"] = bool(cache_hit)
    record["circuit_breaker_state"] = breaker_state
    if error:
        record["error"] = str(error)[:200]
    return record


def _json_ok(text: str) -> bool:
    import json
    try:
        json.loads(text)
        return True
    except Exception:
        return False
