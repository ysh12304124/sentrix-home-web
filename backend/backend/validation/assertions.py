"""Validation assertions over the ModelCallLedger (Phase 12B-FC V2).

Maps a request's ledger records to the per-turn ``validation`` block the final
E2E report needs: expected vs actual roles, model-match proof, degradation
detection.  The evaluator supplies ``expected_roles`` per scenario; the API
produces ``actual_roles`` and the model/proof fields from the ledger.
"""

from __future__ import annotations


def validate_turn(records: list[dict], expected_roles=None, *, required_model=None,
                  parser_failed: bool = False) -> dict:
    """Return the validation block + pass/issues for one E2E turn."""
    expected_roles = list(expected_roles or [])
    actual_roles = sorted({r["role"] for r in records if r.get("role")})
    missing = [r for r in expected_roles if r not in actual_roles]
    model_mismatches = [
        r for r in records
        if r.get("actual_model") and required_model and r["actual_model"] != required_model
    ]
    degraded_calls = [r for r in records if r.get("fallback_used") or r.get("cache_hit")]
    degraded = bool(degraded_calls) or bool(parser_failed)

    issues = []
    if missing:
        issues.append(f"missing_roles={missing}")
    if model_mismatches:
        issues.append(f"model_mismatch={[r['role'] for r in model_mismatches]}")
    if degraded:
        issues.append(f"degradation_used:{'calls=' + str(len(degraded_calls)) if degraded_calls else 'parser_failed'}")

    return {
        "profile": "12b_full_chain_no_fallback",
        "expected_roles": expected_roles,
        "actual_roles": actual_roles,
        "all_expected_roles_called": not missing,
        "all_models_match": not model_mismatches,
        "degradation_used": degraded,
        "model_call_ids": [r.get("call_id") for r in records if r.get("call_id")],
        "passed": not issues,
        "issues": issues,
    }
