"""Validation profile flags (Phase 12B-FC V2).

All flags are OFF unless SENTRIX_12B_FULL_CHAIN_VALIDATION=1 is set AND the
specific flag is on.  The master switch is only enabled on the dedicated
validation API instance (port 8092), never on production 8091.
"""

from __future__ import annotations

import os

_MASTER = "SENTRIX_12B_FULL_CHAIN_VALIDATION"


def _on(name: str) -> bool:
    return os.getenv(_MASTER, "0").strip().lower() in {"1", "true", "on"} and \
        os.getenv(name, "0").strip().lower() in {"1", "true", "on"}


def validation_active() -> bool:
    return os.getenv(_MASTER, "0").strip().lower() in {"1", "true", "on"}


def no_fallback() -> bool:
    """Forbid fallback text on model failure; the case must fail instead."""
    return _on("SENTRIX_AGENT_NO_FALLBACK")


def disable_cache() -> bool:
    """Forbid model response caching (guard; no cache exists today)."""
    return _on("SENTRIX_AGENT_DISABLE_CACHE")


def require_model_trace() -> bool:
    """Every model call must emit a ModelCallRecord; breaker treated as closed."""
    return _on("SENTRIX_AGENT_REQUIRE_MODEL_TRACE")


def require_12b_roles() -> bool:
    """actual_model must equal the profile 12B for every call."""
    return _on("SENTRIX_AGENT_REQUIRE_12B_ROLES")


def fail_on_degradation() -> bool:
    """Any fallback / cache / parser_failed / template-answer fails the case."""
    return _on("SENTRIX_AGENT_FAIL_ON_DEGRADATION")
