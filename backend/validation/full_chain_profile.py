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


# --- RX (Response Experience & Evidence Presentation) flags -----------------
# Independent of the 12B full-chain validation profile.  Enabled only on the
# dedicated RX validation instance (8092) via SENTRIX_RX_V1; never on 8091.

def rx_active() -> bool:
    """Master switch for the RX answer pipeline (AnswerBrief -> Writer -> Validator)."""
    return os.getenv("SENTRIX_RX_V1", "0").strip().lower() in {"1", "true", "on"}


def answer_brief_active() -> bool:
    return rx_active() and os.getenv("SENTRIX_ANSWER_BRIEF_V1", "0").strip().lower() in {"1", "true", "on"}


def response_plan_active() -> bool:
    return rx_active() and os.getenv("SENTRIX_RESPONSE_PLAN_V1", "0").strip().lower() in {"1", "true", "on"}


def visible_evidence_active() -> bool:
    return rx_active() and os.getenv("SENTRIX_VISIBLE_EVIDENCE_V1", "0").strip().lower() in {"1", "true", "on"}


def response_writer_active() -> bool:
    return rx_active() and os.getenv("SENTRIX_RESPONSE_WRITER_V2", "0").strip().lower() in {"1", "true", "on"}


def response_validator_active() -> bool:
    return rx_active() and os.getenv("SENTRIX_RESPONSE_VALIDATOR_V1", "0").strip().lower() in {"1", "true", "on"}


def admin_debug_presentation() -> bool:
    """Admin/debug presentation switch — also gates the frontend debug layer."""
    return os.getenv("SENTRIX_ADMIN_DEBUG_PRESENTATION", "0").strip().lower() in {"1", "true", "on"}


# --- TFPE v2 (Structured Memory) flags --------------------------------------
# Model-driven strategy judgment (parser answer_type/strategy_hint) + a
# deterministic structured executor.  All gated behind rx_active() so the
# structured path reuses the RX AnswerBrief/Writer/Validator boundary.  Default
# off; enabled on the 8092 validation instance only.

def task_contract_active() -> bool:
    return structured_memory_active() and os.getenv("SENTRIX_TASK_CONTRACT_V2", "0").strip().lower() in {"1", "true", "on"}


def retrieval_strategy_active() -> bool:
    return structured_memory_active() and os.getenv("SENTRIX_RETRIEVAL_STRATEGY_V1", "0").strip().lower() in {"1", "true", "on"}


def structured_memory_active() -> bool:
    return rx_active() and os.getenv("SENTRIX_STRUCTURED_MEMORY_V1", "0").strip().lower() in {"1", "true", "on"}
