"""Model routing, unified deadline and circuit breaker (Phase R R5).

Roles:
  - parser  : structural JSON, low latency (may point at a 153 2B backend, D6)
  - answer  : natural expression (Writer / normal chat), stays gemma4:12b (D4)
  - verify  : claim/verifier, only for complex paths

P0-11: every request gets a single ``RequestDeadline`` (default 20s).  Phases
draw from the remaining budget; the deadline is enforced *before* a 30s httpx
timeout can trigger, so the API never waits for a hung model.

Circuit breaker trips per role after ``threshold`` consecutive failures and
routes that role to a deterministic fallback instead of another slow call.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Callable

PARSER = "parser"
ANSWER = "answer"
VERIFY = "verify"
CLAIM = "claim"
REPAIR = "repair"
ROLES = (PARSER, ANSWER, VERIFY, CLAIM, REPAIR)

# 12B-FC: the parser budget is raised for the 12B parser (GPU warm ~4s through
# the full prompt; the old 4s cap made it time out and trip the breaker).  Still
# inside the 20s API deadline.  Overridable via SENTRIX_PARSER_BUDGET.
DEFAULT_PHASE_BUDGETS = {PARSER: 8.0, "retrieval": 5.0, ANSWER: 7.0, "overhead": 2.0}


def _parser_budget():
    try:
        return float(os.environ.get("SENTRIX_PARSER_BUDGET", str(DEFAULT_PHASE_BUDGETS[PARSER])))
    except (TypeError, ValueError):
        return DEFAULT_PHASE_BUDGETS[PARSER]


@dataclass(frozen=True)
class ModelSpec:
    role: str
    model: str
    backend: str = "ollama_local"
    base_url: str = ""


def _env(env, key, default):
    return (env or os.environ).get(key, default)


def resolve_specs(env=None) -> dict[str, ModelSpec]:
    main = _env(env, "OLLAMA_MODEL", "gemma4:12b")
    profile = _env(env, "SENTRIX_AGENT_MODEL_PROFILE", "quality_12b").strip().lower()
    parse_model = _env(env, "SENTRIX_PARSE_MODEL", None)
    parse_backend = _env(env, "SENTRIX_PARSE_BACKEND", None)
    parse_base = _env(env, "SENTRIX_PARSE_BASE_URL", "")
    if parse_model is None and profile == "experimental_2b":
        parse_model = "gemma-4-e2b-it+lora-v2"
    if parse_backend is None and profile == "experimental_2b":
        parse_backend = "e2b"
    if not parse_base and profile == "experimental_2b":
        parse_base = "http://127.0.0.1:8100"
    parse_model = parse_model or main
    parse_backend = parse_backend or "ollama_local"
    answer_model = _env(env, "SENTRIX_ANSWER_MODEL", main)
    verify_model = _env(env, "SENTRIX_VERIFY_MODEL", main)
    claim_model = _env(env, "SENTRIX_CLAIM_MODEL", verify_model)
    repair_model = _env(env, "SENTRIX_REPAIR_MODEL", parse_model)
    return {
        PARSER: ModelSpec(PARSER, parse_model, parse_backend, parse_base),
        ANSWER: ModelSpec(ANSWER, answer_model, "ollama_local", ""),
        VERIFY: ModelSpec(VERIFY, verify_model, "ollama_local", ""),
        CLAIM: ModelSpec(CLAIM, claim_model, "ollama_local", ""),
        REPAIR: ModelSpec(REPAIR, repair_model, parse_backend, parse_base),
    }


@dataclass
class RequestDeadline:
    deadline_seconds: float = 20.0
    phase_budgets: dict = field(default_factory=lambda: dict(DEFAULT_PHASE_BUDGETS))

    def __post_init__(self):
        self.phase_budgets[PARSER] = _parser_budget()
        self._started = time.monotonic()

    def remaining(self) -> float:
        elapsed = time.monotonic() - self._started
        return max(0.0, self.deadline_seconds - elapsed)

    def budget_for(self, phase: str) -> float:
        return self.phase_budgets.get(phase, self.phase_budgets.get("overhead", 2.0))

    def phase_available(self, phase: str) -> float:
        return min(self.remaining(), self.budget_for(phase))


@dataclass
class CircuitBreaker:
    threshold: int = 3
    trip_duration_s: float = 60.0

    def __init__(self, threshold: int = 3, trip_duration_s: float = 60.0):
        self.threshold = threshold
        self.trip_duration_s = trip_duration_s
        self._failures: dict[str, int] = {}
        self._tripped_at: dict[str, float] = {}

    def record_success(self, role: str):
        self._failures.pop(role, None)
        self._tripped_at.pop(role, None)

    def record_failure(self, role: str):
        self._failures[role] = self._failures.get(role, 0) + 1
        if self._failures[role] >= self.threshold:
            self._tripped_at.setdefault(role, time.monotonic())

    def is_tripped(self, role: str) -> bool:
        tripped_at = self._tripped_at.get(role)
        if tripped_at is None:
            return False
        if time.monotonic() - tripped_at >= self.trip_duration_s:
            self._tripped_at.pop(role, None)
            self._failures[role] = 0
            return False
        return True


class ModelRouter:
    """Routes a model role through deadline-aware, breaker-protected calls."""

    def __init__(self, gamma=None, *, env=None, deadline=None, breaker=None):
        self.gamma = gamma
        self.specs = resolve_specs(env)
        self.deadline = deadline or RequestDeadline()
        self.breaker = breaker or CircuitBreaker()

    def chat(self, role: str, prompt: str, *, json_mode: bool = True, fallback=None):
        """Call the role's model with deadline + breaker.

        ``fallback`` is a zero-arg callable returning the deterministic result
        when the role is tripped or the phase budget is exhausted.
        """
        if not self.gamma or not hasattr(self.gamma, "chat"):
            return fallback() if fallback else None
        if self.breaker.is_tripped(role):
            return fallback() if fallback else None
        available = self.deadline.phase_available(role)
        if available <= 0:
            return fallback() if fallback else None
        original_timeout = getattr(self.gamma, "timeout", None)
        if original_timeout:
            self.gamma.timeout = max(0.1, available)
        try:
            result = self.gamma.chat(prompt, json_mode=json_mode, role=role)
            self.breaker.record_success(role)
            return result
        except Exception:
            self.breaker.record_failure(role)
            return fallback() if fallback else None
        finally:
            if original_timeout:
                self.gamma.timeout = original_timeout
