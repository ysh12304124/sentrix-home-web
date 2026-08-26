"""Typed planner task state for the Agent 2 shadow path."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .evidence_contract import PUBLIC_EVIDENCE_TYPES


EVIDENCE_TYPES = PUBLIC_EVIDENCE_TYPES

# P0: Formal unmet reasons to explicitly categorize why a requirement was not satisfied
UNMET_REASONS = frozenset({
    "unnecessary_requirement",
    "no_matching_evidence",
    "capability_missing",
    "capability_not_executed",
    "budget_exhausted",
    "wrong_scope",
    "evidence_incompatible",
    "ambiguous_reference",
    "inherently_unanswerable",
})

# P2: Formal task terminal states decoupled from individual requirement status
TASK_TERMINAL_STATES = frozenset({
    "answered",
    "unsupported_clarified",
    "blocked_scope",
    "budget_exhausted",
    "fallback",
    "shadow_only",
})

_TRANSITIONS = {
    "open": {"running", "ambiguous", "unsupported", "blocked_budget", "partially_supported", "unresolved", "unavailable", "contradicted"},
    "running": {"satisfied", "partially_supported", "unresolved", "ambiguous", "unsupported", "blocked_budget", "unavailable", "contradicted"},
    "partially_supported": {"running", "satisfied", "unresolved", "blocked_budget", "unavailable", "contradicted"},
    "satisfied": set(),
    "ambiguous": set(),
    "unsupported": set(),
    "unresolved": set(),
    "blocked_budget": set(),
    "unavailable": set(),
}


@dataclass(frozen=True)
class EvidenceRequirement:
    id: str
    evidence_type: str
    description: str = ""
    required: bool = True
    parent_id: str = ""
    lineage_reason: str = ""

    def __post_init__(self):
        if not isinstance(self.id, str) or not self.id:
            raise ValueError("requirement id is required")
        if self.evidence_type not in EVIDENCE_TYPES:
            raise ValueError(f"unsupported evidence type: {self.evidence_type}")

    def as_dict(self) -> dict[str, str]:
        data = {
            "id": self.id,
            "evidence_type": self.evidence_type,
            "description": self.description,
            "required": self.required,
        }
        if self.parent_id:
            data["parent_id"] = self.parent_id
        if self.lineage_reason:
            data["lineage_reason"] = self.lineage_reason
        return data

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "EvidenceRequirement":
        return cls(
            id=str(payload.get("id") or ""),
            evidence_type=str(payload.get("evidence_type") or ""),
            description=str(payload.get("description") or ""),
            required=bool(payload.get("required", True)),
            parent_id=str(payload.get("parent_id") or ""),
            lineage_reason=str(payload.get("lineage_reason") or ""),
        )


@dataclass(frozen=True)
class TaskDeclaration:
    goal: str
    scope_id: str
    requirements: tuple[EvidenceRequirement, ...]
    constraints: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not isinstance(self.goal, str) or not self.goal:
            raise ValueError("goal is required")
        if not isinstance(self.scope_id, str) or not self.scope_id:
            raise ValueError("scope_id is required")
        if not self.requirements:
            raise ValueError("at least one requirement is required")
        ids = [requirement.id for requirement in self.requirements]
        if len(ids) != len(set(ids)):
            raise ValueError("requirement ids must be unique")

    def as_dict(self) -> dict[str, Any]:
        return {
            "goal": self.goal,
            "scope_id": self.scope_id,
            "constraints": self.constraints,
            "requirements": [requirement.as_dict() for requirement in self.requirements],
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "TaskDeclaration":
        return cls(
            goal=str(payload.get("goal") or ""),
            scope_id=str(payload.get("scope_id") or ""),
            constraints=dict(payload.get("constraints") or {}),
            requirements=tuple(
                EvidenceRequirement.from_dict(item)
                for item in payload.get("requirements") or []
            ),
        )


@dataclass
class RequirementState:
    requirement: EvidenceRequirement
    status: str = "open"
    evidence_refs: tuple[str, ...] = ()
    unmet_reason: str = ""
    coverage_status: str = "candidate"
    failure_reason: str = ""
    attempt_count: int = 0
    last_attempt: str = ""
    attempts: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self):
        if self.unmet_reason and self.unmet_reason not in UNMET_REASONS:
            raise ValueError(f"unsupported unmet reason: {self.unmet_reason}")

    def as_dict(self) -> dict[str, Any]:
        data = {
            **self.requirement.as_dict(),
            "status": self.status,
            "evidence_refs": list(self.evidence_refs),
            "coverage_status": self.coverage_status,
            "attempt_count": self.attempt_count,
            "attempts": list(self.attempts[-8:]),
        }
        if self.last_attempt:
            data["last_attempt"] = self.last_attempt
        if self.unmet_reason:
            data["unmet_reason"] = self.unmet_reason
        if self.failure_reason:
            data["failure_reason"] = self.failure_reason
        return data


@dataclass
class TaskState:
    declaration: TaskDeclaration
    requirements: dict[str, RequirementState]
    terminal_outcome: str = ""
    status: str = "planned"

    def __post_init__(self):
        if self.terminal_outcome and self.terminal_outcome not in TASK_TERMINAL_STATES:
            raise ValueError(f"unsupported terminal outcome: {self.terminal_outcome}")

    @classmethod
    def from_declaration(cls, declaration: TaskDeclaration) -> "TaskState":
        return cls(
            declaration=declaration,
            requirements={
                requirement.id: RequirementState(requirement=requirement)
                for requirement in declaration.requirements
            },
        )

    def recompute_status(self, *, has_available_tools: bool = True) -> str:
        """Derive the single task status from requirement evidence.

        Legacy per-requirement states are retained for replay compatibility, but
        this status is the only completion decision used by the production path.
        """
        required = [state for state in self.requirements.values()
                    if state.requirement.required]
        if not required:
            self.status = "blocked"
        elif all(state.status == "satisfied" for state in required):
            self.status = "complete"
        elif any(state.status in {"ambiguous", "unsupported", "blocked_budget", "unavailable"}
                 for state in required) and not has_available_tools:
            self.status = "insufficient_evidence"
        elif has_available_tools:
            self.status = "in_progress"
        else:
            self.status = "insufficient_evidence"
        return self.status

    def requirement(self, requirement_id: str) -> RequirementState:
        try:
            return self.requirements[requirement_id]
        except KeyError as exc:
            raise ValueError(f"unknown requirement: {requirement_id}") from exc

    def required_states(self) -> tuple[RequirementState, ...]:
        """Return required evidence states in declaration order."""
        return tuple(state for state in self.requirements.values()
                     if state.requirement.required)

    def unattempted_required(self) -> tuple[RequirementState, ...]:
        """Return required evidence needs with no recorded attempt."""
        return tuple(state for state in self.required_states()
                     if state.attempt_count <= 0)

    def all_required_attempted(self) -> bool:
        """Whether every required evidence need has been tried at least once."""
        return not self.unattempted_required()

    def mark_running(self, requirement_id: str) -> None:
        state = self.requirement(requirement_id)
        self._transition(requirement_id, "running")

    def record_attempt(self, requirement_id: str, attempt: dict[str, Any]) -> None:
        """Record the concrete input/outcome used for one evidence attempt.

        The bounded history lets recovery choose a genuinely new candidate and
        makes a failed visual/OCR attempt auditable without exposing raw model
        prompts.
        """
        state = self.requirement(requirement_id)
        data = dict(attempt or {})
        state.attempt_count += 1
        data.setdefault("attempt", state.attempt_count)
        state.attempts.append(data)
        del state.attempts[:-8]
        state.last_attempt = str(data.get("outcome") or data.get("status") or "")

    def mark_satisfied(self, requirement_id: str, *, evidence_refs: tuple[str, ...]) -> None:
        if not evidence_refs:
            raise ValueError("satisfied requirement needs evidence refs")
        requirement = self.requirement(requirement_id)
        if requirement.status == "open":
            self.mark_running(requirement_id)
        self._transition(requirement_id, "satisfied")
        requirement.evidence_refs = tuple(evidence_refs)
        requirement.unmet_reason = ""
        requirement.coverage_status = "confirmed"
        requirement.failure_reason = ""

    def mark_partially_supported(self, requirement_id: str, *, evidence_refs: tuple[str, ...]) -> None:
        if not evidence_refs:
            raise ValueError("partially supported requirement needs evidence refs")
        requirement = self.requirement(requirement_id)
        if requirement.status == "open":
            self.mark_running(requirement_id)
        self._transition(requirement_id, "partially_supported")
        requirement.evidence_refs = tuple(evidence_refs)
        requirement.coverage_status = "supported"

    def mark_contradicted(self, requirement_id: str, *, evidence_refs: tuple[str, ...]) -> None:
        if not evidence_refs:
            raise ValueError("contradicted requirement needs evidence refs")
        requirement = self.requirement(requirement_id)
        if requirement.status == "open":
            self.mark_running(requirement_id)
        self._transition(requirement_id, "contradicted")
        requirement.evidence_refs = tuple(evidence_refs)
        requirement.coverage_status = "contradicted"
        requirement.failure_reason = ""

    def mark_evidence_failed(self, requirement_id: str, *, reason: str,
                             evidence_refs: tuple[str, ...] = ()) -> None:
        """Keep a failed requirement active while recording why the attempt failed."""
        requirement = self.requirement(requirement_id)
        if requirement.status == "open":
            self.mark_running(requirement_id)
        elif requirement.status == "partially_supported":
            self._transition(requirement_id, "running")
        if requirement.status not in {"running", "partially_supported"}:
            return
        requirement.evidence_refs = tuple(evidence_refs)
        requirement.coverage_status = "failed"
        requirement.failure_reason = str(reason or "evidence_failed")

    def mark_unmet(self, requirement_id: str, *, reason: str, status: str = "unresolved") -> None:
        if reason not in UNMET_REASONS:
            raise ValueError(f"unsupported unmet reason: {reason}")
        requirement = self.requirement(requirement_id)
        if requirement.status == "open":
            self.mark_running(requirement_id)
        self._transition(requirement_id, status)
        requirement.unmet_reason = reason
        requirement.coverage_status = "failed"
        requirement.failure_reason = reason

    def mark_unavailable(self, requirement_id: str, *, reason: str = "capability_missing") -> None:
        """Close a requirement when no registered capability can satisfy it."""
        self.mark_unmet(requirement_id, reason=reason, status="unavailable")

    def set_terminal_outcome(self, outcome: str) -> None:
        if outcome not in TASK_TERMINAL_STATES:
            raise ValueError(f"unsupported terminal outcome: {outcome}")
        self.terminal_outcome = outcome

    def _transition(self, requirement_id: str, target: str) -> None:
        requirement = self.requirement(requirement_id)
        if target not in _TRANSITIONS.get(requirement.status, set()):
            raise ValueError(f"invalid transition: {requirement.status} -> {target}")
        requirement.status = target

    def as_dict(self) -> dict[str, Any]:
        data = {
            "declaration": self.declaration.as_dict(),
            "status": self.status,
            "requirements": [
                self.requirements[requirement.id].as_dict()
                for requirement in self.declaration.requirements
            ],
        }
        if self.terminal_outcome:
            data["terminal_outcome"] = self.terminal_outcome
        return data

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "TaskState":
        declaration = TaskDeclaration.from_dict(dict(payload.get("declaration") or {}))
        state = cls.from_declaration(declaration)
        states_by_id = {
            str(item.get("id") or ""): item
            for item in payload.get("requirements") or []
            if isinstance(item, dict)
        }
        for requirement in declaration.requirements:
            item = states_by_id.get(requirement.id, {})
            restored = state.requirement(requirement.id)
            status = str(item.get("status") or "open")
            if status not in _TRANSITIONS:
                raise ValueError(f"unknown requirement status: {status}")
            restored.status = status
            restored.evidence_refs = tuple(str(ref) for ref in item.get("evidence_refs") or [])
            restored.coverage_status = str(item.get("coverage_status") or (
                "confirmed" if status == "satisfied" else
                "supported" if status == "partially_supported" else "candidate"))
            restored.failure_reason = str(item.get("failure_reason") or "")
            restored.attempt_count = int(item.get("attempt_count") or 0)
            restored.last_attempt = str(item.get("last_attempt") or "")
            restored.attempts = [dict(value) for value in item.get("attempts") or []
                                if isinstance(value, dict)][-8:]
            unmet_reason = str(item.get("unmet_reason") or "")
            if unmet_reason:
                if unmet_reason not in UNMET_REASONS:
                    raise ValueError(f"unsupported unmet reason: {unmet_reason}")
                restored.unmet_reason = unmet_reason
        terminal = str(payload.get("terminal_outcome") or "")
        if terminal:
            state.set_terminal_outcome(terminal)
        state.status = str(payload.get("status") or "planned")
        if state.status not in {"planned", "in_progress", "complete", "insufficient_evidence", "blocked"}:
            state.status = "planned"
        return state
