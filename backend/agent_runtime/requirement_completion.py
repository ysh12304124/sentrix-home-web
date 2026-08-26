"""Requirement-driven completion for the Agent 2 shadow path.

This module deliberately knows nothing about user phrasing.  It matches
the evidence requested by the planner with the evidence a capability actually
produced, supporting Many-to-Many resolution, partial support, and asset binding.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from .evidence_ledger import EvidenceLedger, LedgerEntry
from .task_state import TaskState
from .tool_registry import ToolSpec


class RequirementCompletion:
    def __init__(self, task_state: TaskState, evidence_ledger: EvidenceLedger):
        if task_state.declaration.scope_id != evidence_ledger.scope_id:
            raise ValueError("task state and evidence ledger scope mismatch")
        self.task_state = task_state
        self.evidence_ledger = evidence_ledger

    def allowed_capabilities(self, specs: Iterable[ToolSpec]) -> list[ToolSpec]:
        """Return ready capabilities that can satisfy an open or running planner need."""
        open_types = {
            state.requirement.evidence_type
            for state in self.task_state.requirements.values()
            if state.status in {"open", "running", "partially_supported"}
        }
        return [
            spec for spec in specs
            if spec.readiness == "ready"
            and any(spec.can_satisfy(evidence_type) for evidence_type in open_types)
        ]

    def satisfy_from_entry(
        self,
        requirement_id: str,
        tool_call_id: str,
        *,
        asset_id: str = "",
        subject: str = "",
        allow_partial: bool = True,
    ) -> bool:
        """Satisfy a running or open requirement with compatible recorded evidence and scope binding."""
        requirement = self.task_state.requirement(requirement_id)
        if requirement.status not in {"running", "open", "partially_supported"}:
            return False

        entries = [
            item for item in self.evidence_ledger.entries
            if item.tool_call_id == tool_call_id
            and item.evidence_type == requirement.requirement.evidence_type
        ]
        if not entries:
            return False

        # If asset_id or subject constraint is provided, filter entries
        matching_entry: LedgerEntry | None = None
        for entry in entries:
            if requirement_id not in entry.requirement_refs:
                # Evidence without an explicit requirement binding is valid
                # telemetry, but cannot satisfy a requirement implicitly by
                # sharing only its evidence type.
                continue
            if asset_id and entry.asset_id and entry.asset_id != asset_id:
                continue
            if subject and entry.subject and entry.subject != subject:
                continue
            matching_entry = entry
            break

        if matching_entry is None:
            return False

        # Check coverage
        if matching_entry.coverage.is_partial:
            if allow_partial:
                self.task_state.mark_partially_supported(
                    requirement_id, evidence_refs=(matching_entry.tool_call_id,)
                )
                return True
            return False

        self.task_state.mark_satisfied(
            requirement_id, evidence_refs=(matching_entry.tool_call_id,)
        )
        return True

    def auto_match_all_open_requirements(self) -> int:
        """Match all open/running requirements against available ledger entries (many-to-many)."""
        satisfied_count = 0
        for req_id, req_state in list(self.task_state.requirements.items()):
            if req_state.status in {"open", "running", "partially_supported"}:
                req_type = req_state.requirement.evidence_type
                # Find matching entries
                matching_entries = [
                    entry for entry in self.evidence_ledger.entries
                    if entry.evidence_type == req_type
                    and req_id in entry.requirement_refs
                ]
                if matching_entries:
                    refs = tuple(e.tool_call_id for e in matching_entries)
                    has_partial = any(e.coverage.is_partial for e in matching_entries)
                    if has_partial and req_state.status == "open":
                        self.task_state.mark_partially_supported(req_id, evidence_refs=refs)
                        satisfied_count += 1
                    else:
                        self.task_state.mark_satisfied(req_id, evidence_refs=refs)
                        satisfied_count += 1
        return satisfied_count
