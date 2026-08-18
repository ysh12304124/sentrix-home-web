"""Typed evidence ledger for the Agent 2 shadow path."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .task_state import EVIDENCE_TYPES


@dataclass(frozen=True)
class Coverage:
    requested: int = 0
    processed: int = 0
    skipped_budget: int = 0
    failed: int = 0

    def __post_init__(self):
        values = (self.requested, self.processed, self.skipped_budget, self.failed)
        if any(not isinstance(value, int) or value < 0 for value in values):
            raise ValueError("coverage values must be non-negative integers")
        if self.processed + self.skipped_budget + self.failed > self.requested:
            raise ValueError("coverage exceeds requested items")

    @property
    def is_partial(self) -> bool:
        return self.processed < self.requested

    def as_dict(self) -> dict[str, int]:
        return {
            "requested": self.requested,
            "processed": self.processed,
            "skipped_budget": self.skipped_budget,
            "failed": self.failed,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "Coverage":
        return cls(
            requested=int(payload.get("requested") or 0),
            processed=int(payload.get("processed") or 0),
            skipped_budget=int(payload.get("skipped_budget") or 0),
            failed=int(payload.get("failed") or 0),
        )


@dataclass(frozen=True)
class LedgerEntry:
    tool_call_id: str
    capability: str
    evidence_type: str
    input_refs: tuple[str, ...]
    provenance_refs: tuple[str, ...]
    certainty: str = "uncertain"
    coverage: Coverage = Coverage()
    failure_reason: str = ""
    provenance_scope_id: str = ""
    # P1: Explicit subject & asset scope binding
    subject: str = ""
    asset_id: str = ""
    region_bbox: tuple[float, ...] = ()
    extracted_value: Any = None

    def __post_init__(self):
        if not self.tool_call_id:
            raise ValueError("tool_call_id is required")
        if not self.capability:
            raise ValueError("capability is required")
        if self.evidence_type not in EVIDENCE_TYPES:
            raise ValueError(f"unsupported evidence type: {self.evidence_type}")

    def as_dict(self) -> dict[str, Any]:
        data = {
            "tool_call_id": self.tool_call_id,
            "capability": self.capability,
            "evidence_type": self.evidence_type,
            "input_refs": list(self.input_refs),
            "provenance_refs": list(self.provenance_refs),
            "certainty": self.certainty,
            "coverage": self.coverage.as_dict(),
            "failure_reason": self.failure_reason,
            "provenance_scope_id": self.provenance_scope_id,
        }
        if self.subject:
            data["subject"] = self.subject
        if self.asset_id:
            data["asset_id"] = self.asset_id
        if self.region_bbox:
            data["region_bbox"] = list(self.region_bbox)
        if self.extracted_value is not None:
            data["extracted_value"] = self.extracted_value
        return data

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "LedgerEntry":
        bbox = payload.get("region_bbox") or ()
        return cls(
            tool_call_id=str(payload.get("tool_call_id") or ""),
            capability=str(payload.get("capability") or ""),
            evidence_type=str(payload.get("evidence_type") or ""),
            input_refs=tuple(str(ref) for ref in payload.get("input_refs") or []),
            provenance_refs=tuple(str(ref) for ref in payload.get("provenance_refs") or []),
            certainty=str(payload.get("certainty") or "uncertain"),
            coverage=Coverage.from_dict(dict(payload.get("coverage") or {})),
            failure_reason=str(payload.get("failure_reason") or ""),
            provenance_scope_id=str(payload.get("provenance_scope_id") or ""),
            subject=str(payload.get("subject") or ""),
            asset_id=str(payload.get("asset_id") or ""),
            region_bbox=tuple(float(v) for v in bbox) if bbox else (),
            extracted_value=payload.get("extracted_value"),
        )


class EvidenceLedger:
    def __init__(self, *, scope_id: str, entries: tuple[LedgerEntry, ...] = ()):
        if not scope_id:
            raise ValueError("scope_id is required")
        self.scope_id = scope_id
        self.entries: list[LedgerEntry] = []
        for entry in entries:
            self.append(entry)

    def append(self, entry: LedgerEntry) -> None:
        if entry.provenance_scope_id and entry.provenance_scope_id != self.scope_id:
            raise ValueError("scope mismatch")
        if any(existing.tool_call_id == entry.tool_call_id and existing.evidence_type == entry.evidence_type for existing in self.entries):
            raise ValueError("duplicate tool call")
        self.entries.append(entry)

    def as_dict(self) -> dict[str, Any]:
        return {
            "scope_id": self.scope_id,
            "entries": [entry.as_dict() for entry in self.entries],
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "EvidenceLedger":
        return cls(
            scope_id=str(payload.get("scope_id") or ""),
            entries=tuple(
                LedgerEntry.from_dict(item)
                for item in payload.get("entries") or []
            ),
        )
