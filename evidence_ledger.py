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
    confidence: float | None = None
    requirement_refs: tuple[str, ...] = ()
    # Explicitly distinguish evidence that is valid but cannot satisfy any
    # currently declared requirement from a missing/failed tool result.
    unmatched_reason: str = ""

    def __post_init__(self):
        if not self.tool_call_id:
            raise ValueError("tool_call_id is required")
        if not self.capability:
            raise ValueError("capability is required")
        if self.evidence_type not in EVIDENCE_TYPES:
            raise ValueError(f"unsupported evidence type: {self.evidence_type}")
        if self.confidence is not None and not 0 <= self.confidence <= 1:
            raise ValueError("confidence must be between 0 and 1")

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
            "requirement_refs": list(self.requirement_refs),
            "unmatched_reason": self.unmatched_reason,
        }
        if self.subject:
            data["subject"] = self.subject
        if self.asset_id:
            data["asset_id"] = self.asset_id
        if self.region_bbox:
            data["region_bbox"] = list(self.region_bbox)
        if self.extracted_value is not None:
            data["extracted_value"] = self.extracted_value
            data["value"] = self.extracted_value
        if self.confidence is not None:
            data["confidence"] = self.confidence
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
            confidence=(float(payload["confidence"])
                        if payload.get("confidence") is not None else None),
            requirement_refs=tuple(str(ref) for ref in payload.get("requirement_refs") or []),
            unmatched_reason=str(payload.get("unmatched_reason") or ""),
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

    def build_answer_context(self, question: str, task_state: Any = None) -> dict[str, list]:
        """Compile minimal structured facts for the existing final writer.

        This is intentionally deterministic. It does not rank with a model and it
        never turns an absent or conflicting value into a claim.
        """
        requirements = _task_requirements(task_state)
        relevant_types = {item["evidence_type"] for item in requirements}
        entries = [entry for entry in self.entries if (
            not relevant_types or entry.evidence_type in relevant_types
        )]
        entries.sort(key=_answer_entry_sort_key, reverse=True)

        facts: list[dict[str, Any]] = []
        seen: set[str] = set()
        for entry in entries:
            if entry.extracted_value is None:
                continue
            fact = {
                "evidence_type": entry.evidence_type,
                "value": entry.extracted_value,
                "certainty": entry.certainty,
                "source_refs": list(entry.provenance_refs or (entry.tool_call_id,)),
            }
            if entry.confidence is not None:
                fact["confidence"] = entry.confidence
            if entry.subject:
                fact["subject"] = entry.subject
            if entry.asset_id:
                fact["asset"] = entry.asset_id
            if entry.requirement_refs:
                fact["requirement_refs"] = list(entry.requirement_refs)
            key = _fact_key(fact)
            if key not in seen:
                facts.append(fact)
                seen.add(key)

        conflicts = _find_conflicts(facts)
        facts_by_type = {fact["evidence_type"] for fact in facts}
        unknowns = []
        for item in requirements:
            if item["status"] in {"satisfied", "partially_supported"}:
                continue
            if item["evidence_type"] in facts_by_type:
                continue
            unknowns.append({
                "requirement_id": item["id"],
                "evidence_type": item["evidence_type"],
                "description": item["description"],
                "status": item["status"],
                "reason": item["unmet_reason"] or "no_matching_evidence",
            })
        return {"facts": facts, "unknowns": unknowns, "conflicts": conflicts}

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "EvidenceLedger":
        return cls(
            scope_id=str(payload.get("scope_id") or ""),
            entries=tuple(
                LedgerEntry.from_dict(item)
                for item in payload.get("entries") or []
            ),
        )


def _task_requirements(task_state: Any) -> list[dict[str, str]]:
    if task_state is None:
        return []
    if isinstance(task_state, dict):
        rows = task_state.get("requirements") or []
    else:
        rows = []
        for state in getattr(task_state, "requirements", {}).values():
            requirement = state.requirement
            rows.append({
                "id": requirement.id,
                "evidence_type": requirement.evidence_type,
                "description": requirement.description,
                "status": state.status,
                "unmet_reason": state.unmet_reason,
            })
    return [{
        "id": str(row.get("id") or ""),
        "evidence_type": str(row.get("evidence_type") or ""),
        "description": str(row.get("description") or ""),
        "status": str(row.get("status") or "open"),
        "unmet_reason": str(row.get("unmet_reason") or ""),
    } for row in rows if isinstance(row, dict)]


def _answer_entry_sort_key(entry: LedgerEntry) -> tuple[int, float, int]:
    direct = 1 if entry.evidence_type in {"visible_text", "visual_observation", "structured_fact"} else 0
    confidence = entry.confidence if entry.confidence is not None else (
        1.0 if entry.certainty in {"supported", "confirmed", "full_support"} else 0.5
    )
    return direct, confidence, len(entry.requirement_refs)


def _fact_key(fact: dict[str, Any]) -> str:
    import json
    return json.dumps({
        "type": fact.get("evidence_type"),
        "value": fact.get("value"),
        "subject": fact.get("subject"),
        "asset": fact.get("asset"),
    }, ensure_ascii=False, sort_keys=True, default=str)


def _find_conflicts(facts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for fact in facts:
        key = (
            str(fact.get("evidence_type") or ""),
            str(fact.get("subject") or ""),
            str(fact.get("asset") or ""),
        )
        grouped.setdefault(key, []).append(fact)
    conflicts = []
    for (evidence_type, subject, asset), rows in grouped.items():
        values = {_fact_key({"evidence_type": evidence_type, "value": row.get("value")}) for row in rows}
        if len(values) > 1 and (subject or asset):
            conflicts.append({
                "evidence_type": evidence_type,
                "subject": subject,
                "asset": asset,
                "values": [row.get("value") for row in rows],
                "source_refs": [ref for row in rows for ref in row.get("source_refs") or []],
            })
    return conflicts
