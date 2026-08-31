"""公共证据合同。

Planner、工具注册表、账本和最终门禁必须共享同一份公共类型集合。
``memory_reference`` 等内部传输字段不能绕过这个合同成为可完成任务的证据。
"""

from __future__ import annotations

PUBLIC_EVIDENCE_TYPES = frozenset({
    "memory_asset",
    "location_metadata",
    "temporal_metadata",
    "confirmed_identity",
    "photo_identity",
    "visual_observation",
    "visible_text",
    "structured_fact",
    "user_statement",
})


def is_public_evidence_type(value: str) -> bool:
    return str(value or "") in PUBLIC_EVIDENCE_TYPES


def planner_evidence_types() -> tuple[str, ...]:
    """Stable prompt order; the set remains the source of truth."""
    return tuple(sorted(PUBLIC_EVIDENCE_TYPES))

