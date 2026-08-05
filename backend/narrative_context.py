"""Narrative context packet builder for Thin Agent complex paths (Phase 4).

The Writer never receives canonical Observation text as the only source of
truth.  ``build_narrative_context_packet`` prepares a supporting context view
that the Writer may consult for language, while the Verifier keeps working
against the canonical Evidence Bundle built by
:func:`evidence_retrieval.build_verifier_evidence_bundle`.
"""

from __future__ import annotations


def build_narrative_context_packet(packet, spec):
    """Return a compact narrative view of an EvidencePacket.

    The result contains:

    - ``subject``: the confirmed person/event the user asked about.
    - ``anchors``: at most 8 evidence items with place/activity/time/clothing.
    - ``patterns``: repeated categories across evidence — used as hint, not
      as fact.
    - ``time_range``: earliest/latest ``captured_at``.
    - ``coverage``: count of exact/strong/approximate results.
    """
    subject = None
    for constraint in getattr(spec, "constraints", []) or []:
        if constraint.dimension == "person":
            subject = {"kind": "person", "name": constraint.value}
            break
    if subject is None:
        subject = {"kind": "general", "name": None}
    anchors = []
    times = []
    for item in packet.assets[:8]:
        fields = item.get("observation_fields") or {}
        anchors.append({
            "asset_id": item.get("asset_id"),
            "file_name": item.get("file_name"),
            "captured_at": item.get("captured_at"),
            "place": fields.get("place"),
            "activity": fields.get("activity"),
            "subject_clothing": fields.get("subject_clothing") or [],
            "level": item.get("level"),
            "evidence_ids": item.get("evidence_ids", []),
        })
        if item.get("captured_at"):
            times.append(item["captured_at"])
    patterns = {"places": [], "activities": []}
    for anchor in anchors:
        if anchor["place"] and anchor["place"] not in patterns["places"]:
            patterns["places"].append(anchor["place"])
        if anchor["activity"] and anchor["activity"] not in patterns["activities"]:
            patterns["activities"].append(anchor["activity"])
    return {
        "subject": subject,
        "anchors": anchors,
        "patterns": patterns,
        "time_range": {"earliest": min(times) if times else None, "latest": max(times) if times else None},
        "coverage": {
            "exact": len(packet.exact_results),
            "strong": len(packet.strong_results),
            "approximate": len(packet.approximate_results),
            "gaps": len(packet.gaps),
        },
    }
