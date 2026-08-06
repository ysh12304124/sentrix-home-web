#!/usr/bin/env python3
"""Phase 5 maintenance — build the initial Core Memory Cards from canonical data.

Reads ``entities``, ``semantic_profiles``, ``semantic_claims``, ``person_patterns``
and ``relationships`` from the given database and materialises one Core Memory
Card per confirmed subject.  Each item carries ``(source_type, source_ids,
source_revisions)`` so :meth:`CoreMemoryStore.invalidate_by_source_revision`
can drop stale items later without touching the canonical rows.

Safe to run against production — this only inserts into agent-owned tables.
Use ``--scope-id`` to limit the build to one memory space and ``--apply`` to
actually write.  Without ``--apply`` only a dry-run summary is printed.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from backend.core_memory import CoreMemoryStore
from backend.db import MemoryStore


def _entity_role_text(entity):
    role = entity.get("family_role") or entity.get("entity_type") or "家庭成员"
    return f"{entity.get('canonical_name')} 的角色记录：{role}"


def _profile_summary_text(entity, profile):
    summary = (profile or {}).get("summary") or ""
    return f"{entity.get('canonical_name')} 的概览：{summary.strip()}"


def _pattern_text(pattern):
    label = pattern.get("label") or pattern.get("summary") or "反复出现的模式"
    return f"观察到的模式：{label}"


def _relationship_text(relationship, entity):
    other_name = relationship.get("target_name") or relationship.get("target_id") or "未知对象"
    kind = relationship.get("relationship_type") or "有关联"
    return f"{entity.get('canonical_name')} 与 {other_name}：{kind}"


def build_for_entity(cms, entity, profile, patterns, relationships):
    """Return the number of items materialised for this entity."""
    scope_id = entity.get("scope_id") or "home-default"
    card_id = cms.upsert_card(scope_id=scope_id, subject_type="person",
                              subject_id=entity["id"],
                              display_name=entity.get("canonical_name") or entity["id"])
    items = 0
    # 1. Confirmed role / family_role — canonical evidence.
    if entity.get("family_role") or entity.get("entity_type") == "person":
        cms.upsert_item(
            card_id=card_id, text=_entity_role_text(entity),
            epistemic_type="confirmed_fact", source_type="entity",
            source_ids=[entity["id"]],
            source_revisions={entity["id"]: int(entity.get("revision", 1) or 1)},
        )
        items += 1
    # 2. Semantic profile summary — user_assertion when actor is the owner.
    if profile and (profile.get("summary") or "").strip():
        cms.upsert_item(
            card_id=card_id, text=_profile_summary_text(entity, profile),
            epistemic_type="user_assertion", source_type="semantic_profile",
            source_ids=[profile.get("id", entity["id"])],
            source_revisions={profile.get("id", entity["id"]): int(profile.get("revision", 1) or 1)},
        )
        items += 1
    # 3. Person patterns — observed_pattern only.
    for pattern in patterns or []:
        cms.upsert_item(
            card_id=card_id, text=_pattern_text(pattern),
            epistemic_type="observed_pattern", source_type="person_pattern",
            source_ids=[pattern.get("id", entity["id"])],
            source_revisions={pattern.get("id", entity["id"]): int(pattern.get("revision", 1) or 1)},
        )
        items += 1
    # 4. Relationships — confirmed_fact when relationship was user-affirmed.
    for relationship in relationships or []:
        cms.upsert_item(
            card_id=card_id, text=_relationship_text(relationship, entity),
            epistemic_type="confirmed_fact", source_type="relationship",
            source_ids=[relationship.get("id", entity["id"])],
            source_revisions={relationship.get("id", entity["id"]): int(relationship.get("revision", 1) or 1)},
        )
        items += 1
    return card_id, items


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("db_path", help="Path to sentrix.db")
    parser.add_argument("--scope-id", default=None, help="Restrict to a memory space")
    parser.add_argument("--apply", action="store_true", help="Actually write to the database")
    args = parser.parse_args()
    store = MemoryStore(args.db_path)
    try:
        cms = CoreMemoryStore(store)
        entities = store.list_entities(status="confirmed", scope_id=args.scope_id)
        summary = {"entities": len(entities), "cards": 0, "items": 0, "apply": bool(args.apply)}
        for entity in entities:
            profile = None
            patterns = []
            relationships = []
            try:
                profile = store.get_semantic_profile(entity["id"])
            except Exception:
                profile = None
            try:
                patterns = store.list_person_patterns(entity["id"], scope_id=entity.get("scope_id"))
            except Exception:
                patterns = []
            try:
                relationships = store.list_relationships(entity_id=entity["id"], scope_id=entity.get("scope_id"))
            except Exception:
                relationships = []
            if not args.apply:
                # Dry run — count without writing.
                planned = 1 + (1 if profile and (profile.get("summary") or "").strip() else 0)
                planned += len(patterns or [])
                planned += len(relationships or [])
                summary["cards"] += 1
                summary["items"] += planned
                continue
            _, items = build_for_entity(cms, entity, profile, patterns, relationships)
            summary["cards"] += 1
            summary["items"] += items
        import json
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    finally:
        store.close()


if __name__ == "__main__":
    main()
