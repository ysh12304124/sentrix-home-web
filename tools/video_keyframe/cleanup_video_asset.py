"""Remove one failed isolated test video and its derived records."""

from __future__ import annotations

import argparse
import os

from backend.db import MemoryStore


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default=os.getenv("SENTRIX_DB_PATH", "data/sentrix.db"))
    parser.add_argument("--asset-id", required=True)
    args = parser.parse_args()
    store = MemoryStore(args.db)
    derived = [item["id"] for item in store.list_derived_assets(args.asset_id)]
    ids = [args.asset_id] + derived
    marks = ",".join("?" for _ in ids)
    observations = [item["id"] for item in store._rows(f"SELECT id FROM observations WHERE asset_id IN ({marks})", ids)]
    faces = [item["id"] for item in store._rows(f"SELECT id FROM face_instances WHERE asset_id IN ({marks})", ids)]
    events = [item["id"] for item in store._rows("SELECT id FROM events WHERE source_asset_id = ?", (args.asset_id,))]
    with store.connection:
        if observations:
            om = ",".join("?" for _ in observations)
            store.connection.execute(f"DELETE FROM event_observations WHERE observation_id IN ({om})", observations)
            store.connection.execute(f"DELETE FROM entity_mentions WHERE observation_id IN ({om})", observations)
            store.connection.execute(f"DELETE FROM entity_observations WHERE observation_id IN ({om})", observations)
            store.connection.execute(f"DELETE FROM person_appearance_evidence WHERE observation_id IN ({om})", observations)
            if faces:
                fm = ",".join("?" for _ in faces)
                store.connection.execute(f"DELETE FROM face_prototypes WHERE face_instance_id IN ({fm})", faces)
            store.connection.execute(f"DELETE FROM face_instances WHERE observation_id IN ({om})", observations)
            store.connection.execute(f"DELETE FROM memory_vectors WHERE source_type = 'observation' AND source_id IN ({om})", observations)
            store.connection.execute(f"DELETE FROM observations WHERE id IN ({om})", observations)
        if events:
            em = ",".join("?" for _ in events)
            store.connection.execute(f"DELETE FROM event_entities WHERE event_id IN ({em})", events)
            store.connection.execute(f"DELETE FROM event_participants WHERE event_id IN ({em})", events)
            store.connection.execute(f"DELETE FROM event_revisions WHERE event_id IN ({em})", events)
            store.connection.execute(f"DELETE FROM memory_vectors WHERE source_type = 'event' AND source_id IN ({em})", events)
            store.connection.execute(f"DELETE FROM events WHERE id IN ({em})", events)
        store.connection.execute(f"DELETE FROM memory_vectors WHERE source_type = 'asset' AND source_id IN ({marks})", ids)
        store.connection.execute(f"DELETE FROM assets WHERE id IN ({marks}) OR parent_asset_id = ?", ids + [args.asset_id])
    print(f"removed parent={args.asset_id} derived={len(derived)}")


if __name__ == "__main__":
    main()
