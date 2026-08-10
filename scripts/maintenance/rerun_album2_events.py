"""Rerun event clustering for a single scope without re-running vision models.

Deletes the scope's events and re-merges existing (already enriched)
observations so the new GPS-distance clustering and visual-place event
placement take effect. Back up the DB before running.
"""
import json
import os
import sys

from backend.db import MemoryStore
from backend.pipeline import IngestionPipeline

SCOPE_ID = os.getenv("RERUN_SCOPE", "album_29199af13bff")
DB_PATH = os.getenv("SENTRIX_DB_PATH", "data/sentrix.db")


def main():
    store = MemoryStore(DB_PATH)
    pipeline = IngestionPipeline(store)
    observations = store._rows(
        "SELECT o.id, o.asset_id, o.captured_at, o.place, o.activity, o.event_type, o.canonical_json "
        "FROM observations o JOIN assets a ON a.id = o.asset_id WHERE a.scope_id = ? ORDER BY o.captured_at",
        (SCOPE_ID,),
    )
    if not observations:
        print(f"No observations for scope {SCOPE_ID}")
        return
    # Collect events of this scope and delete their links, then the events.
    event_ids = [row["id"] for row in store._rows("SELECT id FROM events WHERE scope_id = ?", (SCOPE_ID,))]
    store.connection.execute("PRAGMA foreign_keys = OFF")
    try:
        for event_id in event_ids:
            store.connection.execute("DELETE FROM event_observations WHERE event_id = ?", (event_id,))
            store.connection.execute("DELETE FROM event_participants WHERE event_id = ?", (event_id,))
            store.connection.execute("DELETE FROM event_entities WHERE event_id = ?", (event_id,))
            store.connection.execute("DELETE FROM person_event_memory WHERE event_id = ?", (event_id,))
            store.connection.execute("DELETE FROM memory_vectors WHERE source_type = 'event' AND source_id = ?", (event_id,))
            store.connection.execute("DELETE FROM facts WHERE evidence_ids_json LIKE ?", (f"%{event_id}%",))
            store.connection.execute("DELETE FROM events WHERE id = ?", (event_id,))
        store.connection.commit()
    finally:
        store.connection.execute("PRAGMA foreign_keys = ON")
    print(f"Cleared {len(event_ids)} old events")

    merged = 0
    summarize = os.getenv("RERUN_SUMMARIZE", "1") == "1"
    for row in observations:
        observation = store.get_observation(row["id"])
        if not observation:
            continue
        event = store.merge_observation_into_event(observation)
        if event:
            if summarize:
                pipeline.summarize_event(event["id"])
            merged += 1
    print(f"Re-merged {merged}/{len(observations)} observations into events")
    store.connection.commit()
    print("DONE")


if __name__ == "__main__":
    main()
