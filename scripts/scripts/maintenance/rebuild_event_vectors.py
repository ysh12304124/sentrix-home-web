"""Rebuild episodic event vectors for a scope's events (title/summary -> vector)."""
import os

from backend.db import MemoryStore
from backend.model_clients import ClipAdapter

DB_PATH = os.getenv("SENTRIX_DB_PATH", "data/sentrix.db")
SCOPE_ID = os.getenv("RERUN_SCOPE", "album_29199af13bff")


def main():
    store = MemoryStore(DB_PATH)
    clip = ClipAdapter()
    events = store._rows("SELECT id, title, event_type, activity, summary FROM events WHERE scope_id = ?", (SCOPE_ID,))
    updated = 0
    for event in events:
        text = " ".join(str(event.get(key) or "") for key in ("title", "event_type", "activity", "summary"))
        if not text.strip():
            continue
        vector = clip.embed_text(text)
        if not vector:
            print(f"skip empty vector for event {event['id'][-8:]}")
            continue
        store.upsert_vector(
            "episodic", "event", event["id"], vector, clip.model_name,
            {"scope_id": SCOPE_ID, "event_summary": True, "rebuild": True},
        )
        updated += 1
    store.connection.commit()
    print(f"Rebuilt vectors for {updated}/{len(events)} events in scope {SCOPE_ID}")
    print("DONE")


if __name__ == "__main__":
    main()
