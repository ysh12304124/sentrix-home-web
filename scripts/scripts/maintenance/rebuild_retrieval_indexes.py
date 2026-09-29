#!/usr/bin/env python3
"""Rebuild the ``observation_search_terms`` derived table from canonical data.

Safe to run against a production database — only derived rows are touched.
Use ``--scope-id`` to limit the rebuild to a specific memory space.  A backup
should be taken via SQLite ``.backup`` before running against production.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from backend.db import MemoryStore
from backend.retrieval_indexes import RetrievalIndex


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("db_path", help="Path to sentrix.db")
    parser.add_argument("--scope-id", default=None, help="Restrict rebuild to this scope")
    parser.add_argument("--apply", action="store_true", help="Actually write to the database")
    args = parser.parse_args()
    if not args.apply:
        print("dry-run: pass --apply to actually rebuild derived rows")
        return
    store = MemoryStore(args.db_path)
    try:
        index = RetrievalIndex(store)
        count = index.rebuild_all(scope_id=args.scope_id)
        print(f"rebuilt observation_search_terms for {count} observation(s)")
    finally:
        store.close()


if __name__ == "__main__":
    main()
