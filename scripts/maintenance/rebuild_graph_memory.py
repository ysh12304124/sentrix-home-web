#!/usr/bin/env python3
"""Inspect or rebuild the derived MAGMA graph memory index.

The command is intentionally opt-in: without ``--apply`` it never rebuilds.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from backend.graph_memory import GraphMemoryService  # noqa: E402


def parse_args():
    data_dir = Path(os.getenv("SENTRIX_DATA_DIR", Path(__file__).resolve().parents[2] / "data"))
    default_db = os.getenv("SENTRIX_DB_PATH", str(data_dir / "sentrix.db"))
    parser = argparse.ArgumentParser(description="Inspect or rebuild derived graph memory tables.")
    parser.add_argument("--db", default=default_db, help="Path to the existing sentrix.db")
    parser.add_argument("--scope-id", default="home-default")
    parser.add_argument("--stats", action="store_true", help="Print graph stats without rebuilding")
    parser.add_argument("--apply", action="store_true", help="Rebuild the selected scope")
    return parser.parse_args()


def main():
    args = parse_args()
    if not args.stats and not args.apply:
        raise SystemExit("No action requested. Use --stats to inspect or --apply to rebuild.")
    service = GraphMemoryService(args.db)
    try:
        if args.apply:
            result = service.rebuild(args.scope_id)
        else:
            result = service.stats(args.scope_id)
        print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    finally:
        service.close()


if __name__ == "__main__":
    main()
