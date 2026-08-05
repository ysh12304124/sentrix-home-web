"""Phase 3.5 ANN adapter interface.

Real backend selection is deferred until the 153 benchmark chooses a library.
Until then, :class:`SqliteBaselineIndex` implements the interface with the
current row-by-row cosine scan — correctness is guaranteed and Recall@k is
1.0.  Once a library is selected, add a sibling implementation that satisfies
the same protocol and switch via ``SENTRIX_ANN_INDEX_V1``.
"""

from __future__ import annotations

from typing import Protocol
import math


class AnnIndex(Protocol):
    """Minimum surface every ANN backend must satisfy."""

    def build(self, vectors: list[tuple[str, list[float], dict]]) -> None: ...
    def add(self, vectors: list[tuple[str, list[float], dict]]) -> None: ...
    def remove(self, ids: list[str]) -> None: ...
    def search(self, query: list[float], k: int, scope_id: str | None = None) -> list[tuple[str, float, dict]]: ...
    def save(self, path: str) -> None: ...
    def load(self, path: str) -> None: ...


class SqliteBaselineIndex:
    """Correctness-first in-memory baseline used until the ANN library is chosen.

    Not intended for the 100k-vector scale target — its purpose is to let the
    rest of the pipeline exercise the AnnIndex protocol without depending on
    an external library.  Recall@k is exactly the ground truth.
    """

    def __init__(self):
        self._vectors: list[tuple[str, list[float], dict]] = []

    @staticmethod
    def _cosine(a, b):
        if not a or not b:
            return 0.0
        norm_a = math.sqrt(sum(x * x for x in a)) or 1.0
        norm_b = math.sqrt(sum(x * x for x in b)) or 1.0
        return sum(x * y for x, y in zip(a, b)) / (norm_a * norm_b)

    def build(self, vectors):
        self._vectors = [(str(item[0]), list(item[1]), dict(item[2])) for item in vectors]

    def add(self, vectors):
        for item in vectors:
            self._vectors.append((str(item[0]), list(item[1]), dict(item[2])))

    def remove(self, ids):
        target = set(str(item) for item in ids)
        self._vectors = [row for row in self._vectors if row[0] not in target]

    def search(self, query, k, scope_id=None):
        rows = self._vectors
        if scope_id is not None:
            rows = [row for row in rows if (row[2].get("scope_id") or "") == scope_id]
        scored = [(row[0], self._cosine(query, row[1]), row[2]) for row in rows]
        scored.sort(key=lambda item: item[1], reverse=True)
        return scored[:k]

    def save(self, path):
        import json
        with open(path, "w", encoding="utf-8") as handle:
            json.dump([{"id": row[0], "vector": row[1], "metadata": row[2]} for row in self._vectors], handle)

    def load(self, path):
        import json
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        self._vectors = [(str(item["id"]), list(item["vector"]), dict(item.get("metadata") or {})) for item in data]


def create_index(kind: str = "sqlite_baseline") -> AnnIndex:
    """Factory — Phase 3.5.2 default only supports the baseline.

    Once the 153 benchmark picks a library, add another branch here and gate
    by ``SENTRIX_ANN_INDEX_V1``.
    """
    if kind == "sqlite_baseline":
        return SqliteBaselineIndex()
    raise ValueError(f"unknown ANN backend: {kind}")
