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


class HnswlibIndex:
    """Production ANN backend — hnswlib cosine index with native delete.

    Phase 3.5.1 selection: hnswlib chosen over FAISS HNSW32 because it has
    ``mark_deleted`` — matches the family-memory delete semantics (user
    removes photos, applies corrections, merges entities).

    hnswlib requires integer labels; a string-id map preserves the AnnIndex
    contract while translating our IDs.  scope filtering stays code-side:
    the index returns candidates, ``EvidenceRetrievalKernel`` re-filters.
    """

    _DEFAULT_MAX_ELEMENTS = 200_000
    _DEFAULT_EF_CONSTRUCTION = 200
    _DEFAULT_M = 16
    _DEFAULT_EF_SEARCH = 50

    def __init__(self, *, dim=None, max_elements=None, ef_construction=None, M=None, ef_search=None, space="cosine"):
        import hnswlib

        self._hnswlib = hnswlib
        self._space = space
        self._dim = dim
        self._max_elements = int(max_elements or self._DEFAULT_MAX_ELEMENTS)
        self._ef_construction = int(ef_construction or self._DEFAULT_EF_CONSTRUCTION)
        self._M = int(M or self._DEFAULT_M)
        self._ef_search = int(ef_search or self._DEFAULT_EF_SEARCH)
        self._index = None
        self._id_to_label = {}
        self._label_to_id = {}
        self._label_to_meta = {}
        self._next_label = 0
        self._deleted_labels = set()

    def _ensure_index(self, dim):
        if self._index is None:
            self._dim = dim
            self._index = self._hnswlib.Index(space=self._space, dim=dim)
            self._index.init_index(max_elements=self._max_elements,
                                    ef_construction=self._ef_construction, M=self._M)
            self._index.set_ef(self._ef_search)

    def _resize_if_needed(self, additional):
        current_max = self._index.get_max_elements() if self._index is not None else self._max_elements
        needed = self._next_label + additional
        if needed > current_max:
            new_size = max(needed, current_max * 2)
            self._index.resize_index(new_size)
            self._max_elements = new_size

    def build(self, vectors):
        self._index = None
        self._id_to_label = {}
        self._label_to_id = {}
        self._label_to_meta = {}
        self._next_label = 0
        self._deleted_labels = set()
        self.add(vectors)

    def add(self, vectors):
        rows = list(vectors)
        if not rows:
            return
        import numpy as np
        first_dim = len(rows[0][1])
        self._ensure_index(first_dim)
        self._resize_if_needed(len(rows))
        labels = []
        vecs = []
        for our_id, vector, metadata in rows:
            our_id = str(our_id)
            if our_id in self._id_to_label:
                # Overwrite an existing id — mark old label deleted first.
                self.remove([our_id])
            label = self._next_label
            self._next_label += 1
            self._id_to_label[our_id] = label
            self._label_to_id[label] = our_id
            self._label_to_meta[label] = dict(metadata or {})
            labels.append(label)
            vecs.append(vector)
        matrix = np.asarray(vecs, dtype="float32")
        self._index.add_items(matrix, labels)

    def remove(self, ids):
        for our_id in ids:
            label = self._id_to_label.pop(str(our_id), None)
            if label is None:
                continue
            self._label_to_id.pop(label, None)
            self._label_to_meta.pop(label, None)
            try:
                self._index.mark_deleted(label)
            except RuntimeError:
                # Already deleted or invalid label — safe to swallow.
                pass
            self._deleted_labels.add(label)

    def search(self, query, k, scope_id=None):
        if self._index is None or self._next_label == 0:
            return []
        import numpy as np
        current = self._next_label - len(self._deleted_labels)
        if current <= 0:
            return []
        vector = np.asarray([query], dtype="float32")
        # Over-fetch when scope filtering is active — ANN returns candidates,
        # the kernel re-filters.  Cap at the total number of live vectors.
        effective_k = min(current, k if scope_id is None else min(current, k * 4))
        labels_batch, distances_batch = self._index.knn_query(vector, k=effective_k)
        results = []
        for label, distance in zip(labels_batch[0], distances_batch[0]):
            label = int(label)
            our_id = self._label_to_id.get(label)
            if our_id is None:
                continue
            metadata = self._label_to_meta.get(label, {})
            if scope_id is not None and metadata.get("scope_id") != scope_id:
                continue
            # hnswlib cosine returns 1 - cos(sim); convert back for callers.
            similarity = 1.0 - float(distance)
            results.append((our_id, similarity, metadata))
            if len(results) >= k:
                break
        return results

    def save(self, path):
        import json
        if self._index is None:
            return
        self._index.save_index(f"{path}.hnsw")
        sidecar = {
            "space": self._space,
            "dim": self._dim,
            "max_elements": self._max_elements,
            "ef_construction": self._ef_construction,
            "M": self._M,
            "ef_search": self._ef_search,
            "next_label": self._next_label,
            "id_to_label": self._id_to_label,
            "label_to_meta": {str(label): meta for label, meta in self._label_to_meta.items()},
            "deleted_labels": list(self._deleted_labels),
        }
        with open(f"{path}.meta.json", "w", encoding="utf-8") as handle:
            json.dump(sidecar, handle, ensure_ascii=False)

    def load(self, path):
        import json
        with open(f"{path}.meta.json", "r", encoding="utf-8") as handle:
            sidecar = json.load(handle)
        self._space = sidecar["space"]
        self._dim = sidecar["dim"]
        self._max_elements = int(sidecar["max_elements"])
        self._ef_construction = int(sidecar["ef_construction"])
        self._M = int(sidecar["M"])
        self._ef_search = int(sidecar["ef_search"])
        self._next_label = int(sidecar["next_label"])
        self._deleted_labels = set(int(item) for item in sidecar["deleted_labels"])
        self._id_to_label = {str(k): int(v) for k, v in sidecar["id_to_label"].items()}
        self._label_to_id = {v: k for k, v in self._id_to_label.items()}
        self._label_to_meta = {int(label): meta for label, meta in sidecar["label_to_meta"].items()}
        self._index = self._hnswlib.Index(space=self._space, dim=self._dim)
        self._index.load_index(f"{path}.hnsw", max_elements=self._max_elements)
        self._index.set_ef(self._ef_search)


def create_index(kind: str = "sqlite_baseline", **options) -> AnnIndex:
    """Factory selecting an :class:`AnnIndex` backend.

    ``sqlite_baseline`` is always available and used as the correctness
    ground truth.  ``hnswlib`` was selected in Phase 3.5.1 for production use
    and requires the ``hnswlib`` package to be importable.
    """
    if kind == "sqlite_baseline":
        return SqliteBaselineIndex()
    if kind == "hnswlib":
        return HnswlibIndex(**options)
    raise ValueError(f"unknown ANN backend: {kind}")
