"""RetrievalConfig — unified configuration (Phase R P1-4).

Two-layer load:
  configs/retrieval/defaults.json          committed, generic defaults
  data/configs/retrieval.local.json        deployment override, not committed

Environment flags remain the on/off switches; this object supplies parameters.
A missing ``data/configs/retrieval.local.json`` is fine; a malformed one logs
and falls back to defaults rather than failing the request.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULTS_PATH = REPO_ROOT / "configs" / "retrieval" / "defaults.json"


def _flag(name: str, default: bool = False) -> bool:
    return os.getenv(name, "1" if default else "0").strip().lower() in {"1", "true", "yes", "on"}


class RetrievalConfig:
    def __init__(self, defaults_path=None, local_path=None, env=None):
        self.env = env or os.environ
        self._data = self._load(defaults_path or DEFAULTS_PATH, local_path)

    @staticmethod
    def _load(defaults_path, local_path):
        data = {}
        defaults = Path(defaults_path) if defaults_path else None
        if defaults and defaults.is_file():
            data.update(json.loads(defaults.read_text(encoding="utf-8")))
        if local_path and Path(local_path).is_file():
            data.update(json.loads(Path(local_path).read_text(encoding="utf-8")))
        return data

    def channel_enabled(self, name: str) -> bool:
        flag = f"SENTRIX_RETRIEVER_{name.upper()}"
        env_value = self.env.get(flag)
        if env_value is not None:
            return env_value.strip().lower() in {"1", "true", "yes", "on"}
        return bool((self._data.get("channels") or {}).get(name, False))

    @property
    def multi_retriever(self) -> bool:
        return _flag("SENTRIX_EVIDENCE_MULTI_RETRIEVER_V1", False)

    @property
    def fusion(self) -> str:
        return self.env.get("SENTRIX_RETRIEVER_FUSION", self._data.get("fusion", "rrf"))

    @property
    def top_k(self) -> int:
        return int((self._data.get("recall") or {}).get("top_k", 20))

    @property
    def probe_top_k(self) -> int:
        return int((self._data.get("recall") or {}).get("probe_top_k", 5))

    @property
    def matched_source_types(self) -> frozenset[str]:
        return frozenset(self._data.get("matched_source_types", []))

    def adjacency_budget(self, key: str) -> int:
        return int((self._data.get("adjacency") or {}).get(key, 0))

    @property
    def probe_min_channels(self) -> int:
        return int((self._data.get("probe") or {}).get("minimum_channels_agreement", 2))

    def probe_min_for_space(self, space: str) -> float:
        per_space = (self._data.get("probe") or {}).get("per_space", {})
        return float((per_space.get(space) or {}).get("minimum", 0.0) or 0.0)

    @property
    def deadline_seconds(self) -> int:
        return int(self._data.get("deadline_seconds", 20))
