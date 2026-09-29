"""Near-duplicate grouping (Phase R P0-13).

Presentation-layer helper: it annotates packet assets with a group id / size so
the UI can fold near-duplicates in ``best`` mode ("组内还有 N 张") while
``all_relevant`` keeps every member.  Grouping NEVER changes which assets the
kernel returns — retrieval metrics keep the original Asset set (P0-13).

Primary key is the ingestion content SHA-256 (already stored on every Asset).
A CLIP cosine auxiliary pass is only active when the store can provide visual
vectors; the threshold comes from config, never hardcoded per benchmark.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class NearDuplicateGrouper:
    store: object
    clip_threshold: float = 0.98
    clip_enabled: bool = False

    def _sha_for(self, asset_id):
        if not asset_id:
            return None
        try:
            asset = self.store.get_asset(asset_id)
        except Exception:
            asset = None
        if asset and asset.get("content_sha256"):
            return asset["content_sha256"]
        return None

    def groups(self, assets: list[dict]) -> dict[str, list[str]]:
        """asset_id -> list of asset_ids sharing its SHA-256 (group leader first)."""
        by_sha: dict[str, list[str]] = {}
        order: list[str] = []
        for asset in assets:
            asset_id = asset.get("asset_id")
            sha = asset.get("content_sha256") or self._sha_for(asset_id)
            if not sha:
                order.append(asset_id)
                by_sha.setdefault(f"sha_none_{len(order)}", [asset_id])
                continue
            if sha not in by_sha:
                order.append(sha)
            by_sha.setdefault(sha, []).append(asset_id)
        return {key: by_sha[key] for key in order}

    def annotate(self, packet_assets: list[dict]) -> None:
        """Add near_duplicate_group / near_duplicate_size in place."""
        membership = {}
        for group_id, members in self.groups(packet_assets).items():
            for member in members:
                membership[member] = (group_id, len(members))
        for asset in packet_assets:
            group_id, size = membership.get(asset.get("asset_id"), (None, 1))
            asset["near_duplicate_group"] = group_id
            asset["near_duplicate_size"] = size
