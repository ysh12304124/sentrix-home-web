"""Evidence bundle + open-world Evidence condition invariants (Phase 2R-1 + 2R-6)."""

import unittest

from backend.evidence_retrieval import (
    EvidencePacket,
    EvidenceRetrievalKernel,
    build_verifier_evidence_bundle,
)
from backend.query_contracts import Constraint, HARD, SEMANTIC, QuerySpec


class EvidenceBundleTests(unittest.TestCase):
    def test_verifier_bundle_separates_canonical_evidence_from_derived_context(self):
        packet = EvidencePacket(
            "q1", "home", "activity",
            assets=[{
                "asset_id": "asset-1", "file_name": "a.jpg", "media_type": "image",
                "observation_ids": ["obs-1"], "evidence_ids": ["asset-1", "obs-1"],
                "condition_results": {}, "level": "exact", "score": 1.0,
                "observation_fields": {"place": "厨房", "activity": "拿碗"},
            }],
        )
        bundle = build_verifier_evidence_bundle(packet, "claim-1")
        self.assertEqual(bundle["claim_id"], "claim-1")
        self.assertTrue(bundle["canonical_evidence"])
        self.assertTrue(all(item["is_canonical"] for item in bundle["canonical_evidence"]))
        self.assertEqual(bundle["derived_context"], [])


class InMemoryStore:
    """Minimum surface used by ``EvidenceRetrievalKernel.retrieve``."""

    def __init__(self, assets, observations):
        self._assets = assets
        self._observations = observations

    def list_assets(self, scope_id=None):
        return [asset for asset in self._assets if not scope_id or asset["scope_id"] == scope_id]

    def list_observations(self, scope_id=None):
        return [obs for obs in self._observations if not scope_id or obs["scope_id"] == scope_id]


class OpenWorldConditionTests(unittest.TestCase):
    """List-shaped fields must return ``unknown`` on miss, not ``contradicted``.

    Aligned with supplementary plan §10.  Only same-subject contrary evidence
    can produce a real ``contradicted`` status.
    """

    def _spec(self, dimension, value):
        return QuerySpec(
            query_id="q", scope_mode="single", scope_ids=["home"], viewer_id="owner",
            conversation_id="c", intent="find_assets", answer_target="clothing",
            constraints=[Constraint(dimension, value, SEMANTIC, "direct_or_possible", source_text=value)],
        )

    def test_two_people_clothing_miss_is_unknown_not_contradicted(self):
        assets = [{"id": "asset-1", "scope_id": "home", "file_name": "a.jpg",
                   "media_type": "image", "captured_at": "2024-05-12T10:00:00"}]
        observations = [{
            "id": "obs-1", "asset_id": "asset-1", "scope_id": "home",
            "captured_at": "2024-05-12T10:00:00", "caption": "两个人",
            "place": "客厅", "activity": "聊天", "people": ["明哥", "妈妈"],
            "clothing": ["红色外套", "蓝色外套"], "objects": [], "confidence": 0.9,
        }]
        kernel = EvidenceRetrievalKernel(InMemoryStore(assets, observations))
        packet = kernel.retrieve(self._spec("clothing", "绿色"))
        for asset in packet.assets:
            status = asset["condition_results"].get("clothing:绿色", {}).get("status")
            self.assertNotEqual(status, "contradicted",
                                 "multi-person clothing miss must not be contradicted")
            self.assertEqual(status, "unknown", f"expected unknown for two-person miss, got {status}")

    def test_missing_subject_binding_yields_unknown(self):
        assets = [{"id": "asset-1", "scope_id": "home", "file_name": "a.jpg",
                   "media_type": "image", "captured_at": "2024-05-12T10:00:00"}]
        observations = [{
            "id": "obs-1", "asset_id": "asset-1", "scope_id": "home",
            "captured_at": "2024-05-12T10:00:00", "caption": "厨房场景",
            "place": "厨房", "activity": "拿碗", "people": ["明哥"],
            "clothing": ["红色外套"], "objects": [], "confidence": 0.9,
            # subject_clothing binding is not provided by formation
        }]
        kernel = EvidenceRetrievalKernel(InMemoryStore(assets, observations))
        packet = kernel.retrieve(self._spec("clothing", "蓝色"))
        for asset in packet.assets:
            status = asset["condition_results"].get("clothing:蓝色", {}).get("status")
            self.assertEqual(status, "unknown",
                             "without subject binding the clothing question must stay unknown")

    def test_contradicted_requires_same_subject_binding(self):
        assets = [{"id": "asset-1", "scope_id": "home", "file_name": "a.jpg",
                   "media_type": "image", "captured_at": "2024-05-12T10:00:00"}]
        observations = [{
            "id": "obs-1", "asset_id": "asset-1", "scope_id": "home",
            "captured_at": "2024-05-12T10:00:00", "caption": "明哥穿红外套",
            "place": "厨房", "activity": "站立", "people": ["明哥"],
            "clothing": ["红色外套"], "objects": [], "confidence": 0.95,
            "subject_clothing": [{"subject_id": "明哥", "value": "红色外套"}],
        }]
        kernel = EvidenceRetrievalKernel(InMemoryStore(assets, observations))
        packet = kernel.retrieve(self._spec("clothing", "蓝色"))
        # With subject binding proving 明哥 wore 红色，查询"蓝色" 应被 contradicted
        # 从结果里剔除（excluded_count 增加），或以 unknown 保留在 packet.assets。
        if packet.assets:
            status = packet.assets[0]["condition_results"].get("clothing:蓝色", {}).get("status")
            self.assertIn(status, {"contradicted", "unknown"})
        else:
            self.assertGreaterEqual(packet.excluded_count, 1,
                                     "asset must be excluded when subject binding contradicts the constraint")


if __name__ == "__main__":
    unittest.main()
