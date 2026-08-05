"""Evidence bundle + open-world Evidence condition invariants (Phase 2R-1 + 2R-6)."""

import unittest

from backend.evidence_retrieval import (
    EvidencePacket,
    EvidenceRetrievalKernel,
    build_verifier_evidence_bundle,
    _contains,
)
from backend.query_contracts import Constraint, HARD, SEMANTIC, QuerySpec


class ContainsSemanticsTests(unittest.TestCase):
    """Phase R P0-6: containment is full-substring only, never tokenized."""

    def test_full_substring_matches(self):
        self.assertTrue(_contains("卧室睡衣自拍", "睡衣"))

    def test_partial_word_tokens_do_not_match(self):
        # The album1-01 failure mode: a whole visual description must not match
        # a caption that shares only some characters.
        self.assertFalse(_contains("卧室睡衣自拍", "浅黄色拼接毛绒睡衣自拍"))
        self.assertFalse(_contains("厨房", "色"))

    def test_empty_needle_never_matches(self):
        self.assertFalse(_contains("anything", ""))

    def test_case_insensitive(self):
        self.assertTrue(_contains("NEW YEAR", "new"))


class MatchedSourceWhitelistTests(unittest.TestCase):
    """Phase R P1-2: matched only from direct-proof source types."""

    def _kernel(self):
        return EvidenceRetrievalKernel(InMemoryStore([], []))

    def test_generic_observation_source_downgraded_to_possible(self):
        kernel = self._kernel()
        asset = {"id": "asset-1", "scope_id": "home", "file_name": "a.jpg",
                 "media_type": "image", "captured_at": "2024-05-12T10:00:00"}
        observation = {"id": "obs-1", "asset_id": "asset-1", "scope_id": "home",
                       "captured_at": "2024-05-12T10:00:00", "caption": "x", "confidence": 0.9}
        spec = QuerySpec(query_id="q", scope_mode="single", scope_ids=["home"], viewer_id="owner",
                         conversation_id="c", intent="answer", answer_target="general",
                         constraints=[Constraint("clothing", "X", SEMANTIC, "direct_or_possible")])
        # Monkeypatch a hypothetical evaluator that (wrongly) claims matched
        # with the weak "observation" source — the whitelist must downgrade it.
        def fake_condition(asset, observation, constraint):
            return ("matched", "observation", observation.get("id"), 0.9)
        kernel._condition = fake_condition
        result = kernel._evaluate(asset, observation, spec)
        status = result["item"]["condition_results"]["clothing:X"]["status"]
        self.assertEqual(status, "possible", "weak source must be downgraded to possible")

    def test_direct_proof_source_stays_matched(self):
        kernel = self._kernel()
        asset = {"id": "asset-1", "scope_id": "home", "file_name": "a.jpg",
                 "media_type": "image", "captured_at": "2024-05-12T10:00:00"}
        observation = {"id": "obs-1", "asset_id": "asset-1", "scope_id": "home",
                       "captured_at": "2024-05-12T10:00:00", "caption": "x", "confidence": 0.9}
        spec = QuerySpec(query_id="q", scope_mode="single", scope_ids=["home"], viewer_id="owner",
                         conversation_id="c", intent="answer", answer_target="general",
                         constraints=[Constraint("media", "image", HARD, "asset_metadata")])
        result = kernel._evaluate(asset, observation, spec)
        status = result["item"]["condition_results"]["media:image"]["status"]
        self.assertEqual(status, "matched")
        self.assertEqual(result["item"]["condition_results"]["media:image"]["source_type"], "asset_metadata")

    def test_single_value_place_exact_is_matched(self):
        assets = [{"id": "asset-1", "scope_id": "home", "file_name": "a.jpg",
                   "media_type": "image", "captured_at": "2024-05-12T10:00:00"}]
        observations = [{
            "id": "obs-1", "asset_id": "asset-1", "scope_id": "home",
            "captured_at": "2024-05-12T10:00:00", "caption": "x",
            "place": "厨房", "activity": None, "people": [], "clothing": [], "objects": [],
            "confidence": 0.9,
        }]
        spec = QuerySpec(query_id="q", scope_mode="single", scope_ids=["home"], viewer_id="owner",
                         conversation_id="c", intent="answer", answer_target="place",
                         constraints=[Constraint("place", "厨房", SEMANTIC, "direct_or_possible")])
        packet = EvidenceRetrievalKernel(InMemoryStore(assets, observations)).retrieve(spec)
        self.assertEqual(packet.assets[0]["condition_results"]["place:厨房"]["status"], "matched")
        self.assertEqual(packet.assets[0]["condition_results"]["place:厨房"]["source_type"], "observation_field_exact")


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
