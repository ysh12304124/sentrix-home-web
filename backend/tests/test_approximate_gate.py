"""Phase R8-5 — approximate evidence gate tests."""

import os
import unittest

from backend.evidence_retrieval import EvidencePacket
from backend.thin_agent import ThinAgentRuntime
from backend.query_contracts import Constraint, QuerySpec


def _item(asset_id, level, attributions=None):
    return {"asset_id": asset_id, "file_name": f"{asset_id}.jpg", "media_type": "image",
            "level": level, "score": 0.0, "evidence_ids": [asset_id],
            "observation_ids": [], "condition_results": {}, "attributions": attributions or []}


def _spec(anchored=False):
    constraints = []
    if anchored:
        constraints.append(Constraint("person", "明哥", "deterministic_hard", "confirmed_bridge"))
    return QuerySpec(query_id="q", scope_mode="single", scope_ids=["album1"], viewer_id="owner",
                     conversation_id="c", intent="answer", answer_target="general", constraints=constraints)


class ApproximateGateTests(unittest.TestCase):
    def _packet(self, assets):
        p = EvidencePacket("q", "album1", "general")
        p.assets = assets
        p.exact_results = [a for a in assets if a["level"] == "exact"]
        p.strong_results = [a for a in assets if a["level"] == "strong"]
        p.approximate_results = [a for a in assets if a["level"] == "approximate"]
        return p

    def test_weak_approximate_dropped(self):
        packet = self._packet([
            _item("a1", "approximate", [{"retriever": "visual_ann", "score": 0.05, "score_kind": "cosine_similarity"}]),
            _item("a2", "approximate", [{"retriever": "visual_ann", "score": 0.40, "score_kind": "cosine_similarity"}]),
        ])
        ThinAgentRuntime._gate_packet_approximate(packet, _spec())
        self.assertEqual([a["asset_id"] for a in packet.assets], ["a2"])
        self.assertGreaterEqual(packet.excluded_count, 1)

    def test_exact_and_strong_kept(self):
        packet = self._packet([
            _item("a1", "exact"), _item("a2", "strong"),
            _item("a3", "approximate", [{"retriever": "visual_ann", "score": 0.05, "score_kind": "cosine_similarity"}]),
        ])
        ThinAgentRuntime._gate_packet_approximate(packet, _spec())
        self.assertEqual({a["asset_id"] for a in packet.assets}, {"a1", "a2"})

    def test_anchored_query_uses_higher_threshold(self):
        packet = self._packet([
            _item("a1", "approximate", [{"retriever": "visual_ann", "score": 0.20, "score_kind": "cosine_similarity"}]),
        ])
        # anchored + no matched condition -> strict drop (no weak approximate)
        ThinAgentRuntime._gate_packet_approximate(packet, _spec(), anchored=True)
        self.assertEqual(packet.assets, [])

    def test_no_attribution_kept(self):
        packet = self._packet([_item("a1", "approximate")])
        ThinAgentRuntime._gate_packet_approximate(packet, _spec())
        self.assertEqual([a["asset_id"] for a in packet.assets], ["a1"])


if __name__ == "__main__":
    unittest.main()
