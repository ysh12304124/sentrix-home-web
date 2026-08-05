"""Phase 7 advanced tool invariants."""

import unittest

from backend.advanced_memory_tools import (
    build_pattern, compare_memories, summarize_event, summarize_person, trace_timeline,
)
from backend.evidence_retrieval import EvidencePacket
from backend.query_contracts import Constraint, HARD, QueryAction, QuerySpec


class FakeGamma:
    def __init__(self, *_args, **_kwargs):
        pass


def _spec(scope="home", answer_target="person", entity_ids=None, constraints=None):
    return QuerySpec(
        query_id="q1", scope_mode="single", scope_ids=[scope], viewer_id="owner",
        conversation_id="c1", intent="answer", answer_target=answer_target,
        constraints=constraints or [], entity_ids=list(entity_ids or []),
        actions=[QueryAction("summarize_person", "person")],
    )


def _packet(assets=None):
    return EvidencePacket("q1", "home", "person", assets=assets or [])


class AdvancedToolsInvariantTests(unittest.TestCase):
    def test_summarize_person_refuses_without_confirmed_entity(self):
        result = summarize_person(_spec(entity_ids=[]), _packet(), gamma=None)
        self.assertIn("已确认", result["answer"])
        self.assertFalse(result["evidence_ids"])
        self.assertEqual(result["tool_trace"][0]["status"], "requires_anchor")

    def test_trace_timeline_never_crosses_hard_time_bound(self):
        constraints = [Constraint("time", "2024-05", HARD, "asset_metadata")]
        spec = _spec(constraints=constraints)
        packet = _packet(assets=[
            {"asset_id": "in", "file_name": "in.jpg", "captured_at": "2024-05-12T10:00:00", "evidence_ids": ["in"]},
            {"asset_id": "out", "file_name": "out.jpg", "captured_at": "2024-07-12T10:00:00", "evidence_ids": ["out"]},
        ])
        result = trace_timeline(spec, packet, gamma=None)
        asset_ids = [item.get("evidence_ids") for item in result["statements"]]
        self.assertNotIn(["out"], asset_ids)

    def test_compare_memories_reports_sides_independently(self):
        spec_a = _spec(scope="home")
        spec_b = _spec(scope="other")
        packet_a = _packet(assets=[{"asset_id": "a1", "evidence_ids": ["a1"]}])
        packet_b = _packet(assets=[{"asset_id": "b1", "evidence_ids": ["b1"]}, {"asset_id": "b2", "evidence_ids": ["b2"]}])
        result = compare_memories(spec_a, packet_a, spec_b, packet_b, gamma=None)
        self.assertIn("集合 A", result["answer"])
        self.assertIn("集合 B", result["answer"])
        # Merged evidence IDs cover both sides but statements stay side-scoped.
        self.assertIn("a1", result["evidence_ids"])
        self.assertIn("b1", result["evidence_ids"])

    def test_build_pattern_needs_at_least_two_events(self):
        # A single-observation packet cannot become a pattern.
        result = build_pattern(_spec(), _packet(assets=[
            {"asset_id": "a1", "evidence_ids": ["obs-1"]}]), gamma=None, min_events=2)
        self.assertEqual(result["tool_trace"][0]["status"], "requires_more_events")

    def test_build_pattern_stays_soft_even_with_enough_events(self):
        packet = _packet(assets=[
            {"asset_id": "a1", "evidence_ids": ["obs-1"]},
            {"asset_id": "a2", "evidence_ids": ["obs-2"]},
            {"asset_id": "a3", "evidence_ids": ["obs-3"]},
        ])
        result = build_pattern(_spec(), packet, gamma=None, min_events=2)
        # Never promoted to a confirmed fact — status stays soft.
        self.assertEqual(result["tool_trace"][0]["status"], "soft_pattern")
        self.assertIn("柔性", result["answer"])

    def test_summarize_event_returns_gap_message_when_packet_empty(self):
        result = summarize_event(_spec(answer_target="event"), _packet(), gamma=None)
        self.assertEqual(result["tool_trace"][0]["status"], "no_evidence")


if __name__ == "__main__":
    unittest.main()
