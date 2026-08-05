import unittest

from backend.memory_gate import MemoryGate
from backend.query_contracts import build_query_spec, sanitize_query_parse
from backend.evidence_retrieval import EvidenceRetrievalKernel
from backend.answer_composer import compose_answer, validate_statement_plan


class FakeStore:
    def __init__(self):
        self.assets = [
            {"id": "asset-may", "scope_id": "home", "file_name": "may.jpg", "media_type": "image", "captured_at": "2024-05-12T10:00:00", "captured_location": "30.1,120.1"},
            {"id": "asset-july", "scope_id": "home", "file_name": "july.jpg", "media_type": "image", "captured_at": "2024-07-12T10:00:00", "captured_location": "30.1,120.1"},
            {"id": "asset-video", "scope_id": "home", "file_name": "may.mp4", "media_type": "video", "captured_at": "2024-05-12T10:00:00", "captured_location": "30.1,120.1"},
            {"id": "asset-other", "scope_id": "other", "file_name": "other.jpg", "media_type": "image", "captured_at": "2024-05-12T10:00:00", "captured_location": "30.1,120.1"},
        ]
        self.observations = [
            {"id": "obs-may", "asset_id": "asset-may", "scope_id": "home", "place": "厨房", "activity": "拿碗", "caption": "明哥在厨房拿碗", "people": ["明哥"], "clothing": ["红色外套"], "objects": ["碗"], "confidence": 0.9, "revision": 1},
            {"id": "obs-july", "asset_id": "asset-july", "scope_id": "home", "place": "厨房", "activity": "做晚饭", "caption": "明哥在厨房做晚饭", "people": ["明哥"], "clothing": ["蓝色外套"], "objects": [], "confidence": 0.9, "revision": 1},
            {"id": "obs-video", "asset_id": "asset-video", "scope_id": "home", "place": "厨房", "activity": "做晚饭", "caption": "厨房视频", "people": ["明哥"], "clothing": [], "objects": [], "confidence": 0.9, "revision": 1},
            {"id": "obs-other", "asset_id": "asset-other", "scope_id": "other", "place": "贵阳夜晚步行街", "activity": "散步", "caption": "贵阳夜晚步行街", "people": [], "clothing": [], "objects": [], "confidence": 0.9, "revision": 1},
        ]

    def list_assets(self, **kwargs):
        return [asset for asset in self.assets if not kwargs.get("scope_id") or asset["scope_id"] == kwargs["scope_id"]]

    def list_observations(self, **kwargs):
        return [obs for obs in self.observations if not kwargs.get("scope_id") or obs["scope_id"] == kwargs["scope_id"]]

    def list_confirmed_entities(self, scope_id=None):
        return [{"id": "entity-ming", "canonical_name": "明哥", "status": "confirmed", "scope_id": scope_id or "home"}]


class ThinAgentContractTests(unittest.TestCase):
    def test_query_parser_drops_model_identity_and_makes_date_hard(self):
        parsed = sanitize_query_parse(
            {
                "intent": "find_assets",
                "answer_target": "activity",
                "entity_names": ["明哥"],
                "time_expression": "2024 年 5 月",
                "scope_id": "attacker-scope",
                "viewer_id": "attacker-viewer",
                "entity_ids": ["attacker-entity"],
                "semantic_conditions": [{"dimension": "place", "value": "厨房"}],
            },
            message="2024 年 5 月明哥在厨房的照片",
        )
        spec = build_query_spec(parsed, scope_id="home", viewer_id="owner", conversation_id="c1", entity_resolver=lambda name: "entity-ming")
        self.assertEqual(spec.scope_ids, ["home"])
        self.assertEqual(spec.viewer_id, "owner")
        self.assertEqual(spec.entity_ids, ["entity-ming"])
        self.assertEqual([c.strictness for c in spec.constraints if c.dimension == "time"], ["deterministic_hard"])
        self.assertNotIn("attacker", repr(spec))

    def test_empty_scope_is_explicit_all_authorized_mode(self):
        parsed = sanitize_query_parse({"intent": "find_assets"}, message="所有照片")
        spec = build_query_spec(parsed, scope_id="", viewer_id="owner", conversation_id="c1")
        self.assertEqual(spec.scope_mode, "all_authorized")
        self.assertEqual(spec.scope_ids, [])

    def test_gate_fast_path_writing_prompt_returns_none_without_parser(self):
        decision = MemoryGate().fast_path("帮我写一段生日祝福")
        self.assertIsNotNone(decision)
        self.assertEqual(decision.mode, "none")
        self.assertEqual(decision.concrete_memory_reads, 0)
        self.assertEqual(decision.query_parse_calls, 0)

    def test_gate_uses_contextual_mode_from_draft(self):
        from backend.query_contracts import QueryParseDraft
        draft = QueryParseDraft(mode="contextual", intent="answer", answer_target="general")
        decision = MemoryGate().classify("今天很累，突然有点想小黑", draft=draft)
        self.assertEqual(decision.mode, "contextual")
        self.assertEqual(decision.concrete_memory_reads, 0)
        self.assertEqual(decision.core_memory_reads, 1)

    def test_gate_routes_person_introduction_via_draft(self):
        from backend.query_contracts import QueryAction, QueryParseDraft
        draft = QueryParseDraft(mode="evidence", intent="answer", answer_target="person",
                                 actions=[QueryAction("answer_question", "person")])
        decision = MemoryGate().classify("介绍一下明哥", draft=draft)
        self.assertEqual(decision.mode, "evidence")
        self.assertEqual(decision.answer_target, "person")

    def test_retrieval_enforces_scope_time_media_and_semantic_contradiction(self):
        parsed = sanitize_query_parse(
            {"intent": "find_assets", "answer_target": "activity", "time_expression": "2024 年 5 月", "media_expressions": ["照片"], "semantic_conditions": [{"dimension": "activity", "value": "做晚饭"}]},
            message="2024 年 5 月厨房里做晚饭的照片",
        )
        spec = build_query_spec(parsed, scope_id="home", viewer_id="owner", conversation_id="c1")
        packet = EvidenceRetrievalKernel(FakeStore()).retrieve(spec)
        ids = {item["asset_id"] for item in packet.assets}
        self.assertNotIn("asset-july", ids)
        self.assertNotIn("asset-video", ids)
        self.assertNotIn("asset-other", ids)
        self.assertIn("asset-may", ids)
        result = next(item for item in packet.assets if item["asset_id"] == "asset-may")
        self.assertEqual(result["level"], "approximate")
        self.assertEqual(result["condition_results"]["activity:做晚饭"]["status"], "unknown")

    def test_answer_cannot_use_real_asset_id_to_support_unallowed_statement(self):
        allowed = {"allowed_answer_facts": [{"text": "照片拍摄于厨房", "status": "matched", "evidence_ids": ["asset-may", "obs-may"]}], "allowed_possibilities": [], "required_unknowns": []}
        draft = {"answer": "明哥在厨房准备晚饭。", "statements": [{"text": "明哥在厨房准备晚饭", "status": "matched", "evidence_ids": ["asset-may"]}]}
        checked = validate_statement_plan(draft, allowed)
        self.assertFalse(checked.valid)
        answer = compose_answer(draft, allowed)
        self.assertIn("无法确认", answer["answer"])


if __name__ == "__main__":
    unittest.main()
