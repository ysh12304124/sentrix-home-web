"""TFPE v2: thin_agent structured path wiring + _structured_answer end to end."""

import os
import unittest
from unittest import mock

from backend.db import MemoryStore
from backend.query_contracts import QueryParseDraft, QuerySpec
from backend.response_validator import scan_internal_leak
from backend.retrieval_strategy import plan_retrieval_strategy
from backend.router import RouteDecision
from backend.thin_agent import ThinAgentRuntime


def _seed_store():
    store = MemoryStore(":memory:")
    store.create_asset("a1", "a1.jpg", "image", "/x/a1.jpg",
                       metadata={"captured_at": "2024-01-15T10:00:00"}, scope_id="home")
    store.create_asset("a2", "a2.jpg", "image", "/x/a2.jpg",
                       metadata={"captured_at": "2024-03-20T09:00:00"}, scope_id="home")
    store.add_observation("a1", {"id": "obs1", "captured_at": "2024-01-15T10:00:00"}, scope_id="home")
    store.add_observation("a2", {"id": "obs2", "captured_at": "2024-03-20T09:00:00"}, scope_id="home")
    return store


def _runtime(store):
    runtime = object.__new__(ThinAgentRuntime)
    runtime.store = store
    runtime.router = None
    runtime.kernel = None
    return runtime


def _count_draft():
    return QueryParseDraft(answer_type="count",
                           structured={"time_range": {"start": "2024-01-01", "end": "2025-01-01"}})


def _spec():
    return QuerySpec("q", "single", ["home"], "owner", "c", "answer", "general", constraints=[])


class StructuredRoutingTests(unittest.TestCase):
    def test_evidence_path_routes_structured_without_retrieval(self):
        with mock.patch.dict(os.environ, {"SENTRIX_RX_V1": "1", "SENTRIX_STRUCTURED_MEMORY_V1": "1"}):
            runtime = _runtime(_seed_store())
            runtime.kernel = mock.Mock()
            decision = RouteDecision("evidence", "explicit_evidence_action")
            result = runtime._evidence_path("去年拍了多少张照片", "", "c1", "home", "owner",
                                            decision, _count_draft())
            runtime.kernel.retrieve.assert_not_called()
            self.assertEqual(result["response_mode"], "structured_fact")
            self.assertIn("2", result["answer"])

    def test_structured_answer_exact_count_no_images_no_leak(self):
        with mock.patch.dict(os.environ, {"SENTRIX_RX_V1": "1", "SENTRIX_STRUCTURED_MEMORY_V1": "1"}):
            runtime = _runtime(_seed_store())
            decision = RouteDecision("evidence", "structured")
            spec = _spec()
            draft = _count_draft()
            strategy = plan_retrieval_strategy(draft, spec)
            result = runtime._structured_answer("去年拍了多少张照片", "c1", "home", "owner",
                                               decision, spec, draft, strategy)
            self.assertEqual(result["response_mode"], "structured_fact")
            self.assertEqual(result["image_results"], [])
            self.assertEqual(result["structured_result"]["total"], 2)
            self.assertEqual(scan_internal_leak(result["answer"]), [])
            self.assertIn("visual_ann", result["retrieval_strategy"]["skipped_channels"])
            self.assertEqual(result["retrieval_trace"][1]["chosen_strategy"], "structured_fact")

    def test_structured_inactive_falls_back_to_retrieval(self):
        with mock.patch.dict(os.environ, {"SENTRIX_RX_V1": "0", "SENTRIX_STRUCTURED_MEMORY_V1": "1"}):
            from backend.validation.full_chain_profile import structured_memory_active
            self.assertFalse(structured_memory_active())

    def test_asset_set_not_routed_to_structured(self):
        from backend.evidence_retrieval import EvidencePacket
        with mock.patch.dict(os.environ, {"SENTRIX_RX_V1": "1", "SENTRIX_STRUCTURED_MEMORY_V1": "1"}):
            runtime = _runtime(_seed_store())
            runtime.kernel = mock.Mock()
            runtime.kernel.retrieve.return_value = EvidencePacket("q", "home", "general")
            decision = RouteDecision("evidence", "explicit_evidence_action")
            draft = QueryParseDraft(answer_type="asset_set")
            runtime._evidence_path("找去年十月爬山拍的合影", "", "c1", "home", "owner",
                                   decision, draft)
            runtime.kernel.retrieve.assert_called_once()


if __name__ == "__main__":
    unittest.main()
