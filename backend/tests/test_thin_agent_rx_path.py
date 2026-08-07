"""RX integration: _evidence_answer routes to the RX pipeline under the flag."""

import json
import os
import unittest
from unittest import mock

from backend.evidence_retrieval import EvidencePacket
from backend.model_routing import ModelRouter
from backend.query_contracts import Constraint, HARD, QueryAction, QueryParseDraft, QuerySpec
from backend.response_validator import scan_internal_leak
from backend.router import GateDecision
from backend.thin_agent import ThinAgentRuntime


class _StubGamma:
    def chat(self, prompt, json_mode=True, role=None):
        return json.dumps({
            "text": "我找到了去年十月在海边拍的照片，下面是其中最相关的几张。",
            "statements": [{"text": "记录中有「海边」", "fact_id": "fact_1", "certainty": "confirmed"}],
        })


def _runtime():
    gamma = _StubGamma()
    runtime = object.__new__(ThinAgentRuntime)
    runtime.gamma = gamma
    runtime.router = ModelRouter(gamma=gamma)
    runtime.store = None
    runtime.complex_builder = None
    return runtime


def _spec(answer_target="general", actions=None, all_relevant=False):
    requirement = {"mode": "all_relevant"} if all_relevant else {}
    return QuerySpec("q", "single", ["home"], "owner", "c", "answer", answer_target,
                     constraints=[Constraint("time", "2025年10月", HARD, "asset_metadata")],
                     actions=actions or [], result_requirement=requirement)


def _asset(asset_id, conds, level="exact"):
    return {"asset_id": asset_id, "file_name": f"{asset_id}.jpg", "media_type": "image",
            "observation_ids": [f"obs-{asset_id}"], "evidence_ids": [asset_id],
            "condition_results": conds, "level": level, "captured_at": "2025-10-02"}


def _decision():
    return GateDecision("evidence", "explicit_evidence_action", query_parse_calls=1)


class RxPathTests(unittest.TestCase):
    def test_exact_result_pipeline(self):
        asset = _asset("asset-1", {"time:2025年10月": {"status": "matched"}})
        packet = EvidencePacket("q", "home", "general", assets=[asset], exact_results=[asset])
        runtime = _runtime()
        result = runtime._rx_answer("去年十月爬山拍的合影", "c1", "home", "owner",
                                    _decision(), _spec(), packet, QueryParseDraft())
        self.assertEqual(result["response_mode"], "exact_result")
        self.assertTrue(result["image_results"])
        self.assertEqual(result["image_results"][0]["display_handle"], "照片1")
        self.assertEqual(scan_internal_leak(result["answer"]), [])
        self.assertTrue(result["claims"])

    def test_all_unknown_approximate_shows_no_images(self):
        asset = _asset("asset-1", {"semantic:海豚": {"status": "unknown"}}, level="approximate")
        packet = EvidencePacket("q", "home", "general", assets=[asset], approximate_results=[asset])
        runtime = _runtime()
        result = runtime._rx_answer("水族馆海豚跃出水面", "c1", "home", "owner",
                                    _decision(), _spec(), packet, QueryParseDraft())
        # all-unknown -> 0 images, but the answer still discloses no confirmed match
        self.assertEqual(result["image_results"], [])
        self.assertEqual(result["response_mode"], "approximate_result")
        self.assertIn("没有完全匹配", result["answer"])

    def test_asset_delivery_text_and_images_agree(self):
        asset = _asset("asset-1", {"image": {"status": "matched"}})
        packet = EvidencePacket("q", "home", "general", assets=[asset], exact_results=[asset])
        spec = _spec(actions=[QueryAction(type="return_assets")])
        runtime = _runtime()
        result = runtime._rx_answer("把原图给我", "c1", "home", "owner",
                                    _decision(), spec, packet, QueryParseDraft())
        self.assertEqual(result["response_mode"], "asset_delivery")
        self.assertTrue(result["image_results"])
        self.assertEqual(result["answer"].strip(), "我找到了去年十月在海边拍的照片，下面是其中最相关的几张。")

    def test_evidence_answer_routes_to_rx_under_flag(self):
        asset = _asset("asset-1", {"time:2025年10月": {"status": "matched"}})
        packet = EvidencePacket("q", "home", "general", assets=[asset], exact_results=[asset])
        runtime = _runtime()
        with mock.patch.dict(os.environ, {"SENTRIX_RX_V1": "1"}, clear=False):
            result = runtime._evidence_answer("去年十月爬山拍的合影", "c1", "home", "owner",
                                              _decision(), _spec(), packet, QueryParseDraft())
        self.assertIn("response_mode", result)
        self.assertEqual(result["response_mode"], "exact_result")

    def test_person_summary_gap_no_claims(self):
        packet = EvidencePacket("q", "home", "person")
        runtime = _runtime()
        spec = QuerySpec("q", "single", ["home"], "owner", "c", "answer", "person",
                         constraints=[Constraint("person", "明哥", HARD, "confirmed_bridge")],
                         entity_ids=["entity-1"])
        result = runtime._rx_answer("介绍一下明哥", "c1", "home", "owner",
                                    _decision(), spec, packet, QueryParseDraft())
        self.assertEqual(result["response_mode"], "person_summary")
        self.assertEqual(result["evidence_status"], "gap")
        self.assertIn("还没有足够", result["answer"])
        self.assertNotIn("多次出现", result["answer"])


if __name__ == "__main__":
    unittest.main()
