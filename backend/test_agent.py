import tempfile
import unittest

from backend.agent import MemoryAgent, contains
from backend.db import MemoryStore


class FakeGamma:
    model = "test-gamma"

    def answer(self, query, context):
        return {"answer": "证据中有冰箱。", "confidence": 0.9, "evidence": [], "insufficient_evidence": False}

    def embed_text(self, text):
        return []


class RefusingGamma(FakeGamma):
    def answer(self, query, context):
        return {"answer": "证据不足", "confidence": 0.1, "evidence": [], "insufficient_evidence": True}


class AgentEvidenceTests(unittest.TestCase):
    def test_search_terms_do_not_match_every_filename_by_one_common_token(self):
        self.assertTrue(contains("SR_AWS_N_0016.jpg", "SR_AWS_N_0016.jpg"))
        self.assertFalse(contains("SR_AWS_N_0054.jpg", "SR_AWS_N_0016.jpg"))
        self.assertTrue(contains("SR_AWS_N_0016.jpg", "请查看 SR_AWS_N_0016.jpg 中的人做了什么"))
        self.assertFalse(contains("SR_AWS_N_0054.jpg", "请查看 SR_AWS_N_0016.jpg 中的人做了什么"))

    def test_answer_returns_asset_observation_and_raw_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            store = MemoryStore(f"{directory}/memory.db")
            asset = store.create_asset("asset_1", "family.jpg", "image", "/tmp/family.jpg")
            observation = store.add_observation(asset["id"], {"caption": "打开冰箱", "place": "厨房", "raw": {"objects": ["冰箱"]}})
            store.merge_observation_into_event(observation)
            result = MemoryAgent(store, gamma=FakeGamma()).answer("冰箱")
            kinds = {item["kind"] for item in result["evidence"]}
            observation_evidence = next(item for item in result["evidence"] if item["kind"] == "observation")
            self.assertEqual(kinds, {"event", "observation"})
            self.assertEqual(observation_evidence["asset_id"], "asset_1")
            self.assertEqual(observation_evidence["raw"]["objects"], ["冰箱"])

    def test_local_evidence_fallback_answers_when_model_refuses(self):
        with tempfile.TemporaryDirectory() as directory:
            store = MemoryStore(f"{directory}/memory.db")
            asset = store.create_asset("asset_1", "family.jpg", "image", "/tmp/family.jpg")
            observation = store.add_observation(asset["id"], {"caption": "家人在客厅聚会", "place": "客厅"})
            store.merge_observation_into_event(observation)

            result = MemoryAgent(store, gamma=RefusingGamma()).answer("family.jpg")

            self.assertFalse(result["insufficient_evidence"])
            self.assertIn("家人在客厅聚会", result["answer"])
            self.assertEqual(result["evidence"][1]["asset_id"], "asset_1")


if __name__ == "__main__":
    unittest.main()
