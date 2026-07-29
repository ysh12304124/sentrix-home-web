import tempfile
import unittest

from backend.agent import MemoryAgent
from backend.db import MemoryStore


class FakeGamma:
    model = "test-gamma"

    def answer(self, query, context):
        return {"answer": "证据中有冰箱。", "confidence": 0.9, "evidence": [], "insufficient_evidence": False}

    def embed_text(self, text):
        return []


class AgentEvidenceTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
