"""Phase D — D4/D5/D6 工具集成测试（search_conversation_history / get_core_memory / get_person_memory）。"""
import os
import tempfile
import unittest

from backend.db import MemoryStore
from backend.agent_conversation import ConversationStore
from backend.agent_runtime import tools as runtime_tools


def _bind(store, scope_id="album3-v2"):
    runtime_tools._RUNTIME.clear()
    runtime_tools.bind_runtime(store)
    runtime_tools.register_tools()


class PhaseDToolsTest(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.store = MemoryStore(os.path.join(self.dir, "test.db"))
        self.cs = ConversationStore(self.store)
        _bind(self.store)

    def test_search_conversation_history(self):
        cid = self.cs.create_conversation(scope_id="album3-v2")
        self.cs.ensure_title(cid, "我是不是说过喜欢西湖")
        self.cs.add_message(cid, "user", {"text": "我是不是说过喜欢西湖"}, scope_id="album3-v2")
        self.cs.add_message(cid, "assistant", {"text": "你说过喜欢西湖。"}, scope_id="album3-v2")
        runtime_tools.set_conversation_id(cid)
        out = runtime_tools._search_conversation_history({"query": "西湖", "scope": "current"},
                                                         context={"scope_id": "album3-v2"})
        self.assertGreaterEqual(out["total"], 1)
        self.assertTrue(any("西湖" in m["text"] for m in out["matches"]))

    def test_get_core_memory(self):
        e1 = self.store.create_entity("明哥", entity_type="person", status="confirmed",
                                      family_role="丈夫", scope_id="album3-v2")
        e2 = self.store.create_entity("雪儿", entity_type="person", status="confirmed",
                                      family_role="女儿", scope_id="album3-v2")
        self.store.create_relationship(e1["id"], "父亲", e2["id"], status="confirmed",
                                       confidence=1.0, evidence_ids=[])
        out = runtime_tools._get_core_memory({"subject": "明哥"},
                                             context={"scope_id": "album3-v2"})
        self.assertGreaterEqual(out["total"], 1)
        texts = " ".join(i["text"] for c in out["cards"] for i in c["items"])
        self.assertIn("丈夫", texts)

    def test_get_person_memory_limited_unknown(self):
        out = runtime_tools._get_person_memory({"person": "不存在的人"},
                                               context={"scope_id": "album3-v2"})
        self.assertEqual(out["readiness"], "limited")
        self.assertTrue(out.get("insufficient_evidence"))


if __name__ == "__main__":
    unittest.main()
