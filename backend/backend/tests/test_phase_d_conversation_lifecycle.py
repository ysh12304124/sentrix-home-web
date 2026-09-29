import os, tempfile, unittest
from backend.db import MemoryStore
from backend.agent_conversation import ConversationStore

class D2ConversationLifecycleTest(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.store = MemoryStore(os.path.join(self.dir, "test.db"))
        self.cs = ConversationStore(self.store)

    def test_lifecycle(self):
        cid = self.cs.create_conversation(scope_id="album3-v2")
        self.cs.ensure_title(cid, "去年杭州的照片有哪些")
        conv = self.cs.get_conversation(cid)
        self.assertEqual(conv["title"], "去年杭州的照片有哪些")
        self.cs.add_message(cid, "user", {"text": "去年杭州的照片有哪些"}, scope_id="album3-v2")
        self.cs.add_message(cid, "assistant", {"text": "找到了 12 张"}, scope_id="album3-v2")
        self.cs.touch_conversation(cid)
        self.assertEqual(len(self.cs.list_messages(cid)), 2)
        lst = self.cs.list_conversations(scope_id="album3-v2")
        self.assertEqual(len(lst), 1)
        hits = self.cs.search_messages("杭州")
        self.assertTrue(any(h["content"].get("text", "") == "去年杭州的照片有哪些" for h in hits))
        self.cs.delete_conversation(cid)
        self.assertIsNone(self.cs.get_conversation(cid))
        self.assertEqual(self.cs.list_messages(cid), [])
        hits2 = self.cs.search_messages("杭州")
        self.assertFalse(any(h["conversation_id"] == cid for h in hits2))

if __name__ == "__main__":
    unittest.main()
