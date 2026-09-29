import unittest

from backend.agent_runtime import intent
from backend.agent_runtime.capability import judge_status


class AgentIntentTests(unittest.TestCase):
    def test_chat_does_not_require_evidence(self):
        self.assertTrue(intent.chat_only("你好，你会什么"))
        self.assertFalse(intent.evidence_intent("你好，你会什么"))

    def test_photo_question_uses_shared_visual_and_multi_signals(self):
        message = "把所有照片都看一下，孩子穿什么颜色的衣服？"
        self.assertTrue(intent.evidence_intent(message))
        self.assertTrue(intent.visual_intent(message))
        self.assertTrue(intent.multi_image_intent(message))

    def test_ocr_and_delivery_signals_are_independent(self):
        self.assertTrue(intent.ocr_intent("招牌上写了什么电话号码？"))
        self.assertFalse(intent.image_delivery_intent("招牌上写了什么电话号码？"))
        self.assertTrue(intent.image_delivery_intent("把这张照片发给我看看"))

    def test_capability_needs_enough_samples_before_ready(self):
        self.assertEqual(judge_status(n=0, support_rate=None), "untested")
        self.assertEqual(judge_status(n=2, support_rate=1.0), "experimental")
        self.assertEqual(judge_status(n=5, support_rate=0.8, false_confident_rate=0.2), "ready")
        self.assertEqual(judge_status(n=5, support_rate=0.6), "limited")


if __name__ == "__main__":
    unittest.main()
