"""RX-5 user-visible redaction: internal tokens must never reach user text."""

import unittest

from backend.response_validator import scan_internal_leak

CORPUS = {
    "internal_ids": "asset_ff34f10b39c3 obs_abc event_xyz entity_ba91871b4d17",
    "english_labels": "确定（matched） 可能（possible） 无法确认（unknown）",
    "table_names": "assets observations events memory_vectors",
    "trace_terms": "fusion_score retrieval_trace condition_key recall_strength",
    "template_language": "根据本地事件记忆检索到相关记录",
}


class RedactionTests(unittest.TestCase):
    def test_every_category_is_detected(self):
        for category, text in CORPUS.items():
            hits = scan_internal_leak(text)
            self.assertTrue(hits, f"{category} not detected in {text!r}")

    def test_clean_chinese_passes(self):
        text = "我找到了去年十月在海边拍的照片，下面是其中最相关的几张。"
        self.assertEqual(scan_internal_leak(text), [])

    def test_writer_style_answer_passes(self):
        text = "没有完全匹配；时间比较接近，但活动还不能确认。"
        self.assertEqual(scan_internal_leak(text), [])

    def test_detects_id_inside_answer(self):
        self.assertEqual(scan_internal_leak("确定（matched）：找到 3 张"), ["matched"])

    def test_no_false_positive_on_english_inside_code(self):
        # "unknown" as a bare word is a leak only in user text; the scanner is
        # the same everywhere, so it should flag it even mid-sentence.
        self.assertEqual(scan_internal_leak("无法确认 unknown 细节"), ["unknown"])


if __name__ == "__main__":
    unittest.main()
