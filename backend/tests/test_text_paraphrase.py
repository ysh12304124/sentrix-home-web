"""Phase R8-4 — text paraphrase evaluator logic tests (stub embedder)."""

import importlib.util
import math
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
BENCH = REPO_ROOT / "scripts" / "benchmarks"


def _load_module():
    spec = importlib.util.spec_from_file_location("_text_paraphrase", BENCH / "evaluate_text_paraphrase.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["_text_paraphrase"] = module
    spec.loader.exec_module(module)
    return module


class StubTextEmbedder:
    """Bigram-hash embedder: overlapping text => similar vectors."""

    dim = 64

    def embed_query(self, text):
        vector = [0.0] * self.dim
        padded = f" {text} "
        for i in range(len(padded) - 1):
            bucket = (sum(ord(c) for c in padded[i:i + 2]) * 2654435761) % self.dim
            vector[bucket] += 1.0
        norm = math.sqrt(sum(x * x for x in vector)) or 1.0
        return [x / norm for x in vector]


class TextParaphraseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = _load_module()

    def test_paraphrase_map_and_generator(self):
        self.assertEqual(self.module._paraphrase("做晚饭"), "做饭")
        self.assertEqual(self.module._paraphrase("合影留念"), "拍合照")
        # general structural rule: strip suffix
        self.assertEqual(self.module._paraphrase("散步留念"), "散步")
        # split on 或
        self.assertEqual(self.module._paraphrase("休息或拍照"), "休息")

    def test_cross_retrieval_recall_computed(self):
        obs = [
            {"id": "o1", "activity": "用餐", "caption": "一家人吃饭", "place": "餐厅", "ocr_text": ""},
            {"id": "o2", "activity": "用餐", "caption": "吃火锅", "place": "餐厅", "ocr_text": ""},
            {"id": "o3", "activity": "休息", "caption": "躺在沙发", "place": "客厅", "ocr_text": ""},
        ]
        corpus = [self.module._texts(row) for row in obs]
        queries = self.module.build_cross_retrieval_queries(obs)
        self.assertTrue(any(q["query"] == "用餐" and len(q["target_ids"]) == 2 for q in queries))
        results = self.module.evaluate(corpus, queries, StubTextEmbedder())
        summary = self.module.summarize(results)
        # 用餐 query should rank its two members near the top.
        self.assertGreaterEqual(summary["recall@1"], 0.5)

    def test_paraphrase_queries_non_self_match(self):
        obs = [{"id": "o1", "activity": "做晚饭", "caption": "厨房", "place": "家", "ocr_text": ""}]
        queries = self.module.build_paraphrase_queries(obs)
        self.assertTrue(any(q["query"] == "做饭" for q in queries))


if __name__ == "__main__":
    unittest.main()
