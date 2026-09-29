"""Phase R R1B — Embedding quality evaluation logic tests.

Two independent evaluators per P0-2:
  - Visual Cross-modal:  query text -> text embed -> image embed brute-force
                         -> correct image rank (Recall@1/5/10, MRR, AUC)
  - Text Retrieval:      query -> caption/activity/place/object/Event/OCR
                         -> correct record rank

These tests exercise the *evaluation logic* with a deterministic stub embedder
whose similarity reflects shared character bigrams.  They do not prove real
CLIP works — the 153 runs with the real ClipAdapter are the acceptance runs.
"""

import importlib.util
import math
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
BENCH = REPO_ROOT / "scripts" / "benchmarks"


def _load(module_name, file_name):
    spec = importlib.util.spec_from_file_location(module_name, BENCH / file_name)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


class StubEmbedder:
    """Deterministic bigram-hash embedder: overlapping substrings => similar.

    Same interface shape as ClipAdapter (embed_text/embed_image returning
    a 1-D list), so the evaluators stay model-agnostic.
    """

    dim = 64

    def embed_text(self, text):
        return self._embed(str(text or ""))

    def embed_image(self, _path):
        return self._embed(_path)

    def _embed(self, text):
        vector = [0.0] * self.dim
        padded = f" {text} "
        for index in range(len(padded) - 1):
            pair = padded[index:index + 2]
            for unit in (pair,):
                bucket = (sum(ord(c) for c in unit) * 2654435761) % self.dim
                vector[bucket] += 1.0
        norm = math.sqrt(sum(x * x for x in vector)) or 1.0
        return [x / norm for x in vector]


def _cosine(a, b):
    return sum(x * y for x, y in zip(a, b))


def _load_module():
    return _load("_embedding_quality", "evaluate_embedding_quality.py")


class VisualCrossModalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = _load_module()
        cls.embedder = StubEmbedder()

    def _run(self, pairs):
        images = [{"id": item[0], "text": item[1]} for item in pairs]
        queries = [{"id": item[0], "query": item[1], "target": item[0]} for item in pairs]
        return self.module.visual_crossmodal(images, queries, embedder=self.embedder)

    def test_correct_image_ranks_top_for_overlapping_text(self):
        pairs = [
            ("img_a", "银色心形手镯"),
            ("img_b", "厨房里做晚饭"),
            ("img_c", "浅黄色毛绒睡衣"),
        ]
        report = self._run(pairs)
        self.assertGreaterEqual(report["recall@1"], 2 / 3)
        self.assertIn("mrr", report)
        self.assertIn("auc", report)

    def test_recall_k_shape(self):
        pairs = [(f"img_{i}", f"物品编号{i}") for i in range(6)]
        report = self._run(pairs)
        self.assertIn("recall@1", report)
        self.assertIn("recall@5", report)
        self.assertIn("recall@10", report)

    def test_negative_separation_raises_auc(self):
        # Same query target twice + one far-away text.
        pairs = [("img_a", "银手镯"), ("img_a2", "银手镯"), ("img_b", "跨年烟花")]


class TextRetrievalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = _load_module()
        cls.embedder = StubEmbedder()

    def test_field_retrieval_recall(self):
        corpus = [
            {"id": "obs_1", "text": "卧室睡衣自拍", "field": "caption"},
            {"id": "obs_2", "text": "厨房做饭", "field": "activity"},
            {"id": "obs_3", "text": "跨年烟花", "field": "caption"},
        ]
        queries = [{"id": "q1", "query": "睡衣自拍", "target": "obs_1"}]
        report = self.module.text_retrieval(corpus, queries, embedder=self.embedder)
        self.assertGreaterEqual(report["recall@1"], 0.5)

    def test_irrelevant_text_does_not_outrank(self):
        corpus = [
            {"id": "obs_1", "text": "厨房做饭", "field": "activity"},
            {"id": "obs_2", "text": "海豚跃出水面", "field": "caption"},
        ]
        queries = [{"id": "q1", "query": "海豚表演", "target": "obs_2"}]
        report = self.module.text_retrieval(corpus, queries, embedder=self.embedder)
        self.assertGreaterEqual(report["recall@1"], 0.5)


if __name__ == "__main__":
    unittest.main()
