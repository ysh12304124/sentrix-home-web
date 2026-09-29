"""Phase R R2 — lexical retriever: pre-tokenized FTS semantics (P0-1)."""

import tempfile
import unittest
from pathlib import Path

from backend.db import MemoryStore
from backend.retrieval import HardFilterContext, RetrievalQuery
from backend.retrieval.lexical import LexicalRetriever
from backend.retrieval_indexes import pre_tokenize


class PreTokenizeTests(unittest.TestCase):
    def test_cjk_bigrams_and_whole(self):
        tokens = pre_tokenize("浅黄色毛绒睡衣")
        self.assertIn("浅黄色毛绒睡衣", tokens)
        self.assertIn("浅黄", tokens)
        self.assertIn("毛绒", tokens)
        self.assertIn("睡衣", tokens)
        self.assertNotIn("色", tokens)  # single CJK char never a token

    def test_single_char_query_yields_nothing(self):
        self.assertEqual(pre_tokenize("色"), [])


class LexicalRetrieverTests(unittest.TestCase):
    def _store(self):
        store = MemoryStore(":memory:")
        store.create_asset("asset_1", "IMG_1.JPG", "image", "/tmp/1", "image/jpeg", 1, {"scope_id": "album1"})
        store.create_asset("asset_2", "IMG_2.JPG", "image", "/tmp/2", "image/jpeg", 1, {"scope_id": "album1"})
        store.connection.execute(
            """INSERT INTO observations (id, scope_id, asset_id, captured_at, source_type, caption,
                people_json, objects_json, ocr_text, confidence, raw_json, canonical_json,
                clothing_json, spatial_relations_json, revision, created_at, updated_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            ("obs_1", "album1", "asset_1", "2023-11-16T14:00:00+00:00", "vision", "卧室睡衣自拍",
             "[]", "[]", "", 0.9, "{}", "{}", '["毛绒睡衣"]', "[]", 1, "2026-08-06T00:00:00Z", "2026-08-06T00:00:00Z"),
        )
        store.connection.execute(
            """INSERT INTO observations (id, scope_id, asset_id, captured_at, source_type, caption,
                people_json, objects_json, ocr_text, confidence, raw_json, canonical_json,
                clothing_json, spatial_relations_json, revision, created_at, updated_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            ("obs_2", "album1", "asset_2", "2024-05-03T18:30:00+00:00", "vision", "厨房做饭",
             "[]", "[]", "", 0.9, "{}", "{}", "[]", "[]", 1, "2026-08-06T00:00:00Z", "2026-08-06T00:00:00Z"),
        )
        store.connection.commit()
        # Build the derived projection (terms + FTS) exactly like maintenance does.
        from backend.retrieval_indexes import RetrievalIndex
        RetrievalIndex(store).rebuild_all()
        return store

    def _retrieve(self, store, query_text, scope="album1"):
        retriever = LexicalRetriever(store)
        filters = HardFilterContext(scope_ids=(scope,))
        query = RetrievalQuery(whole_query=query_text, facets=[])
        return retriever.retrieve(query, filters, limit=10)

    def test_bigram_match_recalls_asset(self):
        store = self._store()
        try:
            hits = self._retrieve(store, "睡衣")
            self.assertTrue(any(hit.asset_id == "asset_1" for hit in hits))
        finally:
            store.close()

    def test_single_char_query_returns_empty(self):
        store = self._store()
        try:
            hits = self._retrieve(store, "色")
            self.assertEqual(hits, [])
        finally:
            store.close()

    def test_whole_query_exact_boost(self):
        store = self._store()
        try:
            hits = self._retrieve(store, "卧室睡衣自拍")
            self.assertTrue(any(hit.asset_id == "asset_1" for hit in hits))
            self.assertGreaterEqual(hits[0].raw_score, 2.0)  # exact whole boosted
        finally:
            store.close()


if __name__ == "__main__":
    unittest.main()
