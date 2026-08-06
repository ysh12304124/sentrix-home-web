"""Phase R R2 — multi-retriever Kernel integration (channel trace + attributions)."""

import unittest

from backend.db import MemoryStore
from backend.evidence_retrieval import EvidenceRetrievalKernel
from backend.query_contracts import Constraint, QueryParseDraft, build_query_spec
from backend.retrieval import HardFilterContext, RetrievalQuery, fuse
from backend.retrieval.lexical import LexicalRetriever
from backend.retrieval.metadata import MetadataRetriever
from backend.retrieval_indexes import RetrievalIndex


class StubEmbedderRouter:
    visual_available = False
    text_available = False
    visual = None
    text = None

    def embed_visual(self, text):
        return []

    def embed_text(self, text):
        return []


def _seed_store():
    store = MemoryStore(":memory:")
    store.create_asset("asset_1", "IMG_1.JPG", "image", "/tmp/1", "image/jpeg", 1, {"scope_id": "album1"})
    store.create_asset("asset_2", "IMG_2.JPG", "image", "/tmp/2", "image/jpeg", 1, {"scope_id": "album1"})
    now = "2026-08-06T00:00:00Z"
    store.connection.execute(
        """INSERT INTO observations (id, scope_id, asset_id, captured_at, source_type, caption,
            people_json, objects_json, ocr_text, confidence, raw_json, canonical_json,
            clothing_json, spatial_relations_json, revision, created_at, updated_at)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        ("obs_1", "album1", "asset_1", "2023-11-16T14:00:00+00:00", "vision", "卧室睡衣自拍",
         "[]", "[]", "", 0.9, "{}", "{}", '["毛绒睡衣"]', "[]", 1, now, now),
    )
    store.connection.execute(
        """INSERT INTO observations (id, scope_id, asset_id, captured_at, source_type, caption,
            people_json, objects_json, ocr_text, confidence, raw_json, canonical_json,
            clothing_json, spatial_relations_json, revision, created_at, updated_at)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        ("obs_2", "album1", "asset_2", "2024-05-03T18:30:00+00:00", "vision", "厨房做饭",
         "[]", "[]", "", 0.9, "{}", "{}", "[]", "[]", 1, now, now),
    )
    store.connection.commit()
    RetrievalIndex(store).rebuild_all()
    return store


class MultiRetrieverKernelTests(unittest.TestCase):
    def _spec(self, value="毛绒睡衣"):
        draft = QueryParseDraft(intent="answer", answer_target="general")
        draft.semantic_conditions.append({"dimension": "clothing", "value": value, "source_text": value})
        return build_query_spec(draft, scope_id="album1", viewer_id="owner",
                                conversation_id="c", query_id="q")

    def test_kernel_multi_path_produces_channel_trace(self):
        store = _seed_store()
        try:
            retrievers = [MetadataRetriever(store), LexicalRetriever(store)]
            kernel = EvidenceRetrievalKernel(store, retrievers=retrievers,
                                             embedding_router=StubEmbedderRouter())
            spec = self._spec()
            packet = kernel._retrieve_multi(spec)
            self.assertTrue(getattr(packet, "channel_trace", {}))
            self.assertIn("metadata", packet.channel_trace)
            self.assertIn("lexical", packet.channel_trace)
            self.assertTrue(packet.channel_trace["metadata"]["invoked"])
            self.assertGreaterEqual(packet.channel_trace["lexical"]["candidate_count"], 0)
        finally:
            store.close()

    def test_kernel_multi_path_recalls_matching_asset(self):
        store = _seed_store()
        try:
            retrievers = [MetadataRetriever(store), LexicalRetriever(store)]
            kernel = EvidenceRetrievalKernel(store, retrievers=retrievers,
                                             embedding_router=StubEmbedderRouter())
            packet = kernel._retrieve_multi(self._spec())
            ids = {item["asset_id"] for item in packet.assets}
            self.assertIn("asset_1", ids)
            # 厨房做饭 asset_2 must not be recalled by a 睡衣 query.
            self.assertNotIn("asset_2", ids)
        finally:
            store.close()

    def test_item_has_attributions_and_fusion_score(self):
        store = _seed_store()
        try:
            retrievers = [MetadataRetriever(store), LexicalRetriever(store)]
            kernel = EvidenceRetrievalKernel(store, retrievers=retrievers,
                                             embedding_router=StubEmbedderRouter())
            packet = kernel._retrieve_multi(self._spec())
            self.assertTrue(packet.assets)
            item = packet.assets[0]
            self.assertIn("attributions", item)
            self.assertIn("fusion_score", item)
            self.assertTrue(any(attr["retriever"] in {"metadata", "lexical"} for attr in item["attributions"]))
        finally:
            store.close()


if __name__ == "__main__":
    unittest.main()
