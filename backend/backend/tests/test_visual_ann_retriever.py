"""Phase R R2 — visual ANN retriever: embed -> index -> candidates + guards."""

import tempfile
import unittest
from pathlib import Path

from backend.db import MemoryStore
from backend.embeddings import EmbeddingRouter
from backend.retrieval import HardFilterContext, RetrievalQuery
from backend.retrieval.visual_ann import VisualAnnRetriever
from backend.retrieval_ann import create_index


class StubClip:
    model_name = "ViT-B-32"
    evidence_ready = True

    def __init__(self, ready=True):
        self.evidence_ready = ready

    def embed_text(self, text):
        # deterministic: single feature = hash of first char
        return [float((ord(str(text)[0]) % 97) / 97.0), 0.0]


def _build_index(tmp, dim=2, model="ViT-B-32", scope="album1"):
    index = create_index("hnswlib", dim=dim, M=4, ef_construction=50, ef_search=10)
    index.set_manifest_extra(model_id=model, checkpoint_hash="x", source_type="asset",
                             normalized=True, source_revision=1)
    # asset_0 vector near the query embedding for "A", asset_1 far.
    index.build([
        ("asset_0", [0.9, 0.0], {"scope_id": scope, "revision": 1}),
        ("asset_1", [0.1, 0.0], {"scope_id": scope, "revision": 1}),
        ("asset_2", [0.0, 0.0], {"scope_id": "other_scope", "revision": 1}),
    ])
    index.save(str(Path(tmp) / "visual"))


def _router():
    clip = StubClip()
    return EmbeddingRouter.from_clip(clip), clip


class VisualAnnRetrieverTests(unittest.TestCase):
    def test_recalls_nearest_asset(self):
        with tempfile.TemporaryDirectory(prefix="vann-") as tmp:
            _build_index(tmp)
            router, _ = _router()
            store = MemoryStore(":memory:")
            retriever = VisualAnnRetriever(store, embedding_router=router, ann_dir=tmp)
            filters = HardFilterContext(scope_ids=("album1",))
            query = RetrievalQuery(whole_query="A")
            hits = retriever.retrieve(query, filters, limit=5)
            store.close()
            self.assertTrue(hits)
            self.assertEqual(hits[0].asset_id, "asset_0")
            self.assertEqual(hits[0].score_kind, "cosine_similarity")

    def test_scope_filtered_candidates(self):
        with tempfile.TemporaryDirectory(prefix="vann-") as tmp:
            _build_index(tmp)
            router, _ = _router()
            store = MemoryStore(":memory:")
            retriever = VisualAnnRetriever(store, embedding_router=router, ann_dir=tmp)
            filters = HardFilterContext(scope_ids=("album1",))
            query = RetrievalQuery(whole_query="A")
            hits = retriever.retrieve(query, filters, limit=5)
            store.close()
            self.assertTrue(all(hit.asset_id in {"asset_0", "asset_1"} for hit in hits))
            self.assertNotIn("asset_2", {hit.asset_id for hit in hits})

    def test_embedder_unavailable_returns_empty(self):
        with tempfile.TemporaryDirectory(prefix="vann-") as tmp:
            _build_index(tmp)
            router = EmbeddingRouter(visual=None, text=None)
            store = MemoryStore(":memory:")
            retriever = VisualAnnRetriever(store, embedding_router=router, ann_dir=tmp)
            hits = retriever.retrieve(RetrievalQuery(whole_query="A"),
                                      HardFilterContext(scope_ids=("album1",)), 5)
            store.close()
            self.assertEqual(hits, [])
            self.assertEqual(retriever.status, "embedder_unavailable")

    def test_incompatible_manifest_skips(self):
        with tempfile.TemporaryDirectory(prefix="vann-") as tmp:
            _build_index(tmp, model="ViT-B-32")
            router = EmbeddingRouter(visual=_StubEmbedderModel("Chinese-CLIP", 512), text=None)
            store = MemoryStore(":memory:")
            retriever = VisualAnnRetriever(store, embedding_router=router, ann_dir=tmp)
            hits = retriever.retrieve(RetrievalQuery(whole_query="A"),
                                      HardFilterContext(scope_ids=("album1",)), 5)
            store.close()
            self.assertEqual(hits, [])
            self.assertEqual(retriever.status, "incompatible")


class _StubEmbedderModel:
    available = True

    def __init__(self, model_id, dim):
        self.model_id = model_id
        self.dimension = dim

    def embed_query(self, text):
        return [0.5, 0.0]


if __name__ == "__main__":
    unittest.main()
