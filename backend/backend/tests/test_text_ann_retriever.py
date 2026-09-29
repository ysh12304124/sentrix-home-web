"""Phase R R2 — text ANN retriever: observation -> asset mapping + guards."""

import tempfile
import unittest
from pathlib import Path

from backend.db import MemoryStore
from backend.embeddings import EmbeddingRouter
from backend.retrieval import HardFilterContext, RetrievalQuery
from backend.retrieval.text_ann import TextAnnRetriever
from backend.retrieval_ann import create_index


class StubTextClip:
    model_name = "ViT-B-32"
    evidence_ready = True

    def embed_text(self, text):
        return [float((ord(str(text)[0]) % 97) / 97.0), 0.0]


def _build_semantic_index(tmp, dim=2, model="ViT-B-32", scope="album1"):
    index = create_index("hnswlib", dim=dim, M=4, ef_construction=50, ef_search=10)
    index.set_manifest_extra(model_id=model, checkpoint_hash="x", source_type="observation",
                             normalized=True, source_revision=1)
    # observation obs_1 belongs to asset_1; obs_2 belongs to asset_2.
    index.build([
        ("obs_1", [0.9, 0.0], {"scope_id": scope, "asset_id": "asset_1", "revision": 1}),
        ("obs_2", [0.1, 0.0], {"scope_id": scope, "asset_id": "asset_2", "revision": 1}),
    ])
    index.save(str(Path(tmp) / "semantic"))


class TextAnnRetrieverTests(unittest.TestCase):
    def _router(self):
        clip = StubTextClip()
        return EmbeddingRouter(visual=None, text=_TextEmbedder(clip))

    def test_maps_observation_hit_to_asset(self):
        with tempfile.TemporaryDirectory(prefix="tann-") as tmp:
            _build_semantic_index(tmp)
            router = self._router()
            store = MemoryStore(":memory:")
            retriever = TextAnnRetriever(store, embedding_router=router, ann_dir=tmp, spaces=("semantic",))
            hits = retriever.retrieve(RetrievalQuery(whole_query="A"),
                                      HardFilterContext(scope_ids=("album1",)), 5)
            store.close()
            self.assertTrue(hits)
            self.assertEqual(hits[0].asset_id, "asset_1")

    def test_embedder_unavailable_returns_empty(self):
        with tempfile.TemporaryDirectory(prefix="tann-") as tmp:
            _build_semantic_index(tmp)
            router = EmbeddingRouter(visual=None, text=None)
            store = MemoryStore(":memory:")
            retriever = TextAnnRetriever(store, embedding_router=router, ann_dir=tmp, spaces=("semantic",))
            hits = retriever.retrieve(RetrievalQuery(whole_query="A"),
                                      HardFilterContext(scope_ids=("album1",)), 5)
            store.close()
            self.assertEqual(hits, [])

    def test_incompatible_model_skips(self):
        with tempfile.TemporaryDirectory(prefix="tann-") as tmp:
            _build_semantic_index(tmp, model="ViT-B-32")
            router = EmbeddingRouter(visual=None, text=_TextEmbedder(StubTextClip(), model_id="bge-m3"))
            store = MemoryStore(":memory:")
            retriever = TextAnnRetriever(store, embedding_router=router, ann_dir=tmp, spaces=("semantic",))
            hits = retriever.retrieve(RetrievalQuery(whole_query="A"),
                                      HardFilterContext(scope_ids=("album1",)), 5)
            store.close()
            self.assertEqual(hits, [])


class _TextEmbedder:
    available = True

    def __init__(self, clip, model_id=None):
        self._clip = clip
        self.model_id = model_id or clip.model_name
        self.dimension = 512

    def embed_query(self, text):
        return self._clip.embed_text(text)


if __name__ == "__main__":
    unittest.main()
