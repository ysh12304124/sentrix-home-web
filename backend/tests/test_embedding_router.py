"""Phase R R2 — EmbeddingRouter composition and availability (P0-3)."""

import os
import unittest
from unittest import mock

from backend.embeddings import EmbeddingRouter


class StubClip:
    model_name = "ViT-B-32"

    def __init__(self, ready=True):
        self.evidence_ready = ready

    def embed_text(self, text):
        return [1.0, 0.0]


class EmbeddingRouterTests(unittest.TestCase):
    def test_from_clip_builds_both_slots(self):
        with mock.patch.dict(os.environ, {
            "SENTRIX_IMAGE_EMBEDDER": "clip",
            "SENTRIX_TEXT_EMBEDDER": "clip",
        }):
            router = EmbeddingRouter.from_clip(StubClip(ready=True))
        self.assertTrue(router.visual_available)
        self.assertTrue(router.text_available)
        self.assertEqual(router.embed_visual("A"), [1.0, 0.0])
        self.assertEqual(router.embed_text("A"), [1.0, 0.0])

    def test_unready_clip_disables_slots(self):
        with mock.patch.dict(os.environ, {
            "SENTRIX_IMAGE_EMBEDDER": "clip",
            "SENTRIX_TEXT_EMBEDDER": "clip",
        }):
            router = EmbeddingRouter.from_clip(StubClip(ready=False))
        self.assertFalse(router.visual_available)
        self.assertFalse(router.text_available)
        self.assertEqual(router.embed_visual("A"), [])

    def test_qwen_visual_profile_uses_remote_visual_slot(self):
        visual = mock.Mock()
        text = mock.Mock()
        with mock.patch.dict(os.environ, {
            "SENTRIX_IMAGE_EMBEDDER": "qwen3_vl",
            "SENTRIX_TEXT_EMBEDDER": "bge",
        }), mock.patch(
            "backend.embeddings.qwen3_vl_remote.Qwen3VLVisualAdapter.shared",
            return_value=visual,
        ), mock.patch(
            "backend.embeddings.bge_text.LocalTextQueryEmbedder",
            return_value=text,
        ):
            router = EmbeddingRouter.from_clip(StubClip(ready=True))

        self.assertIs(router.visual, visual)
        self.assertIs(router.text, text)

    def test_empty_router_available_false(self):
        router = EmbeddingRouter(visual=None, text=None)
        self.assertFalse(router.visual_available)
        self.assertFalse(router.text_available)


if __name__ == "__main__":
    unittest.main()
