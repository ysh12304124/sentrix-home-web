"""Phase R R2 — EmbeddingRouter composition and availability (P0-3)."""

import unittest

from backend.embeddings import EmbeddingRouter


class StubClip:
    model_name = "ViT-B-32"

    def __init__(self, ready=True):
        self.evidence_ready = ready

    def embed_text(self, text):
        return [1.0, 0.0]


class EmbeddingRouterTests(unittest.TestCase):
    def test_from_clip_builds_both_slots(self):
        router = EmbeddingRouter.from_clip(StubClip(ready=True))
        self.assertTrue(router.visual_available)
        self.assertTrue(router.text_available)
        self.assertEqual(router.embed_visual("A"), [1.0, 0.0])
        self.assertEqual(router.embed_text("A"), [1.0, 0.0])

    def test_unready_clip_disables_slots(self):
        router = EmbeddingRouter.from_clip(StubClip(ready=False))
        self.assertFalse(router.visual_available)
        self.assertFalse(router.text_available)
        self.assertEqual(router.embed_visual("A"), [])

    def test_empty_router_available_false(self):
        router = EmbeddingRouter(visual=None, text=None)
        self.assertFalse(router.visual_available)
        self.assertFalse(router.text_available)


if __name__ == "__main__":
    unittest.main()
