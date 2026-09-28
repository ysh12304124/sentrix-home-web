import os
import sys
import tempfile
import threading
import types
import unittest
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import Mock, patch

from backend.embeddings import EmbeddingRouter
from backend.embeddings.chinese_clip_visual import ChineseClipVisualEmbedder


class ChineseClipVisualEmbedderTests(unittest.TestCase):
    def setUp(self):
        with ChineseClipVisualEmbedder._shared_lock:
            ChineseClipVisualEmbedder._shared_instances.clear()

    def tearDown(self):
        with ChineseClipVisualEmbedder._shared_lock:
            ChineseClipVisualEmbedder._shared_instances.clear()

    def test_router_reuses_process_visual_embedder(self):
        with patch.dict(os.environ, {
            "SENTRIX_IMAGE_EMBEDDER": "chinese_clip",
            "SENTRIX_TEXT_EMBEDDER": "bge",
            "CLIP_DEVICE": "cpu",
        }):
            first = EmbeddingRouter.from_clip(Mock())
            second = EmbeddingRouter.from_clip(Mock())

        self.assertIs(first.visual, second.visual)
        self.assertEqual(len(ChineseClipVisualEmbedder._shared_instances), 1)

    def test_concurrent_load_initializes_model_once(self):
        load_started = threading.Event()
        release_load = threading.Event()
        model = Mock()

        def load_from_name(*args, **kwargs):
            load_started.set()
            self.assertTrue(release_load.wait(timeout=2))
            return model, Mock()

        clip_module = types.ModuleType("cn_clip.clip")
        clip_module.load_from_name = Mock(side_effect=load_from_name)
        package_module = types.ModuleType("cn_clip")
        package_module.clip = clip_module

        with tempfile.NamedTemporaryFile() as checkpoint, patch.dict(
            sys.modules, {"cn_clip": package_module, "cn_clip.clip": clip_module}
        ):
            embedder = ChineseClipVisualEmbedder(checkpoint=checkpoint.name)
            with ThreadPoolExecutor(max_workers=4) as executor:
                futures = [executor.submit(embedder._load) for _ in range(4)]
                self.assertTrue(load_started.wait(timeout=1))
                release_load.set()
                results = [future.result(timeout=2) for future in futures]

        self.assertEqual(results, [model] * 4)
        self.assertEqual(clip_module.load_from_name.call_count, 1)
        model.eval.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
