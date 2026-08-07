import unittest
from unittest.mock import patch

from backend.model_clients import E2BBackend, GammaClient, OllamaBackend


class VLMRouterTests(unittest.TestCase):
    def setUp(self):
        self.gamma = GammaClient(backend="ollama")

    def test_ollama_backend_name(self):
        backend = OllamaBackend(
            base_url="http://127.0.0.1:11434",
            model="gemma4:12b",
            timeout=180,
            keep_alive="0",
        )
        self.assertEqual(backend.name, "ollama_12b")
        self.assertEqual(backend.model_name, "gemma4:12b")

    def test_ollama_backend_endpoint(self):
        backend = OllamaBackend(
            base_url="http://127.0.0.1:11434",
            model="test",
            timeout=60,
            keep_alive="5m",
        )
        self.assertEqual(backend.endpoint, "http://127.0.0.1:11434")

    def test_bind_store_sets_reference(self):
        fake_store = object()
        self.gamma.bind_store(fake_store)
        self.assertIs(self.gamma._store, fake_store)

    def test_invalidate_backend_cache(self):
        self.gamma.invalidate_backend_cache()
        self.assertIsNone(self.gamma._active_cache)

    def test_read_active_name_falls_back_to_default(self):
        self.gamma._store = None
        name = self.gamma._read_active_name()
        self.assertEqual(name, "ollama_12b")

    def test_read_active_name_from_store(self):
        class FakeStore:
            def get_setting(self, key, default=None):
                return "e2b_lora"
        self.gamma._store = FakeStore()
        name = self.gamma._read_active_name()
        self.assertEqual(name, "e2b_lora")

    def test_read_active_name_unknown_falls_back(self):
        class FakeStore:
            def get_setting(self, key, default=None):
                return None
        self.gamma._store = FakeStore()
        name = self.gamma._read_active_name()
        self.assertEqual(name, "ollama_12b")

    def test_active_name_property(self):
        class FakeStore:
            def get_setting(self, key, default=None):
                return "e2b_lora"
        self.gamma._store = FakeStore()
        self.gamma.invalidate_backend_cache()
        self.assertEqual(self.gamma.active_name, "e2b_lora")

    def test_active_returns_ollama_by_default(self):
        self.gamma._store = None
        self.gamma.invalidate_backend_cache()
        active = self.gamma._active()
        self.assertIsInstance(active, OllamaBackend)
        self.assertEqual(active.name, "ollama_12b")

    def test_active_returns_e2b_when_configured(self):
        class FakeStore:
            def get_setting(self, key, default=None):
                return "e2b_lora"
        self.gamma._store = FakeStore()
        self.gamma.invalidate_backend_cache()
        active = self.gamma._active()
        self.assertIsInstance(active, E2BBackend)
        self.assertEqual(active.name, "e2b_lora")

    def test_active_cache_is_reused(self):
        class FakeStore:
            call_count = 0
            def get_setting(self, key, default=None):
                FakeStore.call_count += 1
                return "ollama_12b"
        self.gamma._store = FakeStore()
        self.gamma.invalidate_backend_cache()
        first = self.gamma._active()
        second = self.gamma._active()
        self.assertIs(first, second)
        self.assertEqual(FakeStore.call_count, 1)

    def test_embed_text_hard_pinned_to_ollama(self):
        self.gamma._store = None
        self.gamma.invalidate_backend_cache()
        with patch.object(OllamaBackend, "embed_text", return_value=[0.1, 0.2]) as mock:
            result = self.gamma.embed_text("test")
        self.assertEqual(result, [0.1, 0.2])
        mock.assert_called_once()

    def test_base_url_property_ollama_default(self):
        self.gamma._store = None
        self.gamma.invalidate_backend_cache()
        url = self.gamma.base_url
        self.assertIn("11434", url)

    def test_model_property_ollama_default(self):
        self.gamma._store = None
        self.gamma.invalidate_backend_cache()
        model = self.gamma.model
        self.assertEqual(model, "gemma4:12b")


if __name__ == "__main__":
    unittest.main()
