"""Tests for GET/POST /api/vlm-backend endpoints."""

import unittest
from unittest.mock import MagicMock, patch

# Mock heavy model dependencies before importing app
with patch("backend.model_clients.FunASRClient"),      patch("backend.model_clients.FaceAdapter"),      patch("backend.model_clients.ClipAdapter"),      patch("backend.model_clients.GammaClient") as mock_gc_cls,      patch("backend.pipeline.IngestionPipeline"),      patch("backend.agent.MemoryAgent"):

    # module-level gamma is now a MagicMock from mock_gc_cls()
    import backend.app
    from backend.app import app

    # Configure a proper MagicMock instance
    mock_gamma = MagicMock()
    mock_gamma.active_name = "ollama_12b"
    mock_gamma.model = "gemma4:12b"
    mock_gamma.base_url = "http://127.0.0.1:11434"
    backend.app.gamma = mock_gamma


class VLMBackendAPITests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from fastapi.testclient import TestClient
        cls.client = TestClient(app)
        cls._mock_gamma = backend.app.gamma

    def setUp(self):
        # Reset mock state
        self._mock_gamma.active_name = "ollama_12b"
        self._mock_gamma.model = "gemma4:12b"
        self._mock_gamma.base_url = "http://127.0.0.1:11434"
        self._mock_gamma.reset_mock()

    def test_get_vlm_backend_default(self):
        self._mock_gamma.active_name = "ollama_12b"
        response = self.client.get("/api/vlm-backend")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("available_backends", data)
        self.assertEqual(data["backend"], "ollama_12b")

    def test_post_switch_to_e2b(self):
        self._mock_gamma.active_name = "e2b_lora"
        self._mock_gamma.model = "gemma-4-e2b-it+lora-v2"
        self._mock_gamma.base_url = "http://127.0.0.1:8100"
        response = self.client.post("/api/vlm-backend", json={"backend": "e2b_lora"})
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["backend"], "e2b_lora")
        self.assertEqual(data["model"], "gemma-4-e2b-it+lora-v2")

    def test_post_rejects_invalid_backend(self):
        response = self.client.post("/api/vlm-backend", json={"backend": "invalid"})
        self.assertEqual(response.status_code, 422)

    def test_post_switch_to_ollama(self):
        self._mock_gamma.active_name = "ollama_12b"
        response = self.client.post("/api/vlm-backend", json={"backend": "ollama_12b"})
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["backend"], "ollama_12b")

    def test_health_includes_vlm_key(self):
        with patch("backend.app._current_model_runtime", return_value={"backend": "vllm", "model": "test"}):
            response = self.client.get("/api/health")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("vlm", data["models"])
        self.assertEqual(data["models"]["vlm"]["active"], "ollama_12b")

    def test_get_with_e2b_active(self):
        self._mock_gamma.active_name = "e2b_lora"
        self._mock_gamma.model = "gemma-4-e2b-it+lora-v2"
        self._mock_gamma.base_url = "http://127.0.0.1:8100"
        response = self.client.get("/api/vlm-backend")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["backend"], "e2b_lora")

    def test_post_invalidates_cache(self):
        self._mock_gamma.active_name = "e2b_lora"
        self._mock_gamma.model = "gemma-4-e2b-it+lora-v2"
        self._mock_gamma.base_url = "http://127.0.0.1:8100"
        response = self.client.post("/api/vlm-backend", json={"backend": "e2b_lora"})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["backend"], "e2b_lora")


if __name__ == "__main__":
    unittest.main()
