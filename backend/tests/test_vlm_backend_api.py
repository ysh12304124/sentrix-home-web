"""Tests for the retired /api/vlm-backend compatibility endpoint."""

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

    @patch("backend.app._current_model_runtime", return_value={
        "profile": "gemma4-12b-it", "model": "gemma4-12b-it", "status": "running",
    })
    def test_get_vlm_backend_reports_managed_vllm(self, _runtime):
        response = self.client.get("/api/vlm-backend")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["backend"], "vllm")
        self.assertEqual(data["available_backends"], ["vllm"])
        self.assertEqual(data["profile"], "gemma4-12b-it")
        self.assertTrue(data["deprecated"])

    def test_post_switch_to_e2b_is_gone(self):
        response = self.client.post("/api/vlm-backend", json={"backend": "e2b_lora"})
        self.assertEqual(response.status_code, 410)
        self.assertIn("/api/model-profiles/switch", response.json()["detail"])

    def test_post_rejects_invalid_backend_as_gone(self):
        response = self.client.post("/api/vlm-backend", json={"backend": "invalid"})
        self.assertEqual(response.status_code, 410)

    def test_post_switch_to_ollama_is_gone(self):
        response = self.client.post("/api/vlm-backend", json={"backend": "ollama_12b"})
        self.assertEqual(response.status_code, 410)

    def test_health_includes_vlm_key(self):
        with patch("backend.app._current_model_runtime", return_value={"backend": "vllm", "model": "test"}):
            response = self.client.get("/api/health")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("vlm", data["models"])
        self.assertEqual(data["models"]["vlm"]["active"], "vllm")

    def test_post_does_not_mutate_legacy_setting(self):
        with patch.object(backend.app.store, "set_setting") as set_setting:
            response = self.client.post("/api/vlm-backend", json={"backend": "e2b_lora"})
        self.assertEqual(response.status_code, 410)
        set_setting.assert_not_called()
        self._mock_gamma.invalidate_backend_cache.assert_not_called()


if __name__ == "__main__":
    unittest.main()
