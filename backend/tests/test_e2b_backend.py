import unittest
from unittest.mock import patch

from backend.model_clients import E2BBackend, ModelError


class E2BBackendTests(unittest.TestCase):
    def setUp(self):
        self.backend = E2BBackend(base_url="http://127.0.0.1:8100", timeout=30)

    def test_default_name_is_e2b_lora(self):
        self.assertEqual(self.backend.name, "e2b_lora")

    def test_default_model_name(self):
        self.assertEqual(self.backend.model_name, "gemma-4-e2b-it+lora-v2")

    def test_endpoint_property(self):
        self.assertEqual(self.backend.endpoint, "http://127.0.0.1:8100")

    def test_default_base_url_from_env(self):
        with patch.dict("os.environ", {"E2B_BASE_URL": "http://e2b:8200"}, clear=False):
            b = E2BBackend()
            self.assertEqual(b.endpoint, "http://e2b:8200")

    def test_default_base_url_fallback(self):
        with patch.dict("os.environ", {}, clear=True):
            b = E2BBackend()
        self.assertEqual(b.endpoint, "http://127.0.0.1:8101")

    @patch("backend.model_clients.httpx.post")
    def test_chat_sends_ollama_shape_payload(self, post):
        post.return_value.raise_for_status.return_value = None
        post.return_value.json.return_value = {"message": {"content": "response text"}}

        result = self.backend.chat("hello", images=None, json_mode=True)

        self.assertEqual(result, "response text")
        post.assert_called_once()
        call_url = post.call_args.args[0]
        self.assertIn("/api/chat", call_url)
        payload = post.call_args.kwargs["json"]
        self.assertEqual(payload["model"], "gemma-4-e2b-it+lora-v2")
        self.assertEqual(payload["messages"][0]["content"], "hello")
        self.assertEqual(payload["format"], "json")

    @patch("backend.model_clients.httpx.post")
    def test_chat_with_images(self, post):
        post.return_value.raise_for_status.return_value = None
        post.return_value.json.return_value = {"message": {"content": "described"}}

        result = self.backend.chat(
            "what is this",
            images=[{"base64": "AAAA", "mime_type": "image/jpeg"}],
            json_mode=False,
        )

        self.assertEqual(result, "described")
        payload = post.call_args.kwargs["json"]
        self.assertIn("AAAA", payload["messages"][0]["images"])

    @patch("backend.model_clients.httpx.post")
    def test_chat_with_vision_options(self, post):
        post.return_value.raise_for_status.return_value = None
        post.return_value.json.return_value = {"message": {"content": "{}"}}

        result = self.backend.chat(
            "analyze",
            images=[{"base64": "BBBB"}],
            vision_options={"num_ctx": 4096, "num_predict": 320, "think": False},
            json_mode=True,
        )

        self.assertEqual(result, "{}")
        payload = post.call_args.kwargs["json"]
        self.assertFalse(payload["think"])
        self.assertEqual(payload["options"]["num_ctx"], 4096)
        self.assertEqual(payload["options"]["num_predict"], 320)

    @patch("backend.model_clients.httpx.post")
    def test_chat_role_routing(self, post):
        post.return_value.raise_for_status.return_value = None
        post.return_value.json.return_value = {"message": {"content": "{}"}}

        result = self.backend.chat("parse this", role="parser", json_mode=True)

        self.assertEqual(result, "{}")
        payload = post.call_args.kwargs["json"]
        self.assertEqual(payload["options"]["temperature"], 0)
        self.assertFalse(payload["think"])
        self.assertEqual(payload["options"]["num_ctx"], 4096)
        self.assertEqual(payload["options"]["num_predict"], 512)

    @patch("backend.model_clients.httpx.post")
    def test_chat_http_error_wraps_model_error(self, post):
        post.side_effect = __import__("httpx").HTTPError("connection refused")

        with self.assertRaises(ModelError):
            self.backend.chat("hello")

    def test_embed_text_raises_not_implemented(self):
        with self.assertRaises(NotImplementedError):
            self.backend.embed_text("some text")

    @patch("backend.model_clients.httpx.get")
    def test_health_returns_parsed_json(self, get):
        get.return_value.raise_for_status.return_value = None
        get.return_value.json.return_value = {"status": "ok", "loaded": True}

        result = self.backend.health()
        self.assertEqual(result["status"], "ok")
        self.assertTrue(result["loaded"])
        get.assert_called_once_with("http://127.0.0.1:8100/api/health", timeout=10)

    @patch("backend.model_clients.httpx.get")
    def test_health_error_returns_empty_dict(self, get):
        get.side_effect = Exception("timeout")

        result = self.backend.health()
        self.assertEqual(result, {})


if __name__ == "__main__":
    unittest.main()
