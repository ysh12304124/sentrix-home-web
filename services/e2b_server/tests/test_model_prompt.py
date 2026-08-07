import unittest


class ModelPromptTests(unittest.TestCase):
    def test_image_messages_use_native_gemma_image_blocks(self):
        from services.e2b_server.ollama_shape import build_chat_messages

        image = object()
        messages = build_chat_messages("describe", [image])

        self.assertEqual(messages[0]["role"], "user")
        self.assertEqual(messages[0]["content"][0]["type"], "image")
        self.assertIs(messages[0]["content"][0]["image"], image)
        self.assertEqual(messages[0]["content"][1], {"type": "text", "text": "describe"})


if __name__ == "__main__":
    unittest.main()
