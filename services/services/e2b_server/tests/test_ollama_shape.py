import unittest

from services.e2b_server.ollama_shape import (
    JSON_TRAILING_HINT,
    build_chat_response,
    build_generate_response,
    extract_prompt_and_images,
    map_options,
)


class ExtractPromptAndImagesTests(unittest.TestCase):
    def test_prompt_from_string_content(self):
        prompt, images = extract_prompt_and_images({
            "messages": [{"role": "user", "content": "hello"}],
        })
        self.assertEqual(prompt, "hello")
        self.assertEqual(len(images), 0)

    def test_prompt_from_mixed_content_list(self):
        prompt, images = extract_prompt_and_images({
            "messages": [{"role": "user", "content": [
                {"type": "text", "text": "describe"},
                {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,AAAA"}},
            ]}],
        })
        self.assertEqual(prompt, "describe")
        self.assertEqual(images, ["AAAA"])

    def test_image_list_property(self):
        prompt, images = extract_prompt_and_images({
            "messages": [{"role": "user", "content": "look", "images": ["img1", "img2"]}],
        })
        self.assertEqual(prompt, "look")
        self.assertEqual(images, ["img1", "img2"])

    def test_text_and_images_combined(self):
        prompt, images = extract_prompt_and_images({
            "messages": [
                {"role": "user", "content": "first"},
                {"role": "user", "content": "second", "images": ["imgA"]},
            ],
        })
        self.assertIn("first", prompt)
        self.assertIn("second", prompt)
        self.assertEqual(images, ["imgA"])

    def test_empty_messages(self):
        prompt, images = extract_prompt_and_images({"messages": []})
        self.assertEqual(prompt, "")
        self.assertEqual(images, [])

    def test_data_uri_fallback(self):
        prompt, images = extract_prompt_and_images({
            "messages": [{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": "data:;base64,BBBB"}},
            ]}],
        })
        self.assertEqual(images, ["BBBB"])


class MapOptionsTests(unittest.TestCase):
    def test_none_options(self):
        self.assertEqual(map_options(None), {})

    def test_empty_options(self):
        self.assertEqual(map_options({}), {})

    def test_num_predict(self):
        result = map_options({"num_predict": 512})
        self.assertEqual(result["max_new_tokens"], 512)

    def test_temperature_zero_disables_sample(self):
        result = map_options({"temperature": 0})
        self.assertEqual(result["temperature"], 0.0)
        self.assertFalse(result["do_sample"])

    def test_temperature_nonzero_enables_sample(self):
        result = map_options({"temperature": 0.7})
        self.assertEqual(result["temperature"], 0.7)
        self.assertTrue(result["do_sample"])


class BuildResponseTests(unittest.TestCase):
    def test_chat_response(self):
        resp = build_chat_response("test-model", "hello world")
        self.assertEqual(resp["model"], "test-model")
        self.assertIn("role", resp["message"])
        self.assertEqual(resp["message"]["content"], "hello world")
        self.assertTrue(resp["done"])

    def test_generate_response(self):
        resp = build_generate_response("test-model", "generated text")
        self.assertEqual(resp["model"], "test-model")
        self.assertEqual(resp["response"], "generated text")
        self.assertTrue(resp["done"])


class TrailingHintTests(unittest.TestCase):
    def test_hint_contains_json_only(self):
        self.assertIn("JSON", JSON_TRAILING_HINT)
        self.assertIn("代码块", JSON_TRAILING_HINT)


if __name__ == "__main__":
    unittest.main()
