import json
import unittest
from unittest.mock import patch
from pathlib import Path
import tempfile

from backend.model_clients import GammaClient, as_text, normalize_confidence, parse_json_response


class ModelClientTests(unittest.TestCase):
    def test_parses_json_inside_markdown_fence(self):
        result = parse_json_response('```json\n{"caption":"公园"}\n```')
        self.assertEqual(result["caption"], "公园")

    def test_invalid_model_output_is_empty_object(self):
        self.assertEqual(parse_json_response("无法确定"), {})

    def test_model_scalar_fields_can_be_written_to_sqlite(self):
        self.assertEqual(as_text(["厨房", "客厅"]), "厨房、客厅")
        self.assertEqual(as_text({"name": "厨房"}), '{"name": "厨房"}')

    def test_normalizes_chinese_confidence_labels_from_model_output(self):
        self.assertEqual(normalize_confidence("高", 0.5), 0.85)
        self.assertEqual(normalize_confidence("中等", 0.5), 0.6)
        self.assertEqual(normalize_confidence("80%", 0.5), 0.8)
        self.assertEqual(normalize_confidence("无效值", 0.65), 0.65)

    @patch("backend.model_clients.httpx.post")
    def test_gamma_request_uses_configured_model_keep_alive(self, post):
        post.return_value.raise_for_status.return_value = None
        post.return_value.json.return_value = {"message": {"content": "{}"}}

        GammaClient(base_url="http://sentrix-ollama", model="gemma4:12b").chat("测试")

        self.assertEqual(post.call_args.kwargs["json"]["keep_alive"], "0")

    @patch("backend.model_clients.httpx.post")
    def test_gamma_request_allows_explicit_batch_keep_alive(self, post):
        post.return_value.raise_for_status.return_value = None
        post.return_value.json.return_value = {"message": {"content": "{}"}}

        GammaClient(base_url="http://sentrix-ollama", keep_alive="15m").chat("测试")

        self.assertEqual(post.call_args.kwargs["json"]["keep_alive"], "15m")

    @patch("backend.model_clients.httpx.post")
    def test_core_vision_options_disable_thinking_and_bound_generation(self, post):
        post.return_value.raise_for_status.return_value = None
        post.return_value.json.return_value = {"message": {"content": "{}"}}

        GammaClient(base_url="http://sentrix-ollama").chat(
            "测试", [{"base64": "image", "mime_type": "image/jpeg"}],
            {"think": False, "num_ctx": 4096, "num_predict": 320},
        )

        payload = post.call_args.kwargs["json"]
        self.assertFalse(payload["think"])
        self.assertEqual(payload["options"]["num_ctx"], 4096)
        self.assertEqual(payload["options"]["num_predict"], 320)

    def test_image_prompt_uses_approved_place_taxonomy(self):
        client = GammaClient()
        with tempfile.NamedTemporaryFile(suffix=".jpg") as image:
            image.write(b"synthetic-image")
            image.flush()
            with patch.object(client, "chat", return_value=json.dumps({"scene_type": "餐饮空间"})) as chat:
                client.analyze_image(image.name)

        prompt = chat.call_args_list[0].args[0]
        self.assertIn("医疗与公共服务", prompt)
        self.assertIn("农场与乡村", prompt)
        self.assertIn("semantic", prompt)
        self.assertNotIn("居住室内", prompt)

    def test_image_analysis_returns_normalized_semantic_contract(self):
        client = GammaClient()
        payload = {
            "place": "湖边餐厅",
            "scene_type": "餐饮空间",
            "semantic": {
                "place": {"primary": "餐饮空间", "details": ["室内"]},
                "objects": [{"primary": "食品与饮品", "label": "蛋糕", "details": ["桌面"]}],
                "atmosphere": {"labels": ["温馨"], "details": ["暖色光线"]},
            },
        }
        with tempfile.NamedTemporaryFile(suffix=".jpg") as image:
            image.write(b"synthetic-image")
            image.flush()
            with patch.object(client, "chat", return_value=json.dumps(payload)):
                result = client.analyze_image(image.name)

        self.assertEqual(result["semantic"]["place"]["primary"], "餐饮空间")
        self.assertEqual(result["semantic"]["objects"][0]["label"], "蛋糕")
        self.assertEqual(result["semantic"]["atmosphere"]["labels"], ["温馨"])
        self.assertEqual(result["raw_labels"]["place"], "湖边餐厅")

    def test_image_analysis_recovers_descriptive_observation_when_first_response_only_has_semantics(self):
        client = GammaClient()
        first = {
            "semantic": {
                "place": {"primary": "居住空间", "details": ["室内"]},
                "objects": [],
                "atmosphere": {"labels": [], "details": []},
            },
        }
        recovery = {
            "caption": "孩子在客厅玩耍",
            "activity": "玩耍",
            "place": "家中客厅",
            "event_type": "日常活动",
            "people": ["孩子"],
            "objects": ["玩具"],
            "ocr_text": "",
        }
        with tempfile.NamedTemporaryFile(suffix=".jpg") as image:
            image.write(b"synthetic-image")
            image.flush()
            with patch.object(client, "chat", side_effect=[json.dumps(first), json.dumps(recovery)]) as chat:
                result = client.analyze_image(image.name)

        self.assertEqual(result["caption"], "孩子在客厅玩耍")
        self.assertEqual(result["activity"], "玩耍")
        self.assertEqual(result["place"], "家中客厅")
        self.assertEqual(result["semantic"]["objects"][0]["label"], "玩具")
        self.assertEqual(chat.call_count, 2)

    def test_clip_uses_project_checkpoint_when_environment_is_unset(self):
        checkpoint = Path(__file__).resolve().parents[2] / "data" / "models" / "clip" / "ViT-B-32.bin"
        with patch.dict("os.environ", {"CLIP_CHECKPOINT": ""}, clear=False), patch.object(Path, "is_file", autospec=True) as is_file:
            is_file.side_effect = lambda path: path == checkpoint
            adapter = __import__("backend.model_clients", fromlist=["ClipAdapter"]).ClipAdapter()

        self.assertEqual(adapter.checkpoint, str(checkpoint))

    def test_person_appearance_analysis_returns_only_target_clothing(self):
        with tempfile.NamedTemporaryFile(suffix=".jpg") as image:
            image.write(b"synthetic-crop")
            image.flush()
            client = GammaClient()
            with patch.object(client, "chat", return_value='{"clothing":["红色针织衫"],"confidence":0.88}') as chat:
                result = client.analyze_person_appearance(
                    image.name,
                    {"target_face_bbox": [20, 10, 60, 50], "face_instance_id": "face_1"},
                )

        self.assertEqual(result["clothing"], ["红色针织衫"])
        self.assertEqual(result["confidence"], 0.88)
        self.assertIn("目标人物", chat.call_args.args[0])


if __name__ == "__main__":
    unittest.main()
