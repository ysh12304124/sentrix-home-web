import json
import unittest
from unittest.mock import patch
from pathlib import Path
import tempfile

from backend.model_clients import ContextBudgetExceeded, GammaClient, as_text, build_image_prompt, normalize_confidence, parse_json_response


class ModelClientTests(unittest.TestCase):
    def test_chat_messages_caps_output_to_remaining_context(self):
        client = GammaClient(base_url="http://sentrix-vllm/v1", model="test-model")
        with patch.object(client, "_tokenize_for_budget", return_value={
            "prompt_tokens": 4401, "max_model_len": 4501,
        }), patch.object(client, "_chat_openai_stream", return_value="ok") as stream:
            result = client.chat_messages(
                [{"role": "user", "content": "test"}],
                role="tool_loop", max_tokens=384,
            )

        self.assertEqual(result, "ok")
        payload = stream.call_args.args[1]
        self.assertEqual(payload["max_tokens"], 100)
        self.assertEqual(stream.call_args.kwargs["budget_metrics"]["estimated_total_tokens"], 4501)

    def test_chat_messages_blocks_prompt_that_fills_context(self):
        client = GammaClient(base_url="http://sentrix-vllm/v1", model="test-model")
        with patch.object(client, "_tokenize_for_budget", return_value={
            "prompt_tokens": 4501, "max_model_len": 4501,
        }), patch.object(client, "_chat_openai_stream") as stream:
            with self.assertRaises(ContextBudgetExceeded):
                client.chat_messages(
                    [{"role": "user", "content": "test"}],
                    role="tool_loop", max_tokens=384,
                )

        stream.assert_not_called()
        metrics = client.get_and_clear_call_metrics()
        self.assertEqual(metrics[0]["status"], "context_budget_exceeded")
        self.assertEqual(metrics[0]["prompt_tokens"], 4501)
        self.assertEqual(metrics[0]["estimated_total_tokens"], 4885)

    def test_token_budget_uses_runtime_bound_manager(self):
        client = GammaClient(
            base_url="http://sentrix-vllm/v1",
            model="test-model",
            manager_url="http://manager-8500/",
        )
        self.assertEqual(client.manager_url, "http://manager-8500")

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

    def test_image_prompt_keeps_reverse_geocode_out_of_visual_place_classification(self):
        prompt = build_image_prompt({"location_context": {"label": "测试省测试市测试区"}})

        self.assertIn("测试省测试市测试区", prompt)
        self.assertIn("place 和 semantic.place.primary 必须只依据图片视觉证据", prompt)
        self.assertIn("不能用 GPS 或地点上下文覆盖", prompt)

    @patch("backend.model_clients.httpx.post")
    def test_gamma_default_uses_vllm_openai_chat_completions(self, post):
        post.return_value.raise_for_status.return_value = None
        post.return_value.json.return_value = {"choices": [{"message": {"content": "{}"}}]}

        GammaClient(base_url="http://sentrix-vllm", model="gemma4-12b-it").chat("测试", role="parser")

        self.assertEqual(post.call_args.args[0], "http://sentrix-vllm/v1/chat/completions")
        payload = post.call_args.kwargs["json"]
        self.assertEqual(payload["model"], "gemma4-12b-it")
        self.assertEqual(payload["max_tokens"], 512)
        self.assertEqual(payload["response_format"], {"type": "json_object"})
        self.assertNotIn("keep_alive", payload)

    @patch("backend.model_clients.httpx.post")
    def test_gamma_vllm_multimodal_request_uses_openai_image_content(self, post):
        post.return_value.raise_for_status.return_value = None
        post.return_value.json.return_value = {"choices": [{"message": {"content": "{}"}}]}

        GammaClient(base_url="http://sentrix-vllm/v1", model="gemma4-e2b-it").chat(
            "看图", [{"base64": "image", "mime_type": "image/jpeg"}],
            {"think": False, "num_ctx": 4096, "num_predict": 320},
        )

        payload = post.call_args.kwargs["json"]
        content = payload["messages"][0]["content"]
        self.assertEqual(content[0], {"type": "text", "text": "看图"})
        self.assertEqual(content[1]["type"], "image_url")
        self.assertEqual(content[1]["image_url"]["url"], "data:image/jpeg;base64,image")
        self.assertEqual(payload["max_tokens"], 320)

    @patch("backend.model_clients.httpx.post")
    def test_gamma_request_uses_configured_model_keep_alive(self, post):
        post.return_value.raise_for_status.return_value = None
        post.return_value.json.return_value = {"message": {"content": "{}"}}

        GammaClient(base_url="http://sentrix-ollama", model="gemma4:12b", backend="ollama").chat("测试")

        self.assertEqual(post.call_args.kwargs["json"]["keep_alive"], "0")

    @patch("backend.model_clients.httpx.post")
    def test_gamma_request_allows_explicit_batch_keep_alive(self, post):
        post.return_value.raise_for_status.return_value = None
        post.return_value.json.return_value = {"message": {"content": "{}"}}

        GammaClient(base_url="http://sentrix-ollama", keep_alive="15m", backend="ollama").chat("测试")

        self.assertEqual(post.call_args.kwargs["json"]["keep_alive"], "15m")

    @patch.dict("os.environ", {"OLLAMA_KEEP_ALIVE": "-1"}, clear=False)
    @patch("backend.model_clients.httpx.post")
    def test_indefinite_keep_alive_is_sent_as_numeric_minus_one(self, post):
        post.return_value.raise_for_status.return_value = None
        post.return_value.json.return_value = {"message": {"content": "{}"}}

        GammaClient(base_url="http://sentrix-ollama", backend="ollama").chat("测试")

        self.assertEqual(post.call_args.kwargs["json"]["keep_alive"], -1)

    @patch("backend.model_clients.httpx.post")
    def test_core_vision_options_disable_thinking_and_bound_generation(self, post):
        post.return_value.raise_for_status.return_value = None
        post.return_value.json.return_value = {"message": {"content": "{}"}}

        GammaClient(base_url="http://sentrix-ollama", backend="ollama").chat(
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

    def test_event_summary_uses_semantic_places_and_removes_coordinates(self):
        client = GammaClient()
        response = {
            "title": "在坐标30.2458,120.2989吃饭",
            "event_type": "餐饮",
            "activity": "在30.2458,120.2989用餐",
            "summary": "在坐标30.2458,120.2989处拍摄了多张餐厅照片。",
            "confidence": 0.8,
        }
        with patch.object(client, "chat", return_value=json.dumps(response)) as chat:
            result = client.summarize_event(
                {"time_start": "2026-07-01T18:00:00+08:00", "time_end": "2026-07-01T18:30:00+08:00", "place": "30.2458,120.2989"},
                [{"id": "obs_1", "place": "餐厅", "caption": "桌上有蛋糕", "activity": "用餐", "objects": ["蛋糕"], "canonical": {"semantic": {"place": {"primary": "餐饮空间"}}}}],
            )

        prompt = chat.call_args.args[0]
        self.assertIn("餐厅", prompt)
        self.assertNotRegex(result["summary"], r"30\.2458|120\.2989|坐标")
        self.assertIn("餐厅", result["summary"])


if __name__ == "__main__":
    unittest.main()
