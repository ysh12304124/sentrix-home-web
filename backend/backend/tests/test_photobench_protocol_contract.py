import inspect
import unittest

from backend import app
from backend.agent_runtime import ocr_tool, runtime, tool_policy


class PhotoBenchProtocolContractTests(unittest.TestCase):
    def test_turn_response_keeps_rich_observability_fields(self):
        source = inspect.getsource(app._tool_loop_turn)
        for field in (
            '"model_call_metrics"', '"call_type"', '"step_id"',
            '"parent_step_id"', '"call_observation"', '"turn_outcome"',
            '"parse_status"', '"next_step"',
        ):
            self.assertIn(field, source)

    def test_tool_internal_calls_keep_parent_binding(self):
        source = inspect.getsource(app._tool_loop_turn)
        self.assertIn('metric["call_type"] = "tool_internal"', source)
        self.assertIn('metric["parent_step_id"] = step_id', source)

    def test_extracted_ocr_keeps_internal_model_metrics(self):
        source = inspect.getsource(ocr_tool)
        self.assertIn("gamma.get_and_clear_call_metrics()", source)
        self.assertIn('metric["tool_subtask"] = label', source)
        self.assertIn('cached_result.pop("_model_call_metrics", None)', source)

    def test_tool_results_keep_image_ids_and_private_metrics(self):
        allowed = tool_policy.ToolPolicy._TOOL_ALLOWED["search_memories"]
        self.assertIn("asset_ids", allowed)
        self.assertIn("_model_call_metrics", tool_policy.ToolPolicy._DEFAULT_ALLOWED)

    def test_runtime_keeps_qwen_message_order_fix(self):
        source = inspect.getsource(runtime)
        self.assertIn("_merge_system_constraint", source)
        self.assertNotIn('messages.append({"role": "system", "content": ctext})', source)

    def test_app_keeps_runtime_binding_and_cancel_routes(self):
        paths = {route.path for route in app.app.routes}
        self.assertIn("/api/model-profiles/bind-runtime", paths)
        self.assertIn("/api/model-profiles/bind-external-runtime", paths)
        self.assertIn("/api/runtime-providers", paths)
        self.assertIn("/api/relationships/batch", paths)


if __name__ == "__main__":
    unittest.main()
