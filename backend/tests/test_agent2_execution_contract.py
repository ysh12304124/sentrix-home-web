import unittest

from backend.agent_runtime.result_set import ResultSet
from backend.agent_runtime.runtime import _agent2_answer_context_ready, _next_resolution_handle


class _Requirement:
    def __init__(self, required=True):
        self.required = required


class _State:
    def __init__(self, status, required=True, attempts=1):
        self.status = status
        self.requirement = _Requirement(required)
        self.attempt_count = attempts


class _Task:
    def __init__(self, preview=None, tool_results=None, requirements=None):
        self.result_preview = preview or []
        self.tool_results = tool_results or []
        self.requirements = requirements or {}


class Agent2ExecutionContractTests(unittest.TestCase):
    def test_result_set_public_handles_are_contiguous_after_selection(self):
        result_set = ResultSet(
            result_set_id="rs_test", scope_id="scope", query="q",
            asset_ids=["a3", "a4", "a5", "a6", "a10", "a12"],
        )
        view = result_set.public_view(["a3", "a4", "a5", "a6", "a10", "a12"])
        self.assertEqual(view.preview(), [
            {"handle": "photo_1", "asset_id": "a3"},
            {"handle": "photo_2", "asset_id": "a4"},
            {"handle": "photo_3", "asset_id": "a5"},
            {"handle": "photo_4", "asset_id": "a6"},
            {"handle": "photo_5", "asset_id": "a10"},
            {"handle": "photo_6", "asset_id": "a12"},
        ])

    def test_next_ocr_handle_skips_previously_attempted_photo(self):
        task = _Task(
            preview=["photo_1", "photo_2", "photo_3"],
            tool_results=[{"tool": "read_photo_text", "inspect_handle": "photo_1"}],
        )
        self.assertEqual(_next_resolution_handle(task, "read_photo_text"), "photo_2")

    def test_writer_waits_until_every_required_requirement_is_terminal(self):
        task = _Task(requirements={"req_1": _State("satisfied"), "req_2": _State("open", attempts=0)})
        self.assertFalse(_agent2_answer_context_ready(task, {"facts": [{"value": "2017"}]}))
        task.requirements["req_2"].status = "running"
        task.requirements["req_2"].attempt_count = 1
        self.assertFalse(_agent2_answer_context_ready(task, {"facts": [{"value": "2017"}]}))

    def test_writer_accepts_attempted_unknown_requirement(self):
        task = _Task(requirements={"req_1": _State("satisfied"), "req_2": _State("blocked", attempts=1)})
        self.assertTrue(_agent2_answer_context_ready(task, {"facts": [{"value": "2017"}]}))


if __name__ == "__main__":
    unittest.main()
