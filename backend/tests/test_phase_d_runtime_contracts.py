"""Phase D — answer_grounding 契约 / termination_reason / ResultSet 持久化。"""
import os
import tempfile
import unittest

from backend.agent_runtime import tools as runtime_tools
from backend.agent_runtime.result_set import ResultSet, ResultSetStore, TaskState
from backend.agent_runtime.runtime import _build_answer_grounding, _classify_termination, RuntimeTurn
from backend.db import MemoryStore


def _task(**kw):
    return TaskState(**kw)


class AnswerGroundingTest(unittest.TestCase):
    def test_chat_no_evidence(self):
        g = _build_answer_grounding(message="你好，谢谢", task=_task())
        self.assertEqual(g["display_mode"], "none")

    def test_explicit_image_request_grid(self):
        task = _task(current_result_set="rs_x", result_preview=["photo_1", "photo_2"],
                     result_total=2)
        g = _build_answer_grounding(message="把照片给我看看", task=task)
        self.assertEqual(g["display_mode"], "result_grid")

    def test_implicit_evidence_collapsed(self):
        task = _task(current_result_set="rs_x", result_preview=["photo_1"], result_total=1)
        g = _build_answer_grounding(message="去年去过哪里", task=task)
        self.assertEqual(g["display_mode"], "collapsed")

    def test_inline_question_selected_photo(self):
        task = _task(current_result_set="rs_x", result_preview=["photo_1"], result_total=1)
        g = _build_answer_grounding(message="这张照片里有几个人？", task=task, selected_handle="photo_1")
        self.assertEqual(g["display_mode"], "inline_images")


class TerminationReasonTest(unittest.TestCase):
    def _turn(self, status, reason=""):
        t = RuntimeTurn(profile="tool_loop", budget=None)
        t.status = status
        t.reason = reason
        return t

    def test_coverage(self):
        cases = [
            (self._turn("complete"), "complete"),
            (self._turn("blocked_by_guard", "group_fabrication"), "guard_recovery_exhausted"),
            (self._turn("error", "unparseable_action"), "parse_failure"),
            (self._turn("partial", "model step budget exhausted"), "model_step_limit"),
            (self._turn("timeout", ""), "wall_time_limit"),
            (self._turn("error", "unknown_tool:foo"), "tool_unavailable"),
            (self._turn("partial", "tool_denied:inspect_photo:budget exhausted"), "tool_rejected"),
        ]
        for turn, expected in cases:
            self.assertEqual(_classify_termination(turn), expected, turn.reason)


class TaskStateRestoreTest(unittest.TestCase):
    """D12：跨轮续接必须恢复结果集预览，否则显式要图无法展示证据网格。"""

    def test_from_dict_restores_result_preview(self):
        from backend.agent_runtime.result_set import TaskState
        task = TaskState.from_dict(
            {"current_result_set": "rs_x", "result_preview": ["photo_1", "photo_2"],
             "result_total": 2, "result_remaining": 0},
            user_goal="把照片给我看看")
        self.assertEqual(task.result_preview, ["photo_1", "photo_2"])
        self.assertEqual(task.result_total, 2)
        g = _build_answer_grounding(message="把刚才的照片给我看看", task=task)
        self.assertEqual(g["display_mode"], "result_grid")


class ResultSetPersistenceTest(unittest.TestCase):
    def test_db_restore_after_memory_clear(self):
        d = tempfile.mkdtemp()
        store = MemoryStore(os.path.join(d, "test.db"))
        rss = ResultSetStore(store)
        rs = rss.new(scope_id="album3-v2", query="沙雕", asset_ids=["a1", "a2", "a3"])
        rss._memory.clear()
        restored = rss.get(rs.result_set_id)
        self.assertIsNotNone(restored)
        self.assertEqual(restored.asset_ids, ["a1", "a2", "a3"])


if __name__ == "__main__":
    unittest.main()
