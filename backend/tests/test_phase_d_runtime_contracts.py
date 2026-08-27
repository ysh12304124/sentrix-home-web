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
        # 证据来自 search_memories（找图唯一入口）；聚合工具 query_memory_facts 不再贡献可见图。
        task = _task(current_result_set="rs_x", result_preview=["photo_1", "photo_2"],
                     result_total=2, tool_results=[{"tool": "search_memories", "evidence_asset_ids": ["asset_1"], "retrieved_asset_ids": ["asset_1"]}])
        g = _build_answer_grounding(message="把照片给我看看", task=task)
        self.assertEqual(g["display_mode"], "result_grid")

    def test_implicit_evidence_collapsed(self):
        # 同上：证据必须来自 search；仅聚合工具的来源图不进入模型可见证据。
        task = _task(current_result_set="rs_x", result_preview=["photo_1"], result_total=1, tool_results=[{"tool": "search_memories", "evidence_asset_ids": ["asset_1"], "retrieved_asset_ids": ["asset_1"]}])
        g = _build_answer_grounding(message="去年去过哪里", task=task)
        self.assertEqual(g["display_mode"], "collapsed")

    def test_facts_do_not_inject_visible_images(self):
        # 聚合工具（query_memory_facts）即使带 source_asset_ids，也不注入模型可见证据。
        task = _task(current_result_set="rs_x", result_preview=["photo_1"], result_total=1,
                     tool_results=[{"tool": "query_memory_facts", "evidence_asset_ids": ["asset_1", "asset_2"], "source_asset_ids": ["asset_1", "asset_2"]}])
        g = _build_answer_grounding(message="国庆一共多少张", task=task)
        self.assertEqual(g["display_mode"], "none")

    def test_inline_question_selected_photo(self):
        task = _task(current_result_set="rs_x", result_preview=["photo_1"], result_total=1, tool_results=[{"tool": "inspect_photo", "asset_id": "asset_1", "inspect_handle": "photo_1", "inspect_text": "两个人"}])
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
        task.tool_results = [{"tool": "search_memories", "evidence_asset_ids": ["asset_1"], "retrieved_asset_ids": ["asset_1"]}]
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
