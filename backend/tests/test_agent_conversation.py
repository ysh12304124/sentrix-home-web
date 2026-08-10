"""A1: ConversationStore 与 trajectory 持久化测试。"""
import json
import tempfile

from backend.agent_conversation import ConversationStore
from backend.db import MemoryStore


def _store(tmp_path):
    return MemoryStore(str(tmp_path / "test.db"))


def test_add_and_list_messages(tmp_path):
    store = _store(tmp_path)
    conv = ConversationStore(store)
    conv.add_message("c1", "user", {"text": "你好"}, scope_id="home-default", turn_id="t1")
    conv.add_message("c1", "assistant", {"text": "你好呀"}, scope_id="home-default", turn_id="t1")
    msgs = conv.list_messages("c1")
    assert [m["role"] for m in msgs] == ["user", "assistant"]
    assert msgs[0]["content"]["text"] == "你好"
    assert msgs[0]["turn_id"] == "t1"


def test_last_messages_order(tmp_path):
    store = _store(tmp_path)
    conv = ConversationStore(store)
    for i in range(5):
        conv.add_message("c1", "user", {"text": f"m{i}"}, turn_id=f"t{i}")
    last = conv.last_messages("c1", limit=3)
    assert [m["content"]["text"] for m in last] == ["m2", "m3", "m4"]


def test_trajectory_roundtrip(tmp_path):
    store = _store(tmp_path)
    conv = ConversationStore(store)
    conv.save_trajectory(
        "turn_x", "c1", profile="pipeline",
        steps=[{"stage": "retrieval", "status": "complete"}],
        result={"answer": "ok"}, public_progress=[{"text": "已找到 3 条相关记录。"}],
        scope_id="album1",
    )
    traj = conv.get_trajectory("turn_x")
    assert traj["profile"] == "pipeline"
    assert traj["steps"][0]["stage"] == "retrieval"
    assert traj["public_progress"][0]["text"] == "已找到 3 条相关记录。"
    listed = conv.list_trajectories("c1")
    assert len(listed) == 1 and listed[0]["turn_id"] == "turn_x"


def test_bootstrap_recent(tmp_path):
    store = _store(tmp_path)
    conv = ConversationStore(store)
    for i in range(4):
        conv.add_message("old", "user", {"text": f"old{i}"}, turn_id=f"t{i}")
    recent = conv.bootstrap_recent("old", limit=2)
    assert [m["content"]["text"] for m in recent] == ["old2", "old3"]
