"""Agent Runtime v2 — ConversationStore & Trajectory (A1).

服务端 canonical conversation history + 可复现的 trajectory。
- ``agent_conversation_messages``：用户/助手/工具/观察/UI 事件的线性历史。
- ``agent_trajectories``：每个 turn 的 model/tool/observation 步骤与结果。

``dialogue_states`` 保留为派生状态（active entity / current result set），
不再作为 conversation history 的唯一真相。

本模块只做持久化，不改变 thin_agent 的回答逻辑。
"""

from __future__ import annotations

import json

from .db import now_iso


class ConversationStore:
    """Server-side canonical conversation history + trajectory store."""

    def __init__(self, store):
        self.store = store

    # ---- messages ----
    def add_message(self, conversation_id, role, content, *, scope_id="home-default",
                    turn_id=None, commit=True):
        from .db import make_id
        mid = make_id("msg")
        self.store.connection.execute(
            """INSERT INTO agent_conversation_messages
               (id, conversation_id, scope_id, turn_id, role, content_json, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (mid, conversation_id, scope_id or "home-default", turn_id, role,
             json.dumps(content, ensure_ascii=False), now_iso()),
        )
        if commit:
            self.store.connection.commit()
        return mid

    def list_messages(self, conversation_id, limit=40, after=None):
        params = [conversation_id]
        sql = ("SELECT id, turn_id, role, content_json, created_at "
               "FROM agent_conversation_messages WHERE conversation_id = ?")
        if after:
            sql += " AND created_at > ?"
            params.append(after)
        sql += " ORDER BY created_at ASC LIMIT ?"
        params.append(int(limit))
        rows = self.store.connection.execute(sql, params).fetchall()
        return [self._decode(row) for row in rows]

    def last_messages(self, conversation_id, limit=10):
        rows = self.store.connection.execute(
            """SELECT id, turn_id, role, content_json, created_at
               FROM agent_conversation_messages WHERE conversation_id = ?
               ORDER BY created_at DESC LIMIT ?""",
            (conversation_id, int(limit)),
        ).fetchall()
        return [self._decode(row) for row in reversed(rows)]

    def bootstrap_recent(self, conversation_id, scope_id=None, limit=10):
        """旧会话兼容：无本地历史时从服务端拉最近消息（不迁移前端历史）。"""
        return self.last_messages(conversation_id, limit=limit)

    # ---- trajectory ----
    def save_trajectory(self, turn_id, conversation_id, profile, steps, result,
                        public_progress, *, scope_id="home-default", commit=True):
        self.store.connection.execute(
            """INSERT INTO agent_trajectories
               (turn_id, conversation_id, scope_id, profile, steps_json, result_json,
                public_progress_json, created_at, updated_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(turn_id) DO UPDATE SET
                 steps_json = excluded.steps_json,
                 result_json = excluded.result_json,
                 public_progress_json = excluded.public_progress_json,
                 updated_at = excluded.updated_at""",
            (turn_id, conversation_id, scope_id or "home-default", profile or "tool_loop",
             json.dumps(steps, ensure_ascii=False),
             json.dumps(result, ensure_ascii=False),
             json.dumps(public_progress, ensure_ascii=False),
             now_iso(), now_iso()),
        )
        if commit:
            self.store.connection.commit()

    def get_trajectory(self, turn_id):
        row = self.store.connection.execute(
            "SELECT * FROM agent_trajectories WHERE turn_id = ?", (turn_id,)).fetchone()
        if not row:
            return None
        return {
            "turn_id": row["turn_id"], "conversation_id": row["conversation_id"],
            "scope_id": row["scope_id"], "profile": row["profile"],
            "steps": json.loads(row["steps_json"] or "[]"),
            "result": json.loads(row["result_json"] or "{}"),
            "public_progress": json.loads(row["public_progress_json"] or "[]"),
            "created_at": row["created_at"], "updated_at": row["updated_at"],
        }

    def list_trajectories(self, conversation_id, limit=20):
        rows = self.store.connection.execute(
            """SELECT turn_id, conversation_id, scope_id, profile, created_at
               FROM agent_trajectories WHERE conversation_id = ?
               ORDER BY created_at DESC LIMIT ?""",
            (conversation_id, int(limit)),
        ).fetchall()
        return [dict(row) for row in rows]

    # ---- helpers ----
    def _decode(self, row):
        try:
            content = json.loads(row["content_json"] or "{}")
        except (TypeError, json.JSONDecodeError):
            content = {}
        return {
            "id": row["id"], "turn_id": row["turn_id"], "role": row["role"],
            "content": content, "created_at": row["created_at"],
        }
