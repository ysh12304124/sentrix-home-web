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
        self._ensure_schema()

    def _ensure_schema(self):
        try:
            self.store.connection.executescript(
                """CREATE TABLE IF NOT EXISTS agent_conversations (
                    conversation_id TEXT PRIMARY KEY,
                    scope_id TEXT NOT NULL DEFAULT 'home-default',
                    title TEXT NOT NULL DEFAULT '新对话',
                    summary TEXT NOT NULL DEFAULT '',
                    state TEXT NOT NULL DEFAULT 'active',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    last_message_at TEXT
                );
                CREATE INDEX IF NOT EXISTS idx_agent_conversations_scope
                    ON agent_conversations(scope_id, updated_at);
                CREATE VIRTUAL TABLE IF NOT EXISTS agent_conversation_messages_fts USING fts5(
                    id UNINDEXED, conversation_id UNINDEXED, role UNINDEXED,
                    content, tokenize='unicode61');"""
            )
            self.store.connection.commit()
        except Exception:
            # FTS5 不可用时静默降级（LIKE 检索仍可用）
            pass

    # ---- conversation lifecycle (D2) ----
    def create_conversation(self, scope_id="home-default", title=None):
        from .db import make_id
        cid = make_id("conversation")
        now = now_iso()
        self.store.connection.execute(
            """INSERT INTO agent_conversations
               (conversation_id, scope_id, title, state, created_at, updated_at)
               VALUES (?, ?, ?, 'active', ?, ?)""",
            (cid, scope_id or "home-default", (title or "").strip() or "新对话", now, now))
        self.store.connection.commit()
        return cid

    def list_conversations(self, scope_id=None, limit=50):
        params = []
        clauses = ["state = 'active'"]
        if scope_id:
            clauses.append("scope_id = ?")
            params.append(scope_id)
        params.append(int(limit))
        rows = self.store.connection.execute(
            f"""SELECT conversation_id, scope_id, title, summary, state, created_at,
                       updated_at, last_message_at
                FROM agent_conversations WHERE {' AND '.join(clauses)}
                ORDER BY COALESCE(last_message_at, updated_at) DESC LIMIT ?""",
            params).fetchall()
        return [dict(r) for r in rows]

    def get_conversation(self, conversation_id):
        row = self.store.connection.execute(
            """SELECT conversation_id, scope_id, title, summary, state, created_at,
                      updated_at, last_message_at
               FROM agent_conversations WHERE conversation_id = ?""",
            (conversation_id,)).fetchone()
        if not row or row["state"] == "deleted":
            return None
        return dict(row)

    def rename_conversation(self, conversation_id, title):
        title = (title or "").strip()
        if not title:
            return None
        self.store.connection.execute(
            "UPDATE agent_conversations SET title = ?, updated_at = ? WHERE conversation_id = ?",
            (title, now_iso(), conversation_id))
        self.store.connection.commit()
        return self.get_conversation(conversation_id)

    def touch_conversation(self, conversation_id):
        """更新会话的 updated_at / last_message_at（每次 turn 后调用）。"""
        self.store.connection.execute(
            """UPDATE agent_conversations
               SET updated_at = ?, last_message_at = ? WHERE conversation_id = ?""",
            (now_iso(), now_iso(), conversation_id))
        self.store.connection.commit()

    @staticmethod
    def auto_title(message: str) -> str:
        """从第一条用户消息生成标题（确定性截断，不额外调用模型）。"""
        text = (message or "").strip().replace("\n", " ")
        if not text:
            return "新对话"
        return text[:24] + ("…" if len(text) > 24 else "")

    def ensure_title(self, conversation_id, message: str):
        row = self.store.connection.execute(
            "SELECT title FROM agent_conversations WHERE conversation_id = ?",
            (conversation_id,)).fetchone()
        if row and row["title"] in {"新对话", ""}:
            self.rename_conversation(conversation_id, self.auto_title(message))

    def save_summary(self, conversation_id, summary: str):
        self.store.connection.execute(
            "UPDATE agent_conversations SET summary = ?, updated_at = ? WHERE conversation_id = ?",
            (summary or "", now_iso(), conversation_id))
        self.store.connection.commit()

    def get_summary(self, conversation_id) -> str:
        row = self.store.connection.execute(
            "SELECT summary FROM agent_conversations WHERE conversation_id = ?",
            (conversation_id,)).fetchone()
        return (row["summary"] if row else "") or ""

    def delete_conversation(self, conversation_id):
        """软删除会话：仅清除聊天历史/轨迹/摘要/FTS，不触碰家庭长期记忆与原始 Asset。"""
        now = now_iso()
        self.store.connection.execute(
            "UPDATE agent_conversations SET state = 'deleted', updated_at = ? WHERE conversation_id = ?",
            (now, conversation_id))
        self.store.connection.execute(
            "DELETE FROM agent_conversation_messages WHERE conversation_id = ?",
            (conversation_id,))
        self.store.connection.execute(
            "DELETE FROM agent_trajectories WHERE conversation_id = ?",
            (conversation_id,))
        try:
            self.store.connection.execute(
                "DELETE FROM agent_conversation_messages_fts WHERE conversation_id = ?",
                (conversation_id,))
        except Exception:
            pass
        self.store.connection.commit()
        return True

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
        try:
            text = ""
            if isinstance(content, dict):
                text = str(content.get("text") or content.get("content") or "")
            elif isinstance(content, str):
                text = content
            if text.strip():
                self.store.connection.execute(
                    "INSERT INTO agent_conversation_messages_fts(id, conversation_id, role, content) VALUES (?, ?, ?, ?)",
                    (mid, conversation_id, role or "", text))
        except Exception:
            pass
        if commit:
            self.store.connection.commit()
        return mid

    def search_messages(self, query, *, conversation_id=None, scope_id=None,
                        limit=10, exclude_deleted=True):
        """D4：跨会话对话检索（FTS5 优先，LIKE 降级）。"""
        query = (query or "").strip()
        if not query:
            return []
        rows = []
        try:
            params = [query]
            sql = ("SELECT m.id, m.turn_id, m.role, m.content_json, m.created_at "
                   "FROM agent_conversation_messages_fts fts "
                   "JOIN agent_conversation_messages m ON m.id = fts.id "
                   "WHERE agent_conversation_messages_fts MATCH ?")
            if conversation_id:
                sql += " AND fts.conversation_id = ?"
                params.append(conversation_id)
            if scope_id:
                sql += " AND m.scope_id = ?"
                params.append(scope_id)
            if exclude_deleted:
                sql += (" AND EXISTS (SELECT 1 FROM agent_conversations c "
                        "WHERE c.conversation_id = m.conversation_id AND c.state = 'active')")
            sql += " ORDER BY m.created_at DESC LIMIT ?"
            params.append(int(limit))
            rows = self.store.connection.execute(sql, params).fetchall()
        except Exception:
            rows = []
        if not rows:
            # LIKE 降级检索
            like = f"%{query}%"
            params = [like]
            sql = ("SELECT id, turn_id, role, content_json, created_at "
                   "FROM agent_conversation_messages WHERE content_json LIKE ?")
            if conversation_id:
                sql += " AND conversation_id = ?"
                params.append(conversation_id)
            if scope_id:
                sql += " AND scope_id = ?"
                params.append(scope_id)
            if exclude_deleted:
                sql += (" AND EXISTS (SELECT 1 FROM agent_conversations c "
                        "WHERE c.conversation_id = agent_conversation_messages.conversation_id "
                        "AND c.state = 'active')")
            sql += " ORDER BY created_at DESC LIMIT ?"
            params.append(int(limit))
            rows = self.store.connection.execute(sql, params).fetchall()
        return [self._decode(r) for r in rows]

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
