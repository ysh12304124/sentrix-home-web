import hashlib
import sqlite3, json, uuid
import networkx as nx
try:
    import ujson as _json
except ImportError:
    _json = json
import threading
from datetime import datetime
import numpy as np
from pathlib import Path
from .graph_db import NetworkXGraphDB, EventNode, EpisodeNode, SessionNode, Link, NodeType, LinkType

class LazyLinkStore:
    """A lazy, thread-safe Link mapping backed by SQLite.

    Full-scale graphs contain millions of edges.  Materializing every Link
    object at startup dominates cold-start time, while a query only touches a
    small neighborhood.  This store keeps the count in memory, caches Link
    objects on demand, and still supports len/get/items for existing code.
    """

    def __init__(self, conn):
        self._conn = conn
        self._cache = {}
        self._lock = threading.RLock()
        self._count = int(conn.execute(
            "SELECT COUNT(*) FROM graph_edges").fetchone()[0])

    def __len__(self):
        return self._count

    def __contains__(self, link_id):
        with self._lock:
            if link_id in self._cache:
                return True
            row = self._conn.execute(
                "SELECT 1 FROM graph_edges WHERE id=?", (link_id,)).fetchone()
            return row is not None

    def __getitem__(self, link_id):
        with self._lock:
            if link_id in self._cache:
                return self._cache[link_id]
            row = self._conn.execute(
                "SELECT data FROM graph_edges WHERE id=?", (link_id,)).fetchone()
            if row is None:
                raise KeyError(link_id)
            link = Link.from_dict(_json.loads(row[0]))
            self._cache[link_id] = link
            return link

    def get(self, link_id, default=None):
        try:
            return self[link_id]
        except KeyError:
            return default

    def get_many(self, link_ids):
        """Batch-load Link objects, caching them to avoid per-edge SQL queries."""
        ids = [lid for lid in link_ids if lid not in self._cache]
        if ids:
            chunk_size = 400
            with self._lock:
                for start in range(0, len(ids), chunk_size):
                    part = ids[start:start + chunk_size]
                    placeholders = ",".join("?" * len(part))
                    rows = self._conn.execute(
                        "SELECT id, data FROM graph_edges WHERE id IN (%s)"
                        % placeholders, part).fetchall()
                    for row_id, data in rows:
                        self._cache[row_id] = Link.from_dict(_json.loads(data))
        return [self._cache[lid] for lid in link_ids if lid in self._cache]

    def __iter__(self):
        with self._lock:
            for row in self._conn.execute("SELECT id FROM graph_edges"):
                yield row[0]

    def keys(self):
        return iter(self)

    def items(self):
        for link_id in self:
            yield link_id, self[link_id]

    def values(self):
        for _, link in self.items():
            yield link

    def clear(self):
        with self._lock:
            self._cache.clear()
            self._count = 0


class LazySQLiteMultiDiGraph:
    """Adjacency view that queries SQLite indexes on demand.

    Startup no longer scans 4.5M edge rows.  A retrieval query visits at most
    a few hundred nodes, and each out/in adjacency lookup is an indexed SQLite
    query.  This trades a tiny per-query cost for a much faster cold start.
    """

    def __init__(self, conn, node_ids=None):
        self._conn = conn
        self._lock = threading.RLock()
        self._node_ids = set(node_ids or ())
        self._local_out = {}
        self._local_in = {}

    def add_node(self, node_id, **attrs):
        with self._lock:
            self._node_ids.add(node_id)

    def add_edge(self, source, target, key=None, **attrs):
        with self._lock:
            self._node_ids.add(source)
            self._node_ids.add(target)
            self._local_out.setdefault(source, []).append((target, key))
            self._local_in.setdefault(target, []).append((source, key))

    def __contains__(self, node_id):
        return node_id in self._node_ids

    def __len__(self):
        return len(self._node_ids)

    @staticmethod
    def _rows(source, rows, keys_flag, data_flag, incoming=False):
        for target, key in rows:
            other = target
            if data_flag:
                yield (other, source, key, {}) if incoming else (source, other, key, {})
            elif keys_flag:
                yield (other, source, key) if incoming else (source, other, key)
            else:
                yield (other, source) if incoming else (source, other)

    def out_edges(self, node_id, keys=False, data=False):
        with self._lock:
            rows = self._conn.execute(
                "SELECT target_id, id FROM graph_edges WHERE source_id=?",
                (node_id,)).fetchall()
            rows.extend(self._local_out.get(node_id, ()))
        return self._rows(node_id, rows, keys, data)

    def in_edges(self, node_id, keys=False, data=False):
        with self._lock:
            rows = self._conn.execute(
                "SELECT source_id, id FROM graph_edges WHERE target_id=?",
                (node_id,)).fetchall()
            rows.extend(self._local_in.get(node_id, ()))
        return self._rows(node_id, rows, keys, data, incoming=True)

    def out_edges_bounded(self, node_id, limit):
        """Return at most ``limit`` outgoing (target_id, link_id) rows."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT target_id, id FROM graph_edges WHERE source_id=? LIMIT ?",
                (node_id, limit)).fetchall()
            local = self._local_out.get(node_id, ())
            if local:
                seen = {r[0] for r in rows}
                for target_id, key in local:
                    if target_id not in seen and len(rows) < limit:
                        rows.append((target_id, key))
        return list(rows)

    def in_edges_bounded(self, node_id, limit):
        """Return at most ``limit`` incoming (source_id, link_id) rows."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT source_id, id FROM graph_edges WHERE target_id=? LIMIT ?",
                (node_id, limit)).fetchall()
            local = self._local_in.get(node_id, ())
            if local:
                seen = {r[0] for r in rows}
                for source_id, key in local:
                    if source_id not in seen and len(rows) < limit:
                        rows.append((source_id, key))
        return list(rows)

    def clear(self):
        with self._lock:
            self._node_ids.clear()
            self._local_out.clear()
            self._local_in.clear()


class SQLiteGraphDB(NetworkXGraphDB):
    def __init__(self, db_path="memory.db"):
        super().__init__()
        self.db_path = str(db_path)
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._lock = threading.Lock()
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self._init_tables()

    def _init_tables(self):
        self.conn.executescript("""
            CREATE TABLE IF NOT EXISTS graph_nodes (id TEXT PRIMARY KEY, data TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS graph_edges (id TEXT PRIMARY KEY, source_id TEXT NOT NULL, target_id TEXT NOT NULL, data TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
            CREATE INDEX IF NOT EXISTS idx_ge_source ON graph_edges(source_id);
            CREATE INDEX IF NOT EXISTS idx_ge_target ON graph_edges(target_id);
        """)
        self.conn.commit()


    def save(self, filepath=None):
        """Persist graph rows in one transaction using batched executemany.

        Event embedding vectors are intentionally not serialized: Qdrant is
        the vector source of truth and SQLiteGraphDB.load() already strips
        them on load.  Removing them cuts both JSON serialization time and
        the SQLite graph size substantially.
        """
        c = self.conn
        c.execute("DELETE FROM graph_nodes")
        c.execute("DELETE FROM graph_edges")

        def node_rows():
            for nid, node in self.nodes.items():
                data = node.to_dict()
                data.pop("embedding_vector", None)
                yield nid, json.dumps(data, default=str, ensure_ascii=False)

        def edge_rows():
            for lid, link in self.links.items():
                yield (lid, link.source_node_id, link.target_node_id,
                       _json.dumps(link.to_dict(), default=str, ensure_ascii=False))

        c.executemany("INSERT OR REPLACE INTO graph_nodes VALUES (?,?)", node_rows())
        c.executemany(
            "INSERT OR REPLACE INTO graph_edges VALUES (?,?,?,?)", edge_rows())
        c.commit()

    def load(self, filepath=None, lazy_links=True):
        """Load runtime graph state without scanning all edge rows.

        ``lazy_links=True`` loads nodes and uses SQLite source/target indexes
        for adjacency.  Link JSON is parsed only when traversal touches it.
        Pass ``lazy_links=False`` for tools that require all Link objects.
        """
        import time as _time
        started = _time.perf_counter()
        c = self.conn
        self.graph.clear(); self.nodes.clear(); self.links.clear(); self.node_to_links.clear()
        self.links = LazyLinkStore(c) if lazy_links else {}

        node_sql = (
            "SELECT id, data FROM graph_nodes"
            if lazy_links else
            "SELECT id, data FROM graph_nodes"
        )
        for row in c.execute(node_sql):
            data = _json.loads(row[1])
            data.pop("embedding_vector", None)
            ntype = data.get("node_type", "EVENT")
            if ntype == "EPISODE": node = EpisodeNode.from_dict(data)
            elif ntype == "SESSION": node = SessionNode.from_dict(data)
            else: node = EventNode.from_dict(data)
            self.nodes[node.node_id] = node
            if not lazy_links:
                self.graph.add_node(node.node_id)
                self.node_to_links[node.node_id] = set()

        if lazy_links:
            self.graph = LazySQLiteMultiDiGraph(c, self.nodes.keys())
        else:
            self.graph = nx.MultiDiGraph()
            for node_id in self.nodes:
                self.graph.add_node(node_id)

            for row in c.execute(
                    "SELECT id, source_id, target_id, data FROM graph_edges"):
                link = Link.from_dict(_json.loads(row[3]))
                self.links[row[0]] = link
                self.graph.add_edge(row[1], row[2], key=row[0])
                self.node_to_links.setdefault(row[1], set()).add(row[0])
                self.node_to_links.setdefault(row[2], set()).add(row[0])

        print("SQLite graph loaded: %d nodes, %d links in %.1fs (%s)" % (
            len(self.nodes), len(self.links), _time.perf_counter() - started,
            "lazy adjacency" if lazy_links else "full links"), flush=True)

    def save_keyword_index(self, node_index):
        data = {k: list(v) for k, v in node_index.items()}
        self.conn.execute("INSERT OR REPLACE INTO metadata VALUES (?,?)", ("keyword_index", json.dumps(data, ensure_ascii=False)))
        self.conn.commit()

    def load_keyword_index(self):
        row = self.conn.execute("SELECT value FROM metadata WHERE key='keyword_index'").fetchone()
        if row: return {k: set(v) for k, v in json.loads(row[0]).items()}
        return {}

    def save_json_metadata(self, key, value):
        self.conn.execute(
            "INSERT OR REPLACE INTO metadata VALUES (?,?)",
            (key, json.dumps(value, ensure_ascii=False, default=str)))
        self.conn.commit()

    def load_json_metadata(self, key):
        row = self.conn.execute(
            "SELECT value FROM metadata WHERE key=?", (key,)).fetchone()
        return json.loads(row[0]) if row else None

    def save_vectors(self, vector_db):
        """持久化向量库到 SQLite（替代 index.faiss + metadata.json）"""
        c = self.conn
        c.execute("DELETE FROM vector_entries")
        for vid, entry in vector_db.entries.items():
            vec_bytes = entry.vector.astype(np.float32).tobytes()
            meta = json.dumps(entry.metadata, ensure_ascii=False, default=str)
            c.execute("INSERT INTO vector_entries VALUES (?,?,?)", (vid, vec_bytes, meta))
        c.commit()
        return len(vector_db.entries)

    def load_vectors(self, vector_db):
        """从 SQLite 恢复向量库"""
        rows = self.conn.execute("SELECT id, vector, metadata FROM vector_entries").fetchall()
        entries = []
        for vid, vec_bytes, meta in rows:
            vec = np.frombuffer(vec_bytes, dtype=np.float32)
            entries.append((vid, vec, json.loads(meta)))
        if entries:
            vector_db.add_vectors(entries)
        return len(entries)

    def save_keyframes(self, session_id, keyframes, video_id="default"):
        with self._lock:
            c = self.conn
            if session_id:
                c.execute("DELETE FROM keyframes WHERE session_id=? AND video_id=?", (session_id, video_id))
            for kf in keyframes:
                eid = str(uuid.uuid4())
                c.execute(
                    "INSERT INTO keyframes "
                    "(id, session_id, time_sec, identity, emotion, emotion_score, image, "
                    " face_image, frame_idx, kf_type, video_id, caption, embedding) "
                    "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (
                        eid, session_id or "", kf.get("time_sec"),
                        kf.get("identity"), kf.get("emotion"), kf.get("emotion_score"),
                        kf.get("image"), kf.get("face_image"), kf.get("frame_idx"),
                        kf.get("kf_type") or "emotion", video_id,
                        kf.get("caption"), kf.get("embedding"),
                    ),
                )
            c.commit()
            return len(keyframes)

    def query_keyframes(self, session_id=None, time_sec=None, kf_type=None,
                        video_id=None, window_seconds=30, limit=3):
        """按时间窗 + 类型查询关键帧，返回 dict 列表，按离 time_sec 最近排序。"""
        with self._lock:
            sql = "SELECT * FROM keyframes WHERE 1=1"
            params = []
            if session_id:
                sql += " AND session_id=?"
                params.append(session_id)
            if video_id:
                sql += " AND video_id=?"
                params.append(video_id)
            if kf_type:
                sql += " AND kf_type=?"
                params.append(kf_type)
            if time_sec is not None:
                sql += " AND time_sec BETWEEN ? AND ?"
                params.extend([time_sec - window_seconds, time_sec + window_seconds])
            sql += " ORDER BY ABS(time_sec - ?) LIMIT ?"
            params.extend([time_sec if time_sec is not None else 0, limit])
            rows = self.conn.execute(sql, params).fetchall()
            return [dict(row) for row in rows]

    def query_emotion_at_time(self, session_id=None, time_sec=None):
        with self._lock:
            if time_sec is None:
                return None
            row = None
            if session_id:
                row = self.conn.execute(
                    "SELECT * FROM keyframes WHERE session_id=? AND kf_type='emotion' "
                    "AND time_sec <= ? ORDER BY time_sec DESC LIMIT 1",
                    (session_id, time_sec),
                ).fetchone()
            if not row:
                row = self.conn.execute(
                    "SELECT * FROM keyframes WHERE kf_type='emotion' "
                    "AND time_sec <= ? ORDER BY time_sec DESC LIMIT 1",
                    (time_sec,),
                ).fetchone()
            if row:
                return {key: row[key] for key in row.keys()}
            return None

    def query_emotion_for_turn(self, session_id=None, time_start=None, time_end=None,
                               identity=None, identity_window_seconds=40,
                               fallback_window_seconds=5):
        """按“轮次窗口 + 说话人身份”查找表情关键帧。

        1) 轮次窗口内同身份最近帧；
        2) 无则取同身份最近前帧（含 1 秒前向容忍，兼容关键帧略晚于轮次起点）；
        3) 再无则退回按时间窗最近帧（兼容关键帧没有身份信息的测试数据）。
        """
        if time_start is None:
            return None
        time_end = time_end if time_end is not None else time_start
        with self._lock:
            def _pick(where, params):
                if session_id:
                    where = "session_id=? AND " + where
                    params = [session_id] + list(params)
                sql = ("SELECT * FROM keyframes WHERE kf_type='emotion' AND "
                       + where + " ORDER BY ABS(time_sec - ?) LIMIT 1")
                row = self.conn.execute(sql, list(params) + [time_start]).fetchone()
                return dict(row) if row else None

            if identity:
                row = _pick(
                    "identity=? AND time_sec BETWEEN ? AND ?",
                    [identity, time_start, time_end],
                )
                if row:
                    return row
                row = _pick(
                    "identity=? AND time_sec BETWEEN ? AND ?",
                    [identity, time_start - identity_window_seconds, time_start + 1],
                )
                if row:
                    return row

            row = _pick(
                "time_sec BETWEEN ? AND ?",
                [time_start - fallback_window_seconds, time_start + fallback_window_seconds],
            )
            if row:
                return row
            row = _pick(
                "time_sec BETWEEN ? AND ?",
                [time_start, time_start + 2 * fallback_window_seconds],
            )
            if row:
                return row
            row = _pick(
                "time_sec BETWEEN ? AND ?",
                [time_start - 2 * fallback_window_seconds, time_start],
            )
            if row:
                return row
            return None

    def get_visual_cache(self, question, time_sec=None):
        with self._lock:
            qh = hashlib.md5(question.encode("utf-8")).hexdigest()
            row = self.conn.execute(
                "SELECT answer, confidence FROM visual_evidence_cache "
                "WHERE question_hash=? AND time_sec IS ? "
                "ORDER BY confidence DESC LIMIT 1",
                (qh, time_sec),
            ).fetchone()
            if row:
                return {"answer": row[0], "confidence": row[1]}
            return None

    def save_visual_cache(self, question, keyframe_id, answer, confidence, time_sec=None):
        with self._lock:
            qh = hashlib.md5(question.encode("utf-8")).hexdigest()
            self.conn.execute(
                "INSERT OR REPLACE INTO visual_evidence_cache "
                "VALUES (?,?,?,?,?,?,?)",
                (qh, keyframe_id, question, answer, confidence, time_sec,
                 datetime.now().isoformat()),
            )
            self.conn.commit()

    def close(self):
        self.conn.close()
