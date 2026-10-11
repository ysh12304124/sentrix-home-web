"""
Qdrant REST API Adapter (read-only)

Wraps the Qdrant HTTP API to fetch keyframe data from ego4d_nlq_frames.
Uses only the standard library (urllib) so no extra dependency is needed.
"""

import json
import logging
import urllib.request
import urllib.error
from typing import List, Dict, Any, Optional, Iterator, Tuple

logger = logging.getLogger(__name__)


class QdrantAdapter:
    """Read-only client for the ego4d_nlq_frames collection."""

    def __init__(self, url=None, collection=None, timeout=None):
        import os
        self.base_url = (url or os.getenv("QDRANT_URL", "http://172.31.227.161:6333")).rstrip("/")
        self.collection = collection or os.getenv("QDRANT_COLLECTION", "ego4d_nlq_frames")
        # Bulk scrolls (all keyframes + visual vectors) routinely exceed 5s
        # per page; 60s + retries keeps full builds from dying mid-scroll.
        self.timeout = float(timeout if timeout is not None
                             else os.getenv("QDRANT_TIMEOUT", "60"))
        self._available = None          # cached availability check
        self._available_checked = 0.0   # timestamp of last check

    def is_available(self, force=False):
        """Quick connectivity check with caching (re-check every 60s)."""
        import time as _t
        now = _t.time()
        if self._available is not None and not force and (now - self._available_checked) < 60:
            return self._available
        try:
            req = urllib.request.Request(self.base_url + "/collections/%s" % self.collection)
            with urllib.request.urlopen(req, timeout=3) as resp:
                json.loads(resp.read())
            self._available = True
        except Exception:
            self._available = False
            logger.warning("Qdrant unavailable at %s - skipping network recall channels", self.base_url)
        self._available_checked = now
        return self._available

    def _request(self, req, retries=3):
        """Send a request with timeout-based retries (exponential backoff)."""
        import time as _t
        last_exc = None
        for attempt in range(retries):
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    return json.loads(resp.read().decode("utf-8"))
            except urllib.error.HTTPError as e:
                detail = e.read().decode("utf-8", errors="replace")
                raise RuntimeError("Qdrant %s on %s: %s" % (e.code, req.full_url, detail)) from e
            except (urllib.error.URLError, TimeoutError, OSError) as e:
                last_exc = e
                if attempt < retries - 1:
                    _t.sleep(1.5 * (2 ** attempt))
        raise RuntimeError("Qdrant request failed after %d tries (timeout=%ss): %s"
                           % (retries, self.timeout, last_exc)) from last_exc

    def _post(self, path, body):
        url = self.base_url + path
        data = json.dumps(body).encode("utf-8")
        req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
        try:
            return self._request(req)
        except RuntimeError:
            raise

    def _get(self, path):
        url = self.base_url + path
        req = urllib.request.Request(url)
        try:
            return self._request(req)
        except RuntimeError:
            raise

    def collection_info(self):
        return self._get("/collections/%s" % self.collection)["result"]

    def count_points(self, exact=True):
        return self._post("/collections/%s/points/count" % self.collection, {"exact": exact})["result"]["count"]

    def vector_config(self):
        info = self.collection_info()
        params = info.get("config", {}).get("params", {})
        return params.get("vectors", {})

    def facet(self, key, limit=1000, filter_obj=None):
        body = {"key": key, "limit": limit}
        if filter_obj:
            body["filter"] = filter_obj
        r = self._post("/collections/%s/facet" % self.collection, body)["result"]
        return [{"value": h["value"], "count": h["count"]} for h in r.get("hits", [])]

    def list_videos(self, tasks=None):
        hits = self.facet("video_uid", limit=1000,
                          filter_obj=self._with_task_filter(None, tasks))
        return sorted([(h["value"], h["count"]) for h in hits], key=lambda x: x[1])
    def list_clips(self):
        hits = self.facet("clip_uid", limit=1000)
        return sorted([(h["value"], h["count"]) for h in hits], key=lambda x: x[1])

    def _scroll(self, filter_obj, with_vector=True, page_size=256, max_points=None):
        offset = None
        yielded = 0
        while True:
            body = {"limit": page_size, "with_payload": True, "with_vector": with_vector}
            if filter_obj:
                body["filter"] = filter_obj
            if offset is not None:
                body["offset"] = offset
            r = self._post("/collections/%s/points/scroll" % self.collection, body)["result"]
            for p in r.get("points", []):
                yield p
                yielded += 1
                if max_points is not None and yielded >= max_points:
                    return
            offset = r.get("next_page_offset")
            if offset is None:
                break

    @staticmethod
    def _with_task_filter(filter_obj, tasks):
        values = [str(t).strip() for t in (tasks or []) if str(t).strip()]
        if not values:
            return filter_obj
        filt = json.loads(json.dumps(filter_obj)) if filter_obj else {}
        must = filt.setdefault("must", [])
        if len(values) == 1:
            must.append({"key": "task", "match": {"value": values[0]}})
        else:
            must.append({"key": "task", "match": {"any": values}})
        return filt

    def fetch_frames_by_video(self, video_uid, with_vector=True, tasks=None):
        filt = self._with_task_filter(
            {"must": [{"key": "video_uid", "match": {"value": video_uid}}]}, tasks)
        return list(self._scroll(filt, with_vector=with_vector))

    def fetch_frames_by_tasks(self, tasks, with_vector=True):
        """Fetch all points belonging to one or more payload task values."""
        filt = self._with_task_filter(None, tasks)
        return list(self._scroll(filt, with_vector=with_vector))

    def fetch_frames_by_clip(self, clip_uid, with_vector=True, tasks=None):
        filt = self._with_task_filter(
            {"must": [{"key": "clip_uid", "match": {"value": clip_uid}}]}, tasks)
        return list(self._scroll(filt, with_vector=with_vector))

    def fetch_frames_by_videos(self, video_uids, with_vector=True, tasks=None):
        should = [{"key": "video_uid", "match": {"value": uid}} for uid in video_uids]
        filt = self._with_task_filter({"should": should}, tasks)
        return list(self._scroll(filt, with_vector=with_vector))

    def fetch_all_frames(self, with_vector=True, page_size=256, tasks=None):
        """Scroll through every point, optionally restricted to payload tasks."""
        filt = self._with_task_filter(None, tasks)
        return list(self._scroll(filt, with_vector=with_vector, page_size=page_size))

    def fetch_frames_by_clips(self, clip_uids, with_vector=True, tasks=None):
        """Fetch frames for a list of clip uids, optionally by payload tasks."""
        should = [{"key": "clip_uid", "match": {"value": uid}} for uid in clip_uids]
        filt = self._with_task_filter({"should": should}, tasks)
        return list(self._scroll(filt, with_vector=with_vector))

    def fetch_point(self, point_id, with_vector=False):
        body = {"ids": [point_id], "with_payload": True, "with_vector": with_vector}
        r = self._post("/collections/%s/points" % self.collection, body)["result"]
        return r[0] if r else None

    def search(self, query_vector, vector_name="visual", top_k=10, filter_obj=None):
        # NumPy scalar values are not JSON serializable; always send plain floats.
        vector = [float(x) for x in query_vector]
        # Qdrant 1.x uses the query endpoint for named-vector search. The
        # legacy /points/search NamedVectorStruct is not accepted here.
        body = {
            "query": vector,
            "using": vector_name,
            "limit": top_k,
            "with_payload": True,
            "with_vector": False,
        }
        if filter_obj:
            body["filter"] = filter_obj
        result = self._post("/collections/%s/points/query" % self.collection, body)["result"]
        if isinstance(result, dict) and "points" in result:
            return result["points"]
        return result

    def search_by_point(self, point_id, vector_name="visual", top_k=6, filter_obj=None):
        query_body = {"query": point_id, "using": vector_name, "limit": top_k, "with_payload": True, "with_vector": False}
        if filter_obj:
            query_body["filter"] = filter_obj
        result = self._post("/collections/%s/points/query" % self.collection, query_body)["result"]
        # query endpoint wraps points under result["points"]
        if isinstance(result, dict) and "points" in result:
            return result["points"]
        return result

    @staticmethod
    def payload_summary(payload):
        visual_text = payload.get("visual_text") or ""
        if visual_text:
            t = payload.get("clip_time_sec", "?")
            return "t=%s | %s" % (t, visual_text[:120])
        rels = payload.get("relation_labels") or []
        objs = payload.get("object_labels") or []
        t = payload.get("clip_time_sec", "?")
        return "t=%s | objects=%s | rels=%s" % (t, objs[:5], rels[:3])
