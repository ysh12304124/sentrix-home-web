"""Phase E — QA Dashboard 只读服务。

数据模型：
  <qa_dir>/<run_id>/qa_result.json   # {"meta","summary","rows","asset_map"}
  <qa_dir>/<run_id>/run_meta.json    # {run_id, tag, created_at, note, branch, profile}
  <qa_dir>/<run_id>/review.json      # {qa_id: {verdict, note}}  人工评审（可选）

路由（全部只读，除 review 保存）：
  GET  /qa                          # Dashboard SPA 页面
  GET  /api/qa/runs                 # run 列表（meta+summary）
  GET  /api/qa/runs/{run_id}        # run 详情
  POST /api/qa/runs/upload          # benchmark 跑完后上传新 run
  GET  /api/qa/runs/{run_id}/review # 读取人工评审
  POST /api/qa/runs/{run_id}/review # 保存人工评审
"""
from __future__ import annotations

import json
import re
import shutil
from pathlib import Path

from fastapi import HTTPException
from fastapi.responses import HTMLResponse

_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_\-]{0,127}$")


def _safe_run_id(run_id: str) -> str:
    run_id = (run_id or "").strip()
    if not _SAFE_ID.match(run_id):
        raise HTTPException(status_code=400, detail="invalid run_id")
    return run_id


class QARunStore:
    def __init__(self, qa_dir: Path):
        self.qa_dir = Path(qa_dir)
        self.qa_dir.mkdir(parents=True, exist_ok=True)
        self.manifest = self._load_manifest()

    def _load_manifest(self) -> dict:
        path = self.qa_dir / "manifest.json"
        if not path.is_file():
            return {}
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return {}

    def list_runs(self) -> list[dict]:
        runs = []
        for entry in sorted(self.qa_dir.iterdir(), reverse=True):
            if not entry.is_dir():
                continue
            result_path = entry / "qa_result.json"
            meta_path = entry / "run_meta.json"
            if not result_path.is_file():
                continue
            try:
                result = json.loads(result_path.read_text(encoding="utf-8"))
                meta = {}
                if meta_path.is_file():
                    meta = json.loads(meta_path.read_text(encoding="utf-8"))
                summary = result.get("summary") or {}
                runs.append({
                    "run_id": entry.name,
                    "created_at": meta.get("created_at") or result.get("meta", {}).get("timestamp", ""),
                    "tag": meta.get("tag", entry.name),
                    "note": meta.get("note", ""),
                    "branch": meta.get("branch_153", ""),
                    "profile": meta.get("profile", ""),
                    "qa_checksum_md5": meta.get("qa_checksum_md5") or (result.get("meta") or {}).get("qa_checksum_md5", ""),
                    "manifest_mismatch": bool(
                        self.manifest.get("qa_checksum_md5")
                        and (meta.get("qa_checksum_md5") or (result.get("meta") or {}).get("qa_checksum_md5"))
                        and (meta.get("qa_checksum_md5") or (result.get("meta") or {}).get("qa_checksum_md5"))
                        != self.manifest["qa_checksum_md5"]),
                    "summary": {
                        "total": summary.get("total", 0),
                        "errored": summary.get("errored", 0),
                        "statuses": summary.get("statuses", {}),
                        "verdicts": summary.get("verdicts", {}),
                        "judged": summary.get("judged", 0),
                        "evidence_hit": summary.get("evidence_hit", 0),
                        "evidence_questions": summary.get("evidence_questions", 0),
                        "evidence_recall_avg": summary.get("evidence_recall_avg"),
                        "avg_latency_s": summary.get("avg_latency_s"),
                        "tool_usage": summary.get("tool_usage", {}),
                    },
                    "review_count": (entry / "review.json").is_file(),
                })
            except Exception:
                continue
        return runs

    def get_run(self, run_id: str) -> dict:
        run_id = _safe_run_id(run_id)
        result_path = self.qa_dir / run_id / "qa_result.json"
        if not result_path.is_file():
            raise HTTPException(status_code=404, detail="run not found")
        try:
            return json.loads(result_path.read_text(encoding="utf-8"))
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"corrupt run data: {exc}")

    def upload(self, payload: dict) -> str:
        run_id = _safe_run_id(payload.get("run_id") or "")
        meta = payload.get("meta") or {}
        summary = payload.get("summary") or {}
        rows = payload.get("rows") or []
        asset_map = payload.get("asset_map") or {}
        if not rows and not summary:
            raise HTTPException(status_code=400, detail="empty run payload")
        # G8：manifest/checksum —— 与当前 QA 数据集不一致的 run 拒绝入库
        checksum = payload.get("qa_checksum_md5") or meta.get("qa_checksum_md5") or ""
        expected = self.manifest.get("qa_checksum_md5")
        if expected and checksum and checksum != expected:
            raise HTTPException(
                status_code=409,
                detail=f"run checksum mismatch: {checksum} != manifest {expected}（QA 数据集已变更，拒绝入库）")
        run_dir = self.qa_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        result = {"meta": meta, "summary": summary, "rows": rows, "asset_map": asset_map}
        tmp = run_dir / "qa_result.json.tmp"
        tmp.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        shutil.move(str(tmp), str(run_dir / "qa_result.json"))
        run_meta = {
            "run_id": run_id,
            "tag": meta.get("tag") or payload.get("tag") or run_id,
            "created_at": meta.get("timestamp") or payload.get("created_at") or "",
            "note": payload.get("note", ""),
            "branch_153": payload.get("branch_153", ""),
            "profile": payload.get("profile", ""),
            "qa_checksum_md5": checksum,
        }
        (run_dir / "run_meta.json").write_text(
            json.dumps(run_meta, ensure_ascii=False, indent=2), encoding="utf-8")
        return run_id

    def get_review(self, run_id: str) -> dict:
        run_id = _safe_run_id(run_id)
        path = self.qa_dir / run_id / "review.json"
        if not path.is_file():
            return {"reviews": {}}
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return {"reviews": {}}

    def save_review(self, run_id: str, reviews: dict) -> dict:
        run_id = _safe_run_id(run_id)
        if not isinstance(reviews, dict):
            raise HTTPException(status_code=400, detail="reviews must be object")
        run_dir = self.qa_dir / run_id
        if not (run_dir / "qa_result.json").is_file():
            raise HTTPException(status_code=404, detail="run not found")
        cleaned = {}
        for qa_id, review in reviews.items():
            if not isinstance(review, dict):
                continue
            verdict = review.get("verdict")
            if verdict not in {"correct", "partial", "wrong", "skip"}:
                continue
            cleaned[qa_id] = {
                "verdict": verdict,
                "note": str(review.get("note") or "")[:500],
                "updated_at": _now_iso(),
            }
        (run_dir / "review.json").write_text(
            json.dumps({"reviews": cleaned}, ensure_ascii=False, indent=2), encoding="utf-8")
        return {"reviews": cleaned}


def _now_iso() -> str:
    from datetime import datetime
    return datetime.now().astimezone().isoformat(timespec="seconds")


def register_qa_routes(app, qa_dir: Path, dashboard_html: Path | None = None):
    store = QARunStore(qa_dir)

    @app.get("/qa", include_in_schema=False)
    def qa_dashboard_page():
        if dashboard_html is None or not dashboard_html.is_file():
            raise HTTPException(status_code=404, detail="qa dashboard asset not found")
        return HTMLResponse(dashboard_html.read_text(encoding="utf-8"))

    @app.get("/api/qa/runs")
    def qa_runs():
        return {"runs": store.list_runs()}

    @app.get("/api/qa/manifest")
    def qa_manifest():
        manifest = dict(store.manifest)
        expected = manifest.get("qa_checksum_md5")
        mismatched = [
            r["run_id"] for r in store.list_runs()
            if expected and r.get("qa_checksum_md5") and r["qa_checksum_md5"] != expected
        ]
        manifest["runs_mismatched"] = mismatched
        return manifest

    @app.get("/api/qa/runs/{run_id}")
    def qa_run(run_id: str):
        return store.get_run(run_id)

    @app.post("/api/qa/runs/upload")
    def qa_upload(payload: dict):
        run_id = store.upload(payload)
        return {"status": "ok", "run_id": run_id}

    @app.get("/api/qa/runs/{run_id}/review")
    def qa_review_get(run_id: str):
        return store.get_review(run_id)

    @app.post("/api/qa/runs/{run_id}/review")
    def qa_review_save(run_id: str, payload: dict):
        return store.save_review(run_id, payload.get("reviews") or {})
