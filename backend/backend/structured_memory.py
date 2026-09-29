"""StructuredMemoryExecutor — deterministic structured answers (TFPE2-3, first slice).

Answers time / count / exists / first / last / list / group-by queries directly
from the SQLite tables (assets.captured_at, assets.media_type,
assets.captured_location, observations.place, entity_mentions), never from ANN
Top-K.  Scope/viewer authorization is enforced by the scope filter exactly like
the retrieval kernel's ``list_assets(scope_id=...)`` path.

Everything here is executed, not judged: the 12B parser decided answer_type /
strategy / structured slots; this module only validates and runs the query.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta

from .query_contracts import QuerySpec, parse_time_expression

_GROUP_EXPR = {
    "month": "substr(a.captured_at, 1, 7)",
    "year": "substr(a.captured_at, 1, 4)",
    "date": "substr(a.captured_at, 1, 10)",
    "media": "a.media_type",
    "place": (
        "CASE WHEN TRIM(COALESCE(json_extract(a.metadata_json, '$.reverse_geocode.city'), '')) != '' "
        "THEN json_extract(a.metadata_json, '$.reverse_geocode.city') "
        "WHEN TRIM(COALESCE(o.place, '')) != '' THEN o.place "
        "WHEN TRIM(COALESCE(a.captured_location, '')) != '' THEN '有GPS坐标' "
        "ELSE '未知' END"),

}


@dataclass
class StructuredResult:
    answer_type: str
    value: object
    rows: list[dict] = field(default_factory=list)
    total: int = 0
    filters_applied: dict = field(default_factory=dict)
    strategy: str = ""
    exact: bool = True

    def as_dict(self):
        return {
            "answer_type": self.answer_type, "value": self.value,
            "rows": self.rows, "total": self.total,
            "filters_applied": self.filters_applied,
            "strategy": self.strategy, "exact": self.exact,
        }


class StructuredMemoryExecutor:
    def __init__(self, store):
        self.store = store

    @property
    def conn(self):
        return self.store.connection

    def _rows(self, sql, params):
        return self.store._rows(sql, params)

    # ---- filter derivation (from model-judged slots + QuerySpec) ----

    def _scope(self, spec: QuerySpec) -> tuple[str, list]:
        if spec.scope_mode == "all_authorized":
            return "1=1", []
        scope_id = spec.scope_id or (spec.scope_ids[0] if spec.scope_ids else "home-default")
        return "a.scope_id = ?", [scope_id]

    def _media(self, draft, spec) -> tuple[list[str], list[str]]:
        include: list[str] = []
        exclude: list[str] = []
        structured_media = (draft.structured or {}).get("media_type")
        if structured_media:
            include.append(structured_media)
        for constraint in spec.constraints_for("media"):
            value = str(constraint.value or "").lower()
            if value not in {"image", "video", "audio", "text"}:
                continue
            if constraint.negated:
                exclude.append(value)
            else:
                include.append(value)
        return list(dict.fromkeys(include)), list(dict.fromkeys(exclude))

    @staticmethod
    def _normalize_end(end: str | None) -> str | None:
        """A date-only end means "through that day" (inclusive).

        The model often writes 2025-12-31 for "去年".  Treating it as an open
        bound would drop assets captured on the last day.  Convert to the next
        day so the SQL ``<`` bound includes the whole last day.
        """
        if not end:
            return None
        value = str(end)
        try:
            day = datetime.strptime(value[:10], "%Y-%m-%d")
            return (day + timedelta(days=1)).isoformat()[:10]
        except ValueError:
            return value

    def _time_range(self, draft, spec) -> tuple[str | None, str | None]:
        structured = (draft.structured or {}).get("time_range") or {}
        if structured.get("start") or structured.get("end"):
            return structured.get("start"), self._normalize_end(structured.get("end"))
        for constraint in spec.constraints_for("time"):
            parsed = parse_time_expression(constraint.value)
            if parsed:
                start, end = parsed
                return start.date().isoformat(), end.date().isoformat()
        return None, None

    def _place(self, draft, spec) -> str | None:
        place = (draft.structured or {}).get("place")
        if place:
            return str(place).strip() or None
        for constraint in spec.constraints_for("place"):
            value = str(constraint.value or "").strip()
            if value:
                return value
        return None

    def _entity_ids(self, spec) -> list[str]:
        return list(dict.fromkeys(spec.entity_ids or []))

    def _filters_applied(self, draft, spec) -> dict:
        scope_id = None if spec.scope_mode == "all_authorized" else (spec.scope_id or (spec.scope_ids[0] if spec.scope_ids else "home-default"))
        start, end = self._time_range(draft, spec)
        include, exclude = self._media(draft, spec)
        return {
            "scope_id": scope_id,
            "time_range": {"start": start, "end": end} if (start or end) else None,
            "media_include": include, "media_exclude": exclude,
            "place": self._place(draft, spec),
            "entity_ids": self._entity_ids(spec),
        }

    # ---- base query builder ----

    def _base_query(self, draft, spec) -> tuple[str, str, list]:
        joins: list[str] = []
        entity_ids = self._entity_ids(spec)
        if entity_ids:
            joins.append("JOIN observations o ON o.asset_id = a.id")
            joins.append("JOIN entity_mentions em ON em.observation_id = o.id")
        else:
            joins.append("LEFT JOIN observations o ON o.asset_id = a.id")

        scope_clause, params = self._scope(spec)
        clauses = [scope_clause]
        include, exclude = self._media(draft, spec)
        if include:
            clauses.append(f"a.media_type IN ({', '.join('?' * len(include))})")
            params.extend(include)
        for value in exclude:
            clauses.append("a.media_type != ?")
            params.append(value)
        start, end = self._time_range(draft, spec)
        if start:
            clauses.append("a.captured_at >= ?")
            params.append(start)
        if end:
            clauses.append("a.captured_at < ?")
            params.append(end)
        place = self._place(draft, spec)
        if place:
            place_clause, place_params = self._place_clause(place)
            clauses.append(place_clause)
            params.extend(place_params)
        if entity_ids:
            clauses.append(f"em.entity_id IN ({', '.join('?' * len(entity_ids))})")
            params.extend(entity_ids)
        where = " AND ".join(f"({clause})" for clause in clauses)
        return " ".join(joins), where, params

    @staticmethod
    def _place_clause(value):
        """地点过滤 WHERE 子句（D12）。

        覆盖：
        1) 场景类型 / GPS 原文 / 反编码 label / name 包含约束值；
        2) 存储的行政区（省/市/区/县，去后缀）出现在约束值里
           （'秦皇岛' 匹配约束 '秦皇岛如是海度假村'）；
        3) 中英别名展开（'清迈' 匹配 'Chiang Mai'）。
        """
        from .geocoding import place_alias_names
        value = str(value or "").strip()
        if not value:
            return "1=1", []
        clauses = []
        params = []
        label = "json_extract(a.metadata_json, '$.reverse_geocode.label')"
        name = "json_extract(a.metadata_json, '$.reverse_geocode.name')"
        clauses.append(f"(o.place LIKE ? OR a.captured_location LIKE ? OR {label} LIKE ? OR {name} LIKE ?)")
        params += [f"%{value}%"] * 4
        for field in ("province", "city", "district"):
            expr = (
                f"replace(replace(replace(replace(replace(COALESCE("
                f"json_extract(a.metadata_json, '$.reverse_geocode.{field}'), ''), "
                f"'省', ''), '市', ''), '区', ''), '县', ''), '地区', '')"
            )
            clauses.append(f"(instr(?, {expr}) > 0 AND length({expr}) >= 2)")
            params.append(value)
        for alias in place_alias_names(value):
            clauses.append(f"({name} LIKE ? OR {label} LIKE ?)")
            params += [f"%{alias}%"] * 2
        return "(" + " OR ".join(clauses) + ")", params

    # ---- aggregation ops ----

    def _count(self, draft, spec) -> int:
        joins, where, params = self._base_query(draft, spec)
        row = self._rows(f"SELECT COUNT(DISTINCT a.id) AS n FROM assets a {joins} WHERE {where}", params)
        return int(row[0]["n"] or 0) if row else 0

    def _first_last(self, draft, spec, op):
        joins, where, params = self._base_query(draft, spec)
        row = self._rows(f"SELECT {op}(a.captured_at) AS v FROM assets a {joins} WHERE {where}", params)
        return (row[0]["v"] if row and row[0]["v"] else None)

    def _group_by(self, draft, spec, group_by: str) -> list[dict]:
        expr = _GROUP_EXPR.get(group_by, _GROUP_EXPR["month"])
        joins, where, params = self._base_query(draft, spec)
        time_filter = " AND a.captured_at IS NOT NULL" if group_by in {"month", "year", "date"} else ""
        rows = self._rows(
            f"SELECT {expr} AS g, COUNT(DISTINCT a.id) AS n FROM assets a {joins} "
            f"WHERE {where}{time_filter} GROUP BY g ORDER BY n DESC, g",
            params,
        )
        return [{"group": row["g"], "count": int(row["n"] or 0)} for row in rows]

    def _list(self, draft, spec, group_by: str) -> list[dict]:
        expr = _GROUP_EXPR.get(group_by, _GROUP_EXPR["month"])
        joins, where, params = self._base_query(draft, spec)
        time_filter = " AND a.captured_at IS NOT NULL" if group_by in {"month", "year", "date"} else ""
        rows = self._rows(
            f"SELECT DISTINCT {expr} AS g FROM assets a {joins} "
            f"WHERE {where}{time_filter} ORDER BY g",
            params,
        )
        return [{"group": row["g"]} for row in rows]

    def _sample_observations(self, draft, spec, limit: int = 3) -> list[dict]:
        """取过滤条件下的代表 observation 证据样本（含 asset_id/caption/captured_at）。

        供 query_memory_facts 等事实工具附加证据，前端可据此展示可点击的原始证据。
        """
        joins, where, params = self._base_query(draft, spec)
        rows = self._rows(
            "SELECT a.id AS asset_id, a.captured_at, a.media_type, "
            "(SELECT COALESCE(o.caption, '') FROM observations o "
            " WHERE o.asset_id = a.id ORDER BY o.captured_at DESC LIMIT 1) AS caption "
            f"FROM assets a {joins} WHERE {where} ORDER BY a.captured_at DESC LIMIT ?",
            params + [limit])
        return [dict(row) for row in rows]

    def _matching_assets(self, draft, spec, limit: int = 500) -> list[dict]:
        """纯硬筛选（时间/媒体/地点/人物）的资产列表（search_memories 空 query 用，不依赖 ANN）。"""
        joins, where, params = self._base_query(draft, spec)
        rows = self._rows(
            "SELECT a.id, a.file_name, a.captured_at, a.media_type, a.captured_location "
            f"FROM assets a {joins} WHERE {where} ORDER BY a.captured_at DESC LIMIT ?",
            params + [limit])
        return [dict(row) for row in rows]

    # ---- entry point ----

    def execute(self, draft, spec: QuerySpec, strategy: str = "structured_fact") -> StructuredResult:
        answer_type = draft.answer_type
        filters = self._filters_applied(draft, spec)

        if answer_type == "count":
            total = self._count(draft, spec)
            return StructuredResult(answer_type, total, total=total, filters_applied=filters, strategy=strategy)
        if answer_type in {"exists", "boolean"}:
            total = self._count(draft, spec)
            return StructuredResult(answer_type, total > 0, total=total, filters_applied=filters, strategy=strategy)
        if answer_type in {"first_occurrence", "date"}:
            value = self._first_last(draft, spec, "MIN")
            total = self._count(draft, spec)
            return StructuredResult(answer_type, value, total=total, filters_applied=filters, strategy=strategy)
        if answer_type == "last_occurrence":
            value = self._first_last(draft, spec, "MAX")
            total = self._count(draft, spec)
            return StructuredResult(answer_type, value, total=total, filters_applied=filters, strategy=strategy)
        if answer_type == "date_range":
            first = self._first_last(draft, spec, "MIN")
            last = self._first_last(draft, spec, "MAX")
            total = self._count(draft, spec)
            return StructuredResult(answer_type, {"first": first, "last": last}, total=total,
                                    filters_applied=filters, strategy=strategy)
        if answer_type == "grouped_list":
            group_by = (draft.structured or {}).get("aggregation", {}).get("group_by") or "month"
            rows = self._group_by(draft, spec, group_by)
            total = sum(row["count"] for row in rows)
            return StructuredResult(answer_type, rows, rows=rows, total=total,
                                    filters_applied=filters, strategy=strategy)
        if answer_type == "list":
            group_by = (draft.structured or {}).get("aggregation", {}).get("group_by") or "month"
            rows = self._list(draft, spec, group_by)
            total = len(rows)
            return StructuredResult(answer_type, rows, rows=rows, total=total,
                                    filters_applied=filters, strategy=strategy)

        raise ValueError(f"answer_type {answer_type!r} is not structured-answerable")
