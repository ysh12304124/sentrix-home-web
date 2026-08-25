"""P2: Canonical Retrieval Intent.

确定性从用户问题提取检索约束（时间/地点/人物/关键词），使同一家庭记忆任务
无论措辞如何都映射到同一组结构化约束，消除 search query 的 paraphrase 漂移。

约束提取完全数据驱动：时间来自正则（通用日期模式），地点来自 scope 内已存地点
label + 通用行政后缀变体 + geocoding 通用别名，人物来自 scope 内已确认人物。
不包含任何 benchmark/测试集特定内容。
"""
import re
import os

_RELATIVE = ("这两年", "近两年", "最近两年", "最近一年", "今年", "去年", "前年",
             "上上个月", "上个月", "去年春天", "去年夏天", "去年秋天", "去年冬天")

# 场景/语义噪声词：不作为 canonical 地点（observation.place 的场景类型，非检索地点）
_PLACE_NOISE = ("户外", "户外公共场所", "餐厅", "厨房", "室内", "沙滩", "街道",
                "快餐店", "奶茶店", "咖啡", "室内厨房", "餐厅内部", "天花板", "墙面")

_ADMIN_SUFFIXES = ("省", "市", "区", "县", "地区", "自治州", "自治县")


def canonical_enabled() -> bool:
    return os.getenv("SENTRIX_CANONICAL_SEARCH", "0").strip().lower() in {"1", "true", "on"}


def extract_time(question: str) -> str | None:
    """返回问题中的时间片段（优先完整日期，其次年月，其次年）或相对时间词。"""
    q = re.sub(r"\s+", "", question or "")
    for pattern in (r"20\d{2}年\d{1,2}月\d{1,2}日", r"20\d{2}年\d{1,2}月", r"20\d{2}年"):
        m = re.search(pattern, q)
        if m:
            return m.group(0)
    for expr in _RELATIVE:
        if expr in q:
            return expr
    return None


def _place_variants(token: str) -> list[str]:
    """生成地点的行政后缀变体（'上海市普陀区' → '上海市普陀区/上海市普陀/上海普陀/普陀区/普陀'）。"""
    variants = {token}
    t = token
    while any(t.endswith(s) for s in _ADMIN_SUFFIXES):
        t = t[:-1]
        variants.add(t)
    # 去掉中间的"市/省/县"后再次去后缀（'上海普陀区' 也可被含）
    core = re.sub(r"(省|市|区|县|地区)", "", token)
    variants.add(core)
    return sorted(variants, key=len, reverse=True)


def _place_labels(store, scope_id: str) -> list[str]:
    """收集 scope 内已知地点：reverse_geocode label/district/city + observations.place。"""
    labels = []

    def value(row, key, fallback=None):
        """Read sqlite Row and lightweight test doubles uniformly."""
        try:
            return row[key]
        except (KeyError, IndexError, TypeError):
            if isinstance(row, dict):
                return row.get(key, fallback)
            return fallback

    try:
        rows = store.connection.execute(
            "SELECT DISTINCT json_extract(a.metadata_json,'$.reverse_geocode.label') AS label, "
            "json_extract(a.metadata_json,'$.reverse_geocode.province') AS province, "
            "json_extract(a.metadata_json,'$.reverse_geocode.admin1') AS admin1, "
            "json_extract(a.metadata_json,'$.reverse_geocode.district') AS dist, "
            "json_extract(a.metadata_json,'$.reverse_geocode.city') AS city "
            "FROM assets a WHERE a.scope_id=?",
            (scope_id or "",)).fetchall()
        for r in rows:
            for v in (value(r, "label"), value(r, "province"), value(r, "admin1"),
                      value(r, "dist"), value(r, "city")):
                if v and str(v).strip():
                    labels.append(str(v).strip())
    except Exception:
        pass
    try:
        rows = store.connection.execute(
            "SELECT DISTINCT o.place FROM observations o "
            "JOIN assets a ON a.id=o.asset_id WHERE a.scope_id=? AND o.place IS NOT NULL AND o.place<>''",
            (scope_id or "",)).fetchall()
        for r in rows:
            labels.append(str(value(r, 0, value(r, "place", ""))).strip())
    except Exception:
        pass
    return sorted(set(labels))


def extract_place(question: str, store, scope_id: str) -> str | None:
    """在 scope 已知地点中，找到问题里出现的地点；完全数据驱动，无 benchmark 特定内容。"""
    q = question or ""
    from ..geocoding import place_alias_names
    best = None
    best_len = 0
    best_position = -1
    for label in _place_labels(store, scope_id):
        if any(n in label for n in _PLACE_NOISE):
            continue
        # label 自身变体
        for variant in _place_variants(label):
            if len(variant) >= 2 and variant in q:
                position = q.rfind(variant)
                # When city and district have the same token length, prefer
                # the more specific trailing place phrase instead of
                # the first broad city token.
                if len(variant) > best_len or (len(variant) == best_len and position > best_position):
                    best, best_len, best_position = variant, len(variant), position
        # 通用别名（place_alias_names 是 geocoding 的真实世界地名表，非 benchmark 特定）
        for alias in place_alias_names(label):
            if alias in q:
                position = q.rfind(alias)
                if len(alias) > best_len or (len(alias) == best_len and position > best_position):
                    best, best_len, best_position = alias, len(alias), position
    # 反向：问题里的中文地名词经通用别名展开后，能命中 scope 已知 label
    for alias in place_alias_names(q):
        for known in _place_labels(store, scope_id):
            if alias and (alias in known or known in alias):
                if len(alias) > best_len:
                    best, best_len, best_position = alias, len(alias), q.rfind(alias)
    return best


def extract_person(question: str, store, scope_id: str) -> str | None:
    """在 scope 已确认人物中，找到问题里出现的人物名（数据驱动）。"""
    q = question or ""
    try:
        for ent in store.list_entities(status="confirmed", scope_id=scope_id or None):
            name = ent.get("canonical_name") or ""
            role = ent.get("family_role") or ""
            for alias in [name, role] + (ent.get("aliases") or []):
                if alias and alias != "自己" and alias in q:
                    return name or alias
    except Exception:
        pass
    return None


def extract_constraints(question: str, store, scope_id: str) -> dict:
    """返回 {time, place, person, strong}。strong = 时间+地点都命中（走确定性元数据路径足够）。"""
    t = extract_time(question)
    p = extract_place(question, store, scope_id)
    person = extract_person(question, store, scope_id)
    return {"time": t, "place": p, "person": person, "strong": bool(t and p)}
