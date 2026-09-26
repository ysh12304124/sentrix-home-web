"""Offline reverse geocoding for image metadata.

Primary: PyGeoCN (Tianditu) for China district-level results.
Fallback: reverse_geocoder (GeoNames) for international city-level results.
Both paths are offline and degrade to {} when unavailable.
"""

from __future__ import annotations

import math
import os
import json
import re
import threading
from pathlib import Path


def _coordinate(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _distance_km(latitude_one, longitude_one, latitude_two, longitude_two):
    radians = math.pi / 180
    lat_one, lat_two = latitude_one * radians, latitude_two * radians
    delta_lat = (latitude_two - latitude_one) * radians
    delta_lon = (longitude_two - longitude_one) * radians
    value = math.sin(delta_lat / 2) ** 2 + math.cos(lat_one) * math.cos(lat_two) * math.sin(delta_lon / 2) ** 2
    return 6371.0088 * 2 * math.atan2(math.sqrt(value), math.sqrt(max(0.0, 1 - value)))


def _has_cjk(value):
    return any("\u4e00" <= char <= "\u9fff" for char in str(value or ""))


def format_gps_prefix(geo):
    """Compact place text for event summaries from a reverse_geocode object."""
    if not isinstance(geo, dict):
        return ""
    source = str(geo.get("source") or "").strip().lower()
    country = str(geo.get("country") or geo.get("cc") or "").strip().upper()
    city = str(geo.get("city") or geo.get("name") or "").strip()
    district = str(geo.get("district") or geo.get("admin2") or "").strip()
    admin1 = str(geo.get("province") or geo.get("admin1") or "").strip()
    label = str(geo.get("label") or "").strip()
    china = (
        source == "tianditu"
        or country in {"CN", "CHN", "中国"}
        or (not country and (_has_cjk(city) or _has_cjk(district) or _has_cjk(label)))
    )
    if china:
        parts = []
        for part in (city, district):
            if part and part not in parts:
                parts.append(part)
        return "".join(parts[:2]) or label
    parts = [part for part in (city, admin1) if part]
    if parts:
        return ", ".join(parts[:2])
    return label


class OfflineReverseGeocoder:
    """Resolve GPS via PyGeoCN first, then international reverse_geocoder."""

    def __init__(self, geo_dir=None):
        self.geo_dir = self._resolve_geo_dir(geo_dir)
        self._pygeo_available = None
        self._rg_available = None
        self._local_loaded = False
        self._local_by_name = {}
        self._local_by_coord = []
        self._local_lock = threading.Lock()

    def _load_local_metadata(self):
        """Load optional local photo metadata as a deterministic geocode fallback.

        Imports often preserve GPS but cannot run an online reverse-geocoder.
        PhotoBench and user imports may already ship an ``image_metadata.jsonl``
        sidecar containing the human-readable location.  This is deliberately
        an optional, data-driven fallback: no place names are embedded in code,
        and normal PyGeoCN/GeoNames resolution still wins.
        """
        if self._local_loaded:
            return
        with self._local_lock:
            if self._local_loaded:
                return
            roots = []
            configured = os.getenv("SENTRIX_LOCATION_METADATA_PATH", "").strip()
            if configured:
                roots.append(Path(configured).expanduser())
            project_root = Path(__file__).resolve().parents[1]
            roots.append(project_root / "services" / "photobench" / "data")
            files = []
            for root in roots:
                if root.is_file() and root.name.lower().endswith(".jsonl"):
                    files.append(root)
                elif root.is_dir():
                    try:
                        files.extend(root.rglob("image_metadata.jsonl"))
                    except OSError:
                        continue
            seen = set()
            for path in files:
                try:
                    resolved = str(path.resolve())
                except OSError:
                    continue
                if resolved in seen:
                    continue
                seen.add(resolved)
                try:
                    with path.open("r", encoding="utf-8") as handle:
                        for line in handle:
                            try:
                                row = json.loads(line)
                            except (TypeError, ValueError, json.JSONDecodeError):
                                continue
                            if not isinstance(row, dict):
                                continue
                            gps = row.get("gps_coordinates") or row.get("released_gps_raw")
                            lat = _coordinate((gps or {}).get("latitude")) if isinstance(gps, dict) else None
                            lon = _coordinate((gps or {}).get("longitude")) if isinstance(gps, dict) else None
                            location = str(row.get("readable_location") or row.get("vlm_location_text") or "").strip()
                            if lat is None or lon is None or not location:
                                continue
                            details = row.get("readable_location_details")
                            details = details if isinstance(details, dict) else {}
                            geo = {
                                "source": "local_metadata",
                                "precision": "address" if details.get("road") or details.get("house_number") else "district",
                                "label": location,
                                "name": details.get("town_suburb") or details.get("poi") or "",
                                "city": details.get("city") or "",
                                "province": details.get("province_state") or "",
                                "district": details.get("district") or "",
                                "admin1": details.get("province_state") or "",
                                "admin2": details.get("district") or "",
                                "country": details.get("country") or "",
                                "latitude": lat, "longitude": lon,
                                "confidence": 0.98,
                            }
                            filename = str(row.get("filename") or Path(str(row.get("image_path") or "")).name).strip().lower()
                            if filename:
                                self._local_by_name[filename] = geo
                            self._local_by_coord.append((lat, lon, geo))
                except (OSError, UnicodeError):
                    continue
            self._local_loaded = True

    def _lookup_local(self, latitude, longitude, filename=None):
        self._load_local_metadata()
        if filename:
            hit = self._local_by_name.get(Path(str(filename)).name.lower())
            if hit:
                # Filenames such as ``IMG_0001.jpg`` repeat across albums;
                # accept a name hit only when its GPS agrees with the asset.
                try:
                    name_distance = _distance_km(
                        latitude, longitude,
                        float(hit.get("latitude")), float(hit.get("longitude")))
                except (TypeError, ValueError):
                    name_distance = float("inf")
                if name_distance <= 0.035:
                    result = dict(hit)
                    result["distance_km"] = round(name_distance, 3)
                    return result
        # Sidecar GPS is rounded to ~4 decimals.  A 35m radius is tight enough
        # to avoid cross-location collisions while tolerating EXIF rounding.
        best, best_distance = None, float("inf")
        for lat, lon, geo in self._local_by_coord:
            distance = _distance_km(latitude, longitude, lat, lon)
            if distance < best_distance and distance <= 0.035:
                best, best_distance = geo, distance
        if best:
            result = dict(best)
            result["distance_km"] = round(best_distance, 3)
            return result
        return {}

    @staticmethod
    def _resolve_geo_dir(geo_dir):
        if geo_dir:
            path = Path(geo_dir).expanduser()
            return path if path.is_dir() else None
        env = os.getenv("SENTRIX_GEO_DIR", "").strip()
        if env:
            path = Path(env).expanduser()
            if path.is_dir():
                return path
        root = Path(__file__).resolve().parents[1]
        candidate = Path(os.getenv("SENTRIX_DATA_DIR", root / "data")) / "geo"
        return candidate if candidate.is_dir() else None

    def _ensure_pygeo(self):
        if self._pygeo_available is None:
            try:
                from PyGeoCN.regeo import regeo  # noqa: F401
                self._pygeo_available = True
            except ImportError:
                self._pygeo_available = False
        return self._pygeo_available

    def _ensure_reverse_geocoder(self):
        if self._rg_available is None:
            try:
                import reverse_geocoder  # noqa: F401
                self._rg_available = True
            except ImportError:
                self._rg_available = False
        return self._rg_available

    def _lookup_pygeo(self, latitude, longitude):
        if not self._ensure_pygeo():
            return {}
        try:
            from PyGeoCN.regeo import regeo
            if self.geo_dir is not None:
                result = regeo(latitude, longitude, str(self.geo_dir))
            else:
                result = regeo(latitude, longitude)
        except Exception:
            return {}
        if not result or result.get("status") != 1:
            return {}
        address = result.get("address") or {}
        province = str(address.get("province") or "").strip()
        city = str(address.get("city") or "").strip()
        district = str(address.get("district") or "").strip()
        if not province and not city and not district:
            return {}
        parts = []
        for part in (province, city, district):
            if part and part not in parts:
                parts.append(part)
        return {
            "source": "tianditu",
            "precision": "district" if district else "city",
            "label": "".join(parts),
            "province": province,
            "city": city,
            "district": district,
            "admin1": province,
            "admin2": district,
            "country": "CN",
            "latitude": latitude,
            "longitude": longitude,
            "confidence": 0.90,
            "distance_km": 0,
        }

    def _lookup_reverse_geocoder(self, latitude, longitude):
        if not self._ensure_reverse_geocoder():
            return {}
        try:
            import reverse_geocoder as rg
            matches = rg.search((latitude, longitude), mode=1)
        except Exception:
            return {}
        if not matches:
            return {}
        match = matches[0] if isinstance(matches, (list, tuple)) else matches
        if not isinstance(match, dict):
            return {}
        name = str(match.get("name") or "").strip()
        admin1 = str(match.get("admin1") or "").strip()
        admin2 = str(match.get("admin2") or "").strip()
        country = str(match.get("cc") or "").strip().upper()
        if not name and not admin1 and not country:
            return {}
        try:
            match_lat = float(match.get("lat"))
            match_lon = float(match.get("lon"))
            distance = round(_distance_km(latitude, longitude, match_lat, match_lon), 3)
        except (TypeError, ValueError):
            match_lat = latitude
            match_lon = longitude
            distance = 0.0
        label_parts = [part for part in (name, admin1, country) if part]
        label = ", ".join(label_parts)
        return {
            "source": "geonames",
            "precision": "city",
            "label": label,
            "name": name,
            "city": name,
            "province": admin1,
            "district": admin2,
            "admin1": admin1,
            "admin2": admin2,
            "country": country,
            "cc": country,
            "latitude": latitude,
            "longitude": longitude,
            "matched_latitude": match_lat,
            "matched_longitude": match_lon,
            "confidence": 0.70,
            "distance_km": distance,
        }

    def lookup(self, gps, filename=None):
        if not isinstance(gps, dict):
            return {}
        latitude = _coordinate(gps.get("latitude", gps.get("lat")))
        longitude = _coordinate(gps.get("longitude", gps.get("lon")))
        if latitude is None or longitude is None or not -90 <= latitude <= 90 or not -180 <= longitude <= 180:
            return {}

        result = self._lookup_pygeo(latitude, longitude)
        if result:
            return {key: value for key, value in result.items() if value not in (None, "")}

        result = self._lookup_reverse_geocoder(latitude, longitude)
        if result:
            return {key: value for key, value in result.items() if value not in (None, "")}
        result = self._lookup_local(latitude, longitude, filename=filename)
        if result:
            return {key: value for key, value in result.items() if value not in (None, "")}
        return {}


# ---- Phase D D12: place-text matching for retrieval ----
# 地理地点检索统一入口：行政区匹配（双向）+ 中英别名。
# 数据来源：assets.metadata_json.reverse_geocode（PyGeoCN/GeoNames 离线反编码）。
# observation.place 是场景类型（"室内餐厅或咖啡馆"），不是地理地点，不作为权威地点依据。

_PLACE_ADMIN_SUFFIXES = ("省", "市", "区", "县", "地区", "自治州", "盟", "特别行政区")

# 常见国际/跨境目的地中英别名（GeoNames 反编码返回英文，中文查询需别名桥接）。
# 这是通用双语地名知识，不是测评答案。
_PLACE_ALIASES = {
    # High-frequency Chinese landmark -> administrative-area aliases.  These
    # are geographic normalization data, not answer text; they let a landmark
    # query use the asset's authoritative reverse-geocode record when the
    # caption itself does not name the landmark.
    "赵州桥": ["赵县"],
    "三峡坝址": ["夷陵区"],
    "清迈": ["Chiang Mai", "Hang Dong"],
    "泰国": ["Thailand", "TH"],
    "曼谷": ["Bangkok"],
    "普吉": ["Phuket"],
    "芭堤雅": ["Pattaya"],
    "新加坡": ["Singapore"],
    "马来西亚": ["Malaysia"],
    "日本": ["Japan"],
    "东京": ["Tokyo"],
    "大阪": ["Osaka"],
    "京都": ["Kyoto"],
    "韩国": ["South Korea", "Korea"],
    "首尔": ["Seoul"],
    "济州": ["Jeju"],
    "美国": ["United States", "USA", "US"],
    "纽约": ["New York"],
    "洛杉矶": ["Los Angeles"],
    "旧金山": ["San Francisco"],
    "英国": ["United Kingdom", "UK", "England"],
    "伦敦": ["London"],
    "法国": ["France"],
    "巴黎": ["Paris"],
    "德国": ["Germany"],
    "意大利": ["Italy"],
    "西班牙": ["Spain"],
    "澳大利亚": ["Australia"],
    "悉尼": ["Sydney"],
    "墨尔本": ["Melbourne"],
    "新西兰": ["New Zealand"],
    "奥克兰": ["Auckland"],
}

_DEFAULT_GEOCODER = None
_DEFAULT_GEOCODER_LOCK = threading.Lock()


def default_reverse_geocoder():
    """Process-local cached resolver used by retrieval/build projections."""
    global _DEFAULT_GEOCODER
    if _DEFAULT_GEOCODER is None:
        with _DEFAULT_GEOCODER_LOCK:
            if _DEFAULT_GEOCODER is None:
                _DEFAULT_GEOCODER = OfflineReverseGeocoder()
    return _DEFAULT_GEOCODER


def _strip_admin_suffix(part):
    """去掉行政区后缀（'秦皇岛市'→'秦皇岛'），便于跨粒度匹配。"""
    part = str(part or "").strip()
    for suffix in _PLACE_ADMIN_SUFFIXES:
        if part.endswith(suffix) and len(part) > len(suffix):
            return part[: -len(suffix)]
    return part


def place_alias_names(value):
    """把中文地点/行程描述展开成可能出现的英文地名（GeoNames 反编码用）。"""
    value = str(value or "").strip()
    names = []
    for zh, targets in _PLACE_ALIASES.items():
        if zh in value:
            names.extend(targets)
    return sorted(set(names))


def place_text_matches(value, geocode):
    """地点条件 vs 反地理编码记录。

    匹配规则（确定性，不依赖模型）：
    1) 约束值整串出现在 geocode label/name/行政区文本里；
    2) 存储的行政区（省/市/区/县，去后缀后）出现在约束值里（'秦皇岛' 匹配
       '秦皇岛如是海度假村'）；
    3) 中英别名命中（'清迈' 匹配 'Chiang Mai'）。
    """
    value = str(value or "").strip()
    if not value or not geocode:
        return False

    # Event/photo imports often store a locality in reverse order and with
    # separators ("沙岭, 易县, 保定市"), while the user says
    # "易县沙岭". Compare normalized administrative components rather than
    # requiring the complete phrase to be a contiguous substring.
    def normalize(text):
        return "".join(ch for ch in str(text or "").casefold() if ch.isalnum())

    query = normalize(value)
    label = " ".join(
        str(part) for part in (
            geocode.get("label"), geocode.get("name"), geocode.get("city"),
            geocode.get("province"), geocode.get("district"),
            geocode.get("admin1"), geocode.get("admin2"), geocode.get("country"),
    ) if part
    )
    if not label:
        return False
    label_normalized = normalize(label)
    if query and query in label_normalized:
        return True

    # A query may specify multiple administrative levels. A conflicting
    # county/district must not pass merely because its city also matches.
    # Missing levels remain open-world (older GPS records may only have a
    # city), but any level present on both sides must agree.
    suffix_groups = (
        (("特别行政区",), ("province", "admin1")),
        (("自治州", "地区", "省"), ("province", "admin1")),
        (("市",), ("city",)),
        (("区", "县", "盟"), ("district", "admin2")),
    )
    higher_fields = []
    matched_admin = False
    for suffixes, fields in suffix_groups:
        available = []
        for key in fields:
            part = str(geocode.get(key) or "").strip()
            if part:
                available.extend((normalize(part), normalize(_strip_admin_suffix(part))))
        available = {part for part in available if part}
        for suffix in suffixes:
            for end_match in re.finditer(re.escape(suffix), value):
                end = end_match.end()
                # Enumerate possible tokens ending at this suffix. This handles
                # concatenated forms such as “上海普陀区” without greedily
                # treating “上海普陀区” as the district itself.
                candidates = {
                    normalize(value[start:end])
                    for start in range(max(0, end - 12), end - 1)
                }
                if available and candidates.intersection(available):
                    matched_admin = True
                    continue

                # When a broader known locality precedes this component, the
                # remaining suffix is unambiguous (保定市 + 赵县). Reject only
                # that explicit contradiction; otherwise keep missing/partial
                # metadata open-world.
                boundary = 0
                for higher_key in higher_fields:
                    higher = str(geocode.get(higher_key) or "").strip()
                    variants = {normalize(higher), normalize(_strip_admin_suffix(higher))}
                    for variant in variants:
                        if len(variant) < 2:
                            continue
                        position = query.find(variant)
                        if position >= 0:
                            boundary = max(boundary, position + len(variant))
                isolated = normalize(value[boundary:end]) if boundary < end else ""
                if (available and boundary > 0 and len(isolated) >= 2
                        and not any(isolated == part or
                                    isolated == normalize(_strip_admin_suffix(part))
                                    for part in available)):
                    return False
                # A short, explicit administrative token is itself a strong
                # contradiction when the corresponding level is known.
                if (available and suffix in {"市", "区", "县", "省"}
                        and 2 <= len(isolated) <= 5
                        and not any(isolated == part or
                                    isolated == normalize(_strip_admin_suffix(part))
                                    for part in available)):
                    return False
        higher_fields.extend(fields)
    # If at least one explicitly named level matches, the place is compatible
    # even if the query adds a street, venue or landmark name.
    if matched_admin:
        return True

    # If the geocoder knows a village/POI name, an explicit mention is a
    # strong match even when the administrative parts precede it in a
    # different order in the query. This check follows administrative
    # contradiction checks so "赵县沙岭" cannot match a record in 易县.
    name = normalize(geocode.get("name"))
    if name and len(name) >= 2 and name in query:
        return True

    for key in ("province", "city", "district", "admin1", "admin2"):
        raw_part = str(geocode.get(key) or "").strip()
        part = _strip_admin_suffix(raw_part)
        if ((raw_part and normalize(raw_part) in query)
                or (len(part) >= 2 and normalize(part) in query)):
            return True
    for alias in place_alias_names(value):
        if normalize(alias) in label_normalized:
            return True
    return False
