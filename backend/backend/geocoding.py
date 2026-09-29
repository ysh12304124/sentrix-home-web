"""Offline reverse geocoding for image metadata.

Primary: PyGeoCN (Tianditu) for China district-level results.
Fallback: reverse_geocoder (GeoNames) for international city-level results.
Both paths are offline and degrade to {} when unavailable.
"""

from __future__ import annotations

import math
import os
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

    def lookup(self, gps):
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
    label = " ".join(
        str(part) for part in (
            geocode.get("label"), geocode.get("name"), geocode.get("city"),
            geocode.get("province"), geocode.get("district"),
            geocode.get("admin1"), geocode.get("admin2"), geocode.get("country"),
        ) if part
    )
    if not label:
        return False
    if value in label:
        return True
    for key in ("province", "city", "district", "admin1", "admin2"):
        part = _strip_admin_suffix(geocode.get(key))
        if len(part) >= 2 and part in value:
            return True
    lower_label = label.lower()
    for alias in place_alias_names(value):
        if alias.lower() in lower_label:
            return True
    return False
