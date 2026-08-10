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
