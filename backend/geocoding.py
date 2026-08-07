"""Offline reverse geocoding for image metadata using PyGeoCN (Tianditu data).

Returns Chinese administrative division names (province/city/district) without
requiring a network connection. Falls back gracefully when the optional
dependency is unavailable.
"""

import math
import os


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


class OfflineReverseGeocoder:
    """Resolve GPS to Chinese administrative divisions via PyGeoCN."""

    def __init__(self):
        self._pygeo_available = None

    def _ensure_pygeo(self):
        if self._pygeo_available is None:
            try:
                from PyGeoCN.regeo import regeo  # noqa: F401
                self._pygeo_available = True
            except ImportError:
                self._pygeo_available = False
        return self._pygeo_available

    def lookup(self, gps):
        if not isinstance(gps, dict):
            return {}
        latitude = _coordinate(gps.get("latitude", gps.get("lat")))
        longitude = _coordinate(gps.get("longitude", gps.get("lon")))
        if latitude is None or longitude is None or not -90 <= latitude <= 90 or not -180 <= longitude <= 180:
            return {}

        if not self._ensure_pygeo():
            return {}

        try:
            from PyGeoCN.regeo import regeo
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

        # Build a clean label: "浙江省绍兴市越城区"
        parts = []
        for part in (province, city, district):
            if part and part not in parts:
                parts.append(part)
        label = "".join(parts)

        context = {
            "source": "tianditu",
            "precision": "district" if district else "city",
            "label": label,
            "province": province,
            "city": city,
            "district": district,
            "latitude": latitude,
            "longitude": longitude,
            "confidence": 0.90,
        }

        # For backwards compatibility with code expecting reverse_geocoder keys
        context.setdefault("admin1", province)
        context.setdefault("admin2", district)
        context.setdefault("country", "CN")
        context.setdefault("distance_km", 0)

        return {key: value for key, value in context.items() if value not in (None, "")}
