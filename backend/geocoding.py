"""Offline reverse geocoding for image metadata.

The default city lookup uses the bundled data from reverse_geocoder when the
optional dependency is installed. A small local POI JSON file can be supplied
through SENTRIX_GEO_POI_PATH without introducing a network dependency.
"""

import json
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


class OfflineReverseGeocoder:
    """Resolve GPS to a conservative city/admin context and optional POI."""

    def __init__(self, city_records=None, poi_records=None, poi_radius_km=1.5, city_radius_km=200.0, poi_path=None):
        self.city_records = list(city_records or [])
        self.poi_records = list(poi_records or self._load_pois(poi_path or os.getenv("SENTRIX_GEO_POI_PATH")))
        self.poi_radius_km = float(poi_radius_km)
        self.city_radius_km = float(city_radius_km)
        self._reverse_geocoder = None

    @staticmethod
    def _load_pois(path):
        if not path:
            return []
        try:
            payload = json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            return []
        return payload.get("places", []) if isinstance(payload, dict) else payload if isinstance(payload, list) else []

    @staticmethod
    def _record_coordinate(record):
        latitude = _coordinate(record.get("lat", record.get("latitude")))
        longitude = _coordinate(record.get("lon", record.get("longitude")))
        if latitude is None or longitude is None or not -90 <= latitude <= 90 or not -180 <= longitude <= 180:
            return None
        return latitude, longitude

    def _nearest(self, records, latitude, longitude):
        candidates = []
        for record in records:
            coordinates = self._record_coordinate(record)
            if coordinates:
                distance = _distance_km(latitude, longitude, *coordinates)
                candidates.append((distance, record, coordinates))
        return min(candidates, key=lambda item: item[0]) if candidates else None

    def _city_record(self, latitude, longitude):
        if self.city_records:
            nearest = self._nearest(self.city_records, latitude, longitude)
            return nearest if nearest and nearest[0] <= self.city_radius_km else None
        try:
            if self._reverse_geocoder is None:
                import reverse_geocoder
                self._reverse_geocoder = reverse_geocoder
            result = self._reverse_geocoder.search((latitude, longitude), mode=1)
        except (ImportError, OSError, TypeError, ValueError):
            return None
        if not result:
            return None
        record = result[0]
        coordinates = self._record_coordinate(record)
        if not coordinates:
            return None
        distance = _distance_km(latitude, longitude, *coordinates)
        return distance, record, coordinates

    @staticmethod
    def _label(record):
        parts = []
        for key in ("admin1", "name", "admin2"):
            value = str(record.get(key) or "").strip()
            if value and value not in parts:
                parts.append(value)
        return "".join(parts)

    @staticmethod
    def _confidence(distance_km, scale_km):
        return round(max(0.2, min(0.95, 0.95 - (distance_km / max(scale_km, 1.0)) * 0.7)), 2)

    def lookup(self, gps):
        if not isinstance(gps, dict):
            return {}
        latitude = _coordinate(gps.get("latitude", gps.get("lat")))
        longitude = _coordinate(gps.get("longitude", gps.get("lon")))
        if latitude is None or longitude is None or not -90 <= latitude <= 90 or not -180 <= longitude <= 180:
            return {}

        city_match = self._city_record(latitude, longitude)
        if not city_match:
            return {}
        city_distance, city, _ = city_match
        context = {
            "source": "offline",
            "precision": "city",
            "label": self._label(city),
            "city": str(city.get("name") or ""),
            "district": str(city.get("admin2") or ""),
            "admin1": str(city.get("admin1") or ""),
            "admin2": str(city.get("admin2") or ""),
            "country": str(city.get("cc") or city.get("country") or ""),
            "latitude": latitude,
            "longitude": longitude,
            "distance_km": round(city_distance, 3),
            "confidence": self._confidence(city_distance, self.city_radius_km),
        }
        poi_match = self._nearest(self.poi_records, latitude, longitude)
        if poi_match and poi_match[0] <= self.poi_radius_km:
            poi_distance, poi, _ = poi_match
            poi_name = str(poi.get("name") or poi.get("label") or "").strip()
            if poi_name:
                context.update({
                    "precision": "poi",
                    "label": poi_name,
                    "poi_name": poi_name,
                    "poi_kind": str(poi.get("kind") or ""),
                    "poi_distance_km": round(poi_distance, 3),
                    "confidence": self._confidence(poi_distance, self.poi_radius_km),
                })
        return {key: value for key, value in context.items() if value not in (None, "")}
