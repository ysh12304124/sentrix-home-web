import unittest
from unittest.mock import patch

from backend.geocoding import OfflineReverseGeocoder, format_gps_prefix


class GeocodingTests(unittest.TestCase):
    def test_returns_empty_for_invalid_coordinates(self):
        geocoder = OfflineReverseGeocoder()
        self.assertEqual(geocoder.lookup({"latitude": 181, "longitude": 1}), {})
        self.assertEqual(geocoder.lookup(None), {})

    def test_format_gps_prefix_joins_china_without_separators(self):
        prefix = format_gps_prefix({
            "source": "tianditu",
            "city": "深圳市",
            "district": "福田区",
            "country": "CN",
            "label": "广东省深圳市福田区",
        })
        self.assertEqual(prefix, "深圳市福田区")

    def test_format_gps_prefix_uses_comma_for_international(self):
        prefix = format_gps_prefix({
            "source": "geonames",
            "city": "Kyoto",
            "admin1": "Kyoto Prefecture",
            "country": "JP",
            "label": "Kyoto, Kyoto Prefecture, JP",
        })
        self.assertEqual(prefix, "Kyoto, Kyoto Prefecture")

    def test_pygeo_success_short_circuits_fallback(self):
        geocoder = OfflineReverseGeocoder()
        with patch.object(geocoder, "_lookup_pygeo", return_value={
            "source": "tianditu",
            "precision": "district",
            "label": "广东省深圳市福田区",
            "province": "广东省",
            "city": "深圳市",
            "district": "福田区",
            "country": "CN",
            "latitude": 22.53,
            "longitude": 114.05,
            "confidence": 0.9,
        }) as pygeo, patch.object(geocoder, "_lookup_reverse_geocoder") as fallback:
            result = geocoder.lookup({"latitude": 22.53, "longitude": 114.05})
        self.assertEqual(result["source"], "tianditu")
        self.assertEqual(result["label"], "广东省深圳市福田区")
        pygeo.assert_called_once()
        fallback.assert_not_called()

    def test_falls_back_to_reverse_geocoder_outside_china(self):
        geocoder = OfflineReverseGeocoder()
        with patch.object(geocoder, "_lookup_pygeo", return_value={}), patch.object(
            geocoder,
            "_lookup_reverse_geocoder",
            return_value={
                "source": "geonames",
                "precision": "city",
                "label": "Kyoto, Kyoto Prefecture, JP",
                "city": "Kyoto",
                "admin1": "Kyoto Prefecture",
                "country": "JP",
                "latitude": 35.02,
                "longitude": 135.77,
                "confidence": 0.7,
            },
        ) as fallback:
            result = geocoder.lookup({"latitude": 35.02, "longitude": 135.77})
        self.assertEqual(result["source"], "geonames")
        self.assertEqual(result["label"], "Kyoto, Kyoto Prefecture, JP")
        fallback.assert_called_once()


if __name__ == "__main__":
    unittest.main()
