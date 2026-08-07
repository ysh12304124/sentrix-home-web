import unittest

from backend.geocoding import OfflineReverseGeocoder


class GeocodingTests(unittest.TestCase):
    def setUp(self):
        self.geocoder = OfflineReverseGeocoder(
            city_records=[
                {"name": "测试市", "admin1": "测试省", "admin2": "测试区", "cc": "CN", "lat": 31.2304, "lon": 121.4737},
            ],
            poi_records=[
                {"name": "测试湖公园", "kind": "park", "lat": 31.2310, "lon": 121.4740},
            ],
        )

    def test_returns_city_context_from_offline_index(self):
        result = self.geocoder.lookup({"latitude": 31.2304, "longitude": 121.4737})

        self.assertEqual(result["source"], "offline")
        self.assertEqual(result["precision"], "poi")
        self.assertEqual(result["label"], "测试湖公园")
        self.assertEqual(result["city"], "测试市")
        self.assertEqual(result["district"], "测试区")
        self.assertGreater(result["confidence"], 0)

    def test_returns_empty_for_invalid_coordinates(self):
        self.assertEqual(self.geocoder.lookup({"latitude": 181, "longitude": 1}), {})
        self.assertEqual(self.geocoder.lookup(None), {})

    def test_poi_is_not_used_when_it_is_too_far_away(self):
        geocoder = OfflineReverseGeocoder(
            city_records=self.geocoder.city_records,
            poi_records=self.geocoder.poi_records,
            poi_radius_km=0.01,
        )

        result = geocoder.lookup({"latitude": 31.2304, "longitude": 121.4737})

        self.assertEqual(result["precision"], "city")
        self.assertEqual(result["label"], "测试省测试市测试区")


if __name__ == "__main__":
    unittest.main()
