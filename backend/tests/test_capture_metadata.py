import tempfile
import unittest
from pathlib import Path

from backend.app import _normalized_capture_metadata
from backend.db import MemoryStore
from backend.pipeline import IngestionPipeline


class CaptureMetadataTests(unittest.TestCase):
    def test_normalizes_explicit_time_and_gps(self):
        self.assertEqual(
            _normalized_capture_metadata({"capturedAt": "2025-05-20T20:30:00", "latitude": 30.25, "longitude": 120.0}),
            {"captured_at": "2025-05-20T20:30:00", "gps": {"latitude": 30.25, "longitude": 120.0}},
        )

    def test_rejects_unpaired_or_out_of_range_coordinates(self):
        with self.assertRaises(ValueError):
            _normalized_capture_metadata({"latitude": 30.0})
        with self.assertRaises(ValueError):
            _normalized_capture_metadata({"latitude": 91, "longitude": 120})

    def test_gps_is_retained_and_reverse_geocode_fills_location(self):
        with tempfile.TemporaryDirectory() as directory:
            image = Path(directory) / "photo.jpg"
            image.write_bytes(b"not-a-real-image")
            store = MemoryStore(str(Path(directory) / "memory.db"))

            class Geocoder:
                def lookup(self, gps):
                    self.gps = gps
                    return {"label": "测试省测试市", "precision": "city", "source": "offline"}

            geocoder = Geocoder()
            pipeline = IngestionPipeline(store, geocoder=geocoder)
            asset = pipeline.create_asset(image, metadata={"captured_at": "2025-05-20T20:30:00", "gps": {"latitude": 30.25, "longitude": 120.0}})
            self.assertEqual(asset["captured_location"], "测试省测试市")
            self.assertEqual(asset["metadata_json"]["gps"]["latitude"], 30.25)
            self.assertEqual(asset["metadata_json"]["reverse_geocode"]["label"], "测试省测试市")
            self.assertEqual(geocoder.gps["longitude"], 120.0)


if __name__ == "__main__":
    unittest.main()
