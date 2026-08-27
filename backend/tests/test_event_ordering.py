import unittest

from backend.event_ordering import sort_video_scene_observations


class VideoSceneObservationOrderingTests(unittest.TestCase):
    def test_orders_observations_by_embedded_source_timestamp(self):
        observations = [
            {"id": "late", "asset_id": "frame-late", "asset": {"source_timestamp_sec": 15.96}},
            {"id": "early", "asset_id": "frame-early", "asset": {"source_timestamp_sec": 12.46}},
        ]

        ordered = sort_video_scene_observations(observations)

        self.assertEqual([item["id"] for item in ordered], ["early", "late"])

    def test_falls_back_to_chronological_keyframe_asset_order(self):
        observations = [
            {"id": "middle", "asset_id": "frame-middle", "asset": {}},
            {"id": "late", "asset_id": "frame-late", "asset": {}},
            {"id": "early", "asset_id": "frame-early", "asset": {}},
        ]
        frames = [
            {"id": "frame-early", "source_timestamp_sec": 34.98},
            {"id": "frame-middle", "source_timestamp_sec": 43.48},
            {"id": "frame-late", "source_timestamp_sec": 47.47},
        ]

        ordered = sort_video_scene_observations(observations, frames)

        self.assertEqual([item["id"] for item in ordered], ["early", "middle", "late"])


if __name__ == "__main__":
    unittest.main()
