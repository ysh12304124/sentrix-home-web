import unittest
from contextlib import nullcontext

from scripts.maintenance.re_enrich_missing_observations import (
    _blank, _user_media, _write_reenrichment,
)


class MissingObservationReenrichmentTests(unittest.TestCase):
    def test_blank_selector_requires_no_searchable_visual_detail(self):
        self.assertTrue(_blank({"caption": "", "activity": "", "objects": [], "detail": {}}))
        self.assertFalse(_blank({"caption": "婚礼合影"}))
        self.assertFalse(_blank({"detail": {"visible_details": ["红色拱门"]}}))

    def test_derived_identity_and_event_cards_are_not_backfilled(self):
        self.assertFalse(_user_media({"file_name": "faceid_1.jpg"}))
        self.assertFalse(_user_media({"file_name": "event_event_00001.webp"}))
        self.assertTrue(_user_media({"file_name": "2018-11-02 135412.jpg"}))

    def test_reenrichment_updates_observation_and_retrieval_vectors_together(self):
        writes = []

        class Store:
            @staticmethod
            def transaction():
                return nullcontext()

            @staticmethod
            def enrich_observation(*args, **kwargs):
                writes.append(("observation", args[0]))

            @staticmethod
            def upsert_vector(space, source_type, source_id, vector, model, metadata):
                writes.append((space, source_id))

        class Pipeline:
            @staticmethod
            def _text_embed(text):
                assert "出库单" in text
                return [1.0, 0.0], "bge-m3"

            @staticmethod
            def _field_desc_text(merged):
                return merged["caption"]

            @staticmethod
            def _field_desc_embed(text):
                return [0.0, 1.0], "bge-m3"

        updated = _write_reenrichment(
            Store(), Pipeline(),
            {"id": "obs-1", "caption": "", "activity": ""},
            {"id": "asset-1", "metadata_json": {"event_id": "event-1"}},
            {"caption": "产品出库单"},
        )
        self.assertTrue(updated)
        self.assertEqual(writes, [
            ("observation", "obs-1"), ("episodic", "obs-1"),
            ("semantic", "obs-1"), ("field_desc", "asset-1"),
        ])


if __name__ == "__main__":
    unittest.main()
