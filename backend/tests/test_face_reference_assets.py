import unittest

from backend.agent_runtime.tools import _is_face_reference_asset


class _Store:
    def __init__(self, assets):
        self.assets = assets

    def get_asset(self, asset_id):
        return self.assets.get(asset_id)


class FaceReferenceAssetTests(unittest.TestCase):
    def test_faceid_filename_is_not_a_retrieval_asset(self):
        self.assertTrue(_is_face_reference_asset({"file_name": "faceid_10.jpg"}))

    def test_identity_seed_metadata_is_not_a_retrieval_asset(self):
        store = _Store({"seed-1": {"metadata_json": {"source_type": "identity_seed"}}})
        self.assertTrue(_is_face_reference_asset({"asset_id": "seed-1"}, store))

    def test_derived_face_crop_is_not_a_retrieval_asset(self):
        store = _Store({"crop-1": {"metadata_json": {"derived_kind": "face_crop"}}})
        self.assertTrue(_is_face_reference_asset({"asset_id": "crop-1"}, store))

    def test_original_photo_remains_a_valid_retrieval_asset(self):
        store = _Store({"photo-1": {"metadata_json": {"source_type": "user_upload"}}})
        self.assertFalse(_is_face_reference_asset({
            "asset_id": "photo-1", "file_name": "wedding.jpg",
        }, store))


if __name__ == "__main__":
    unittest.main()
