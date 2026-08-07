import unittest

from backend.image_io import (
    guess_mime_type,
    media_type_for_path,
    media_type_from_upload,
    needs_browser_transcode,
)


class ImageIoTests(unittest.TestCase):
    def test_heic_suffix_is_image(self):
        self.assertEqual(media_type_for_path("IMG_0001.HEIC"), "image")
        self.assertEqual(media_type_for_path("photo.heif"), "image")
        self.assertEqual(guess_mime_type("IMG_0001.HEIC"), "image/heic")
        self.assertEqual(guess_mime_type("photo.heif"), "image/heif")

    def test_upload_falls_back_to_filename_for_heic(self):
        self.assertEqual(media_type_from_upload("application/octet-stream", "IMG_0001.heic"), "image")
        self.assertEqual(media_type_from_upload("image/heic", "IMG_0001.heic"), "image")
        self.assertEqual(media_type_from_upload(None, "clip.wav"), "audio")

    def test_heic_needs_browser_transcode(self):
        self.assertTrue(needs_browser_transcode("IMG_0001.HEIC", "image/heic"))
        self.assertTrue(needs_browser_transcode("photo.heif", None))
        self.assertFalse(needs_browser_transcode("photo.jpg", "image/jpeg"))


if __name__ == "__main__":
    unittest.main()
