import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

from scripts.maintenance.rebuild_memory import SUPPORTED, rebuild


class RebuildInputTests(unittest.TestCase):
    def test_metadata_json_files_are_not_rebuild_assets(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory)
            (source / "asset_001.jpg").write_bytes(b"image")
            (source / "sentrix_metadata.json").write_text("{}", encoding="utf-8")
            (source / "virtual_album_manifest.json").write_text("{}", encoding="utf-8")

            files = sorted(
                path.name
                for path in source.rglob("*")
                if path.is_file() and path.suffix.lower() in SUPPORTED
            )

        self.assertEqual(files, ["asset_001.jpg"])

    def test_rebuild_fails_before_deleting_data_when_identity_embedding_is_unavailable(self):
        class UnavailableFace:
            enabled = True
            identity_model = "adaface"
            identity_configured = False
            identity_error = "checkpoint is unavailable"

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = root / "data"
            data.mkdir()
            database = data / "sentrix.db"
            database.write_bytes(b"keep")
            source = root / "source"
            source.mkdir()

            with patch("scripts.maintenance.rebuild_memory.FaceAdapter", return_value=UnavailableFace()):
                with self.assertRaisesRegex(RuntimeError, "identity embedding"):
                    rebuild(root, source)

            self.assertEqual(database.read_bytes(), b"keep")


if __name__ == "__main__":
    unittest.main()
