import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

from scripts.maintenance.rebuild_memory import SUPPORTED, benchmark_imports, metadata_for_path, rebuild


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

    def test_metadata_uses_source_relative_path_before_duplicate_filename(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory)
            first = source / "album-a" / "IMG_0001.jpg"
            second = source / "album-b" / "IMG_0001.jpg"
            first.parent.mkdir()
            second.parent.mkdir()
            first.write_bytes(b"first")
            second.write_bytes(b"second")
            metadata = {
                "album-a/IMG_0001.jpg": {"captured_location": "客厅"},
                "album-b/IMG_0001.jpg": {"captured_location": "厨房"},
                "IMG_0001.jpg": {"captured_location": "legacy fallback"},
            }

            self.assertEqual(metadata_for_path(metadata, source, first)["captured_location"], "客厅")
            self.assertEqual(metadata_for_path(metadata, source, second)["captured_location"], "厨房")

    def test_benchmark_imports_resolve_album_files_and_allowlist_provenance(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = root / "album1" / "images" / "keep.jpg"
            image.parent.mkdir(parents=True)
            image.write_bytes(b"image")
            manifest = {
                "source_root": str(root),
                "spaces": [{
                    "scope_id": "album1",
                    "import": {"files": [{
                        "relative_path": "album1/images/keep.jpg",
                        "file_name": "keep.jpg",
                        "captured_at": "2025-01-01T10:00:00",
                        "captured_location": "1.000000,2.000000",
                        "source_album_id": "album1",
                        "scope_id": "album1",
                    }]},
                    "evaluation": {"face_id_to_nicknames": {"1": ["不得导入"]}},
                }],
            }

            values = list(benchmark_imports(manifest))

            self.assertEqual(values[0][0], "album1")
            self.assertTrue(values[0][1].is_file())
            self.assertEqual(values[0][2], {
                "captured_at": "2025-01-01T10:00:00",
                "captured_location": "1.000000,2.000000",
                "source_album_id": "album1",
                "scope_id": "album1",
            })


if __name__ == "__main__":
    unittest.main()
