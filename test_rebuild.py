import tempfile
import unittest
from pathlib import Path

from scripts.rebuild_memory import SUPPORTED


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


if __name__ == "__main__":
    unittest.main()
