"""Phase R R2 — ANN manifest: save/load/validate (P0-4)."""

import tempfile
import unittest
from pathlib import Path

from backend.retrieval_ann import create_index


def _build(tmp, dim=4, model="ViT-B-32"):
    index = create_index("hnswlib", dim=dim, M=4, ef_construction=50, ef_search=10)
    index.set_manifest_extra(model_id=model, checkpoint_hash="abc123", source_type="asset",
                             normalized=True, source_revision=3)
    index.build([(f"asset_{i}", [float(i) * 0.5, 0.0, 0.0, 0.0], {"scope_id": "album1", "revision": 3})
                 for i in range(5)])
    path = str(Path(tmp) / "visual")
    index.save(path)
    return path


class AnnManifestTests(unittest.TestCase):
    def test_save_writes_manifest(self):
        with tempfile.TemporaryDirectory(prefix="ann-manifest-") as tmp:
            path = _build(tmp)
            manifest_path = Path(f"{path}.manifest.json")
            self.assertTrue(manifest_path.is_file())
            import json
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(manifest["model_id"], "ViT-B-32")
            self.assertEqual(manifest["dimension"], 4)
            self.assertEqual(manifest["space"], "cosine")
            self.assertEqual(manifest["source_count"], 5)
            self.assertIn("id_map_checksum", manifest)

    def test_load_matching_manifest_validates(self):
        with tempfile.TemporaryDirectory(prefix="ann-manifest-") as tmp:
            path = _build(tmp)
            index = create_index("hnswlib", dim=4)
            index.load(path)
            self.assertTrue(index.validate(expected_model_id="ViT-B-32", expected_dim=4))

    def test_load_mismatched_model_rejected(self):
        with tempfile.TemporaryDirectory(prefix="ann-manifest-") as tmp:
            path = _build(tmp)
            index = create_index("hnswlib", dim=4)
            index.load(path)
            self.assertFalse(index.validate(expected_model_id="Chinese-CLIP", expected_dim=4))
            self.assertIn("model_mismatch", index.incompatible_reason)

    def test_load_mismatched_dimension_rejected(self):
        with tempfile.TemporaryDirectory(prefix="ann-manifest-") as tmp:
            path = _build(tmp)
            index = create_index("hnswlib", dim=4)
            index.load(path)
            self.assertFalse(index.validate(expected_dim=512))
            self.assertIn("dimension_mismatch", index.incompatible_reason)

    def test_load_reload_recall(self):
        with tempfile.TemporaryDirectory(prefix="ann-manifest-") as tmp:
            path = _build(tmp)
            index = create_index("hnswlib", dim=4)
            index.load(path)
            rows = index.search([0.0, 0.0, 0.0, 0.0], k=5)
            self.assertEqual(len(rows), 5)
            self.assertEqual(rows[0][0], "asset_0")  # closest to zero vector


if __name__ == "__main__":
    unittest.main()
