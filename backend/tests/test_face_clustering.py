import math
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

try:
    import numpy as np
except ImportError:
    np = None

from backend.db import MemoryStore
from backend.face_clustering import FaceClusterer, FaceSample, pairwise_metrics
from backend.face_embeddings import (
    AdaFaceAdapter,
    EmbeddingResult,
    FaceEmbeddingAdapter,
    FaceEmbeddingUnavailable,
    compute_face_quality,
    pose_bucket,
)
from backend.model_clients import FaceAdapter


def vector(angle_degrees, radius=1.0):
    angle = math.radians(angle_degrees)
    return [radius * math.cos(angle), radius * math.sin(angle)]


class FaceEmbeddingContractTests(unittest.TestCase):
    def test_adapter_normalizes_embedding_and_exposes_model_version(self):
        adapter = FaceEmbeddingAdapter(
            model_name="adaface",
            model_version="test-1",
            backend=lambda _: [3.0, 4.0],
        )

        result = adapter.embed("aligned-face")

        self.assertIsInstance(result, EmbeddingResult)
        self.assertEqual(result.model_name, "adaface")
        self.assertEqual(result.model_version, "test-1")
        self.assertAlmostEqual(result.embedding[0], 0.6)
        self.assertAlmostEqual(result.embedding[1], 0.8)
        self.assertAlmostEqual(result.quality_signal, 5.0)

    def test_quality_keeps_profile_faces_as_weak_evidence(self):
        frontal = compute_face_quality(0.95, 0.08, 0.9, [0.0, 4.0, 1.0])
        profile = compute_face_quality(0.95, 0.08, 0.9, [0.0, 48.0, 1.0])

        self.assertGreater(frontal, profile)
        self.assertGreater(profile, 0.0)
        self.assertLessEqual(frontal, 1.0)
        self.assertEqual(pose_bucket([0.0, 4.0, 1.0]), "frontal")
        self.assertEqual(pose_bucket([0.0, 48.0, 1.0]), "profile_right")

    @unittest.skipUnless(np, "NumPy is required for InsightFace runtime pose data")
    def test_quality_accepts_insightface_numpy_pose_array(self):
        pose = np.asarray([0.0, 24.0, 1.0], dtype="float32")

        quality = compute_face_quality(0.90, 0.12, 0.0, pose)

        self.assertGreater(quality, 0.0)
        self.assertEqual(pose_bucket(pose), "profile_right")

    def test_missing_adaface_checkpoint_fails_loudly_without_arcface_fallback(self):
        adapter = AdaFaceAdapter(model_path="/tmp/sentrix-missing-adaface.ckpt")

        with self.assertRaises(FaceEmbeddingUnavailable):
            adapter.embed("aligned-face")

    def test_official_checkpoint_loader_requests_full_trusted_checkpoint(self):
        adapter = AdaFaceAdapter(model_path="/tmp/official.ckpt")
        adapter.model_path = type("PathLike", (), {"is_file": lambda self: True, "name": "official.ckpt"})()
        calls = {}

        class FakeTorch:
            @staticmethod
            def load(path, map_location=None, **kwargs):
                calls.update(kwargs)
                return {"model.weight": None}

        class FakeNet:
            @staticmethod
            def build_model(_):
                class Model:
                    def load_state_dict(self, _):
                        raise RuntimeError("stop after load contract")
                return Model()

        with patch.dict("sys.modules", {"torch": FakeTorch(), "net": FakeNet()}):
            with self.assertRaises(FaceEmbeddingUnavailable):
                adapter._load_model()

        self.assertFalse(calls["weights_only"])

    def test_official_checkpoint_loader_discovers_adjacent_official_net_module(self):
        with tempfile.TemporaryDirectory() as directory:
            repo_root = Path(directory)
            checkpoint = repo_root / "pretrained" / "official.ckpt"
            checkpoint.parent.mkdir()
            checkpoint.write_bytes(b"checkpoint")
            (repo_root / "net.py").write_text("# official adapter entry\n", encoding="utf-8")
            adapter = AdaFaceAdapter(model_path=checkpoint)

            self.assertEqual(adapter._repository_root(), str(repo_root))

    def test_face_adapter_reports_detection_and_identity_readiness_separately(self):
        with patch.dict("os.environ", {"FACE_EMBEDDING_MODE": "adaface", "ADAFACE_MODEL_PATH": "/tmp/missing.ckpt"}, clear=False):
            adapter = FaceAdapter()

        self.assertTrue(adapter.enabled)
        self.assertFalse(adapter.identity_ready)
        self.assertEqual(adapter.identity_model, "adaface")
        self.assertIn("checkpoint", adapter.identity_error.lower())

    def test_identity_ready_requires_successful_runtime_inference(self):
        with patch.dict("os.environ", {"FACE_EMBEDDING_MODE": "adaface", "ADAFACE_MODEL_PATH": "/tmp/missing.ckpt"}, clear=False):
            adapter = FaceAdapter()

        self.assertFalse(adapter.identity_configured)
        self.assertFalse(adapter.identity_ready)
        adapter.identity_runtime_error = "runtime load failed"
        self.assertFalse(adapter.identity_ready)

    def test_face_adapter_uses_configured_identity_adapter_result(self):
        class FakeDetection:
            bbox = [0, 0, 60, 60]
            det_score = 0.95
            pose = [0, 0, 0]

            class Embedding:
                @staticmethod
                def tolist():
                    return [0.0, 1.0]

            embedding = Embedding()

        class FakeApp:
            def __init__(self):
                self.image = None

            def prepare(self, **kwargs):
                return None

            def get(self, image):
                self.image = image
                return [FakeDetection()]

        class FakeIdentity:
            model_version = "ada-test"
            available = True

            def embed(self, crop):
                return EmbeddingResult([3.0, 4.0], "adaface", "ada-test", 7.5)

        class FakeImage:
            shape = (64, 64, 3)

            def __getitem__(self, key):
                return self

        class FakeCv2:
            @staticmethod
            def imread(path):
                return FakeImage()

        with tempfile.TemporaryDirectory() as directory:
            image_path = f"{directory}/face.jpg"
            open(image_path, "wb").close()
            adapter = FaceAdapter.__new__(FaceAdapter)
            adapter.enabled = True
            adapter._app = FakeApp()
            adapter.error = None
            adapter.identity_model = "adaface"
            adapter.identity_adapter = FakeIdentity()
            adapter.identity_error = None

            with patch.dict(os.environ, {"FACE_MIN_SIZE": "1"}), patch.dict(
                sys.modules, {"cv2": FakeCv2()}
            ), patch("PIL.Image.fromarray", return_value="crop"):
                results = adapter.detect(image_path)

        self.assertTrue(results, f"detect error={adapter.error!r}")
        self.assertEqual(results[0]["embedding"], [3.0, 4.0])
        self.assertEqual(results[0]["embedding_version"], "ada-test")
        self.assertEqual(results[0]["quality_signal"], 7.5)

    def test_face_adapter_limits_insightface_modules_when_adaface_owns_identity(self):
        calls = {}

        class FakeAnalysis:
            def __init__(self, **kwargs):
                calls.update(kwargs)

            def prepare(self, **kwargs):
                return None

        adapter = FaceAdapter.__new__(FaceAdapter)
        adapter.enabled = True
        adapter._app = None
        adapter._load_lock = __import__("threading").Lock()
        adapter.error = None
        adapter.identity_model = "adaface"
        adapter.identity_adapter = type("Identity", (), {"available": True, "model_version": "test"})()
        adapter.identity_error = None

        class FakeCv2:
            @staticmethod
            def imread(_):
                return None

        with patch.dict(sys.modules, {"insightface.app": type("App", (), {"FaceAnalysis": FakeAnalysis})(), "cv2": FakeCv2()}):
            adapter.detect("unused.jpg")

        self.assertEqual(calls["allowed_modules"], ["detection", "landmark_2d_106"])
        self.assertEqual(calls["providers"], ["CUDAExecutionProvider", "CPUExecutionProvider"])

    def test_face_adapter_passes_five_point_landmarks_to_alignment(self):
        class FakeDetection:
            bbox = [0, 0, 60, 60]
            det_score = 0.95
            pose = [0, 0, 0]
            kps = [[10, 12], [42, 12], [26, 28], [14, 46], [40, 46]]

            class Embedding:
                @staticmethod
                def tolist():
                    return [0.0, 1.0]

            embedding = Embedding()

        class FakeApp:
            def __init__(self):
                self.image = None

            def prepare(self, **kwargs):
                return None

            def get(self, image):
                self.image = image
                return [FakeDetection()]

        class FakeIdentity:
            model_version = "ada-test"
            available = True

            def embed(self, crop):
                self.crop = crop
                return EmbeddingResult([3.0, 4.0], "adaface", "ada-test", 7.5)

        class FakeImage:
            shape = (64, 64, 3)

            def __getitem__(self, key):
                return self

        class FakeCv2:
            @staticmethod
            def imread(path):
                return FakeImage()

        identity = FakeIdentity()
        app = FakeApp()
        adapter = FaceAdapter.__new__(FaceAdapter)
        adapter.enabled = True
        adapter._app = app
        adapter.error = None
        adapter.identity_model = "adaface"
        adapter.identity_adapter = identity
        adapter.identity_error = None

        with tempfile.TemporaryDirectory() as directory:
            image_path = f"{directory}/face.jpg"
            open(image_path, "wb").close()
            with patch.dict(os.environ, {"FACE_MIN_SIZE": "1"}), patch.dict(
                sys.modules, {"cv2": FakeCv2()}
            ), patch(
                "backend.model_clients.align_face_crop", return_value="aligned"
            ) as align:
                results = adapter.detect(image_path)

        self.assertTrue(results)
        align.assert_called_once_with(app.image, [0.0, 0.0, 60.0, 60.0], [[10.0, 12.0], [42.0, 12.0], [26.0, 28.0], [14.0, 46.0], [40.0, 46.0]])
        self.assertEqual(identity.crop, "aligned")

    def test_unavailable_identity_model_does_not_create_empty_embedding_cluster(self):
        with tempfile.TemporaryDirectory() as directory:
            store = MemoryStore(f"{directory}/memory.db")
            store.create_asset("asset-identity", "face.jpg", "image", "/tmp/face.jpg", "image/jpeg")
            observation = store.add_observation("asset-identity", {"caption": "人脸"})

            saved = store.add_face_instance(
                "asset-identity",
                observation["id"],
                {"bbox": [1, 2, 30, 40], "embedding": [], "confidence": 0.95, "identity_ready": False},
            )

            self.assertIsNone(saved)
            self.assertEqual(store.count("face_instances"), 0)
            self.assertEqual(store.count("face_clusters"), 0)
            store.close()

    def test_detection_marks_small_low_confidence_face_as_evidence_only(self):
        class FakeDetection:
            bbox = [0, 0, 50, 50]
            det_score = 0.60
            pose = [0, 0, 0]
            kps = []

            class Embedding:
                @staticmethod
                def tolist():
                    return [0.0, 1.0]

            embedding = Embedding()

        class FakeApp:
            def prepare(self, **kwargs):
                return None

            def get(self, image):
                return [FakeDetection()]

        class FakeIdentity:
            model_version = "ada-test"
            available = True

            def embed(self, crop):
                return EmbeddingResult([3.0, 4.0], "adaface", "ada-test", 20.0)

        class FakeImage:
            shape = (480, 480, 3)

            def __getitem__(self, key):
                return self

        class FakeCv2:
            @staticmethod
            def imread(path):
                return FakeImage()

        adapter = FaceAdapter.__new__(FaceAdapter)
        adapter.enabled = True
        adapter._app = FakeApp()
        adapter.error = None
        adapter.identity_model = "adaface"
        adapter.identity_adapter = FakeIdentity()
        adapter.identity_error = None
        adapter.identity_runtime_error = None

        with tempfile.TemporaryDirectory() as directory:
            image_path = f"{directory}/face.jpg"
            open(image_path, "wb").close()
            with patch.dict(
                os.environ,
                {"FACE_MIN_SIZE": "1", "FACE_MIN_DETECTION_SCORE": "0.01"},
            ), patch.dict(sys.modules, {"cv2": FakeCv2()}), patch("PIL.Image.fromarray", return_value="crop"):
                result = adapter.detect(image_path)[0]

        self.assertFalse(result["identity_eligible"])
        self.assertLess(result["quality"], 1.0)
        self.assertEqual(result["quality_signal"], 20.0)


class FaceClustererTests(unittest.TestCase):
    def test_frontal_and_profile_views_share_multi_view_cluster(self):
        samples = [
            FaceSample("frontal", vector(0), quality=0.98, pose_bucket="frontal"),
            FaceSample("left", vector(24), quality=0.90, pose_bucket="profile_left"),
            FaceSample("right", vector(-24), quality=0.88, pose_bucket="profile_right"),
        ]

        result = FaceClusterer(match_threshold=0.78).fit(samples)

        self.assertEqual(len(result.clusters), 1)
        self.assertEqual(result.labels["frontal"], result.labels["left"])
        self.assertEqual(result.labels["frontal"], result.labels["right"])
        self.assertEqual(
            {item.pose_bucket for item in result.clusters[0].prototypes},
            {"frontal", "profile_left", "profile_right"},
        )

    def test_low_quality_bridge_does_not_merge_two_identities(self):
        samples = [
            FaceSample("person_a", vector(0), quality=0.98, pose_bucket="frontal"),
            FaceSample("person_b", vector(48), quality=0.98, pose_bucket="frontal"),
            FaceSample("low_quality_bridge", vector(24), quality=0.08, pose_bucket="unknown"),
        ]

        result = FaceClusterer(match_threshold=0.78, minimum_quality=0.30).fit(samples)

        self.assertNotEqual(result.labels["person_a"], result.labels["person_b"])
        self.assertNotEqual(result.labels["low_quality_bridge"], result.labels["person_a"])
        self.assertNotEqual(result.labels["low_quality_bridge"], result.labels["person_b"])

    def test_pairwise_metrics_report_missed_and_false_merges(self):
        metrics = pairwise_metrics(
            predicted={"a1": "cluster-1", "a2": "cluster-1", "b1": "cluster-2", "b2": "cluster-1"},
            truth={"a1": "person-a", "a2": "person-a", "b1": "person-b", "b2": "person-b"},
        )

        self.assertEqual(metrics["true_positive"], 1)
        self.assertEqual(metrics["false_positive"], 2)
        self.assertEqual(metrics["false_negative"], 1)
        self.assertAlmostEqual(metrics["precision"], 1 / 3)
        self.assertAlmostEqual(metrics["recall"], 1 / 2)
        self.assertAlmostEqual(metrics["f1"], 2 / 5)

    def test_different_embedding_models_do_not_share_a_cluster(self):
        samples = [
            FaceSample("ada", vector(0), quality=0.98, model_name="adaface", model_version="v1"),
            FaceSample("mag", vector(0), quality=0.98, model_name="magface", model_version="v1"),
        ]

        result = FaceClusterer(match_threshold=0.78).fit(samples)

        self.assertNotEqual(result.labels["ada"], result.labels["mag"])

    def test_clusterer_default_threshold_supports_aligned_adaface_views(self):
        samples = [
            FaceSample("first", vector(0), quality=0.90, pose_bucket="frontal"),
            FaceSample("second", vector(60), quality=0.88, pose_bucket="frontal"),
        ]

        result = FaceClusterer().fit(samples)

        self.assertEqual(result.labels["first"], result.labels["second"])


class FacePersistenceTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.store = MemoryStore(f"{self.temp_dir.name}/sentrix.db")
        self.store.create_asset("asset-1", "one.jpg", "image", "/tmp/one.jpg", "image/jpeg")
        self.store.create_asset("asset-2", "two.jpg", "image", "/tmp/two.jpg", "image/jpeg")
        self.obs_one = self.store.add_observation("asset-1", {"caption": "人物一"})
        self.obs_two = self.store.add_observation("asset-2", {"caption": "人物一"})

    def tearDown(self):
        self.store.close()
        self.temp_dir.cleanup()

    def test_face_instance_persists_quality_pose_and_embedding_provenance(self):
        face = self.store.add_face_instance(
            "asset-1",
            self.obs_one["id"],
            {
                "bbox": [1, 2, 101, 122],
                "embedding": vector(0),
                "confidence": 0.96,
                "quality": 0.88,
                "area_ratio": 0.08,
                "sharpness": 0.91,
                "pose": [0, 8, 1],
                "pose_bucket": "frontal",
                "embedding_model": "adaface",
                "embedding_version": "ir50-ms1mv2",
            },
        )

        saved = self.store.get_face_instance(face["id"])

        self.assertAlmostEqual(saved["quality"], 0.88)
        self.assertEqual(saved["pose"], [0.0, 8.0, 1.0])
        self.assertEqual(saved["pose_bucket"], "frontal")
        self.assertEqual(saved["embedding_model"], "adaface")
        self.assertEqual(saved["embedding_version"], "ir50-ms1mv2")

    def test_cluster_keeps_one_high_quality_prototype_per_view_bucket(self):
        first = self.store.add_face_instance(
            "asset-1", self.obs_one["id"],
            {"bbox": [1, 2, 101, 122], "embedding": vector(0), "confidence": 0.96, "quality": 0.98, "pose_bucket": "frontal", "embedding_model": "adaface", "embedding_version": "test"},
        )
        self.store.add_face_instance(
            "asset-2", self.obs_two["id"],
            {"bbox": [1, 2, 101, 122], "embedding": vector(24), "confidence": 0.90, "quality": 0.90, "pose_bucket": "profile_left", "embedding_model": "adaface", "embedding_version": "test"},
            threshold=0.78,
        )
        prototypes = self.store.list_face_prototypes(first["cluster_id"])

        self.assertEqual({item["pose_bucket"] for item in prototypes}, {"frontal", "profile_left"})
        self.assertEqual({item["model_name"] for item in prototypes}, {"adaface"})

    def test_online_low_quality_face_does_not_join_existing_cluster(self):
        first = self.store.add_face_instance(
            "asset-1", self.obs_one["id"],
            {"bbox": [1, 2, 101, 122], "embedding": vector(0), "confidence": 0.96, "quality": 0.98, "pose_bucket": "frontal"},
            threshold=0.78,
        )
        low_quality = self.store.add_face_instance(
            "asset-2", self.obs_two["id"],
            {"bbox": [1, 2, 101, 122], "embedding": vector(24), "confidence": 0.90, "quality": 0.08, "pose_bucket": "unknown"},
            threshold=0.78,
        )

        self.assertNotEqual(first["cluster_id"], low_quality["cluster_id"])
        cluster = self.store._row("SELECT status, confidence FROM face_clusters WHERE id = ?", (low_quality["cluster_id"],))
        self.assertEqual(cluster["status"], "pending")
        self.assertLess(cluster["confidence"], 0.30)

    def test_evidence_only_face_has_no_cluster_or_candidate_entity(self):
        saved = self.store.add_face_instance(
            "asset-1", self.obs_one["id"],
            {
                "bbox": [1, 2, 31, 32], "embedding": vector(0), "confidence": 0.56,
                "quality": 0.36, "area_ratio": 0.01, "identity_eligible": False,
                "embedding_model": "adaface", "embedding_version": "test",
            },
        )

        self.assertIsNone(saved["cluster_id"])
        self.assertEqual(self.store.count("face_instances"), 1)
        self.assertEqual(self.store.count("face_clusters"), 0)
        self.assertEqual(self.store.count("entities"), 0)

    def test_global_recluster_does_not_join_two_people_through_low_quality_bridge(self):
        samples = [
            ("a", vector(0), 0.98),
            ("b", vector(48), 0.98),
            ("bridge", vector(24), 0.08),
        ]
        for index, (name, embedding, quality) in enumerate(samples):
            asset_id = f"asset-{index + 1}"
            if not self.store.get_asset(asset_id):
                self.store.create_asset(asset_id, f"{name}.jpg", "image", f"/tmp/{name}.jpg", "image/jpeg")
            observation = self.store.add_observation(asset_id, {"caption": name})
            self.store.add_face_instance(
                asset_id,
                observation["id"],
                {"bbox": [1, 2, 30, 40], "embedding": embedding, "confidence": 0.95, "quality": quality, "pose_bucket": "frontal"},
                threshold=0.78,
            )

        self.store.recluster_faces(threshold=0.78, minimum_quality=0.30)
        rows = self.store._rows("SELECT fi.id, a.file_name, fi.cluster_id FROM face_instances fi JOIN assets a ON a.id = fi.asset_id")
        labels = {row["file_name"]: row["cluster_id"] for row in rows}

        self.assertNotEqual(labels["one.jpg"], labels["two.jpg"])

    def test_global_recluster_preserves_confirmed_entity_when_old_clusters_merge(self):
        first = self.store.add_face_instance(
            "asset-1", self.obs_one["id"],
            {"bbox": [1, 2, 101, 122], "embedding": vector(0), "confidence": 0.90, "quality": 0.80, "pose_bucket": "frontal"},
            threshold=0.78,
        )
        second = self.store.add_face_instance(
            "asset-2", self.obs_two["id"],
            {"bbox": [1, 2, 101, 122], "embedding": vector(8), "confidence": 0.99, "quality": 0.99, "pose_bucket": "frontal"},
            threshold=0.999,
        )
        source_entity_id = self.store._row("SELECT entity_id FROM face_clusters WHERE id = ?", (second["cluster_id"],))["entity_id"]
        confirmed = self.store.confirm_face_cluster(first["cluster_id"], "妈妈", "母亲")
        result = self.store.recluster_faces(threshold=0.78, minimum_quality=0.30)

        self.assertEqual(result["clusters"], 1)
        cluster = self.store._row("SELECT * FROM face_clusters WHERE id = ?", (first["cluster_id"],))
        other = self.store._row("SELECT * FROM face_clusters WHERE id = ?", (second["cluster_id"],))
        self.assertEqual(cluster["status"], "confirmed")
        self.assertEqual(cluster["entity_id"], confirmed["entity"]["id"])
        self.assertEqual(other["status"], "rejected")
        self.assertEqual(self.store.get_entity(source_entity_id)["status"], "rejected")
        self.assertNotIn(source_entity_id, {item["id"] for item in self.store.list_entities()})

    def test_global_recluster_never_merges_two_confirmed_people(self):
        first = self.store.add_face_instance(
            "asset-1", self.obs_one["id"],
            {"bbox": [1, 2, 101, 122], "embedding": vector(0), "confidence": 0.98, "quality": 0.98, "pose_bucket": "frontal"},
            threshold=0.78,
        )
        second = self.store.add_face_instance(
            "asset-2", self.obs_two["id"],
            {"bbox": [1, 2, 101, 122], "embedding": vector(8), "confidence": 0.98, "quality": 0.98, "pose_bucket": "frontal"},
            threshold=0.999,
        )
        first_detail = self.store.confirm_face_cluster(first["cluster_id"], "妈妈", "母亲")
        second_detail = self.store.confirm_face_cluster(second["cluster_id"], "爸爸", "父亲")

        result = self.store.recluster_faces(threshold=0.78, minimum_quality=0.30)
        clusters = self.store._rows("SELECT id, status, entity_id FROM face_clusters WHERE status != 'rejected'")

        self.assertEqual(result["clusters"], 2)
        self.assertEqual({item["entity_id"] for item in clusters}, {first_detail["entity"]["id"], second_detail["entity"]["id"]})
        self.assertEqual({item["status"] for item in clusters}, {"confirmed"})

    def test_user_merge_and_split_write_auditable_revisions(self):
        first = self.store.add_face_instance(
            "asset-1", self.obs_one["id"],
            {"bbox": [1, 2, 101, 122], "embedding": vector(0), "confidence": 0.95, "quality": 0.95, "pose_bucket": "frontal"},
            threshold=0.999,
        )
        second = self.store.add_face_instance(
            "asset-2", self.obs_two["id"],
            {"bbox": [1, 2, 101, 122], "embedding": vector(48), "confidence": 0.95, "quality": 0.95, "pose_bucket": "frontal"},
            threshold=0.999,
        )
        self.assertNotEqual(first["cluster_id"], second["cluster_id"])

        merged = self.store.merge_face_clusters(first["cluster_id"], second["cluster_id"], source="user_merge")
        self.assertEqual(merged["member_count"], 2)
        self.assertEqual(self.store.count("entity_revisions"), 1)

        split = self.store.split_face_instance(merged["id"], second["id"], source="user_split")
        self.assertNotEqual(split["id"], merged["id"])
        self.assertEqual(self.store.count("entity_revisions"), 2)


if __name__ == "__main__":
    unittest.main()
