"""Tests for Machine Learning pipeline, dataset manifest, evaluator, and latency profiler."""
import tempfile
import unittest
from pathlib import Path

from plant_rover.ml import DatasetManifest, ModelInferenceEngine, ModelEvaluator, profile_inference


class MLPipelineTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root_path = Path(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_dataset_manifest_group_splitting_prevents_leakage(self):
        """Verifies that plants are partitioned strictly so no plant crosses train/val/test splits."""
        manifest = DatasetManifest.create_synthetic_evaluation_manifest(self.root_path / "data", sample_count=30, seed=42)
        train_items = manifest.get_split("train")
        val_items = manifest.get_split("val")
        test_items = manifest.get_split("test")

        train_plants = {item.plant_id for item in train_items}
        val_plants = {item.plant_id for item in val_items}
        test_plants = {item.plant_id for item in test_items}

        # Mutual exclusivity check - zero data leakage
        self.assertTrue(train_plants.isdisjoint(val_plants))
        self.assertTrue(train_plants.isdisjoint(test_plants))
        self.assertTrue(val_plants.isdisjoint(test_plants))

    def test_model_inference_engine_uncertainty_handling(self):
        """Verifies that low-confidence or corrupt inputs trigger explicit UNCERTAIN output."""
        engine = ModelInferenceEngine(confidence_threshold=0.99)  # High threshold guarantees uncertainty
        manifest = DatasetManifest.create_synthetic_evaluation_manifest(self.root_path / "data", sample_count=10, seed=42)
        sample = manifest.items[0]

        result = engine.infer(sample.path)
        self.assertTrue(result.uncertain)
        self.assertEqual(result.predicted_class, "uncertain")
        self.assertIsNotNone(result.rejection_reason)

    def test_evaluator_computes_macro_f1_and_confusion_matrix(self):
        """Verifies evaluation pipeline calculations."""
        manifest = DatasetManifest.create_synthetic_evaluation_manifest(self.root_path / "data", sample_count=20, seed=42)
        engine = ModelInferenceEngine(mode=ModelInferenceEngine.MODE_FUSION, confidence_threshold=0.50)

        report = ModelEvaluator.evaluate(engine, manifest, split="test")
        self.assertGreaterEqual(report.total_samples, 1)
        self.assertGreaterEqual(report.accuracy, 0.0)
        self.assertLessEqual(report.accuracy, 1.0)
        self.assertIn("healthy", report.confusion_matrix)

    def test_compare_baselines_runs_all_three_modes(self):
        """Verifies comparison across Image-Only, Sensor-Only, and Fusion on held-out test set."""
        manifest = DatasetManifest.create_synthetic_evaluation_manifest(self.root_path / "data", sample_count=20, seed=42)
        comparisons = ModelEvaluator.compare_baselines(manifest, split="test")

        self.assertIn("image_only", comparisons)
        self.assertIn("sensor_only", comparisons)
        self.assertIn("fusion", comparisons)
        self.assertEqual(comparisons["image_only"].mode, "image_only")
        self.assertEqual(comparisons["sensor_only"].mode, "sensor_only")
        self.assertEqual(comparisons["fusion"].mode, "fusion")

    def test_profiler_measures_latency_on_pc(self):
        """Verifies CPU inference profiling."""
        manifest = DatasetManifest.create_synthetic_evaluation_manifest(self.root_path / "data", sample_count=5, seed=42)
        engine = ModelInferenceEngine()
        stats = profile_inference(engine, manifest.items[0].path, iterations=10)

        self.assertIn("mean_latency_ms", stats)
        self.assertIn("p95_latency_ms", stats)
        self.assertIn("fps_estimate", stats)
        self.assertGreater(stats["mean_latency_ms"], 0.0)


if __name__ == "__main__":
    unittest.main()
