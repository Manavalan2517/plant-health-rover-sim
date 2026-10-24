"""Comprehensive tests for model preprocessing contracts and Keras vs ONNX numerical parity.

Verifies:
1. Canonical MobileNetV2 preprocessing formula: (pixel / 127.5) - 1.0.
2. Parity between train_classifier.py and image_classifier.py preprocessing.
3. Equivalence with official tf.keras.applications.mobilenet_v2.preprocess_input.
4. Proof that ONNX graph performs preprocessing externally.
5. Numerical parity of class probabilities and feature embeddings between Keras and ONNX models.
6. Sensitivity proof that incorrect (e.g. [0, 255]) scaling degrades model outputs.
7. Unit norm and shape validation for class centroids.
"""
from __future__ import annotations

import json
import unittest
from pathlib import Path
import numpy as np
from PIL import Image

from plant_rover.ml.image_classifier import PlantLeafClassifier
from plant_rover.ml.train_classifier import preprocess_image_file


class PreprocessingContractAndParityTests(unittest.TestCase):
    """Test suite ensuring end-to-end preprocessing consistency and ONNX/Keras parity."""

    def setUp(self):
        self.models_dir = Path(__file__).parent.parent / "plant_rover" / "models"
        self.samples_dir = Path(__file__).parent.parent / "plant_rover" / "static" / "samples"
        self.classifier = PlantLeafClassifier(models_dir=self.models_dir)

    def test_01_canonical_preprocessing_formula_range_and_values(self):
        """1. Canonical formula (x / 127.5) - 1.0 scales [0, 255] exactly to [-1.0, 1.0]."""
        test_inputs = np.array([0.0, 63.75, 127.5, 191.25, 255.0], dtype=np.float32)
        expected = np.array([-1.0, -0.5, 0.0, 0.5, 1.0], dtype=np.float32)
        scaled = (test_inputs / 127.5) - 1.0
        np.testing.assert_allclose(scaled, expected, atol=1e-7)

        # Full uint8 dynamic range
        all_pixels = np.arange(256, dtype=np.float32)
        all_scaled = (all_pixels / 127.5) - 1.0
        self.assertEqual(all_scaled.min(), -1.0)
        self.assertEqual(all_scaled.max(), 1.0)
        self.assertAlmostEqual(all_scaled[128], (128 / 127.5) - 1.0, places=6)

    def test_02_train_and_inference_preprocessing_parity(self):
        """2. Preprocessing in train_classifier.py and image_classifier.py are bit-for-bit identical."""
        sample_path = self.samples_dir / "sample_septoria_leaf_spot.jpg"
        if not sample_path.exists():
            self.skipTest(f"Sample {sample_path} not found")

        # 1. Via train_classifier.preprocess_image_file
        t_train = preprocess_image_file(sample_path)

        # 2. Via image_classifier.PlantLeafClassifier.preprocess
        with Image.open(sample_path) as im:
            from PIL import ImageOps
            try:
                im = ImageOps.exif_transpose(im)
            except Exception:
                pass
            rgb = im.convert("RGB")
            t_infer = self.classifier.preprocess(rgb)[0]

        self.assertEqual(t_train.shape, (224, 224, 3))
        self.assertEqual(t_infer.shape, (224, 224, 3))
        max_diff = float(np.max(np.abs(t_train - t_infer)))
        self.assertEqual(max_diff, 0.0, f"Discrepancy detected between training and inference preprocessing: {max_diff}")

    def test_03_reference_keras_mobilenetv2_preprocess_input_equivalence(self):
        """3. Canonical formula matches tf.keras.applications.mobilenet_v2.preprocess_input exactly."""
        try:
            import tensorflow as tf
            from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as keras_preprocess
        except ImportError:
            self.skipTest("TensorFlow/Keras not available in this environment")

        # Grid of all possible 256 pixel values across 3 channels
        raw_pixels = np.arange(256, dtype=np.float32).reshape(1, 16, 16, 1)
        raw_rgb = np.repeat(raw_pixels, 3, axis=-1)

        keras_out = keras_preprocess(raw_rgb.copy())
        formula_out = (raw_rgb / 127.5) - 1.0

        max_abs_diff = float(np.max(np.abs(keras_out - formula_out)))
        self.assertEqual(max_abs_diff, 0.0, f"Discrepancy between Keras reference and formula: {max_abs_diff}")
        self.assertEqual(keras_out.min(), -1.0)
        self.assertEqual(keras_out.max(), 1.0)

    def test_04_onnx_graph_preprocessing_is_external(self):
        """4. ONNX graph expects pre-scaled [-1.0, 1.0] input and contains no embedded normalization."""
        try:
            import onnx
        except ImportError:
            self.skipTest("onnx package not available in this environment")

        onnx_path = self.models_dir / "plant_disease_mobilenetv2.onnx"
        self.assertTrue(onnx_path.exists())
        model = onnx.load(str(onnx_path))

        # Check model input signature
        input_names = [inp.name for inp in model.graph.input]
        self.assertIn("image_input", input_names)

        # Check first nodes: image_input connects to Transpose -> Conv1
        # Proves no Sub, Div, or Mul normalization node precedes Conv1
        first_nodes = model.graph.node[:5]
        input_consumers = [n for n in model.graph.node if "image_input" in n.input]
        self.assertEqual(len(input_consumers), 1)
        self.assertEqual(input_consumers[0].op_type, "Transpose")

        # Following node is the first convolution
        conv_nodes = [n for n in model.graph.node if input_consumers[0].output[0] in n.input]
        self.assertTrue(any(n.op_type == "Conv" for n in conv_nodes))

    def test_05_onnx_and_keras_predictions_numerical_parity(self):
        """5. ONNX class probabilities match Keras model output within numerical tolerance (1e-5)."""
        try:
            from tensorflow import keras
            import onnxruntime as ort
        except ImportError:
            self.skipTest("TensorFlow or ONNX Runtime not available")

        keras_path = self.models_dir / "plant_disease_mobilenetv2.keras"
        onnx_path = self.models_dir / "plant_disease_mobilenetv2.onnx"
        if not keras_path.exists() or not onnx_path.exists():
            self.skipTest("Model files missing")

        keras_model = keras.models.load_model(str(keras_path))
        onnx_sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])

        test_files = [
            self.samples_dir / "sample_septoria_leaf_spot.jpg",
            self.samples_dir / "sample_healthy.jpg",
            self.samples_dir / "sample_yellow_leaf_curl.jpg",
        ]

        for p in test_files:
            if not p.exists():
                continue
            tensor = preprocess_image_file(p).reshape(1, 224, 224, 3)

            # Keras inference
            k_probs = keras_model(tensor, training=False).numpy()[0]

            # ONNX inference
            o_outs = onnx_sess.run(None, {"image_input": tensor})
            o_probs = o_outs[0][0]

            # Maximum absolute probability error must be < 1e-5 (typically < 1e-7)
            prob_diff = float(np.max(np.abs(k_probs - o_probs)))
            self.assertLess(prob_diff, 1e-5, f"Probability diff on {p.name}: {prob_diff}")
            self.assertEqual(int(np.argmax(k_probs)), int(np.argmax(o_probs)))

    def test_06_onnx_and_keras_feature_embeddings_numerical_parity(self):
        """6. ONNX feature embeddings match Keras backbone output (cos_sim >= 0.99999, diff < 1e-4)."""
        try:
            from tensorflow import keras
            import onnxruntime as ort
        except ImportError:
            self.skipTest("TensorFlow or ONNX Runtime not available")

        keras_path = self.models_dir / "plant_disease_mobilenetv2.keras"
        onnx_path = self.models_dir / "plant_disease_mobilenetv2.onnx"
        if not keras_path.exists() or not onnx_path.exists():
            self.skipTest("Model files missing")

        keras_model = keras.models.load_model(str(keras_path))
        backbone_layer = keras_model.get_layer("mobilenetv2_1.00_224")
        onnx_sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])

        test_files = [
            self.samples_dir / "sample_septoria_leaf_spot.jpg",
            self.samples_dir / "sample_healthy.jpg",
            self.samples_dir / "sample_yellow_leaf_curl.jpg",
        ]

        for p in test_files:
            if not p.exists():
                continue
            tensor = preprocess_image_file(p).reshape(1, 224, 224, 3)

            # Keras backbone output
            k_feats = backbone_layer(tensor, training=False).numpy()[0]

            # ONNX dual output (output 1 is the 1280-dim feature vector)
            o_outs = onnx_sess.run(None, {"image_input": tensor})
            self.assertGreaterEqual(len(o_outs), 2, "ONNX model must have at least 2 outputs (probs, feats)")
            o_feats = o_outs[1][0]

            self.assertEqual(k_feats.shape, (1280,))
            self.assertEqual(o_feats.shape, (1280,))

            max_diff = float(np.max(np.abs(k_feats - o_feats)))
            cos_sim = float(np.dot(k_feats, o_feats) / (np.linalg.norm(k_feats) * np.linalg.norm(o_feats)))

            self.assertLess(max_diff, 1e-4, f"Max embedding diff on {p.name}: {max_diff}")
            self.assertGreaterEqual(cos_sim, 0.99999, f"Cosine similarity on {p.name}: {cos_sim}")

    def test_07_class_manifest_preprocessing_contract(self):
        """7. class_manifest.json explicitly records the [-1.0, 1.0] formula contract."""
        manifest_path = self.models_dir / "class_manifest.json"
        self.assertTrue(manifest_path.exists())
        with open(manifest_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.assertIn("preprocessing", data)
        prep = data["preprocessing"]
        self.assertEqual(prep.get("format"), "RGB")
        self.assertEqual(prep.get("resize"), [224, 224])
        self.assertEqual(prep.get("scaling"), "[-1.0, 1.0]")
        self.assertEqual(prep.get("formula"), "(pixel / 127.5) - 1.0")

    def test_08_sensitivity_to_incorrect_preprocessing(self):
        """8. Feeding raw [0, 255] pixels produces degenerate outputs, proving [-1, 1] requirement."""
        sample_path = self.samples_dir / "sample_septoria_leaf_spot.jpg"
        if not sample_path.exists():
            self.skipTest(f"Sample {sample_path} not found")

        with Image.open(sample_path) as im:
            resized = im.convert("RGB").resize((224, 224), Image.Resampling.BILINEAR)
            arr = np.array(resized, dtype=np.float32)

        t_correct = np.expand_dims((arr / 127.5) - 1.0, axis=0)
        t_raw_255 = np.expand_dims(arr, axis=0)

        # Run ONNX inference
        prob_correct = self.classifier._predict_raw(t_correct)
        prob_raw = self.classifier._predict_raw(t_raw_255)

        # Correct preprocessing correctly identifies Septoria (class 4) with high confidence (>90%)
        self.assertEqual(int(np.argmax(prob_correct)), 4)
        self.assertGreater(prob_correct[4], 0.90)

        # Raw [0, 255] fails completely (Septoria probability collapses to < 5%)
        self.assertNotEqual(int(np.argmax(prob_raw)), 4)
        self.assertLess(prob_raw[4], 0.05)

    def test_09_centroids_unit_norm_and_separation(self):
        """9. class_centroids.npy has shape (10, 1280) and unit norm."""
        centroids_path = self.models_dir / "class_centroids.npy"
        self.assertTrue(centroids_path.exists())
        centroids = np.load(centroids_path)
        self.assertEqual(centroids.shape, (10, 1280))

        norms = np.linalg.norm(centroids, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-5)


if __name__ == "__main__":
    unittest.main()
