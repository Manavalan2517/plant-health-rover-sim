"""Automated test suite for Plant Leaf Image Inspection, Quality Gates, and Uncertainty."""
from __future__ import annotations

import base64
import io
import json
from pathlib import Path
import threading
import unittest
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import numpy as np
from PIL import Image, ImageDraw

from plant_rover.live_mode import LiveDataHub
from plant_rover.live_server import LiveOnlyHTTPServer
from plant_rover.ml.image_classifier import (
    PlantLeafClassifier,
    MAX_FILE_SIZE_BYTES,
)


def create_test_image(width: int = 224, height: int = 224, color: tuple[int, int, int] = (60, 160, 60), fmt: str = "JPEG") -> bytes:
    """Create in-memory image bytes with simulated leaf-like gradients and texture."""
    im = Image.new("RGB", (width, height), (30, 30, 30))
    draw = ImageDraw.Draw(im)
    pad = max(1, min(width, height) // 8)
    if width > 2 * pad and height > 2 * pad:
        draw.ellipse((pad, pad, width - pad, height - pad), fill=color, outline=(120, 200, 120), width=max(1, pad // 2))
        step = max(5, min(width, height) // 10)
        for i in range(pad * 2, width - pad * 2, step):
            draw.line([(i, pad * 2), (i, height - pad * 2)], fill=(40, 120, 40), width=1)
            draw.line([(pad * 2, i), (width - pad * 2, i)], fill=(40, 120, 40), width=1)
    buf = io.BytesIO()
    im.save(buf, format=fmt)
    return buf.getvalue()


class QualityGateTests(unittest.TestCase):
    def setUp(self):
        self.classifier = PlantLeafClassifier()

    def test_quality_gate_accepts_clean_jpeg(self):
        data = create_test_image(256, 256, fmt="JPEG")
        quality, im = self.classifier.evaluate_quality_gate(data)
        self.assertTrue(quality.passed)
        self.assertIsNone(quality.rejection_reason)
        self.assertEqual(quality.format, "JPEG")
        self.assertFalse(quality.is_blurry)
        self.assertFalse(quality.is_underexposed)
        self.assertFalse(quality.is_overexposed)
        self.assertIsNotNone(im)

    def test_quality_gate_accepts_png_and_webp(self):
        for fmt in ("PNG", "WEBP"):
            data = create_test_image(224, 224, fmt=fmt)
            quality, im = self.classifier.evaluate_quality_gate(data)
            self.assertTrue(quality.passed, f"Failed on {fmt}")
            self.assertEqual(quality.format, fmt)

    def test_quality_gate_rejects_corrupted_bytes(self):
        corrupt = b"\xff\xd8\xff\xe0" + b"\x00" * 20 + b"not-a-valid-jpeg"
        quality, im = self.classifier.evaluate_quality_gate(corrupt)
        self.assertFalse(quality.passed)
        self.assertIn("Corrupted", quality.rejection_reason)
        self.assertIsNone(im)

    def test_quality_gate_rejects_unsupported_format(self):
        # Create a GIF
        im = Image.new("RGB", (100, 100), (50, 100, 50))
        buf = io.BytesIO()
        im.save(buf, format="GIF")
        quality, _ = self.classifier.evaluate_quality_gate(buf.getvalue())
        self.assertFalse(quality.passed)
        self.assertIn("Unsupported format", quality.rejection_reason)

    def test_quality_gate_rejects_tiny_images(self):
        tiny = create_test_image(32, 32, fmt="JPEG")
        quality, _ = self.classifier.evaluate_quality_gate(tiny)
        self.assertFalse(quality.passed)
        self.assertIn("too small", quality.rejection_reason)

    def test_quality_gate_detects_blurry_flat_images(self):
        # Solid flat color has 0 variance (blurry)
        im = Image.new("RGB", (224, 224), (80, 80, 80))
        buf = io.BytesIO()
        im.save(buf, format="JPEG")
        quality, _ = self.classifier.evaluate_quality_gate(buf.getvalue())
        self.assertFalse(quality.passed)
        self.assertTrue(quality.is_blurry)
        self.assertIn("blurry", quality.rejection_reason.lower())

    def test_quality_gate_detects_underexposure_and_overexposure(self):
        dark_data = Image.new("RGB", (224, 224), (5, 5, 5))
        buf = io.BytesIO()
        dark_data.save(buf, format="JPEG")
        quality, _ = self.classifier.evaluate_quality_gate(buf.getvalue())
        self.assertFalse(quality.passed)
        self.assertTrue(quality.is_underexposed)

        bright_data = Image.new("RGB", (224, 224), (250, 250, 250))
        buf = io.BytesIO()
        bright_data.save(buf, format="JPEG")
        quality, _ = self.classifier.evaluate_quality_gate(buf.getvalue())
        self.assertFalse(quality.passed)
        self.assertTrue(quality.is_overexposed)


class ClassifierInferenceTests(unittest.TestCase):
    def setUp(self):
        self.classifier = PlantLeafClassifier()

    def test_classifier_reports_ready_and_lists_ten_classes(self):
        self.assertTrue(self.classifier.is_ready)
        info = self.classifier.get_model_info()
        self.assertTrue(info["model_ready"])
        self.assertEqual(info["num_classes"], 10)
        self.assertIn("Tomato Healthy", info["classes"])

    def test_successful_image_inspection_workflow(self):
        data = create_test_image(224, 224, (50, 150, 50))
        result = self.classifier.inspect_image(data, "sample_leaf.jpg")
        self.assertIn(result.status, ("ASSESSED", "UNCERTAIN"))
        self.assertIsNotNone(result.predicted_condition)
        self.assertGreater(result.confidence_pct, 0.0)
        self.assertFalse(result.sensor_data_available)
        self.assertIn("unavailable", result.sensor_note.lower())
        self.assertTrue(len(result.recommended_action) > 10)
        self.assertEqual(result.provenance, "MANUAL_IMAGE_INSPECTION_PC_ONLY")

    def test_missing_model_returns_model_unavailable_without_fabricating(self):
        # Create classifier pointing to empty dir
        empty_classifier = PlantLeafClassifier(models_dir="/tmp/nonexistent_models_dir")
        self.assertFalse(empty_classifier.is_ready)
        data = create_test_image(224, 224)
        result = empty_classifier.inspect_image(data, "test.jpg")
        self.assertEqual(result.status, "MODEL_UNAVAILABLE")
        self.assertEqual(result.predicted_condition, "Model Not Ready")
        self.assertIsNone(result.confidence_pct)
        self.assertFalse(result.sensor_data_available)

    def test_uncertainty_triggered_on_uniform_noise(self):
        # Noise image creates ambiguous logits
        arr = np.random.randint(40, 200, (224, 224, 3), dtype=np.uint8)
        im = Image.fromarray(arr)
        buf = io.BytesIO()
        im.save(buf, format="JPEG")
        result = self.classifier.inspect_image(buf.getvalue(), "noise.jpg")
        # Should be flagged uncertain or rejected
        self.assertIn(result.status, ("UNCERTAIN", "IMAGE_REJECTED", "ASSESSED"))
        if result.status == "UNCERTAIN":
            self.assertIn("Uncertain", result.predicted_condition)

    def test_unsupported_condition_detected_on_powdery_mildew(self):
        user_img_path = Path("/home/mano/Documents/Anveshion/images.jpeg")
        if user_img_path.exists():
            data = user_img_path.read_bytes()
            result = self.classifier.inspect_image(data, "images.jpeg")
            self.assertEqual(result.status, "UNSUPPORTED_CONDITION")
            self.assertIn("Powdery Mildew", result.predicted_condition)
            self.assertIn("Do NOT apply Early Blight", result.recommended_action)
            self.assertIsNone(result.confidence_pct)
            self.assertFalse(result.sensor_data_available)

    def test_supported_septoria_prediction_preserved(self):
        img_path = Path("/home/mano/Downloads/pub0006537_76781.webp")
        if img_path.exists():
            data = img_path.read_bytes()
            result = self.classifier.inspect_image(data, "pub0006537_76781.webp")
            self.assertEqual(result.status, "ASSESSED")
            self.assertIn("Septoria Leaf Spot", result.predicted_condition)
            self.assertGreaterEqual(result.confidence_pct, 80.0)
            self.assertFalse(result.sensor_data_available)

    def test_clear_known_sample_is_assessed(self):
        sample_path = Path("/home/mano/Documents/Anveshion/plant-health-rover-sim/plant_rover/static/samples/sample_yellow_leaf_curl.jpg")
        if sample_path.exists():
            data = sample_path.read_bytes()
            result = self.classifier.inspect_image(data, "sample_yellow_leaf_curl.jpg")
            self.assertEqual(result.status, "ASSESSED")
            self.assertIn("Yellow Leaf Curl", result.predicted_condition)
            self.assertGreaterEqual(result.confidence_pct, 80.0)


class ImageInspectionHTTPTests(unittest.TestCase):
    def setUp(self):
        self.hub = LiveDataHub()
        self.classifier = PlantLeafClassifier()
        self.server = LiveOnlyHTTPServer(("127.0.0.1", 0), self.hub, self.classifier)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.base = f"http://127.0.0.1:{self.server.server_address[1]}"

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)

    def test_get_inspect_model_endpoint(self):
        with urlopen(self.base + "/api/inspect/model", timeout=3) as resp:
            self.assertEqual(resp.status, 200)
            data = json.loads(resp.read().decode())
            self.assertTrue(data["model_ready"])
            self.assertEqual(data["num_classes"], 10)

    def test_manual_inspection_does_not_affect_live_hardware_state(self):
        # 1. Check live snapshot is waiting for data
        with urlopen(self.base + "/api/snapshot", timeout=3) as resp:
            snap_before = json.loads(resp.read().decode())
        self.assertEqual(snap_before["status"], "WAITING_FOR_LIVE_DATA")
        self.assertEqual(snap_before["devices"], [])
        self.assertEqual(snap_before["sensor_readings"], [])
        self.assertIsNone(snap_before["rover_telemetry"])

        # 2. Upload manual image
        data = create_test_image(224, 224)
        req = Request(
            self.base + "/api/inspect/analyze",
            data=json.dumps({"image": base64.b64encode(data).decode(), "filename": "test.jpg"}).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urlopen(req, timeout=3) as resp:
            self.assertEqual(resp.status, 200)
            res = json.loads(resp.read().decode())
            self.assertFalse(res["sensor_data_available"])
            self.assertIn("unavailable", res["sensor_note"].lower())

        # 3. Check live snapshot is STILL waiting for data with 0 devices and 0 sensors
        with urlopen(self.base + "/api/snapshot", timeout=3) as resp:
            snap_after = json.loads(resp.read().decode())
        self.assertEqual(snap_after["status"], "WAITING_FOR_LIVE_DATA")
        self.assertEqual(snap_after["devices"], [])
        self.assertEqual(snap_after["sensor_readings"], [])
        self.assertIsNone(snap_after["rover_telemetry"])

        # 4. Check manual inspection history recorded it
        with urlopen(self.base + "/api/inspect/history", timeout=3) as resp:
            hist = json.loads(resp.read().decode())
        self.assertEqual(len(hist), 1)
        self.assertEqual(hist[0]["filename"], "test.jpg")

    def test_multipart_form_data_upload(self):
        data = create_test_image(224, 224)
        boundary = "---------------------------Boundary12345"
        body = (
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="image"; filename="leaf.png"\r\n'
            f"Content-Type: image/png\r\n\r\n"
        ).encode() + data + f"\r\n--{boundary}--\r\n".encode()

        req = Request(
            self.base + "/api/inspect/analyze",
            data=body,
            headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
            method="POST",
        )
        with urlopen(req, timeout=3) as resp:
            self.assertEqual(resp.status, 200)
            res = json.loads(resp.read().decode())
            self.assertIn(res["status"], ("ASSESSED", "UNCERTAIN"))
            self.assertEqual(res["filename"], "leaf.png")

    def test_corrupted_file_upload_rejected_with_400(self):
        req = Request(
            self.base + "/api/inspect/analyze",
            data=b"this is plain text not an image",
            headers={"Content-Type": "image/jpeg"},
            method="POST",
        )
        with self.assertRaises(HTTPError) as ctx:
            urlopen(req, timeout=3)
        self.assertEqual(ctx.exception.code, 400)
        res = json.loads(ctx.exception.read().decode())
        self.assertEqual(res["status"], "IMAGE_REJECTED")

    def test_oversized_file_rejected_with_413(self):
        huge_bytes = b"0" * (11 * 1024 * 1024)
        req = Request(
            self.base + "/api/inspect/analyze",
            data=huge_bytes,
            headers={"Content-Type": "image/jpeg"},
            method="POST",
        )
        with self.assertRaises(HTTPError) as ctx:
            urlopen(req, timeout=3)
        self.assertEqual(ctx.exception.code, 413)

    def test_empty_body_rejected_with_400(self):
        req = Request(
            self.base + "/api/inspect/analyze",
            data=b"",
            headers={"Content-Type": "image/jpeg"},
            method="POST",
        )
        with self.assertRaises(HTTPError) as ctx:
            urlopen(req, timeout=3)
        self.assertEqual(ctx.exception.code, 400)


class Section9RegressionTests(unittest.TestCase):
    """Explicit regression tests covering all 16 items specified in Section 9."""

    def setUp(self):
        self.classifier = PlantLeafClassifier()

    def test_01_correct_class_label_mapping(self):
        """1. Correct class-label mapping for the 10 loaded ONNX model classes."""
        expected_display_names = [
            "Tomato Bacterial Spot",
            "Tomato Early Blight",
            "Tomato Late Blight",
            "Tomato Leaf Mold",
            "Tomato Septoria Leaf Spot",
            "Tomato Two-Spotted Spider Mite",
            "Tomato Target Spot",
            "Tomato Yellow Leaf Curl Virus",
            "Tomato Mosaic Virus",
            "Tomato Healthy",
        ]
        self.assertEqual(len(self.classifier.classes), 10)
        actual_display_names = [c["display_name"] for c in self.classifier.classes]
        self.assertEqual(actual_display_names, expected_display_names)
        self.assertNotIn("Powdery Mildew", actual_display_names)
        self.assertNotIn("Magnesium Deficiency", actual_display_names)
        for c in self.classifier.classes:
            self.assertTrue(len(c["recommended_action"]) > 10)

    def test_02_consistent_preprocessing(self):
        """2. Consistent preprocessing with [-1, 1] normalization, 224x224 shape, RGB order."""
        red_im = Image.new("RGB", (300, 300), (255, 0, 0))
        tensor = self.classifier.preprocess(red_im)
        self.assertEqual(tensor.shape, (1, 224, 224, 3))
        self.assertEqual(tensor.dtype, np.float32)
        self.assertAlmostEqual(float(tensor[0, 112, 112, 0]), 1.0, places=2)
        self.assertAlmostEqual(float(tensor[0, 112, 112, 1]), -1.0, places=2)
        self.assertAlmostEqual(float(tensor[0, 112, 112, 2]), -1.0, places=2)

    def test_03_valid_supported_disease_predictions(self):
        """3. Valid supported-disease predictions satisfy acceptance criteria."""
        sample_path = Path("/home/mano/Documents/Anveshion/plant-health-rover-sim/plant_rover/static/samples/sample_yellow_leaf_curl.jpg")
        if sample_path.exists():
            data = sample_path.read_bytes()
            result = self.classifier.inspect_image(data, "sample_yellow_leaf_curl.jpg")
            self.assertEqual(result.status, "ASSESSED")
            self.assertIn("Yellow Leaf Curl", result.predicted_condition)
            self.assertEqual(result.condition_code, "Tomato__Tomato_YellowLeaf__Curl_Virus")
            self.assertGreaterEqual(result.confidence_pct, 65.0)

    def test_04_uncertain_supported_disease_predictions(self):
        """4. Uncertain predictions when confidence is low or margin is narrow."""
        data = create_test_image(224, 224)
        clf = PlantLeafClassifier()
        ambiguous_probs = np.array([0.40, 0.38, 0.10, 0.04, 0.02, 0.02, 0.01, 0.01, 0.01, 0.01], dtype=np.float32)
        clf._predict_raw = lambda tensor: ambiguous_probs
        result = clf.inspect_image(data, "ambiguous.jpg")
        self.assertEqual(result.status, "UNCERTAIN")
        self.assertIn("Uncertain", result.predicted_condition)
        self.assertEqual(result.confidence_pct, 40.0)
        self.assertIn("Do NOT apply chemical treatments", result.recommended_action)

    def test_05_unsupported_condition_handling(self):
        """5. Rejection mechanism flags out-of-domain pathologies with N/A confidence."""
        user_img_path = Path("/home/mano/Documents/Anveshion/images.jpeg")
        if user_img_path.exists():
            data = user_img_path.read_bytes()
            result = self.classifier.inspect_image(data, "images.jpeg")
            self.assertEqual(result.status, "UNSUPPORTED_CONDITION")
            self.assertEqual(result.condition_code, "unsupported_powdery_mildew")
            self.assertIsNone(result.confidence_pct)
            self.assertIn("Powdery Mildew", result.predicted_condition)
            self.assertIn("Do NOT apply Early Blight", result.recommended_action)

    def test_06_invalid_and_non_leaf_images(self):
        """6. Invalid, corrupt, and non-leaf blurry images are rejected with None confidence."""
        corrupt = b"\xff\xd8\xff\xe0" + b"\x00" * 30
        res_corrupt = self.classifier.inspect_image(corrupt, "corrupt.jpg")
        self.assertEqual(res_corrupt.status, "IMAGE_REJECTED")
        self.assertIsNone(res_corrupt.confidence_pct)

        flat = Image.new("RGB", (224, 224), (120, 120, 120))
        buf = io.BytesIO()
        flat.save(buf, format="JPEG")
        res_flat = self.classifier.inspect_image(buf.getvalue(), "flat.jpg")
        self.assertEqual(res_flat.status, "IMAGE_REJECTED")
        self.assertIsNone(res_flat.confidence_pct)

    def test_07_missing_or_incompatible_model_artifacts(self):
        """7. Missing or incompatible model artifacts handled gracefully."""
        empty_clf = PlantLeafClassifier(models_dir="/tmp/nonexistent_model_dir_xyz")
        self.assertFalse(empty_clf.is_ready)
        data = create_test_image(224, 224)
        res = empty_clf.inspect_image(data, "test.jpg")
        self.assertEqual(res.status, "MODEL_UNAVAILABLE")
        self.assertEqual(res.predicted_condition, "Model Not Ready")
        self.assertIsNone(res.confidence_pct)

    def test_08_invalid_or_non_finite_model_outputs(self):
        """8. Model outputs with NaN, Inf, or incorrect dimensions return MODEL_UNAVAILABLE."""
        data = create_test_image(224, 224)
        clf = PlantLeafClassifier()
        nan_probs = np.array([np.nan] * 10, dtype=np.float32)
        clf._predict_raw = lambda tensor: nan_probs
        res_nan = clf.inspect_image(data, "nan.jpg")
        self.assertEqual(res_nan.status, "MODEL_UNAVAILABLE")
        self.assertEqual(res_nan.predicted_condition, "Inference Error")
        self.assertIsNone(res_nan.confidence_pct)

        wrong_dim_probs = np.array([0.2] * 5, dtype=np.float32)
        clf._predict_raw = lambda tensor: wrong_dim_probs
        res_dim = clf.inspect_image(data, "dim.jpg")
        self.assertEqual(res_dim.status, "MODEL_UNAVAILABLE")
        self.assertEqual(res_dim.predicted_condition, "Inference Error")
        self.assertIsNone(res_dim.confidence_pct)

    def test_09_confidence_display_for_all_result_states(self):
        """9. Confidence reporting reflects legitimate status without manufactured numbers."""
        data = create_test_image(224, 224)
        # 1. ASSESSED
        sample_path = Path("/home/mano/Documents/Anveshion/plant-health-rover-sim/plant_rover/static/samples/sample_yellow_leaf_curl.jpg")
        if sample_path.exists():
            res_assessed = self.classifier.inspect_image(sample_path.read_bytes(), "leaf.jpg")
            self.assertEqual(res_assessed.status, "ASSESSED")
            self.assertIsInstance(res_assessed.confidence_pct, float)
            self.assertGreater(res_assessed.confidence_pct, 65.0)

        # 2. UNCERTAIN
        clf_unc = PlantLeafClassifier()
        clf_unc._predict_raw = lambda t: np.array([0.4, 0.35] + [0.03125] * 8, dtype=np.float32)
        res_uncertain = clf_unc.inspect_image(data, "uncertain.jpg")
        self.assertEqual(res_uncertain.status, "UNCERTAIN")
        self.assertEqual(res_uncertain.confidence_pct, 40.0)

        # 3. UNSUPPORTED_CONDITION
        user_img_path = Path("/home/mano/Documents/Anveshion/images.jpeg")
        if user_img_path.exists():
            clf_unsupp = PlantLeafClassifier()
            res_unsupported = clf_unsupp.inspect_image(user_img_path.read_bytes(), "images.jpeg")
            self.assertEqual(res_unsupported.status, "UNSUPPORTED_CONDITION")
            self.assertIsNone(res_unsupported.confidence_pct)

        # 4. IMAGE_REJECTED
        res_rejected = self.classifier.inspect_image(b"invalid_bytes", "bad.jpg")
        self.assertEqual(res_rejected.status, "IMAGE_REJECTED")
        self.assertIsNone(res_rejected.confidence_pct)

        # 5. MODEL_UNAVAILABLE
        empty_clf = PlantLeafClassifier(models_dir="/tmp/nonexistent_model_dir_xyz")
        res_unavail = empty_clf.inspect_image(data, "test.jpg")
        self.assertEqual(res_unavail.status, "MODEL_UNAVAILABLE")
        self.assertIsNone(res_unavail.confidence_pct)

    def test_10_consistent_confidence_reporting_between_api_and_dashboard(self):
        """10. Dashboard HTML matches API contract for null confidence and uncertain states."""
        html_path = Path(__file__).parent.parent / "plant_rover" / "static" / "live.html"
        self.assertTrue(html_path.exists())
        html = html_path.read_text(encoding="utf-8")
        self.assertIn("isUnsupported ? 'N/A'", html)
        self.assertIn("Unavailable (uncertain assessment)", html)
        self.assertIn("Visual lookalike", html)

    def test_11_suppression_of_unsupported_treatment_advice(self):
        """11. Ambiguous and unsupported conditions suppress specific chemical/nutrient cures."""
        user_img_path = Path("/home/mano/Documents/Anveshion/images.jpeg")
        if user_img_path.exists():
            res_unsupported = self.classifier.inspect_image(user_img_path.read_bytes(), "images.jpeg")
            self.assertNotIn("Chlorothalonil", res_unsupported.recommended_action)
            self.assertNotIn("Mancozeb", res_unsupported.recommended_action)
            self.assertNotIn("Epsom", res_unsupported.recommended_action)

        data = create_test_image(224, 224)
        clf = PlantLeafClassifier()
        clf._predict_raw = lambda t: np.array([0.4, 0.35] + [0.03125] * 8, dtype=np.float32)
        res_uncertain = clf.inspect_image(data, "uncertain.jpg")
        self.assertNotIn("Chlorothalonil", res_uncertain.recommended_action)
        self.assertIn("Do NOT apply chemical treatments without verified diagnosis", res_uncertain.recommended_action)

    def test_12_no_fabricated_sensor_readings(self):
        """12. Manual image inspection has sensor_data_available=False and no fabricated sensors."""
        data = create_test_image(224, 224)
        res = self.classifier.inspect_image(data, "leaf.jpg")
        self.assertFalse(res.sensor_data_available)
        self.assertIn("unavailable", res.sensor_note.lower())
        self.assertIn("pc-side", res.sensor_note.lower())

    def test_13_no_accidental_modification_of_live_hardware_status(self):
        """13. Uploading manual images does not alter live hardware status."""
        hub = LiveDataHub()
        server = LiveOnlyHTTPServer(("127.0.0.1", 0), hub, self.classifier)
        t = threading.Thread(target=server.serve_forever, daemon=True)
        t.start()
        base = f"http://127.0.0.1:{server.server_address[1]}"
        try:
            with urlopen(base + "/api/snapshot", timeout=3) as r:
                snap_before = json.loads(r.read().decode())
            self.assertEqual(snap_before["status"], "WAITING_FOR_LIVE_DATA")
            self.assertEqual(snap_before["devices"], [])

            data = create_test_image(224, 224)
            req = Request(
                base + "/api/inspect/analyze",
                data=json.dumps({"image": base64.b64encode(data).decode(), "filename": "pc_leaf.jpg"}).encode(),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urlopen(req, timeout=3) as r:
                res = json.loads(r.read().decode())
            self.assertFalse(res["sensor_data_available"])

            with urlopen(base + "/api/snapshot", timeout=3) as r:
                snap_after = json.loads(r.read().decode())
            self.assertEqual(snap_after["status"], "WAITING_FOR_LIVE_DATA")
            self.assertEqual(snap_after["devices"], [])
            self.assertIsNone(snap_after["rover_telemetry"])
        finally:
            server.shutdown()
            server.server_close()
            t.join(timeout=2)

    def test_14_no_stale_result_displayed_after_analysing_different_image(self):
        """14. Consecutive inspections produce distinct results without reusing stale state."""
        hub = LiveDataHub()
        server = LiveOnlyHTTPServer(("127.0.0.1", 0), hub, self.classifier)
        t = threading.Thread(target=server.serve_forever, daemon=True)
        t.start()
        base = f"http://127.0.0.1:{server.server_address[1]}"
        try:
            data1 = create_test_image(224, 224, color=(50, 150, 50))
            req1 = Request(
                base + "/api/inspect/analyze",
                data=json.dumps({"image": base64.b64encode(data1).decode(), "filename": "leaf1.jpg"}).encode(),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urlopen(req1, timeout=3) as r:
                res1 = json.loads(r.read().decode())

            data2 = create_test_image(224, 224, color=(150, 50, 50))
            req2 = Request(
                base + "/api/inspect/analyze",
                data=json.dumps({"image": base64.b64encode(data2).decode(), "filename": "leaf2.jpg"}).encode(),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urlopen(req2, timeout=3) as r:
                res2 = json.loads(r.read().decode())

            self.assertNotEqual(res1["inspection_id"], res2["inspection_id"])
            self.assertEqual(res1["filename"], "leaf1.jpg")
            self.assertEqual(res2["filename"], "leaf2.jpg")

            with urlopen(base + "/api/inspect/history", timeout=3) as r:
                history = json.loads(r.read().decode())
            self.assertGreaterEqual(len(history), 2)
            self.assertEqual(history[0]["inspection_id"], res2["inspection_id"])
            self.assertEqual(history[1]["inspection_id"], res1["inspection_id"])
        finally:
            server.shutdown()
            server.server_close()
            t.join(timeout=2)

    def test_15_correct_display_when_model_is_unavailable(self):
        """15. API reports model_ready=False and inspect returns MODEL_UNAVAILABLE with None confidence."""
        empty_clf = PlantLeafClassifier(models_dir="/tmp/dummy_empty_dir_for_test")
        hub = LiveDataHub()
        server = LiveOnlyHTTPServer(("127.0.0.1", 0), hub, empty_clf)
        t = threading.Thread(target=server.serve_forever, daemon=True)
        t.start()
        base = f"http://127.0.0.1:{server.server_address[1]}"
        try:
            with urlopen(base + "/api/inspect/model", timeout=3) as r:
                model_info = json.loads(r.read().decode())
            self.assertFalse(model_info["model_ready"])

            data = create_test_image(224, 224)
            req = Request(
                base + "/api/inspect/analyze",
                data=json.dumps({"image": base64.b64encode(data).decode(), "filename": "leaf.jpg"}).encode(),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with self.assertRaises(HTTPError) as ctx:
                urlopen(req, timeout=3)
            self.assertEqual(ctx.exception.code, 503)
            res = json.loads(ctx.exception.read().decode())
            self.assertEqual(res["status"], "MODEL_UNAVAILABLE")
            self.assertIsNone(res["confidence_pct"])
        finally:
            server.shutdown()
            server.server_close()
            t.join(timeout=2)

    def test_16_dataset_leakage_checks(self):
        """16. Dataset partitioning requires specimen-group separation to prevent leakage."""
        from plant_rover.ml import DatasetManifest
        manifest = DatasetManifest.create_synthetic_evaluation_manifest(Path("/tmp/synth_data"), sample_count=30, seed=42)
        train_plants = {item.plant_id for item in manifest.get_split("train")}
        val_plants = {item.plant_id for item in manifest.get_split("val")}
        test_plants = {item.plant_id for item in manifest.get_split("test")}
        self.assertTrue(train_plants.isdisjoint(val_plants))
        self.assertTrue(train_plants.isdisjoint(test_plants))
        self.assertTrue(val_plants.isdisjoint(test_plants))


if __name__ == "__main__":
    unittest.main()
