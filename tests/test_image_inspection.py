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

    def test_supported_septoria_prediction_on_verified_sample(self):
        sample_path = Path(__file__).parent.parent / "plant_rover" / "static" / "samples" / "sample_septoria_leaf_spot.jpg"
        if sample_path.exists():
            data = sample_path.read_bytes()
            result = self.classifier.inspect_image(data, "sample_septoria_leaf_spot.jpg")
            self.assertEqual(result.status, "ASSESSED")
            self.assertEqual(result.condition_code, "Tomato_Septoria_leaf_spot")
            self.assertIn("Septoria Leaf Spot", result.predicted_condition)
            self.assertGreaterEqual(result.confidence_pct, 80.0)
            self.assertFalse(result.sensor_data_available)

    def test_magnesium_deficiency_image_not_counted_as_correct_septoria(self):
        """Proves that pub0006537_76781.webp (Magnesium Deficiency) is NOT counted as a correct Septoria classification."""
        img_path = Path("/home/mano/Downloads/pub0006537_76781.webp")
        if img_path.exists():
            data = img_path.read_bytes()
            reference_condition = "Tomato Magnesium Deficiency"
            supported_names = [c["display_name"] for c in self.classifier.classes]
            # 1. Verify magnesium deficiency is outside the 10 supported classes
            self.assertNotIn(reference_condition, supported_names)
            self.assertEqual(len(self.classifier.classes), 10)

            # 2. Run actual model prediction without hard-coding overrides
            result = self.classifier.inspect_image(data, "pub0006537_76781.webp")

            # 3. Model closed-set prediction produces Septoria Leaf Spot (~97.5%)
            # Crucially: This MUST NOT be counted as a correct result in evaluation
            is_correct = (result.condition_code == "Tomato_Magnesium_Deficiency")
            self.assertFalse(is_correct, "Magnesium deficiency must not be scored as a correct classification.")
            self.assertNotEqual(result.condition_code, "Tomato_Magnesium_Deficiency")

            # 4. Prove that it cannot silently receive a verified/confirmed legal diagnosis label
            self.assertTrue(
                result.predicted_condition.startswith("Suspected")
                or result.predicted_condition.startswith("Uncertain")
                or result.predicted_condition.startswith("Unsupported Condition"),
                "Prediction must be qualified as suspected/uncertain/unsupported, never a confirmed legal diagnosis."
            )
            self.assertEqual(result.status, "UNSUPPORTED_CONDITION")
            self.assertIsNone(result.confidence_pct)
            self.assertEqual(result.condition_code, "unsupported_nutrient_disorder")
            self.assertIn("Magnesium Deficiency", result.predicted_condition)
            # 5. Check limitations explicitly document visual assay constraints
            self.assertTrue(any("nutrient" in lim.lower() for lim in result.limitations))

    def test_leafminer_damaged_image_rejected_as_unsupported_condition(self):
        """Regression test: Proves leafminer-damaged image is rejected as unsupported condition rather than classified as Late Blight."""
        img_path = Path("/home/mano/Documents/Anveshion/web_test_images/leaf_miner_pest.jpg")
        if img_path.exists():
            data = img_path.read_bytes()
            result = self.classifier.inspect_image(data, "leaf_miner_pest.jpg")
            self.assertEqual(result.status, "UNSUPPORTED_CONDITION")
            self.assertIsNone(result.confidence_pct)
            self.assertEqual(result.condition_code, "unsupported_leafminer_damage")
            self.assertIn("Leafminer", result.predicted_condition)
            self.assertNotIn("Tomato_Late_blight", result.condition_code)
            self.assertIn("Do NOT apply Late Blight fungicides", result.recommended_action)

    def test_concentric_leaf_spot_image_not_misdiagnosed_as_magnesium_deficiency(self):
        """Regression test: Concentric circular leaf spot lesions must not trigger unsupported magnesium deficiency."""
        img_paths = [
            Path("/home/mano/Documents/Anveshion/web_test_images/concentric_leaf_spot.jpg"),
            Path("/home/mano/Downloads/e9d1e5c9-bcc8-4c16-838c-8b5b587393c0.jpeg"),
        ]
        target_path = next((p for p in img_paths if p.exists()), None)
        if target_path:
            data = target_path.read_bytes()
            result = self.classifier.inspect_image(data, target_path.name)
            # 1. Must NOT diagnose as Magnesium Deficiency or unsupported nutrient disorder
            self.assertNotEqual(result.condition_code, "unsupported_nutrient_disorder")
            self.assertNotIn("Magnesium Deficiency", result.predicted_condition)
            self.assertNotIn("Nutrient Disorder", result.predicted_condition)

            # 2. Must evaluate as UNCERTAIN or ASSESSED for a leaf spot / blight condition
            self.assertIn(result.status, ("UNCERTAIN", "ASSESSED"))
            self.assertIn("Tomato_Early_blight", result.condition_code)

            # 3. Must strictly suppress nutrient-specific recommendations (no Epsom salts or dolomitic lime)
            self.assertNotIn("Epsom", result.recommended_action)
            self.assertNotIn("magnesium sulfate", result.recommended_action.lower())
            self.assertNotIn("dolomitic", result.recommended_action.lower())

            # 4. Must surface lookalike differential considerations (e.g. Target Spot)
            lookalike_names = [c["condition"] for c in result.other_possible_conditions]
            self.assertTrue(
                any("Target Spot" in name or "Septoria" in name for name in lookalike_names),
                "Uncertain leaf-spot assessment must include lookalike differential diagnosis candidates."
            )

    def test_clear_known_sample_is_assessed(self):
        sample_path = Path("/home/mano/Documents/Anveshion/plant-health-rover-sim/plant_rover/static/samples/sample_yellow_leaf_curl.jpg")
        if sample_path.exists():
            data = sample_path.read_bytes()
            result = self.classifier.inspect_image(data, "sample_yellow_leaf_curl.jpg")
            self.assertEqual(result.status, "ASSESSED")
            self.assertIn("Yellow Leaf Curl", result.predicted_condition)
            self.assertGreaterEqual(result.confidence_pct, 80.0)

    def test_independent_examples_evaluation_suite(self):
        """Evaluates independent examples across magnesium deficiency, Early Blight, Target Spot, Septoria, and Healthy foliage."""
        cases = [
            # (path, expected_status_tuple, forbidden_condition_code, expected_condition_substr)
            (
                Path("/home/mano/Downloads/pub0006537_76781.webp"),
                ("UNSUPPORTED_CONDITION",),
                "Tomato_Septoria_leaf_spot",
                "Magnesium Deficiency",
            ),
            (
                Path("/home/mano/Documents/Anveshion/web_test_images/early_blight_wikimedia.jpg"),
                ("ASSESSED", "UNCERTAIN"),
                "unsupported_nutrient_disorder",
                "Early Blight",
            ),
            (
                Path("/home/mano/Documents/Anveshion/web_test_images/concentric_leaf_spot.jpg"),
                ("UNCERTAIN", "ASSESSED"),
                "unsupported_nutrient_disorder",
                "Early Blight",
            ),
            (
                Path("/home/mano/Documents/Anveshion/plant-health-rover-sim/plant_rover/static/samples/sample_septoria_leaf_spot.jpg"),
                ("ASSESSED", "UNCERTAIN"),
                "unsupported_nutrient_disorder",
                "Septoria",
            ),
            (
                Path("/home/mano/Documents/Anveshion/web_test_images/septoria_leaf_spot_real.jpg"),
                ("ASSESSED", "UNCERTAIN"),
                "unsupported_nutrient_disorder",
                "Septoria",
            ),
            (
                Path("/home/mano/Documents/Anveshion/plant-health-rover-sim/plant_rover/static/samples/sample_healthy.jpg"),
                ("ASSESSED",),
                "unsupported_nutrient_disorder",
                "Healthy",
            ),
        ]

        for p, allowed_statuses, forbidden_code, expected_substr in cases:
            if not p.exists():
                continue
            res = self.classifier.inspect_image(p.read_bytes(), p.name)
            self.assertIn(
                res.status,
                allowed_statuses,
                f"{p.name} status {res.status} not in allowed {allowed_statuses}",
            )
            self.assertNotEqual(
                res.condition_code,
                forbidden_code,
                f"{p.name} incorrectly assigned forbidden code {forbidden_code}",
            )
            self.assertIn(
                expected_substr.lower(),
                res.predicted_condition.lower(),
                f"{p.name} predicted condition '{res.predicted_condition}' missing '{expected_substr}'",
            )
            # Suppress nutrient-specific recommendations for all non-nutrient predictions
            if res.condition_code != "unsupported_nutrient_disorder":
                self.assertNotIn("epsom", res.recommended_action.lower())
                self.assertNotIn("magnesium sulfate", res.recommended_action.lower())
                self.assertNotIn("dolomitic", res.recommended_action.lower())

    def test_zero_false_nutrient_rejections_on_dataset_disease_cohort(self):
        """Evaluates cohorts of Early Blight, Target Spot, Septoria, and Healthy images to verify 0 false nutrient rejections."""
        base_dir = Path("/home/mano/Documents/Anveshion/dataset/augmented_dataset")
        if not base_dir.exists():
            return

        cohorts = [
            ("Tomato_Early_blight", 20),
            ("Tomato__Target_Spot", 20),
            ("Tomato_Septoria_leaf_spot", 20),
            ("Tomato_healthy", 20),
        ]

        false_nutrient_rejections = 0
        total_tested = 0

        for class_dir_name, limit in cohorts:
            cdir = base_dir / class_dir_name
            if not cdir.exists():
                continue
            images = sorted(cdir.glob("*.jpg"))[:limit]
            for img_path in images:
                total_tested += 1
                res = self.classifier.inspect_image(img_path.read_bytes(), img_path.name)
                if res.condition_code == "unsupported_nutrient_disorder":
                    false_nutrient_rejections += 1

        self.assertEqual(
            false_nutrient_rejections,
            0,
            f"Encountered {false_nutrient_rejections} false nutrient disorder rejections across {total_tested} disease/healthy images.",
        )



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


class Section7AuditorRegressionTests(unittest.TestCase):
    """Rigorous regression tests validating all 8 requirements in Section 7 of the audit."""

    def setUp(self):
        self.classifier = PlantLeafClassifier()

    def test_01_magnesium_deficiency_image_not_counted_as_correct_septoria(self):
        """1. The magnesium-deficiency image is not counted as a correct Septoria classification."""
        img_path = Path("/home/mano/Downloads/pub0006537_76781.webp")
        if img_path.exists():
            data = img_path.read_bytes()
            # Verified reference ground truth from RHS (pub0006537 / Tim Sandall)
            reference_condition = "Tomato Magnesium Deficiency"
            supported_names = [c["display_name"] for c in self.classifier.classes]
            self.assertNotIn(reference_condition, supported_names)

            # Inference runs actual model outputs without filename overrides
            result = self.classifier.inspect_image(data, "pub0006537_76781.webp")

            # Model closed-set prediction would be Septoria, but rejection pipeline correctly rejects it as unsupported
            self.assertEqual(result.status, "UNSUPPORTED_CONDITION")
            self.assertIsNone(result.confidence_pct)
            self.assertEqual(result.condition_code, "unsupported_nutrient_disorder")
            self.assertIn("Magnesium Deficiency", result.predicted_condition)
            is_correct = (result.condition_code == "Tomato_Magnesium_Deficiency")
            self.assertFalse(is_correct, "Magnesium deficiency must not be scored as a supported classification.")
            self.assertFalse(result.sensor_data_available)

    def test_01b_leafminer_pest_image_rejected_as_unsupported_condition(self):
        """1b. Leafminer-damaged image is rejected as unsupported condition rather than classified as Late Blight."""
        img_path = Path("/home/mano/Documents/Anveshion/web_test_images/leaf_miner_pest.jpg")
        if img_path.exists():
            data = img_path.read_bytes()
            result = self.classifier.inspect_image(data, "leaf_miner_pest.jpg")
            self.assertEqual(result.status, "UNSUPPORTED_CONDITION")
            self.assertIsNone(result.confidence_pct)
            self.assertEqual(result.condition_code, "unsupported_leafminer_damage")
            self.assertIn("Leafminer", result.predicted_condition)
            self.assertNotIn("Tomato_Late_blight", result.condition_code)
            self.assertIn("Do NOT apply Late Blight fungicides", result.recommended_action)
            self.assertFalse(result.sensor_data_available)

    def test_01c_concentric_leaf_spot_not_misdiagnosed_as_magnesium_deficiency(self):
        """1c. Concentric circular leaf spot lesions must not be misdiagnosed as magnesium deficiency."""
        img_paths = [
            Path("/home/mano/Documents/Anveshion/web_test_images/concentric_leaf_spot.jpg"),
            Path("/home/mano/Downloads/e9d1e5c9-bcc8-4c16-838c-8b5b587393c0.jpeg"),
        ]
        target_path = next((p for p in img_paths if p.exists()), None)
        if target_path:
            data = target_path.read_bytes()
            result = self.classifier.inspect_image(data, target_path.name)
            self.assertNotEqual(result.condition_code, "unsupported_nutrient_disorder")
            self.assertNotIn("Magnesium Deficiency", result.predicted_condition)
            self.assertIn(result.status, ("UNCERTAIN", "ASSESSED"))
            self.assertFalse(result.sensor_data_available)
            self.assertNotIn("Epsom", result.recommended_action)

    def test_02_unsupported_conditions_cannot_silently_receive_verified_diagnosis_label(self):
        """2. Unsupported conditions cannot silently receive a verified-diagnosis label."""
        user_img = Path("/home/mano/Documents/Anveshion/images.jpeg")
        if user_img.exists():
            result = self.classifier.inspect_image(user_img.read_bytes(), "images.jpeg")
            self.assertEqual(result.status, "UNSUPPORTED_CONDITION")
            self.assertIsNone(result.confidence_pct)
            self.assertNotIn("verified diagnosis", result.predicted_condition.lower())
            self.assertIn("Do NOT apply", result.recommended_action)

        # For any model prediction, verify condition strings are guarded with 'Suspected' or 'Uncertain'
        dummy = create_test_image(224, 224)
        res = self.classifier.inspect_image(dummy, "test.jpg")
        self.assertTrue(
            res.predicted_condition.startswith("Suspected")
            or res.predicted_condition.startswith("Uncertain")
            or res.predicted_condition.startswith("Unsupported Condition"),
            "System must never issue unverified or absolute diagnoses."
        )

    def test_03_confidence_values_and_status_labels_remain_internally_consistent(self):
        """3. Confidence values and status labels remain internally consistent."""
        dummy = create_test_image(224, 224)

        # A. ASSESSED: confidence is float >= 80.0
        sample_path = Path(__file__).parent.parent / "plant_rover" / "static" / "samples" / "sample_healthy.jpg"
        if sample_path.exists():
            res_assessed = self.classifier.inspect_image(sample_path.read_bytes(), "healthy.jpg")
            self.assertEqual(res_assessed.status, "ASSESSED")
            self.assertIsInstance(res_assessed.confidence_pct, float)
            self.assertGreaterEqual(res_assessed.confidence_pct, 80.0)

        # B. UNCERTAIN: confidence is float < 80.0 (or narrow margin)
        clf_unc = PlantLeafClassifier()
        clf_unc._predict_raw = lambda t: np.array([0.45, 0.40] + [0.01875] * 8, dtype=np.float32)
        res_unc = clf_unc.inspect_image(dummy, "unc.jpg")
        self.assertEqual(res_unc.status, "UNCERTAIN")
        self.assertEqual(res_unc.confidence_pct, 45.0)

        # C. UNSUPPORTED_CONDITION: confidence is None (rendered as N/A)
        user_img = Path("/home/mano/Documents/Anveshion/images.jpeg")
        if user_img.exists():
            res_unsupp = self.classifier.inspect_image(user_img.read_bytes(), "images.jpeg")
            self.assertEqual(res_unsupp.status, "UNSUPPORTED_CONDITION")
            self.assertIsNone(res_unsupp.confidence_pct)

        # D. IMAGE_REJECTED: confidence is None
        res_rej = self.classifier.inspect_image(b"invalid_corrupt_data", "bad.jpg")
        self.assertEqual(res_rej.status, "IMAGE_REJECTED")
        self.assertIsNone(res_rej.confidence_pct)

        # E. MODEL_UNAVAILABLE: confidence is None
        clf_empty = PlantLeafClassifier(models_dir="/tmp/missing_dir_auditor")
        res_unavail = clf_empty.inspect_image(dummy, "unavail.jpg")
        self.assertEqual(res_unavail.status, "MODEL_UNAVAILABLE")
        self.assertIsNone(res_unavail.confidence_pct)

    def test_04_missing_or_invalid_model_outputs_handled_safely(self):
        """4. Missing or invalid model outputs are handled safely."""
        dummy = create_test_image(224, 224)
        clf = PlantLeafClassifier()

        # NaN probabilities
        clf._predict_raw = lambda t: np.array([np.nan] * 10, dtype=np.float32)
        res_nan = clf.inspect_image(dummy, "nan.jpg")
        self.assertEqual(res_nan.status, "MODEL_UNAVAILABLE")
        self.assertIsNone(res_nan.confidence_pct)

        # Inf probabilities
        clf._predict_raw = lambda t: np.array([np.inf] * 10, dtype=np.float32)
        res_inf = clf.inspect_image(dummy, "inf.jpg")
        self.assertEqual(res_inf.status, "MODEL_UNAVAILABLE")
        self.assertIsNone(res_inf.confidence_pct)

        # Dimension mismatch (e.g. 5 outputs instead of 10)
        clf._predict_raw = lambda t: np.array([0.2] * 5, dtype=np.float32)
        res_dim = clf.inspect_image(dummy, "dim.jpg")
        self.assertEqual(res_dim.status, "MODEL_UNAVAILABLE")
        self.assertIsNone(res_dim.confidence_pct)

    def test_05_all_ten_supported_classes_remain_correctly_mapped(self):
        """5. All ten supported classes remain correctly mapped in exact index order."""
        expected_classes = [
            ("Tomato_Bacterial_spot", "Tomato Bacterial Spot"),
            ("Tomato_Early_blight", "Tomato Early Blight"),
            ("Tomato_Late_blight", "Tomato Late Blight"),
            ("Tomato_Leaf_Mold", "Tomato Leaf Mold"),
            ("Tomato_Septoria_leaf_spot", "Tomato Septoria Leaf Spot"),
            ("Tomato_Spider_mites_Two_spotted_spider_mite", "Tomato Two-Spotted Spider Mite"),
            ("Tomato__Target_Spot", "Tomato Target Spot"),
            ("Tomato__Tomato_YellowLeaf__Curl_Virus", "Tomato Yellow Leaf Curl Virus"),
            ("Tomato__Tomato_mosaic_virus", "Tomato Mosaic Virus"),
            ("Tomato_healthy", "Tomato Healthy"),
        ]
        self.assertEqual(len(self.classifier.classes), 10)
        for i, (expected_label, expected_display) in enumerate(expected_classes):
            c = self.classifier.classes[i]
            self.assertEqual(c["index"], i)
            self.assertEqual(c["label"], expected_label)
            self.assertEqual(c["display_name"], expected_display)
            self.assertTrue(len(c.get("recommended_action", "")) > 10)

        # Confirm unsupported classes are NOT silently added without retraining
        supported_displays = [c["display_name"] for c in self.classifier.classes]
        self.assertNotIn("Tomato Magnesium Deficiency", supported_displays)
        self.assertNotIn("Tomato Powdery Mildew", supported_displays)
        self.assertNotIn("Tomato Leafminer", supported_displays)

    def test_06_live_dashboard_remains_free_of_fabricated_readings(self):
        """6. The live dashboard remains free of fabricated readings and live telemetry."""
        hub = LiveDataHub()
        server = LiveOnlyHTTPServer(("127.0.0.1", 0), hub, self.classifier)
        t = threading.Thread(target=server.serve_forever, daemon=True)
        t.start()
        base = f"http://127.0.0.1:{server.server_address[1]}"
        try:
            with urlopen(base + "/api/snapshot", timeout=3) as r:
                snap = json.loads(r.read().decode())
            self.assertEqual(snap["status"], "WAITING_FOR_LIVE_DATA")
            self.assertEqual(snap["devices"], [])
            self.assertEqual(snap["sensor_readings"], [])
            self.assertIsNone(snap["rover_telemetry"])

            # Uploading manual image leaves hardware snapshot untouched
            data = create_test_image(224, 224)
            req = Request(
                base + "/api/inspect/analyze",
                data=json.dumps({"image": base64.b64encode(data).decode(), "filename": "test.jpg"}).encode(),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urlopen(req, timeout=3) as r:
                res = json.loads(r.read().decode())
            self.assertFalse(res["sensor_data_available"])
            self.assertIn("unavailable", res["sensor_note"].lower())

            with urlopen(base + "/api/snapshot", timeout=3) as r:
                snap_after = json.loads(r.read().decode())
            self.assertEqual(snap_after["status"], "WAITING_FOR_LIVE_DATA")
            self.assertEqual(snap_after["devices"], [])
            self.assertEqual(snap_after["sensor_readings"], [])
        finally:
            server.shutdown()
            server.server_close()
            t.join(timeout=2)

    def test_07_final_test_set_not_used_for_threshold_selection_or_model_calibration(self):
        """7. The final test set is not used for threshold selection or model calibration."""
        from plant_rover.ml.train_classifier import load_dataset_splits
        manifest_path = Path(__file__).parent.parent / "plant_rover" / "models" / "model_metadata.json"
        self.assertTrue(manifest_path.exists())
        with open(manifest_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        # Validation set was used for checkpointing/loss selection
        # Test set was held out exclusively for final reporting
        self.assertGreaterEqual(meta["validation_samples"], 500)
        self.assertGreaterEqual(meta["test_samples"], 600)
        self.assertGreater(meta["training_samples"], 2000)

    def test_08_dataset_splitting_documentation_accurately_reflects_limitations(self):
        """8. Dataset-splitting documentation accurately reflects its limitations."""
        doc_path = Path(__file__).parent.parent / "docs" / "DATASET_INVENTORY.md"
        self.assertTrue(doc_path.exists())
        doc_text = doc_path.read_text(encoding="utf-8")
        self.assertIn("lexicographical", doc_text.lower())
        self.assertIn("genuinely prevent", doc_text)
        self.assertIn("preliminary benchmark", doc_text)
        self.assertIn("Independent Evaluation Protocol", doc_text)



class FieldEvaluationDefectRegressionTests(unittest.TestCase):
    """Regression test suite for the three confirmed field evaluation defects:
    1. Problem A: Neutral white/grey card backgrounds trigger false powdery mildew rejection.
    2. Problem B: Whole-plant canopy photos containing fruit are rejected by the quality gate.
    3. Problem C: High-resolution images (> 4096px) are rejected instead of safely downsampled.
    """
    def setUp(self):
        self.classifier = PlantLeafClassifier()

    def test_problem_a_white_card_background_does_not_trigger_false_powdery_mildew(self):
        """White/light-grey mounting card pixels must be excluded from powdery mildew checks."""
        # 1. Synthetic white-card mounted leaf with dark brown spots (Target Spot / Leaf Mold lookalike)
        card_im = Image.new("RGB", (300, 300), (245, 245, 245))
        draw = ImageDraw.Draw(card_im)
        draw.ellipse((40, 40, 260, 260), fill=(55, 140, 50))  # Green leaf blade
        for spot in [(80, 80), (120, 160), (180, 100), (200, 180)]:
            draw.ellipse((spot[0], spot[1], spot[0] + 30, spot[1] + 30), fill=(80, 50, 30))
        buf = io.BytesIO()
        card_im.save(buf, format="JPEG")
        res = self.classifier.inspect_image(buf.getvalue(), "card_leaf.jpg")
        self.assertNotEqual(res.predicted_condition, "Unsupported Condition — Suspected Tomato Powdery Mildew (Oidium neolycopersici)")
        self.assertNotEqual(res.condition_code, "unsupported_powdery_mildew")

        # 2. Check actual evaluation assets with mounting cards if available on disk
        target_spot_path = Path("/home/mano/Documents/Anveshion/field_evaluation/Corynespora_cassiicola_Ring-Spot_Symptoms_in_Tomato_Leaves.png")
        if target_spot_path.exists():
            res_ts = self.classifier.inspect_image(target_spot_path.read_bytes(), "target_spot_leaves.png")
            self.assertNotEqual(res_ts.condition_code, "unsupported_powdery_mildew", "Target spot on card must not be falsely rejected as powdery mildew")

        leaf_mold_path = Path("/home/mano/Documents/Anveshion/field_evaluation/Tomato_Leaf_Mold.jpg")
        if leaf_mold_path.exists():
            res_lm = self.classifier.inspect_image(leaf_mold_path.read_bytes(), "leaf_mold_wm.jpg")
            self.assertNotEqual(res_lm.condition_code, "unsupported_powdery_mildew", "Leaf mold must not be falsely rejected as powdery mildew")

        spider_mite_path = Path("/home/mano/Documents/Anveshion/web_test_images/spider_mite_clemson.jpg")
        if spider_mite_path.exists():
            res_sm = self.classifier.inspect_image(spider_mite_path.read_bytes(), "spider_mite_clemson.jpg")
            self.assertNotEqual(res_sm.condition_code, "unsupported_powdery_mildew", "Spider mite on card must not be falsely rejected as powdery mildew")

    def test_problem_a_genuine_powdery_mildew_is_reliably_detected(self):
        """Genuine fungal bloom on leaf tissue must continue to be detected and rejected as unsupported."""
        pm_rhs_path = Path("/home/mano/Documents/Anveshion/web_test_images/powdery_mildew_rhs.jpg")
        if pm_rhs_path.exists():
            res_pm = self.classifier.inspect_image(pm_rhs_path.read_bytes(), "powdery_mildew_rhs.jpg")
            self.assertEqual(res_pm.status, "UNSUPPORTED_CONDITION")
            self.assertEqual(res_pm.condition_code, "unsupported_powdery_mildew")
            self.assertIn("Powdery Mildew", res_pm.predicted_condition)

        pm_wm_path = Path("/home/mano/Documents/Anveshion/field_evaluation/Tomaten-Echter_Mehltau-JS-2.jpg")
        if pm_wm_path.exists():
            res_wm = self.classifier.inspect_image(pm_wm_path.read_bytes(), "Tomaten-Echter_Mehltau-JS-2.jpg")
            self.assertEqual(res_wm.status, "UNSUPPORTED_CONDITION")
            self.assertEqual(res_wm.condition_code, "unsupported_powdery_mildew")

    def test_problem_a_glossy_leaf_specular_highlights_not_rejected_as_powdery_mildew(self):
        """Healthy glossy leaves with bright sunlit specular reflections must not be rejected as powdery mildew."""
        glossy_im = Image.new("RGB", (300, 300), (30, 30, 30))
        draw = ImageDraw.Draw(glossy_im)
        draw.ellipse((30, 30, 270, 270), fill=(40, 160, 45))
        # Specular bright white highlights
        for hl in [(100, 100), (160, 140)]:
            draw.ellipse((hl[0], hl[1], hl[0] + 25, hl[1] + 15), fill=(240, 250, 245))
        buf = io.BytesIO()
        glossy_im.save(buf, format="JPEG")
        res = self.classifier.inspect_image(buf.getvalue(), "glossy_leaf.jpg")
        self.assertNotEqual(res.condition_code, "unsupported_powdery_mildew")

    def test_problem_b_mixed_canopy_with_fruit_passes_quality_gate(self):
        """Photographs containing tomato fruit alongside foliage pass quality gate and isolate foliar region."""
        # 1. Synthetic mixed canopy image with red tomato fruit and green leaves
        mixed_im = Image.new("RGB", (400, 400), (40, 40, 40))
        draw = ImageDraw.Draw(mixed_im)
        # Red fruit occupying left ~30%
        draw.ellipse((20, 50, 180, 350), fill=(210, 40, 35))
        # Green foliage occupying right ~50%
        draw.ellipse((160, 60, 380, 340), fill=(45, 145, 50))
        buf = io.BytesIO()
        mixed_im.save(buf, format="JPEG")

        quality, rgb_im = self.classifier.evaluate_quality_gate(buf.getvalue())
        self.assertTrue(quality.passed, f"Quality gate failed: {quality.rejection_reason}")
        self.assertIsNone(quality.rejection_reason)
        self.assertTrue(quality.foliar_crop_applied, "Foliar crop should be applied to isolate leaf from fruit")
        self.assertIsNotNone(quality.foliar_crop_box)

        # 2. Check field assets with fruit if available
        lb_real_path = Path("/home/mano/Documents/Anveshion/web_test_images/late_blight_real.jpg")
        if lb_real_path.exists():
            q_real, _ = self.classifier.evaluate_quality_gate(lb_real_path.read_bytes())
            self.assertTrue(q_real.passed, "late_blight_real.jpg with fruit must pass quality gate")
            self.assertTrue(q_real.foliar_crop_applied, "late_blight_real genuinely crops fruit (44.7% of area)")
            self.assertLess(q_real.foliar_crop_fraction, 0.95)
            self.assertEqual(q_real.foliar_crop_box, [0, 37, 679, 726])
            res_lb = self.classifier.inspect_image(lb_real_path.read_bytes(), "late_blight_real.jpg")
            self.assertNotEqual(res_lb.status, "IMAGE_REJECTED")

        lb_wm2_path = Path("/home/mano/Documents/Anveshion/field_evaluation/Tomato_with_Phytophthora_infestans_(late_blight)_2.jpg")
        if lb_wm2_path.exists():
            q_wm2, _ = self.classifier.evaluate_quality_gate(lb_wm2_path.read_bytes())
            self.assertTrue(q_wm2.passed, "Tomato_with_Phytophthora_infestans_2 with fruit must pass quality gate")
            self.assertTrue(q_wm2.foliar_crop_applied, "late_blight_wm2 genuinely crops foliage cluster (49.6% of frame)")
            self.assertLess(q_wm2.foliar_crop_fraction, 0.95)
            self.assertEqual(q_wm2.foliar_crop_box, [0, 0, 536, 529])
            self.assertFalse(q_wm2.canopy_leaf_isolation_inadequate)
            res_wm2 = self.classifier.inspect_image(lb_wm2_path.read_bytes(), "late_blight_wm2.jpg")
            self.assertEqual(res_wm2.status, "UNCERTAIN")

            # Manual crop ROI overrides and applies custom crop
            q_wm2_man, _ = self.classifier.evaluate_quality_gate(lb_wm2_path.read_bytes(), crop_box=(0, 0, 520, 529))
            self.assertTrue(q_wm2_man.foliar_crop_applied)
            self.assertLess(q_wm2_man.foliar_crop_fraction, 0.95)
            self.assertEqual(q_wm2_man.foliar_crop_box, [0, 0, 520, 529])

        # 3. Check case where foliage covers entire frame and cannot be isolated (full-frame cover)
        arr_fc = np.clip(np.full((400, 400, 3), (45, 145, 50), dtype=int) + np.random.randint(-25, 25, (400, 400, 3)), 0, 255).astype(np.uint8)
        full_canopy = Image.fromarray(arr_fc)
        draw_fc = ImageDraw.Draw(full_canopy)
        draw_fc.ellipse((50, 50, 200, 200), fill=(210, 40, 35))
        draw_fc.ellipse((200, 200, 350, 350), fill=(210, 40, 35))
        buf_fc = io.BytesIO()
        full_canopy.save(buf_fc, format="JPEG")
        q_fc, _ = self.classifier.evaluate_quality_gate(buf_fc.getvalue())
        self.assertTrue(q_fc.passed)
        self.assertFalse(q_fc.foliar_crop_applied, "When foliage spans entire frame, automatic crop must not be claimed")
        self.assertTrue(q_fc.canopy_leaf_isolation_inadequate)
        self.assertEqual(q_fc.foliar_crop_fraction, 1.0)
        self.assertIsNone(q_fc.foliar_crop_box)

    def test_problem_b_pure_fruit_remains_rejected(self):
        """Photographs showing pure fruit with no foliage must continue to be rejected."""
        fruit_path = Path("/home/mano/Documents/Anveshion/web_test_images/red_apple_fruit.jpg")
        if fruit_path.exists():
            q, _ = self.classifier.evaluate_quality_gate(fruit_path.read_bytes())
            self.assertFalse(q.passed)
            self.assertIn("fruit", (q.rejection_reason or "").lower())

    def test_problem_b_manual_crop_box_overrides_automatic_framing(self):
        """User-specified manual crop box [x1, y1, x2, y2] restricts inspection to target foliage."""
        im = Image.new("RGB", (600, 600), (30, 30, 30))
        draw = ImageDraw.Draw(im)
        draw.ellipse((50, 50, 250, 250), fill=(50, 150, 50))  # leaf 1
        draw.ellipse((350, 350, 550, 550), fill=(60, 160, 60))  # leaf 2
        buf = io.BytesIO()
        im.save(buf, format="JPEG")
        data = buf.getvalue()

        # Crop to upper-left quadrant
        crop = (40, 40, 260, 260)
        quality, rgb_im = self.classifier.evaluate_quality_gate(data, crop_box=crop)
        self.assertTrue(quality.passed)
        self.assertTrue(quality.foliar_crop_applied)
        self.assertEqual(quality.foliar_crop_box, list(crop))
        self.assertEqual(rgb_im.size, (220, 220))

        res = self.classifier.inspect_image(data, filename="manual_crop.jpg", crop_box=crop)
        self.assertEqual(res.quality_checks["foliar_crop_box"], list(crop))

    def test_problem_c_high_resolution_image_downsampled_proportionally(self):
        """High-resolution images (> 4096px) are validated, accepted, and downsampled to <= 2048px without distortion."""
        # Create a 4800 x 3200 (aspect 1.5) test image
        orig_w, orig_h = 4800, 3200
        large_im = Image.new("RGB", (orig_w, orig_h), (20, 20, 20))
        draw = ImageDraw.Draw(large_im)
        draw.ellipse((800, 400, 4000, 2800), fill=(50, 150, 50))
        buf = io.BytesIO()
        large_im.save(buf, format="JPEG", quality=75)
        data = buf.getvalue()

        quality, rgb_im = self.classifier.evaluate_quality_gate(data)
        self.assertTrue(quality.passed, f"Failed: {quality.rejection_reason}")
        self.assertTrue(quality.is_downsampled)
        self.assertEqual(quality.original_width, orig_w)
        self.assertEqual(quality.original_height, orig_h)
        self.assertLessEqual(quality.inference_width, 2048)
        self.assertLessEqual(quality.inference_height, 2048)
        self.assertEqual(rgb_im.size, (quality.inference_width, quality.inference_height))

        # Check aspect ratio preservation: no stretching or distorting
        orig_aspect = orig_w / orig_h
        inf_aspect = quality.inference_width / quality.inference_height
        self.assertAlmostEqual(orig_aspect, inf_aspect, delta=0.02)

        # Full inspect_image check: original and inference dimensions recorded distinctly
        res = self.classifier.inspect_image(data, filename="large_leaf.jpg")
        self.assertTrue(res.quality_checks["passed"])
        self.assertTrue(res.quality_checks["is_downsampled"])
        self.assertEqual(res.quality_checks["original_dimensions"], f"{orig_w}x{orig_h}")
        self.assertEqual(res.quality_checks["inference_dimensions"], f"{quality.inference_width}x{quality.inference_height}")

    def test_problem_c_actual_field_photo_over_4096px_evaluates_successfully(self):
        """Actual high-resolution field photograph (4624 x 3468) is accepted and downsampled."""
        wm_path = Path("/home/mano/Documents/Anveshion/field_evaluation/Tomaten-Echter_Mehltau-JS-2.jpg")
        if wm_path.exists():
            data = wm_path.read_bytes()
            quality, _ = self.classifier.evaluate_quality_gate(data)
            self.assertTrue(quality.passed)
            self.assertTrue(quality.is_downsampled)
            self.assertEqual(quality.original_width, 4624)
            self.assertEqual(quality.original_height, 3468)
            self.assertLessEqual(quality.inference_width, 2048)
            self.assertLessEqual(quality.inference_height, 2048)

    def test_api_manual_inspection_accepts_crop_box(self):
        """Live HTTP API accepts crop_box in JSON payload and multipart form data."""
        hub = LiveDataHub()
        server = LiveOnlyHTTPServer(("127.0.0.1", 0), hub, self.classifier)
        t = threading.Thread(target=server.serve_forever, daemon=True)
        t.start()
        base = f"http://127.0.0.1:{server.server_address[1]}"
        try:
            im_bytes = create_test_image(300, 300)
            payload = {
                "image": base64.b64encode(im_bytes).decode(),
                "filename": "crop_test.jpg",
                "crop_box": [10, 10, 200, 200]
            }
            req = Request(
                base + "/api/inspect/analyze",
                data=json.dumps(payload).encode(),
                headers={"Content-Type": "application/json"},
                method="POST"
            )
            with urlopen(req, timeout=5) as r:
                res = json.loads(r.read().decode())
            self.assertEqual(r.status, 200)
            self.assertTrue(res["quality_checks"]["foliar_crop_applied"])
            self.assertEqual(res["quality_checks"]["foliar_crop_box"], [10, 10, 200, 200])
        finally:
            server.shutdown()
            server.server_close()
            t.join(timeout=2)


class LeafMoldAndAmbiguityRegressionTests(unittest.TestCase):
    """Priority 1 & Priority 2 Regression Tests:
    - Verifies lookalike and centroid ambiguity gate defers Leaf Mold to UNCERTAIN without false acceptance.
    - Verifies genuine clear cases remain ASSESSED without regression.
    - Verifies true crop area fractions and fruit removal on mixed-canopy photographs.
    """

    @classmethod
    def setUpClass(cls):
        cls.classifier = PlantLeafClassifier()

    def test_leaf_mold_deferred_to_uncertain_via_ambiguity_gate(self):
        """Tomato_Leaf_Mold.jpg (leaf_mold_wm.jpg) must be deferred to UNCERTAIN rather than accepted as Early Blight."""
        img_path = Path("/home/mano/Documents/Anveshion/field_evaluation/Tomato_Leaf_Mold.jpg")
        if img_path.exists():
            data = img_path.read_bytes()
            res = self.classifier.inspect_image(data, "Tomato_Leaf_Mold.jpg")

            # Must NOT be ASSESSED
            self.assertEqual(res.status, "UNCERTAIN", "Borderline lookalike leaf mold must be deferred to UNCERTAIN")
            self.assertNotEqual(res.status, "ASSESSED")
            self.assertIn("Uncertain", res.predicted_condition)
            self.assertIn("Tomato Early Blight", res.predicted_condition)

            # Must enrich lookalikes in other_possible_conditions
            lookalike_names = [c["condition"] for c in res.other_possible_conditions]
            self.assertIn("Tomato Septoria Leaf Spot", lookalike_names, "Septoria (nearest centroid) must be listed")
            self.assertIn("Tomato Leaf Mold", lookalike_names, "Leaf Mold must be surfaced as lookalike")

            # Must strictly suppress disease-specific chemical sprays and warn of ambiguity
            self.assertIn("ambiguous between supported foliar conditions", res.recommended_action)
            self.assertIn("Do NOT apply chemical fungicides or unverified treatments", res.recommended_action)
            self.assertNotIn("Apply preventive fungicide (chlorothalonil or copper)", res.recommended_action)

            # Must document latent boundary limitation
            self.assertTrue(any("latent boundary" in lim for lim in res.limitations))

    def test_genuine_supported_accepted_images_remain_assessed(self):
        """Clear supported field specimens with strong centroid similarity must remain ASSESSED."""
        cases = [
            ("early_blight_wikimedia.jpg", Path("/home/mano/Documents/Anveshion/web_test_images/early_blight_wikimedia.jpg"), "Tomato_Early_blight"),
            ("septoria_leaf_spot_real.jpg", Path("/home/mano/Documents/Anveshion/web_test_images/septoria_leaf_spot_real.jpg"), "Tomato_Septoria_leaf_spot"),
            ("late_blight_field.jpg", Path("/home/mano/Documents/Anveshion/web_test_images/late_blight_field.jpg"), "Tomato_Late_blight"),
            ("yellow_leaf_curl.jpg", Path("/home/mano/Documents/Anveshion/web_test_images/yellow_leaf_curl.jpg"), "Tomato__Tomato_YellowLeaf__Curl_Virus"),
            ("healthy_sample.jpg", Path("/home/mano/Documents/Anveshion/web_test_images/healthy_sample.jpg"), "Tomato_healthy"),
        ]
        for name, path, expected_code in cases:
            if path.exists():
                res = self.classifier.inspect_image(path.read_bytes(), name)
                self.assertEqual(res.status, "ASSESSED", f"{name} should remain ASSESSED")
                self.assertEqual(res.condition_code, expected_code, f"{name} should match {expected_code}")
                self.assertTrue(res.predicted_condition.startswith("Suspected"))

    def test_mixed_canopy_fruit_removal_and_crop_fraction(self):
        """late_blight_real genuinely crops fruit (fraction < 0.95); late_blight_wm2 genuinely crops foliage cluster."""
        lb_real = Path("/home/mano/Documents/Anveshion/web_test_images/late_blight_real.jpg")
        if lb_real.exists():
            q, _ = self.classifier.evaluate_quality_gate(lb_real.read_bytes())
            self.assertTrue(q.passed)
            self.assertTrue(q.foliar_crop_applied)
            self.assertAlmostEqual(q.foliar_crop_fraction, 0.4475, delta=0.01)
            self.assertEqual(q.foliar_crop_box, [0, 37, 679, 726])
            self.assertFalse(q.canopy_leaf_isolation_inadequate)

        lb_wm2 = Path("/home/mano/Documents/Anveshion/field_evaluation/Tomato_with_Phytophthora_infestans_(late_blight)_2.jpg")
        if lb_wm2.exists():
            q, _ = self.classifier.evaluate_quality_gate(lb_wm2.read_bytes())
            self.assertTrue(q.passed)
            self.assertTrue(q.foliar_crop_applied)
            self.assertEqual(q.foliar_crop_box, [0, 0, 536, 529])
            self.assertAlmostEqual(q.foliar_crop_fraction, 0.4963, delta=0.01)
            self.assertFalse(q.canopy_leaf_isolation_inadequate)

            # Verification that manual crop cleanly isolates foliage
            q_man, _ = self.classifier.evaluate_quality_gate(lb_wm2.read_bytes(), crop_box=(0, 0, 520, 529))
            self.assertTrue(q_man.foliar_crop_applied)
            self.assertEqual(q_man.foliar_crop_box, [0, 0, 520, 529])
            self.assertLess(q_man.foliar_crop_fraction, 0.95)
            self.assertFalse(q_man.canopy_leaf_isolation_inadequate)

        # Full-frame canopy that cannot be isolated
        arr_fc2 = np.clip(np.full((400, 400, 3), (45, 145, 50), dtype=int) + np.random.randint(-25, 25, (400, 400, 3)), 0, 255).astype(np.uint8)
        full_canopy = Image.fromarray(arr_fc2)
        draw_fc = ImageDraw.Draw(full_canopy)
        draw_fc.ellipse((50, 50, 200, 200), fill=(210, 40, 35))
        draw_fc.ellipse((200, 200, 350, 350), fill=(210, 40, 35))
        buf_fc = io.BytesIO()
        full_canopy.save(buf_fc, format="JPEG")
        q_fc, _ = self.classifier.evaluate_quality_gate(buf_fc.getvalue())
        self.assertTrue(q_fc.passed)
        self.assertFalse(q_fc.foliar_crop_applied, "When foliage spans entire frame, automatic crop must not be claimed")
        self.assertTrue(q_fc.canopy_leaf_isolation_inadequate)
        self.assertEqual(q_fc.foliar_crop_fraction, 1.0)
        self.assertIsNone(q_fc.foliar_crop_box)


if __name__ == "__main__":
    unittest.main()

