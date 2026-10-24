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
        self.assertEqual(result.confidence_pct, 0.0)
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


if __name__ == "__main__":
    unittest.main()
