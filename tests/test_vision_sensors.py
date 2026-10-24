import tempfile
import time
import unittest
from pathlib import Path

from plant_rover.scenarios import SCENARIOS
from plant_rover.sensors import VirtualSensorNode, build_history_summary
from plant_rover.vision import ImageQualityGate, SyntheticCamera


class VisionTests(unittest.TestCase):
    def test_clear_synthetic_image_passes_quality_gate(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = SyntheticCamera(tmp).capture("run", "R01-P001", "healthy_looking_fixture")
            result = ImageQualityGate().evaluate(path)
            self.assertTrue(result.accepted, result.reason)
            self.assertGreaterEqual(result.score, 0)

    def test_blurry_and_no_leaf_images_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            camera = SyntheticCamera(tmp)
            gate = ImageQualityGate()
            blur = camera.capture("run", "R01-P001", "healthy_looking_fixture", "blur")
            no_leaf = camera.capture("run", "R01-P002", "healthy_looking_fixture", "no_leaf")
            self.assertFalse(gate.evaluate(blur).accepted)
            self.assertFalse(gate.evaluate(no_leaf).accepted)

    def test_overexposure_and_underexposure_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            camera = SyntheticCamera(tmp); gate = ImageQualityGate()
            over = camera.capture("run", "R01-P001", "healthy_looking_fixture", "overexposed")
            under = camera.capture("run", "R01-P002", "healthy_looking_fixture", "underexposed")
            self.assertFalse(gate.evaluate(over).accepted)
            self.assertFalse(gate.evaluate(under).accepted)

    def test_capture_failure_has_no_image(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertIsNone(SyntheticCamera(tmp).capture("run", "R01-P001", "healthy_looking_fixture", "capture_failure"))


class SensorTests(unittest.TestCase):
    def test_drought_time_series_decreases_over_24_hours(self):
        node = VirtualSensorNode(1, "zone-1", seed=5)
        early = node.sample(1_700_000_000, 0, "drought")
        late = node.sample(1_700_086_200, 1430, "drought")
        self.assertGreater(early.soil_moisture_pct, late.soil_moisture_pct + 20)

    def test_history_summary_captures_trend(self):
        node = VirtualSensorNode(1, "zone-1", seed=5)
        rows = [node.sample(1_700_000_000 + i*600, i*10, "drought") for i in range(144)]
        summary = build_history_summary(rows, 1, rows[-1].timestamp_epoch)
        self.assertEqual(summary.sample_count, 144)
        self.assertLess(summary.soil_moisture_trend_pct_per_hour, 0)
        self.assertFalse(summary.stale)


if __name__ == "__main__":
    unittest.main()
