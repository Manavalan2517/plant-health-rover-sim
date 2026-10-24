"""Tests for Hardware Abstraction Layer (HAL) interfaces and navigation state machine."""
import tempfile
import unittest
from pathlib import Path
from PIL import Image

from plant_rover.hal import NavigationState, NavigationStateMachine
from plant_rover.vision import RecordedImageCamera


class HALTests(unittest.TestCase):
    def test_navigation_state_machine_happy_path(self):
        sm = NavigationStateMachine(NavigationState.IDLE)
        self.assertEqual(sm.current_state, NavigationState.IDLE)

        # Transition to READY
        self.assertTrue(sm.transition(NavigationState.READY, "system_startup"))
        self.assertEqual(sm.current_state, NavigationState.READY)

        # Transition to NAVIGATING
        self.assertTrue(sm.transition(NavigationState.NAVIGATING, "start_row_inspection"))
        self.assertEqual(sm.current_state, NavigationState.NAVIGATING)

        # Approaching plant
        self.assertTrue(sm.transition(NavigationState.APPROACHING_PLANT, "plant_marker_detected"))
        self.assertTrue(sm.transition(NavigationState.STOPPED_FOR_INSPECTION, "aligned_with_plant"))
        self.assertTrue(sm.transition(NavigationState.CAPTURING, "trigger_camera"))
        self.assertTrue(sm.transition(NavigationState.ANALYZING, "inference_in_progress"))
        self.assertTrue(sm.transition(NavigationState.RECORDING, "persisting_report"))
        self.assertTrue(sm.transition(NavigationState.RESUMING, "inspection_complete"))
        self.assertTrue(sm.transition(NavigationState.NAVIGATING, "next_checkpoint"))
        self.assertEqual(sm.current_state, NavigationState.NAVIGATING)

    def test_navigation_state_machine_invalid_transition(self):
        sm = NavigationStateMachine(NavigationState.IDLE)
        # Cannot jump straight from IDLE to RECORDING
        self.assertFalse(sm.transition(NavigationState.RECORDING, "illegal_jump"))
        self.assertEqual(sm.current_state, NavigationState.IDLE)

    def test_navigation_state_machine_emergency_stop(self):
        sm = NavigationStateMachine(NavigationState.NAVIGATING)
        sm.force_emergency_stop("hardware_bumper_pressed")
        self.assertEqual(sm.current_state, NavigationState.EMERGENCY_STOP)
        history = sm.get_history()
        self.assertIn(("EMERGENCY_STOP", "hardware_bumper_pressed"), history)

    def test_recorded_image_camera(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            rec_dir = tmp_path / "recorded"
            out_dir = tmp_path / "output"
            rec_dir.mkdir()

            # Create a mock recorded image file
            sample_img = rec_dir / "sample_healthy_looking_fixture.png"
            Image.new("RGB", (100, 100), (0, 200, 0)).save(sample_img)

            cam = RecordedImageCamera(rec_dir, out_dir)
            captured = cam.capture("test_session", "P001", "healthy_looking_fixture")
            self.assertIsNotNone(captured)
            self.assertTrue(captured.exists())


if __name__ == "__main__":
    unittest.main()
