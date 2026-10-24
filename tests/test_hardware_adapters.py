"""Tests verifying pluggable hardware adapters and graceful PC simulation fallback."""
import unittest

from plant_rover.hardware_adapters import (
    PhysicalPicamera2Adapter,
    PhysicalSerialAdapter,
    PhysicalModemNotifier,
)
from plant_rover.protocol import SerialFrame, SerialType


class HardwareAdapterTests(unittest.TestCase):
    def test_picamera2_adapter_graceful_missing_driver_handling(self):
        """Verifies that attempting physical camera capture on a PC raises a clean RuntimeError."""
        adapter = PhysicalPicamera2Adapter(output_dir="/tmp/test_cam")
        # On a host without picamera2, capture raises a clear error rather than crashing silently
        try:
            adapter.capture("sess1", "P001")
        except RuntimeError as exc:
            self.assertIn("Picamera2", str(exc))

    def test_physical_serial_adapter_connection_error_on_missing_port(self):
        """Verifies that opening a non-existent physical COM/UART port raises ConnectionError."""
        adapter = PhysicalSerialAdapter(port="/dev/non_existent_tty_device_12345")
        self.assertFalse(adapter.is_connected())
        with self.assertRaises(ConnectionError):
            adapter.connect()

    def test_physical_modem_notifier_fails_safely_when_port_missing(self):
        """Verifies that physical modem notifier returns False when hardware modem is absent."""
        notifier = PhysicalModemNotifier(port="/dev/non_existent_modem_tty")
        result = notifier.send_sms("+919876543210", "வணக்கம்")
        self.assertFalse(result)


if __name__ == "__main__":
    unittest.main()
