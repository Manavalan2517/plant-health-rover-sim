"""Verifies golden byte-level vectors for serial and radio wire protocols."""
import json
from pathlib import Path
import unittest

from plant_rover.protocol import crc16_ccitt_false, SerialCodec, RadioCodec
from plant_rover.models import SensorReading, PlantReport, HistorySummary


class GoldenVectorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        golden_path = Path(__file__).parent / "golden_vectors.json"
        with open(golden_path, "r", encoding="utf-8") as f:
            cls.golden = json.load(f)

    def test_crc16_golden_vectors(self):
        for item in self.golden["crc16_ccitt_false"]:
            inp = bytes.fromhex(item["input_hex"])
            expected = item["expected_crc"]
            self.assertEqual(crc16_ccitt_false(inp), expected, f"CRC mismatch on input: {inp}")

    def test_serial_golden_vectors(self):
        for item in self.golden["serial_frames"]:
            encoded = bytes.fromhex(item["encoded_hex"])
            decoded = SerialCodec.decode_exact(encoded)
            self.assertEqual(decoded.sequence, item["sequence"])
            self.assertEqual(int(decoded.frame_type), item["type"])
            self.assertEqual(decoded.payload.hex(), item["payload_hex"])
            # Re-encode check
            re_encoded = SerialCodec.encode(decoded)
            self.assertEqual(re_encoded, encoded)

    def test_radio_golden_vectors(self):
        for item in self.golden["radio_frames"]:
            encoded = bytes.fromhex(item["encoded_hex"])
            decoded = RadioCodec.decode_exact(encoded)
            self.assertEqual(decoded.sequence, item["sequence"])
            self.assertEqual(int(decoded.message_type), item["type"])
            self.assertEqual(decoded.payload.hex(), item["payload_hex"])
            # Verify models can unpack payloads
            if item["name"] == "sensor_reading":
                sr = SensorReading.from_payload(decoded.payload)
                self.assertEqual(sr.sequence, 42)
            elif item["name"] == "plant_report":
                pr = PlantReport.from_payload(decoded.payload)
                self.assertEqual(pr.plant_index, 5)
            elif item["name"] == "history_summary":
                hs = HistorySummary.from_payload(decoded.payload)
                self.assertEqual(hs.node_id, 2)


if __name__ == "__main__":
    unittest.main()
