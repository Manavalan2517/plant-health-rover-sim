import unittest

from plant_rover.protocol import (ProtocolError, RadioCodec, RadioFrame, SerialCodec,
                                  SerialFrame, SerialFrameParser, SerialType, crc16_ccitt_false)
from plant_rover.models import (AssessmentCode, CueCode, HistorySummary, PlantReport,
                                RadioType, SensorReading)


class CRCAndSerialTests(unittest.TestCase):
    def test_crc16_ccitt_false_known_vector(self):
        self.assertEqual(crc16_ccitt_false(b"123456789"), 0x29B1)

    def test_serial_round_trip_and_corruption_detection(self):
        frame = SerialFrame(37, SerialType.COMMAND, b"MOVE:28")
        raw = SerialCodec.encode(frame)
        self.assertEqual(SerialCodec.decode_exact(raw), frame)
        bad = bytearray(raw)
        bad[-3] ^= 0x08
        with self.assertRaises(ProtocolError):
            SerialCodec.decode_exact(bytes(bad))

    def test_parser_handles_fragmentation_noise_and_resynchronises(self):
        first = SerialCodec.encode(SerialFrame(1, SerialType.HEARTBEAT, b""))
        second = SerialCodec.encode(SerialFrame(2, SerialType.COMMAND, b"MOVE:10"))
        parser = SerialFrameParser()
        self.assertEqual(parser.feed(b"noise" + first[:3]), [])
        self.assertEqual(parser.feed(first[3:] + b"\x00\x01" + second[:5]), [SerialFrame(1, SerialType.HEARTBEAT, b"")])
        self.assertEqual(parser.feed(second[5:]), [SerialFrame(2, SerialType.COMMAND, b"MOVE:10")])
        self.assertGreaterEqual(parser.discarded_bytes, 5)

    def test_crc_failure_does_not_prevent_following_valid_frame(self):
        broken = bytearray(SerialCodec.encode(SerialFrame(4, SerialType.COMMAND, b"MOVE:1")))
        broken[-1] ^= 0x01
        good = SerialCodec.encode(SerialFrame(5, SerialType.HEARTBEAT))
        parser = SerialFrameParser()
        frames = parser.feed(bytes(broken) + good)
        self.assertEqual(frames, [SerialFrame(5, SerialType.HEARTBEAT)])
        self.assertGreater(parser.bad_frames, 0)


class RadioAndPayloadTests(unittest.TestCase):
    def test_radio_round_trip_and_crc(self):
        frame = RadioFrame(65220, RadioType.SENSOR_READING, b"abc\x00xyz")
        self.assertEqual(RadioCodec.decode_exact(RadioCodec.encode(frame)), frame)
        raw = bytearray(RadioCodec.encode(frame)); raw[7] ^= 0x20
        with self.assertRaises(ProtocolError):
            RadioCodec.decode_exact(bytes(raw))

    def test_sensor_payload_round_trip(self):
        row = SensorReading(3, 400, 1_798_000_000, 23.45, 21.27, 32.11, 84.36,
                            91, 28, 142, 6.23, 880, 73, 0)
        decoded = SensorReading.from_payload(row.to_payload())
        self.assertEqual(decoded.node_id, row.node_id)
        self.assertEqual(decoded.sequence, row.sequence)
        self.assertAlmostEqual(decoded.soil_moisture_pct, row.soil_moisture_pct)
        self.assertAlmostEqual(decoded.air_temperature_c, row.air_temperature_c)
        self.assertEqual(decoded.nitrogen_mg_kg, row.nitrogen_mg_kg)

    def test_plant_report_payload_round_trip(self):
        report = PlantReport(1, 7, 1_798_000_000, CueCode.SPOT_LIKE_FIXTURE,
            AssessmentCode.UNCERTAIN, 0, True, 64, 47, 26, 61, 89, 82, 1)
        self.assertEqual(PlantReport.from_payload(report.to_payload()), report)

    def test_history_summary_payload_round_trip(self):
        summary = HistorySummary(2, 1_798_000_000, 144, 23, 61, -1, 21, 34, 92, False)
        self.assertEqual(HistorySummary.from_payload(summary.to_payload()), summary)

    def test_payload_length_must_be_exact(self):
        with self.assertRaises(ProtocolError):
            SensorReading.from_payload(b"short")


if __name__ == "__main__":
    unittest.main()
