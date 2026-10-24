import unittest

from plant_rover.models import RadioType
from plant_rover.protocol import RadioCodec, RadioFrame, SerialCodec, SerialFrame, SerialType
from plant_rover.transports import (FaultInjectingLink, FaultProfile, SimulatedSerialSession,
                                    VirtualESP32, reliable_radio_send)


def ack(sequence):
    return RadioCodec.encode(RadioFrame(sequence, RadioType.ACK, sequence.to_bytes(2, "big")))


def receiver(raw):
    try:
        frame = RadioCodec.decode_exact(raw)
    except ValueError:
        return None
    return ack(frame.sequence)


def ack_validator(raw, expected):
    try:
        f = RadioCodec.decode_exact(raw)
    except ValueError:
        return None
    if f.message_type == RadioType.ACK and f.sequence == expected and f.payload == expected.to_bytes(2, "big"):
        return b"accepted"
    return None


class TransportTests(unittest.TestCase):
    def test_reliable_radio_delivers_encoded_frame_and_ack(self):
        link = FaultInjectingLink("test", FaultProfile(max_frame_bytes=64), seed=1)
        packet = RadioCodec.encode(RadioFrame(9, RadioType.PLANT_REPORT, b"binary"))
        success, attempts, reason = reliable_radio_send(link, packet, 9, receiver,
            lambda raw: ack_validator(raw, 9), retries=3)
        self.assertTrue(success, reason)
        self.assertEqual(attempts, 1)
        self.assertGreaterEqual(link.stats.accepted, 2)  # report + actual ACK frame

    def test_radio_drop_exhausts_retry_budget(self):
        link = FaultInjectingLink("drop", FaultProfile(drop_probability=1.0, max_frame_bytes=64), seed=2)
        packet = RadioCodec.encode(RadioFrame(10, RadioType.PLANT_REPORT, b"x"))
        success, attempts, reason = reliable_radio_send(link, packet, 10, receiver,
            lambda raw: ack_validator(raw, 10), retries=3)
        self.assertFalse(success)
        self.assertEqual(attempts, 3)
        self.assertEqual(link.stats.network_drops, 3)

    def test_radio_rejects_oversized_packet_before_receiver(self):
        link = FaultInjectingLink("small", FaultProfile(max_frame_bytes=10), seed=3)
        result = link.deliver(b"x" * 11, lambda raw: b"ack")
        self.assertFalse(result.accepted)
        self.assertEqual(result.reason, "frame_too_large")

    def test_serial_protocol_acknowledgement(self):
        session = SimulatedSerialSession(FaultProfile(max_frame_bytes=260), seed=4)
        raw = SerialCodec.encode(SerialFrame(1, SerialType.COMMAND, b"MOVE:28"))
        ok, attempts, reason = session.send(raw, 1, runtime_ms=100)
        self.assertTrue(ok, reason)
        self.assertEqual(attempts, 1)
        self.assertTrue(session.esp32.motor_enabled)
        self.assertEqual(session.esp32.speed_pct, 28)

    def test_serial_loss_causes_watchdog_safe_stop(self):
        esp32 = VirtualESP32()
        move = SerialCodec.encode(SerialFrame(1, SerialType.COMMAND, b"MOVE:28"))
        self.assertIsNotNone(esp32.receive_serial(move, runtime_ms=100))
        self.assertTrue(esp32.motor_enabled)
        self.assertFalse(esp32.check_watchdog(500))  # timeout is strictly greater than 400ms
        self.assertTrue(esp32.check_watchdog(501))
        self.assertFalse(esp32.motor_enabled)
        self.assertTrue(esp32.watchdog_tripped)


if __name__ == "__main__":
    unittest.main()
