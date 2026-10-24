"""Hardware-independent transports with repeatable fault injection."""
from __future__ import annotations

from dataclasses import dataclass
import random
import time
from typing import Callable

from .protocol import ProtocolError, RadioCodec, RadioFrame, SerialCodec, SerialFrame, SerialType
from .models import RadioType


@dataclass
class FaultProfile:
    enabled: bool = True
    drop_probability: float = 0.0
    corrupt_probability: float = 0.0
    duplicate_probability: float = 0.0
    latency_ms: int = 0
    max_frame_bytes: int = 64

    def validate(self) -> None:
        for name in ("drop_probability", "corrupt_probability", "duplicate_probability"):
            value = getattr(self, name)
            if not 0 <= value <= 1:
                raise ValueError(f"{name} must be between 0 and 1")
        if self.latency_ms < 0:
            raise ValueError("latency_ms cannot be negative")
        if self.max_frame_bytes < 1:
            raise ValueError("max_frame_bytes must be positive")


@dataclass
class TransferStats:
    attempts: int = 0
    network_drops: int = 0
    corruptions: int = 0
    duplicates: int = 0
    size_rejections: int = 0
    disconnects: int = 0
    accepted: int = 0


@dataclass
class DeliveryResult:
    accepted: bool
    responses: list[bytes]
    reason: str
    latency_ms: int = 0


class FaultInjectingLink:
    """A byte-only link: callers must serialize messages before using it.

    A receiver returns an encoded response (usually an ACK) for a valid packet,
    or None when the packet is dropped/rejected. The same fault model is applied
    in both directions, so ACK loss and duplicate delivery are testable too.
    """

    def __init__(self, name: str, profile: FaultProfile | None = None, seed: int = 17) -> None:
        self.name = name
        self.profile = profile or FaultProfile()
        self.profile.validate()
        self._rng = random.Random(seed)
        self.stats = TransferStats()
        self.last_reason = "idle"

    def reset_random_seed(self, seed: int) -> None:
        self._rng = random.Random(seed)
        self.stats = TransferStats()

    def deliver(self, frame: bytes, receiver: Callable[[bytes], bytes | None]) -> DeliveryResult:
        self.stats.attempts += 1
        if not self.profile.enabled:
            self.stats.disconnects += 1
            self.last_reason = "disconnected"
            return DeliveryResult(False, [], "disconnected")
        if len(frame) > self.profile.max_frame_bytes:
            self.stats.size_rejections += 1
            self.last_reason = "frame_too_large"
            return DeliveryResult(False, [], "frame_too_large")
        if self._rng.random() < self.profile.drop_probability:
            self.stats.network_drops += 1
            self.last_reason = "network_drop"
            return DeliveryResult(False, [], "network_drop")
        packet = bytearray(frame)
        corrupted = self._rng.random() < self.profile.corrupt_probability
        if corrupted and packet:
            # Mutate a deterministic random bit. CRC validation is the receiver's job.
            idx = self._rng.randrange(0, len(packet))
            packet[idx] ^= 1 << self._rng.randrange(0, 8)
            self.stats.corruptions += 1
        if self.profile.latency_ms:
            time.sleep(self.profile.latency_ms / 1000.0)
        copies = 2 if self._rng.random() < self.profile.duplicate_probability else 1
        if copies == 2:
            self.stats.duplicates += 1
        responses: list[bytes] = []
        for _ in range(copies):
            try:
                response = receiver(bytes(packet))
            except (ProtocolError, ValueError, struct_error_types()) as exc:  # type: ignore[misc]
                response = None
                self.last_reason = f"receiver_rejected:{type(exc).__name__}"
            except Exception as exc:  # receiver errors become link diagnostics, not hidden success
                response = None
                self.last_reason = f"receiver_error:{type(exc).__name__}"
            if response is not None:
                responses.append(response)
        accepted = bool(responses)
        if accepted:
            self.stats.accepted += 1
            self.last_reason = "accepted"
        elif self.last_reason == "idle":
            self.last_reason = "receiver_rejected"
        return DeliveryResult(accepted, responses, self.last_reason, self.profile.latency_ms)


def struct_error_types():
    # Import lazily to keep the module's public API independent of struct details.
    import struct
    return struct.error


def _ack_frame(sequence: int, serial: bool = False) -> bytes:
    if serial:
        # Echo the 8-bit acknowledged sequence in both SEQUENCE and PAYLOAD.
        return SerialCodec.encode(SerialFrame(sequence & 0xFF, SerialType.ACK, bytes((sequence & 0xFF,))))
    return RadioCodec.encode(RadioFrame(sequence & 0xFFFF, RadioType.ACK, sequence.to_bytes(2, "big")))


def reliable_radio_send(
    link: FaultInjectingLink,
    frame: bytes,
    sequence: int,
    receiver: Callable[[bytes], bytes | None],
    ack_handler: Callable[[bytes], bytes | None],
    retries: int = 3,
) -> tuple[bool, int, str]:
    """Send a serialized radio frame; ACK is also serialized and fault-injected."""
    for attempt in range(1, retries + 1):
        data_result = link.deliver(frame, receiver)
        for ack_frame in data_result.responses:
            ack_result = link.deliver(ack_frame, ack_handler)
            if ack_result.accepted:
                return True, attempt, "acknowledged"
    return False, retries, link.last_reason


class VirtualESP32:
    """Host-side control-firmware simulator. No GPIO or serial device is opened."""

    WATCHDOG_TIMEOUT_MS = 400

    def __init__(self) -> None:
        self.motor_enabled = False
        self.speed_pct = 0
        self.heading_deg = 0.0
        self.last_valid_frame_ms: int | None = None
        self.watchdog_tripped = False
        self.emergency_stop = False
        self._seen_sequences: set[int] = set()
        self._seen_order: list[int] = []

    def receive_serial(self, raw: bytes, runtime_ms: int) -> bytes | None:
        try:
            frame = SerialCodec.decode_exact(raw)
        except ProtocolError:
            return None
        if frame.frame_type not in (SerialType.COMMAND, SerialType.HEARTBEAT, SerialType.STOP):
            return None
        self.last_valid_frame_ms = runtime_ms
        duplicate = frame.sequence in self._seen_sequences
        if not duplicate:
            self._seen_sequences.add(frame.sequence)
            self._seen_order.append(frame.sequence)
            if len(self._seen_order) > 128:
                old = self._seen_order.pop(0)
                self._seen_sequences.discard(old)
            if frame.frame_type == SerialType.STOP:
                self.motor_enabled = False
                self.speed_pct = 0
                self.emergency_stop = True
            elif not self.emergency_stop and frame.frame_type == SerialType.COMMAND:
                # This host-side contract accepts a simple ASCII command for easy inspection.
                command = frame.payload.decode("utf-8", errors="replace")
                if command.startswith("MOVE:"):
                    try:
                        self.speed_pct = max(0, min(100, int(command.split(":", 1)[1])))
                        self.motor_enabled = self.speed_pct > 0
                        self.watchdog_tripped = False
                    except ValueError:
                        return None
                elif command == "STOP":
                    self.motor_enabled = False
                    self.speed_pct = 0
            elif frame.frame_type == SerialType.HEARTBEAT:
                pass
        return _ack_frame(frame.sequence, serial=True)

    def check_watchdog(self, runtime_ms: int) -> bool:
        if self.motor_enabled and self.last_valid_frame_ms is not None:
            if runtime_ms - self.last_valid_frame_ms > self.WATCHDOG_TIMEOUT_MS:
                self.motor_enabled = False
                self.speed_pct = 0
                self.watchdog_tripped = True
                return True
        return False

    def clear_emergency_stop(self) -> None:
        self.emergency_stop = False
        self.watchdog_tripped = False
        self.motor_enabled = False
        self.speed_pct = 0
        self._seen_sequences.clear()
        self._seen_order.clear()


class SimulatedSerialSession:
    def __init__(self, profile: FaultProfile | None = None, seed: int = 31) -> None:
        self.link = FaultInjectingLink("pi-esp32-serial", profile or FaultProfile(max_frame_bytes=260), seed)
        self.esp32 = VirtualESP32()
        self._received_ack: int | None = None

    def send(self, frame: bytes, sequence: int, runtime_ms: int, retries: int = 3) -> tuple[bool, int, str]:
        for attempt in range(1, retries + 1):
            data_result = self.link.deliver(frame, lambda raw: self.esp32.receive_serial(raw, runtime_ms))
            for ack in data_result.responses:
                ack_result = self.link.deliver(ack, lambda raw: self._validate_ack(raw, sequence))
                if ack_result.accepted:
                    return True, attempt, "acknowledged"
        return False, retries, self.link.last_reason

    def _validate_ack(self, raw: bytes, expected_sequence: int) -> bytes | None:
        try:
            frame = SerialCodec.decode_exact(raw)
        except ProtocolError:
            return None
        if frame.frame_type != SerialType.ACK or frame.sequence != expected_sequence:
            return None
        if frame.payload != bytes((expected_sequence & 0xFF,)):
            return None
        self._received_ack = expected_sequence
        return b"ACK_ACCEPTED"
