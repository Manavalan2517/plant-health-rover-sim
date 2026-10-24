"""Wire formats for the simulation-first Plant-Health Inspection Rover.

Serial format (Pi <-> virtual ESP32):
    START(0xAA) | LENGTH | SEQUENCE | TYPE | PAYLOAD | CRC16_BE
LENGTH counts SEQUENCE + TYPE + PAYLOAD bytes (not START, LENGTH, or CRC).
CRC is CRC-16/CCITT-FALSE over LENGTH through the final payload byte
(poly=0x1021, init=0xFFFF, no reflection, xorout=0x0000).

Radio format (virtual LoRa / ESP-NOW):
    MAGIC(0xC3) | VERSION | TYPE | LENGTH | SEQUENCE_BE16 | PAYLOAD | CRC16_BE
CRC covers VERSION through PAYLOAD, excluding MAGIC and CRC.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
import struct


class ProtocolError(ValueError):
    """Invalid, incomplete, or corrupt wire frame."""


def crc16_ccitt_false(data: bytes) -> int:
    crc = 0xFFFF
    for byte in data:
        crc ^= byte << 8
        for _ in range(8):
            crc = ((crc << 1) ^ 0x1021) & 0xFFFF if crc & 0x8000 else (crc << 1) & 0xFFFF
    return crc


class SerialType(IntEnum):
    COMMAND = 0x01
    TELEMETRY = 0x02
    HEARTBEAT = 0x03
    STOP = 0x04
    ACK = 0x7F
    FAULT = 0x80


@dataclass(frozen=True)
class SerialFrame:
    sequence: int
    frame_type: int
    payload: bytes = b""


class SerialCodec:
    START = 0xAA
    HEADER_BODY_BYTES = 2  # sequence + type, counted by LENGTH
    MAX_LENGTH = 255

    @classmethod
    def encode(cls, frame: SerialFrame) -> bytes:
        if not 0 <= frame.sequence <= 255:
            raise ProtocolError("serial sequence must fit in one byte")
        if not 0 <= int(frame.frame_type) <= 255:
            raise ProtocolError("serial frame type must fit in one byte")
        length = cls.HEADER_BODY_BYTES + len(frame.payload)
        if not cls.HEADER_BODY_BYTES <= length <= cls.MAX_LENGTH:
            raise ProtocolError("serial payload exceeds one-byte LENGTH field")
        protected = bytes((length, frame.sequence, int(frame.frame_type))) + frame.payload
        crc = crc16_ccitt_false(protected)
        return bytes((cls.START,)) + protected + struct.pack("!H", crc)

    @classmethod
    def decode_exact(cls, data: bytes) -> SerialFrame:
        if len(data) < 6:
            raise ProtocolError("serial frame is too short")
        if data[0] != cls.START:
            raise ProtocolError("invalid serial START byte")
        length = data[1]
        expected = 1 + 1 + length + 2
        if length < cls.HEADER_BODY_BYTES:
            raise ProtocolError("serial LENGTH is smaller than SEQUENCE + TYPE")
        if len(data) != expected:
            raise ProtocolError(f"serial frame length mismatch: expected {expected}, got {len(data)}")
        protected = data[1:-2]
        actual_crc = struct.unpack("!H", data[-2:])[0]
        if crc16_ccitt_false(protected) != actual_crc:
            raise ProtocolError("serial CRC16 mismatch")
        return SerialFrame(sequence=data[2], frame_type=data[3], payload=data[4:-2])


class SerialFrameParser:
    """Incremental parser that can consume arbitrary serial byte chunks."""

    def __init__(self) -> None:
        self._buffer = bytearray()
        self.bad_frames = 0
        self.discarded_bytes = 0

    def feed(self, chunk: bytes) -> list[SerialFrame]:
        self._buffer.extend(chunk)
        frames: list[SerialFrame] = []
        while self._buffer:
            try:
                start = self._buffer.index(SerialCodec.START)
                if start:
                    self.discarded_bytes += start
                    del self._buffer[:start]
            except ValueError:
                self.discarded_bytes += len(self._buffer)
                self._buffer.clear()
                break
            if len(self._buffer) < 2:
                break
            length = self._buffer[1]
            if length < SerialCodec.HEADER_BODY_BYTES:
                self.bad_frames += 1
                del self._buffer[0]
                continue
            frame_size = 1 + 1 + length + 2
            if len(self._buffer) < frame_size:
                break
            candidate = bytes(self._buffer[:frame_size])
            try:
                frame = SerialCodec.decode_exact(candidate)
            except ProtocolError:
                self.bad_frames += 1
                # Drop only the start byte so a subsequent START can resynchronise.
                del self._buffer[0]
                continue
            frames.append(frame)
            del self._buffer[:frame_size]
        return frames


@dataclass(frozen=True)
class RadioFrame:
    sequence: int
    message_type: int
    payload: bytes = b""
    version: int = 1


class RadioCodec:
    MAGIC = 0xC3
    VERSION = 1
    HEADER = struct.Struct("!BBBBH")  # magic, version, type, length, sequence

    @classmethod
    def encode(cls, frame: RadioFrame) -> bytes:
        if not 0 <= frame.sequence <= 65535:
            raise ProtocolError("radio sequence must fit in uint16")
        if not 0 <= int(frame.message_type) <= 255:
            raise ProtocolError("radio message type must fit in uint8")
        if len(frame.payload) > 255:
            raise ProtocolError("radio payload exceeds one-byte length")
        header = cls.HEADER.pack(cls.MAGIC, frame.version, int(frame.message_type), len(frame.payload), frame.sequence)
        protected = header[1:] + frame.payload
        return header + frame.payload + struct.pack("!H", crc16_ccitt_false(protected))

    @classmethod
    def decode_exact(cls, data: bytes) -> RadioFrame:
        if len(data) < cls.HEADER.size + 2:
            raise ProtocolError("radio frame too short")
        magic, version, message_type, length, sequence = cls.HEADER.unpack(data[:cls.HEADER.size])
        expected = cls.HEADER.size + length + 2
        if magic != cls.MAGIC:
            raise ProtocolError("invalid radio MAGIC byte")
        if version != cls.VERSION:
            raise ProtocolError(f"unsupported radio frame version: {version}")
        if len(data) != expected:
            raise ProtocolError(f"radio frame length mismatch: expected {expected}, got {len(data)}")
        payload_end = cls.HEADER.size + length
        protected = data[1:payload_end]
        expected_crc = struct.unpack("!H", data[payload_end:])[0]
        if crc16_ccitt_false(protected) != expected_crc:
            raise ProtocolError("radio CRC16 mismatch")
        return RadioFrame(sequence=sequence, message_type=message_type, payload=data[cls.HEADER.size:payload_end], version=version)
