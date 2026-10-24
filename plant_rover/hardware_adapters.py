"""Pluggable hardware adapters for physical deployment.

These classes implement the HAL interfaces. They use lazy imports so that
running on a PC in simulation mode never requires physical libraries
(such as picamera2 or RPi.GPIO) to be installed.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from .hal import BaseCamera, BaseSerialTransport, BaseRadioTransport, BaseNotifier
from .protocol import SerialCodec, SerialFrame, SerialType


class PhysicalPicamera2Adapter(BaseCamera):
    """Raspberry Pi Camera Module 3 / HQ adapter using Picamera2."""

    def __init__(self, output_dir: str | Path, resolution: tuple[int, int] = (640, 480)) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.resolution = resolution
        self._cam = None

    def _init_camera(self) -> None:
        if self._cam is not None:
            return
        try:
            from picamera2 import Picamera2  # type: ignore
            self._cam = Picamera2()
            config = self._cam.create_still_configuration(main={"size": self.resolution})
            self._cam.configure(config)
            self._cam.start()
        except ImportError as exc:
            raise RuntimeError("Picamera2 library is not installed on this host. Run in simulation mode.") from exc

    def capture(self, session_id: str, plant_id: str, cue: str = "", condition: str = "normal", attempt: int = 1) -> Optional[Path]:
        self._init_camera()
        dest = self.output_dir / session_id / f"{plant_id}-attempt-{attempt}.jpg"
        dest.parent.mkdir(parents=True, exist_ok=True)
        self._cam.capture_file(str(dest))
        return dest

    def close(self) -> None:
        if self._cam is not None:
            self._cam.stop()
            self._cam = None


class PhysicalSerialAdapter(BaseSerialTransport):
    """Physical USB / UART connection to the ESP32 controller using pySerial."""

    def __init__(self, port: str = "/dev/ttyUSB0", baudrate: int = 115200, timeout: float = 0.5) -> None:
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self._serial = None

    def connect(self) -> None:
        if self._serial is not None and self._serial.is_open:
            return
        try:
            import serial  # type: ignore
            self._serial = serial.Serial(self.port, self.baudrate, timeout=self.timeout)
        except Exception as exc:
            raise ConnectionError(f"Failed to open physical serial port {self.port}: {exc}") from exc

    def is_connected(self) -> bool:
        return self._serial is not None and self._serial.is_open

    def send_command(self, frame: SerialFrame, retries: int = 3) -> bool:
        self.connect()
        encoded = SerialCodec.encode(frame)
        for _ in range(retries):
            self._serial.write(encoded)
            self._serial.flush()
            # Read ACK response
            raw = self._serial.read(6)
            if len(raw) == 6:
                try:
                    resp = SerialCodec.decode_exact(raw)
                    if resp.frame_type == SerialType.ACK and resp.sequence == frame.sequence:
                        return True
                except ValueError:
                    continue
        return False

    def read_telemetry(self, timeout_seconds: float = 0.5) -> Optional[SerialFrame]:
        self.connect()
        self._serial.timeout = timeout_seconds
        header = self._serial.read(2)
        if len(header) < 2 or header[0] != SerialCodec.START:
            return None
        length = header[1]
        rest = self._serial.read(length + 2)
        if len(rest) != length + 2:
            return None
        return SerialCodec.decode_exact(header + rest)

    def close(self) -> None:
        if self._serial is not None and self._serial.is_open:
            self._serial.close()
            self._serial = None


class PhysicalModemNotifier(BaseNotifier):
    """Physical GSM SMS gateway using SIM800L module over UART AT commands."""

    def __init__(self, port: str = "/dev/ttyAMA0", baudrate: int = 9600) -> None:
        self.port = port
        self.baudrate = baudrate
        self._ser = None

    def _connect(self) -> None:
        if self._ser is None or not self._ser.is_open:
            import serial  # type: ignore
            self._ser = serial.Serial(self.port, self.baudrate, timeout=2.0)
            self._ser.write(b"AT\r\n")
            self._ser.readline()

    def send_sms(self, recipient: str, text: str) -> bool:
        try:
            self._connect()
            # Set UCS-2 hex mode for Tamil Unicode text
            self._ser.write(b'AT+CMGF=1\r\n')
            self._ser.readline()
            self._ser.write(b'AT+CSCS="HEX"\r\n')
            self._ser.readline()
            self._ser.write(b'AT+CSMP=17,167,0,8\r\n')
            self._ser.readline()
            hex_msg = text.encode("utf-16-be").hex().upper()
            cmd = f'AT+CMGS="{recipient}"\r\n'.encode()
            self._ser.write(cmd)
            self._ser.readline()
            self._ser.write(hex_msg.encode() + b"\x1A")
            resp = self._ser.read_until(b"OK")
            return b"OK" in resp
        except Exception:
            return False

    def play_voice(self, phrase_key: str, language: str = "ta") -> None:
        pass  # Physical I2S DAC playback

    def trigger_buzzer(self, duration_ms: int, frequency_hz: int = 2000) -> None:
        pass  # Physical GPIO buzzer
