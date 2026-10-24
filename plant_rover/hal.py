"""Hardware Abstraction Layer (HAL) for the Plant-Health Inspection Rover.

Defines standard abstract interfaces and states so domain logic, navigation,
and sensor processing remain decoupled from whether the rover runs on
pure PC software simulation or physical Raspberry Pi & ESP32 hardware.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path
from typing import Any, Optional

from .models import SensorReading
from .protocol import SerialFrame


class NavigationState(Enum):
    IDLE = auto()
    READY = auto()
    NAVIGATING = auto()
    APPROACHING_PLANT = auto()
    STOPPED_FOR_INSPECTION = auto()
    CAPTURING = auto()
    ANALYZING = auto()
    RECORDING = auto()
    RESUMING = auto()
    OBSTACLE_HALT = auto()
    LOW_BATTERY_RETURN = auto()
    SAFE_STOP = auto()
    FAULT = auto()
    EMERGENCY_STOP = auto()


class NavigationStateMachine:
    """Explicit state machine governing rover movement, inspection cycle, and safety."""

    VALID_TRANSITIONS: dict[NavigationState, set[NavigationState]] = {
        NavigationState.IDLE: {NavigationState.READY, NavigationState.FAULT, NavigationState.EMERGENCY_STOP},
        NavigationState.READY: {NavigationState.NAVIGATING, NavigationState.IDLE, NavigationState.SAFE_STOP, NavigationState.EMERGENCY_STOP},
        NavigationState.NAVIGATING: {
            NavigationState.APPROACHING_PLANT, NavigationState.OBSTACLE_HALT, NavigationState.LOW_BATTERY_RETURN,
            NavigationState.SAFE_STOP, NavigationState.FAULT, NavigationState.EMERGENCY_STOP
        },
        NavigationState.APPROACHING_PLANT: {
            NavigationState.STOPPED_FOR_INSPECTION, NavigationState.NAVIGATING, NavigationState.SAFE_STOP, NavigationState.EMERGENCY_STOP
        },
        NavigationState.STOPPED_FOR_INSPECTION: {
            NavigationState.CAPTURING, NavigationState.SAFE_STOP, NavigationState.FAULT, NavigationState.EMERGENCY_STOP
        },
        NavigationState.CAPTURING: {
            NavigationState.ANALYZING, NavigationState.STOPPED_FOR_INSPECTION, NavigationState.SAFE_STOP, NavigationState.EMERGENCY_STOP
        },
        NavigationState.ANALYZING: {
            NavigationState.RECORDING, NavigationState.SAFE_STOP, NavigationState.FAULT, NavigationState.EMERGENCY_STOP
        },
        NavigationState.RECORDING: {
            NavigationState.RESUMING, NavigationState.SAFE_STOP, NavigationState.FAULT, NavigationState.EMERGENCY_STOP
        },
        NavigationState.RESUMING: {
            NavigationState.NAVIGATING, NavigationState.READY, NavigationState.SAFE_STOP, NavigationState.EMERGENCY_STOP
        },
        NavigationState.OBSTACLE_HALT: {NavigationState.NAVIGATING, NavigationState.SAFE_STOP, NavigationState.EMERGENCY_STOP},
        NavigationState.LOW_BATTERY_RETURN: {NavigationState.SAFE_STOP, NavigationState.IDLE, NavigationState.EMERGENCY_STOP},
        NavigationState.SAFE_STOP: {NavigationState.READY, NavigationState.IDLE, NavigationState.EMERGENCY_STOP},
        NavigationState.FAULT: {NavigationState.SAFE_STOP, NavigationState.IDLE, NavigationState.EMERGENCY_STOP},
        NavigationState.EMERGENCY_STOP: {NavigationState.IDLE, NavigationState.READY},
    }

    def __init__(self, initial_state: NavigationState = NavigationState.IDLE) -> None:
        self._state = initial_state
        self._history: list[tuple[NavigationState, str]] = [(initial_state, "initialization")]

    @property
    def current_state(self) -> NavigationState:
        return self._state

    def can_transition_to(self, target: NavigationState) -> bool:
        return target in self.VALID_TRANSITIONS.get(self._state, set())

    def transition(self, target: NavigationState, reason: str = "") -> bool:
        if not self.can_transition_to(target):
            return False
        self._state = target
        self._history.append((target, reason))
        return True

    def force_emergency_stop(self, reason: str = "hardware_e_stop") -> None:
        self._state = NavigationState.EMERGENCY_STOP
        self._history.append((NavigationState.EMERGENCY_STOP, reason))

    def get_history(self) -> list[tuple[str, str]]:
        return [(state.name, reason) for state, reason in self._history]


class BaseCamera(ABC):
    """Interface for camera capture (synthetic, recorded directory, or physical Picamera2)."""

    @abstractmethod
    def capture(self, session_id: str, plant_id: str, cue: str, condition: str = "normal", attempt: int = 1) -> Optional[Path]:
        """Capture an image. Returns Path to the stored frame or None on capture failure."""
        raise NotImplementedError


class BaseSerialTransport(ABC):
    """Interface for serial link to real-time MCU."""

    @abstractmethod
    def send_command(self, frame: SerialFrame, retries: int = 3) -> bool:
        raise NotImplementedError

    @abstractmethod
    def read_telemetry(self, timeout_seconds: float = 0.5) -> Optional[SerialFrame]:
        raise NotImplementedError

    @abstractmethod
    def is_connected() -> bool:
        raise NotImplementedError


class BaseRadioTransport(ABC):
    """Interface for wireless transmission (LoRa / ESP-NOW)."""

    @abstractmethod
    def send(self, data: bytes, destination: str = "") -> bool:
        raise NotImplementedError

    @abstractmethod
    def poll_received() -> Optional[bytes]:
        raise NotImplementedError

    @abstractmethod
    def is_online() -> bool:
        raise NotImplementedError


class BaseMotorController(ABC):
    """Interface for low-level rover drivetrain."""

    @abstractmethod
    def set_drive(self, linear_mps: float, angular_radps: float) -> bool:
        raise NotImplementedError

    @abstractmethod
    def stop(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def get_encoder_counts() -> tuple[int, int]:
        raise NotImplementedError

    @abstractmethod
    def is_watchdog_active() -> bool:
        raise NotImplementedError


class BaseSensorNode(ABC):
    """Interface for field micro-climate & soil nodes."""

    @abstractmethod
    def sample(self, epoch: int, scenario: str = "healthy") -> SensorReading:
        raise NotImplementedError

    @abstractmethod
    def get_battery_pct() -> int:
        raise NotImplementedError

    @abstractmethod
    def is_online() -> bool:
        raise NotImplementedError


class BaseNotifier(ABC):
    """Interface for farmer-facing alerts (SMS, audio voice prompts, buzzer)."""

    @abstractmethod
    def send_sms(self, recipient: str, text: str) -> bool:
        raise NotImplementedError

    @abstractmethod
    def play_voice(self, phrase_key: str, language: str = "ta") -> None:
        raise NotImplementedError

    @abstractmethod
    def trigger_buzzer(self, duration_ms: int, frequency_hz: int = 2000) -> None:
        raise NotImplementedError


class BaseClock(ABC):
    """Interface for deterministic simulation and hardware RTC."""

    @abstractmethod
    def now_epoch(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def now_iso(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def advance_seconds(self, seconds: int) -> None:
        raise NotImplementedError
