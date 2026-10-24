"""Typed binary payloads for radio messages. All payloads use network byte order."""
from __future__ import annotations

from dataclasses import dataclass
import struct
from enum import IntEnum

from .protocol import ProtocolError


class RadioType(IntEnum):
    SENSOR_READING = 0x01
    PLANT_REPORT = 0x02
    HISTORY_SUMMARY = 0x03
    ACK = 0x7F


class CueCode(IntEnum):
    HEALTHY_LOOKING = 0
    SPOT_LIKE_FIXTURE = 1
    WILT_LIKE_FIXTURE = 2
    YELLOWING_LIKE_FIXTURE = 3
    UNKNOWN = 4


class AssessmentCode(IntEnum):
    HEALTHY = 0
    WATER_STRESS_RISK = 1
    HEAT_NUTRIENT_RISK = 2
    SUSPECTED_SYMPTOMS = 3
    UNCERTAIN = 4
    SYSTEM_FAULT = 5


ASSESSMENT_NAMES = {
    AssessmentCode.HEALTHY: "healthy",
    AssessmentCode.WATER_STRESS_RISK: "water_stress_risk",
    AssessmentCode.HEAT_NUTRIENT_RISK: "heat_or_nutrient_risk",
    AssessmentCode.SUSPECTED_SYMPTOMS: "suspected_symptoms",
    AssessmentCode.UNCERTAIN: "uncertain",
    AssessmentCode.SYSTEM_FAULT: "system_fault",
}
CUE_NAMES = {
    CueCode.HEALTHY_LOOKING: "healthy_looking_fixture",
    CueCode.SPOT_LIKE_FIXTURE: "spot_like_fixture",
    CueCode.WILT_LIKE_FIXTURE: "wilt_like_fixture",
    CueCode.YELLOWING_LIKE_FIXTURE: "yellowing_like_fixture",
    CueCode.UNKNOWN: "unknown",
}


@dataclass(frozen=True)
class SensorReading:
    node_id: int
    sequence: int
    timestamp_epoch: int
    soil_moisture_pct: float
    soil_temperature_c: float
    air_temperature_c: float
    relative_humidity_pct: float
    nitrogen_mg_kg: int
    phosphorus_mg_kg: int
    potassium_mg_kg: int
    ph: float
    ec_us_cm: int
    battery_pct: int
    quality_flags: int = 0

    # B node, H seq, I epoch, h soil C*100, h air C*100,
    # H moisture*100, H RH*100, H N, H P, H K, H pH*100, H EC, B battery, B quality.
    _STRUCT = struct.Struct("!BHIhhHHHHHHHBB")

    def to_payload(self) -> bytes:
        values = (
            self.node_id, self.sequence, self.timestamp_epoch,
            round(self.soil_temperature_c * 100), round(self.air_temperature_c * 100),
            round(self.soil_moisture_pct * 100), round(self.relative_humidity_pct * 100),
            self.nitrogen_mg_kg, self.phosphorus_mg_kg, self.potassium_mg_kg,
            round(self.ph * 100), self.ec_us_cm, self.battery_pct, self.quality_flags,
        )
        try:
            return self._STRUCT.pack(*values)
        except struct.error as exc:
            raise ProtocolError(f"sensor reading out of binary range: {exc}") from exc

    @classmethod
    def from_payload(cls, payload: bytes) -> "SensorReading":
        if len(payload) != cls._STRUCT.size:
            raise ProtocolError(f"sensor payload must be {cls._STRUCT.size} bytes")
        (node, seq, epoch, soil_t, air_t, moisture, humidity, n, p, k, ph, ec, battery, flags) = cls._STRUCT.unpack(payload)
        return cls(node, seq, epoch, moisture / 100, soil_t / 100, air_t / 100, humidity / 100, n, p, k, ph / 100, ec, battery, flags)


@dataclass(frozen=True)
class PlantReport:
    row_index: int
    plant_index: int
    timestamp_epoch: int
    cue_code: CueCode
    assessment_code: AssessmentCode
    confidence_pct: int
    quality_passed: bool
    risk_flags: int
    soil_moisture_pct: int
    air_temperature_c: int
    relative_humidity_pct: int
    rover_battery_pct: int
    image_quality_score: int
    provenance_code: int = 1  # 1 = scenario fixture substitute, not a trained ML model

    _STRUCT = struct.Struct("!BHIBBBBHBbBBBB")

    def to_payload(self) -> bytes:
        values = (
            self.row_index, self.plant_index, self.timestamp_epoch,
            int(self.cue_code), int(self.assessment_code), self.confidence_pct,
            int(self.quality_passed), self.risk_flags, self.soil_moisture_pct,
            self.air_temperature_c, self.relative_humidity_pct, self.rover_battery_pct,
            self.image_quality_score, self.provenance_code,
        )
        try:
            return self._STRUCT.pack(*values)
        except struct.error as exc:
            raise ProtocolError(f"plant report out of binary range: {exc}") from exc

    @classmethod
    def from_payload(cls, payload: bytes) -> "PlantReport":
        if len(payload) != cls._STRUCT.size:
            raise ProtocolError(f"plant report payload must be {cls._STRUCT.size} bytes")
        row, plant, epoch, cue, assessment, conf, quality, flags, moisture, air, rh, battery, qscore, provenance = cls._STRUCT.unpack(payload)
        try:
            cue_code = CueCode(cue)
            assessment_code = AssessmentCode(assessment)
        except ValueError as exc:
            raise ProtocolError(f"unknown plant report code: {exc}") from exc
        return cls(row, plant, epoch, cue_code, assessment_code, conf, bool(quality), flags, moisture, air, rh, battery, qscore, provenance)


@dataclass(frozen=True)
class HistorySummary:
    node_id: int
    timestamp_epoch: int
    sample_count: int
    soil_moisture_min_pct: int
    soil_moisture_max_pct: int
    soil_moisture_trend_pct_per_hour: int
    air_temp_min_c: int
    air_temp_max_c: int
    humidity_max_pct: int
    stale: bool

    _STRUCT = struct.Struct("!BIHBBbbbBB")

    def to_payload(self) -> bytes:
        values = (self.node_id, self.timestamp_epoch, self.sample_count,
                  self.soil_moisture_min_pct, self.soil_moisture_max_pct,
                  self.soil_moisture_trend_pct_per_hour, self.air_temp_min_c,
                  self.air_temp_max_c, self.humidity_max_pct, int(self.stale))
        try:
            return self._STRUCT.pack(*values)
        except struct.error as exc:
            raise ProtocolError(f"history summary out of binary range: {exc}") from exc

    @classmethod
    def from_payload(cls, payload: bytes) -> "HistorySummary":
        if len(payload) != cls._STRUCT.size:
            raise ProtocolError(f"history summary payload must be {cls._STRUCT.size} bytes")
        node, epoch, count, low, high, trend, air_low, air_high, rh_max, stale = cls._STRUCT.unpack(payload)
        return cls(node, epoch, count, low, high, trend, air_low, air_high, rh_max, bool(stale))
