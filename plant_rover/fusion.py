"""Conservative rule-based demo fusion. Thresholds are placeholders, not validated advice."""
from __future__ import annotations

from dataclasses import dataclass, asdict
from statistics import mean
from typing import Iterable

from .models import AssessmentCode, CueCode, SensorReading

RISK_LOW_MOISTURE = 1
RISK_HEAT = 2
RISK_LOW_NITROGEN = 4
RISK_HIGH_HUMIDITY = 8
RISK_MISSING_CONTEXT = 16
RISK_IMAGE_INVALID = 32
RISK_CONFLICT = 64


@dataclass(frozen=True)
class FusionDecision:
    assessment_code: AssessmentCode
    demo_confidence_pct: int
    risk_flags: int
    evidence: dict
    recommended_action: str
    provenance: str = "DEMO_RULES_NOT_VALIDATED"

    def as_dict(self) -> dict:
        result = asdict(self)
        result["assessment_code"] = int(self.assessment_code)
        return result


class DemoFusionEngine:
    """Executable pipeline substitute; not an ML classifier or agronomic diagnostic."""

    # Deliberately configurable prototype thresholds; they must be tuned/validated with data.
    LOW_MOISTURE_PCT = 27.0
    HIGH_AIR_TEMP_C = 35.0
    LOW_NITROGEN_MG_KG = 25
    HIGH_HUMIDITY_PCT = 85.0
    MAX_READING_AGE_SECONDS = 1800

    def assess(self, cue: CueCode, image_accepted: bool, readings: Iterable[SensorReading], now_epoch: int) -> FusionDecision:
        valid = [r for r in readings if r.quality_flags == 0 and 0 <= now_epoch - r.timestamp_epoch <= self.MAX_READING_AGE_SECONDS]
        if not image_accepted:
            return FusionDecision(AssessmentCode.UNCERTAIN, 0, RISK_IMAGE_INVALID,
                {"image_quality": "rejected", "valid_sensor_nodes": len(valid), "note": "No classification attempted after a failed quality gate."},
                "Retake the image; inspect the plant manually if repeated captures fail.")
        if not valid:
            return FusionDecision(AssessmentCode.UNCERTAIN, 0, RISK_MISSING_CONTEXT,
                {"image_quality": "accepted", "valid_sensor_nodes": 0, "note": "No fresh, valid sensor context."},
                "Check the field nodes and inspect the plant manually; do not infer healthy from missing data.")

        moisture = mean(r.soil_moisture_pct for r in valid)
        air_temp = max(r.air_temperature_c for r in valid)
        humidity = mean(r.relative_humidity_pct for r in valid)
        nitrogen = mean(r.nitrogen_mg_kg for r in valid)
        evidence = {
            "image_quality": "accepted", "valid_sensor_nodes": len(valid),
            "soil_moisture_mean_pct": round(moisture, 2), "air_temperature_max_c": round(air_temp, 2),
            "relative_humidity_mean_pct": round(humidity, 2), "nitrogen_mean_mg_kg": round(nitrogen, 2),
            "fixture_cue": cue.name.lower(), "fresh_sensor_context": True,
        }
        flags = 0
        if moisture < self.LOW_MOISTURE_PCT:
            flags |= RISK_LOW_MOISTURE
        if air_temp >= self.HIGH_AIR_TEMP_C:
            flags |= RISK_HEAT
        if nitrogen < self.LOW_NITROGEN_MG_KG:
            flags |= RISK_LOW_NITROGEN
        if humidity >= self.HIGH_HUMIDITY_PCT:
            flags |= RISK_HIGH_HUMIDITY

        if flags & (RISK_LOW_MOISTURE | RISK_HEAT | RISK_LOW_NITROGEN):
            if flags & RISK_LOW_MOISTURE:
                code, confidence = AssessmentCode.WATER_STRESS_RISK, 76
                action = "Check irrigation and confirm moisture at the plant root zone. Threshold is provisional."
            else:
                code, confidence = AssessmentCode.HEAT_NUTRIENT_RISK, 72
                action = "Check temperature and calibrated soil/nutrient measurements before changing inputs."
            if cue == CueCode.SPOT_LIKE_FIXTURE:
                flags |= RISK_CONFLICT
                evidence["interpretation"] = "Environmental risk coexists with spot-like fixture; symptoms are not confirmed disease."
            return FusionDecision(code, confidence, flags, evidence, action)

        if cue == CueCode.SPOT_LIKE_FIXTURE:
            if humidity >= self.HIGH_HUMIDITY_PCT:
                return FusionDecision(AssessmentCode.SUSPECTED_SYMPTOMS, 68, flags | RISK_HIGH_HUMIDITY, evidence,
                    "Treat as suspected symptoms only; inspect the plant and seek confirmation before treatment.")
            return FusionDecision(AssessmentCode.UNCERTAIN, 0, flags | RISK_CONFLICT, evidence,
                "Evidence is conflicting: inspect the leaf closely and collect more observations before deciding.")
        if flags & RISK_HIGH_HUMIDITY:
            return FusionDecision(AssessmentCode.UNCERTAIN, 0, flags, evidence,
                "Humidity is elevated but no symptom evidence is present in this fixture. Check sensor placement, airflow and the plant manually.")
        if cue in (CueCode.WILT_LIKE_FIXTURE, CueCode.YELLOWING_LIKE_FIXTURE, CueCode.UNKNOWN):
            return FusionDecision(AssessmentCode.UNCERTAIN, 0, flags | RISK_CONFLICT, evidence,
                "The visual fixture is ambiguous and sensor thresholds do not explain it; inspect manually.")
        return FusionDecision(AssessmentCode.HEALTHY, 82, flags, evidence,
            "No issue was triggered by this demo rule set. Continue routine monitoring; this is not proof of plant health.")
