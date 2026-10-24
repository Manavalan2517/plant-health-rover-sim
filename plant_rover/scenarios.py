"""Named, reproducible demonstration scenarios (not agronomic ground truth)."""
from __future__ import annotations

from .models import CueCode

SCENARIOS = {
    "mixed_field": {"title": "Mixed field", "description": "Healthy-looking, wilt-like, spot-like and yellowing fixtures with normal synthetic sensor variation."},
    "healthy": {"title": "Healthy baseline", "description": "Normal synthetic readings and healthy-looking fixtures."},
    "drought": {"title": "Drying soil", "description": "Synthetic soil-moisture trend declines across the simulated 24-hour history."},
    "disease_like": {"title": "Disease-like symptoms", "description": "Spot-like image fixtures; outcome remains suspected/uncertain, never a confirmed diagnosis."},
    "conflicting_evidence": {"title": "Conflicting evidence", "description": "Spot-like fixtures without supporting environmental context; should remain uncertain."},
    "heat_stress": {"title": "Heat stress", "description": "Synthetic air temperature is elevated; used to test warnings, not agricultural thresholds."},
    "high_humidity": {"title": "High humidity", "description": "Synthetic humidity remains elevated and demonstrates a contextual risk flag."},
    "nutrient_stress": {"title": "Low-N fixture", "description": "Synthetic nitrogen estimate is low. NPK values are not real soil measurements."},
    "blurry_capture": {"title": "Blur / retake", "description": "All captures are intentionally blurred; quality gate should reject them after retries."},
    "node_dropout": {"title": "Field node dropout", "description": "Node 2 disconnects during the run; other nodes continue reporting."},
    "radio_loss": {"title": "LoRa packet loss", "description": "LoRa reports/ACKs face seeded packet loss and corruption; retransmissions are exercised."},
    "serial_loss": {"title": "Pi–ESP32 serial loss", "description": "Serial frames are dropped; the virtual ESP32 watchdog must stop simulated motion."},
    "low_battery": {"title": "Low battery", "description": "Simulated battery begins low and exercises safe-stop policy."},
    "capture_failure": {"title": "Camera capture failure", "description": "Camera fails to provide an image; unusable captures must not be marked healthy."},
    "no_leaf": {"title": "No visible leaf", "description": "Camera returns a clear but uninformative fixture; it should be treated as uncertain."},
}


def cue_for(scenario: str, plant_number: int) -> CueCode:
    if scenario in ("disease_like", "conflicting_evidence"):
        return CueCode.SPOT_LIKE_FIXTURE
    if scenario == "drought":
        return CueCode.WILT_LIKE_FIXTURE
    if scenario in ("heat_stress", "nutrient_stress"):
        return CueCode.YELLOWING_LIKE_FIXTURE
    if scenario == "mixed_field":
        return (CueCode.HEALTHY_LOOKING, CueCode.WILT_LIKE_FIXTURE, CueCode.SPOT_LIKE_FIXTURE,
                CueCode.YELLOWING_LIKE_FIXTURE, CueCode.UNKNOWN)[(plant_number - 1) % 5]
    return CueCode.HEALTHY_LOOKING
