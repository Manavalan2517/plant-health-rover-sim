"""Deterministic virtual field sensors and 24-hour summary features."""
from __future__ import annotations

from dataclasses import dataclass
import math
import random
from statistics import mean
from typing import Iterable

from .models import HistorySummary, SensorReading


@dataclass
class VirtualSensorNode:
    node_id: int
    zone: str
    seed: int = 2026
    battery_pct: int = 96
    online: bool = True
    sequence: int = 0

    def sample(self, timestamp_epoch: int, minutes_since_run_start: int, scenario: str) -> SensorReading:
        rng = random.Random(self.seed + self.node_id * 100_003 + minutes_since_run_start * 17)
        hours = minutes_since_run_start / 60.0
        moisture = 47.0 + 2.8 * math.sin(hours / 3.4 + self.node_id)
        air_temp = 26.0 + 2.2 * math.sin(hours / 5.0 + self.node_id * 0.2)
        soil_temp = air_temp - 1.8
        humidity = 62.0 + 6.0 * math.sin(hours / 4.4 + self.node_id * 0.3)
        nitrogen, phosphorus, potassium = 92, 36, 144

        if scenario == "drought":
            moisture = 54.0 - 1.22 * hours + 1.3 * math.sin(hours / 2.8 + self.node_id)
        if scenario == "heat_stress":
            air_temp, soil_temp = 38.0 + rng.uniform(-0.5, 0.5), 32.5 + rng.uniform(-0.5, 0.5)
            humidity = 35 + rng.uniform(-2, 2)
        elif scenario == "high_humidity":
            humidity = 90.0 + rng.uniform(-2, 2)
        elif scenario == "nutrient_stress":
            nitrogen, phosphorus, potassium = 18, 21, 34
        elif scenario == "conflicting_evidence":
            moisture, air_temp, humidity = 49.0 + rng.uniform(-1, 1), 26.0, 59.0
        elif scenario == "low_battery":
            self.battery_pct = max(4, 16 - int(hours / 2))

        moisture = max(0.0, min(100.0, moisture + rng.uniform(-0.8, 0.8)))
        air_temp += rng.uniform(-0.35, 0.35)
        soil_temp += rng.uniform(-0.3, 0.3)
        humidity = max(0.0, min(100.0, humidity + rng.uniform(-1.0, 1.0)))
        self.sequence = (self.sequence + 1) & 0xFFFF
        if scenario != "low_battery":
            self.battery_pct = max(20, self.battery_pct - (1 if minutes_since_run_start % 360 == 0 else 0))
        return SensorReading(
            node_id=self.node_id, sequence=self.sequence, timestamp_epoch=timestamp_epoch,
            soil_moisture_pct=round(moisture, 2), soil_temperature_c=round(soil_temp, 2),
            air_temperature_c=round(air_temp, 2), relative_humidity_pct=round(humidity, 2),
            nitrogen_mg_kg=max(0, nitrogen + rng.randint(-3, 3)),
            phosphorus_mg_kg=max(0, phosphorus + rng.randint(-2, 2)),
            potassium_mg_kg=max(0, potassium + rng.randint(-4, 4)),
            ph=round(6.2 + rng.uniform(-0.15, 0.15), 2),
            ec_us_cm=max(0, int(880 + rng.randint(-40, 40))), battery_pct=self.battery_pct,
            quality_flags=0,
        )


def build_history_summary(readings: Iterable[SensorReading], node_id: int, now_epoch: int) -> HistorySummary:
    rows = sorted((r for r in readings if r.node_id == node_id and 0 <= now_epoch-r.timestamp_epoch <= 86400), key=lambda r: r.timestamp_epoch)
    if not rows:
        return HistorySummary(node_id, now_epoch, 0, 0, 0, 0, 0, 0, 0, True)
    moisture = [r.soil_moisture_pct for r in rows]
    temps = [r.air_temperature_c for r in rows]
    humidities = [r.relative_humidity_pct for r in rows]
    hours = max((rows[-1].timestamp_epoch - rows[0].timestamp_epoch) / 3600.0, 1 / 60)
    trend = round((moisture[-1] - moisture[0]) / hours)
    stale = now_epoch - rows[-1].timestamp_epoch > 1800
    return HistorySummary(node_id=node_id, timestamp_epoch=rows[-1].timestamp_epoch, sample_count=len(rows),
        soil_moisture_min_pct=round(min(moisture)), soil_moisture_max_pct=round(max(moisture)),
        soil_moisture_trend_pct_per_hour=max(-127, min(127, trend)),
        air_temp_min_c=round(min(temps)), air_temp_max_c=round(max(temps)),
        humidity_max_pct=round(max(humidities)), stale=stale)
