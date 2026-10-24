"""Live-only data hub. It never creates simulated readings or sample records.

The dashboard starts empty and changes only when an authenticated device bridge
forwards measurements/telemetry received from real hardware.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hmac
import math
import threading
import time
from typing import Any


DEVICE_TYPES = {"rover", "esp32_controller", "base_station", "sensor_node", "camera", "rtc", "gsm"}
LINK_IDS = {"serial", "lora", "esp_now", "camera", "rtc", "gsm"}
ASSESSMENTS = {"healthy", "disease_like", "water_stress", "heat_or_nutrient_stress", "uncertain"}

# A node can report every ten minutes according to the system plan. These are
# visibility/freshness limits, not device-emulation timers.
DEVICE_TTL_SECONDS = {
    "rover": 12,
    "esp32_controller": 12,
    "base_station": 30,
    "sensor_node": 15 * 60,
    "camera": 30,
    "rtc": 60,
    "gsm": 5 * 60,
}
LINK_TTL_SECONDS = {"serial": 12, "lora": 120, "esp_now": 15 * 60, "camera": 30, "rtc": 60, "gsm": 5 * 60}


def utc_iso(epoch: float | None = None) -> str:
    dt = datetime.fromtimestamp(epoch if epoch is not None else time.time(), tz=timezone.utc)
    return dt.isoformat(timespec="seconds").replace("+00:00", "Z")


def parse_timestamp(value: Any, field: str = "timestamp") -> float:
    if isinstance(value, bool):
        raise ValueError(f"{field} must be an ISO-8601 timestamp or Unix epoch")
    if isinstance(value, (int, float)):
        stamp = float(value)
    elif isinstance(value, str) and value.strip():
        try:
            normalized = value.strip().replace("Z", "+00:00")
            dt = datetime.fromisoformat(normalized)
            if dt.tzinfo is None:
                raise ValueError
            stamp = dt.timestamp()
        except (ValueError, OverflowError) as exc:
            raise ValueError(f"{field} must include a timezone, e.g. 2026-10-22T10:00:00Z") from exc
    else:
        raise ValueError(f"{field} must be an ISO-8601 timestamp or Unix epoch")
    if not math.isfinite(stamp):
        raise ValueError(f"{field} must be finite")
    # Do not accept future-dated values as current telemetry.
    if stamp > time.time() + 60:
        raise ValueError(f"{field} is more than 60 seconds in the future")
    if stamp < 0:
        raise ValueError(f"{field} must be after the Unix epoch")
    return stamp


def _number(body: dict[str, Any], key: str, *, required: bool = False,
            minimum: float | None = None, maximum: float | None = None) -> float | None:
    value = body.get(key)
    if value is None:
        if required:
            raise ValueError(f"{key} is required for a live reading")
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise ValueError(f"{key} must be a finite number")
    value = float(value)
    if minimum is not None and value < minimum:
        raise ValueError(f"{key} must be >= {minimum}")
    if maximum is not None and value > maximum:
        raise ValueError(f"{key} must be <= {maximum}")
    return value


def _required_string(body: dict[str, Any], key: str, maximum: int = 128) -> str:
    value = body.get(key)
    if not isinstance(value, str) or not value.strip() or len(value.strip()) > maximum:
        raise ValueError(f"{key} must be a non-empty string (max {maximum} characters)")
    return value.strip()


class LiveDataHub:
    """Thread-safe, transient cache of genuine incoming device messages only."""

    def __init__(self, ingest_token: str | None = None) -> None:
        self._lock = threading.RLock()
        self._token = ingest_token.strip() if ingest_token else ""
        self.devices: dict[str, dict[str, Any]] = {}
        self.links: dict[str, dict[str, Any]] = {}
        self.sensor_readings: dict[str, dict[str, Any]] = {}
        self.rover_telemetry: dict[str, Any] | None = None
        self.inspections: list[dict[str, Any]] = []
        self.events: list[dict[str, Any]] = []
        self.manual_inspections: list[dict[str, Any]] = []

    def record_manual_inspection(self, inspection: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            self.manual_inspections.insert(0, inspection)
            del self.manual_inspections[100:]
        return inspection

    @property
    def ingestion_enabled(self) -> bool:
        return len(self._token) >= 16

    def authenticate(self, token: str | None) -> bool:
        return self.ingestion_enabled and bool(token) and hmac.compare_digest(self._token, token)

    def _event(self, component: str, event_type: str, severity: str = "info", **details: Any) -> None:
        self.events.insert(0, {
            "timestamp": utc_iso(), "component": component, "event_type": event_type,
            "severity": severity, "details": details, "provenance": "live_device_message_or_server_observation",
        })
        del self.events[250:]

    def heartbeat(self, body: dict[str, Any]) -> dict[str, Any]:
        device_id = _required_string(body, "device_id")
        device_type = _required_string(body, "device_type", 40)
        if device_type not in DEVICE_TYPES:
            raise ValueError(f"device_type must be one of: {', '.join(sorted(DEVICE_TYPES))}")
        transport = body.get("transport", "unknown")
        if not isinstance(transport, str) or len(transport) > 40:
            raise ValueError("transport must be a string of at most 40 characters")
        now = time.time()
        firmware = body.get("firmware_version")
        if firmware is not None and (not isinstance(firmware, str) or len(firmware) > 80):
            raise ValueError("firmware_version must be a string of at most 80 characters")
        with self._lock:
            previous = self.devices.get(device_id)
            was_fresh = bool(previous and now - previous["received_epoch"] <= DEVICE_TTL_SECONDS.get(previous["device_type"], 30))
            entry = {
                "device_id": device_id, "device_type": device_type, "transport": transport,
                "firmware_version": firmware, "device_timestamp": body.get("timestamp"),
                "received_epoch": now, "last_seen_at": utc_iso(now),
            }
            self.devices[device_id] = entry
            if not was_fresh:
                self._event(device_id, "live_heartbeat_received", device_type=device_type, transport=transport,
                            firmware_version=firmware)
        return {"accepted": True, "device_id": device_id, "received_at": utc_iso(now)}

    def link_status(self, body: dict[str, Any]) -> dict[str, Any]:
        link_id = _required_string(body, "link_id", 40)
        if link_id not in LINK_IDS:
            raise ValueError(f"link_id must be one of: {', '.join(sorted(LINK_IDS))}")
        source_device_id = _required_string(body, "source_device_id")
        connected = body.get("connected")
        if not isinstance(connected, bool):
            raise ValueError("connected must be true or false from a real link-status observation")
        stamp = parse_timestamp(body.get("timestamp"), "timestamp")
        error = body.get("error")
        if error is not None and (not isinstance(error, str) or len(error) > 300):
            raise ValueError("error must be a string of at most 300 characters")
        now = time.time()
        with self._lock:
            self.links[link_id] = {
                "link_id": link_id, "source_device_id": source_device_id,
                "connected": connected, "timestamp_epoch": stamp, "timestamp": utc_iso(stamp),
                "received_epoch": now, "received_at": utc_iso(now), "error": error,
            }
            self._event(source_device_id, "link_status_received", "info" if connected else "warning",
                        link_id=link_id, connected=connected, error=error)
        return {"accepted": True, "link_id": link_id, "received_at": utc_iso(now)}

    def sensor_reading(self, body: dict[str, Any]) -> dict[str, Any]:
        node_id = _required_string(body, "node_id")
        stamp = parse_timestamp(body.get("timestamp"), "timestamp")
        seq = body.get("sequence")
        if seq is not None and (isinstance(seq, bool) or not isinstance(seq, int) or seq < 0):
            raise ValueError("sequence must be a non-negative integer when supplied")
        reading = {
            "node_id": node_id, "sequence": seq, "timestamp_epoch": stamp, "timestamp": utc_iso(stamp),
            "received_epoch": time.time(), "received_at": utc_iso(),
            "soil_moisture_pct": _number(body, "soil_moisture_pct", minimum=0, maximum=100),
            "soil_temperature_c": _number(body, "soil_temperature_c", minimum=-40, maximum=100),
            "air_temperature_c": _number(body, "air_temperature_c", required=True, minimum=-60, maximum=100),
            "relative_humidity_pct": _number(body, "relative_humidity_pct", required=True, minimum=0, maximum=100),
            "nitrogen_mg_kg": _number(body, "nitrogen_mg_kg", minimum=0, maximum=100000),
            "phosphorus_mg_kg": _number(body, "phosphorus_mg_kg", minimum=0, maximum=100000),
            "potassium_mg_kg": _number(body, "potassium_mg_kg", minimum=0, maximum=100000),
            "ph": _number(body, "ph", minimum=0, maximum=14),
            "ec_us_cm": _number(body, "ec_us_cm", minimum=0, maximum=100000),
            "battery_pct": _number(body, "battery_pct", minimum=0, maximum=100),
            "quality_flags": body.get("quality_flags", 0),
            "source": "authenticated_live_hardware_bridge",
        }
        if isinstance(reading["quality_flags"], bool) or not isinstance(reading["quality_flags"], int) or reading["quality_flags"] < 0:
            raise ValueError("quality_flags must be a non-negative integer")
        now = time.time()
        with self._lock:
            previous = self.sensor_readings.get(node_id)
            # Ignore an older packet rather than replacing a newer current reading.
            if previous and stamp < previous["timestamp_epoch"]:
                raise ValueError("out-of-order sensor packet rejected; current reading was not replaced")
            self.sensor_readings[node_id] = reading
            self.devices[node_id] = {
                "device_id": node_id, "device_type": "sensor_node", "transport": str(body.get("transport", "esp_now")),
                "firmware_version": body.get("firmware_version"), "device_timestamp": body.get("timestamp"),
                "received_epoch": now, "last_seen_at": utc_iso(now),
            }
            self._event(node_id, "live_sensor_reading_received", severity="warning" if reading["quality_flags"] else "info",
                        sensor_timestamp=reading["timestamp"], quality_flags=reading["quality_flags"])
        return {"accepted": True, "node_id": node_id, "received_at": utc_iso(now)}

    def rover_telemetry_packet(self, body: dict[str, Any]) -> dict[str, Any]:
        device_id = _required_string(body, "device_id")
        stamp = parse_timestamp(body.get("timestamp"), "timestamp")
        state = body.get("state")
        if state is not None and (not isinstance(state, str) or len(state) > 60):
            raise ValueError("state must be a string of at most 60 characters")
        telemetry = {
            "device_id": device_id, "timestamp_epoch": stamp, "timestamp": utc_iso(stamp),
            "received_epoch": time.time(), "received_at": utc_iso(), "state": state,
            "plant_id": body.get("plant_id") if isinstance(body.get("plant_id"), str) else None,
            "position_m": _number(body, "position_m", minimum=0),
            "heading_deg": _number(body, "heading_deg", minimum=0, maximum=360),
            "speed_m_s": _number(body, "speed_m_s", minimum=0, maximum=10),
            "battery_pct": _number(body, "battery_pct", minimum=0, maximum=100),
            "motor_enabled": body.get("motor_enabled") if isinstance(body.get("motor_enabled"), bool) else None,
            "obstacle_detected": body.get("obstacle_detected") if isinstance(body.get("obstacle_detected"), bool) else None,
            "source": "authenticated_live_hardware_bridge",
        }
        now = time.time()
        with self._lock:
            if self.rover_telemetry and stamp < self.rover_telemetry["timestamp_epoch"]:
                raise ValueError("out-of-order rover telemetry rejected; current telemetry was not replaced")
            self.rover_telemetry = telemetry
            self.devices[device_id] = {
                "device_id": device_id, "device_type": "rover", "transport": str(body.get("transport", "serial")),
                "firmware_version": body.get("firmware_version"), "device_timestamp": body.get("timestamp"),
                "received_epoch": now, "last_seen_at": utc_iso(now),
            }
            self._event(device_id, "live_rover_telemetry_received", state=state, plant_id=telemetry["plant_id"])
        return {"accepted": True, "device_id": device_id, "received_at": utc_iso(now)}

    def inspection_report(self, body: dict[str, Any]) -> dict[str, Any]:
        device_id = _required_string(body, "device_id")
        plant_id = _required_string(body, "plant_id", 80)
        stamp = parse_timestamp(body.get("timestamp"), "timestamp")
        assessment = _required_string(body, "assessment", 60)
        if assessment not in ASSESSMENTS:
            raise ValueError(f"assessment must be one of: {', '.join(sorted(ASSESSMENTS))}")
        confidence = _number(body, "confidence_pct", required=True, minimum=0, maximum=100)
        model_version = _required_string(body, "model_version", 100)
        quality = body.get("image_quality_passed")
        if not isinstance(quality, bool):
            raise ValueError("image_quality_passed must be a boolean from the real inspection pipeline")
        report_id = _required_string(body, "report_id", 100)
        now = time.time()
        with self._lock:
            if any(r["report_id"] == report_id for r in self.inspections):
                return {"accepted": True, "duplicate": True, "report_id": report_id}
            report = {
                "report_id": report_id, "device_id": device_id, "plant_id": plant_id,
                "timestamp_epoch": stamp, "timestamp": utc_iso(stamp), "received_epoch": now,
                "received_at": utc_iso(now), "assessment": assessment,
                "confidence_pct": confidence, "model_version": model_version,
                "image_quality_passed": quality, "source": "authenticated_live_hardware_bridge",
            }
            self.inspections.insert(0, report)
            del self.inspections[200:]
            # The authenticated report is also evidence that this source was active now.
            self.devices[device_id] = {
                "device_id": device_id, "device_type": "rover", "transport": str(body.get("transport", "lora")),
                "firmware_version": body.get("firmware_version"), "device_timestamp": body.get("timestamp"),
                "received_epoch": now, "last_seen_at": utc_iso(now),
            }
            self._event(device_id, "live_inspection_report_received", "warning" if assessment not in ("healthy", "uncertain") else "info",
                        plant_id=plant_id, assessment=assessment, model_version=model_version, report_id=report_id)
        return {"accepted": True, "report_id": report_id, "received_at": utc_iso(now)}

    def snapshot(self) -> dict[str, Any]:
        now = time.time()
        with self._lock:
            devices = []
            for entry in sorted(self.devices.values(), key=lambda d: (d["device_type"], d["device_id"])):
                age = max(0.0, now - entry["received_epoch"])
                ttl = DEVICE_TTL_SECONDS.get(entry["device_type"], 30)
                devices.append({
                    "device_id": entry["device_id"], "device_type": entry["device_type"],
                    "transport": entry.get("transport"), "firmware_version": entry.get("firmware_version"),
                    "last_seen_at": entry["last_seen_at"], "seconds_since_last_seen": round(age, 1),
                    "status": "online" if age <= ttl else "stale",
                    "status_detail": "Fresh authenticated heartbeat" if age <= ttl else "Last packet is stale; not considered live",
                })
            latest_sensors = []
            stale_sensor_count = 0
            for reading in self.sensor_readings.values():
                age = max(0.0, now - reading["timestamp_epoch"])
                if age <= DEVICE_TTL_SECONDS["sensor_node"]:
                    latest_sensors.append({k: v for k, v in reading.items() if k not in ("received_epoch",)})
                else:
                    stale_sensor_count += 1
            latest_sensors.sort(key=lambda r: r["node_id"])

            rover = self.rover_telemetry
            if rover:
                rover_age = max(0.0, now - rover["timestamp_epoch"])
                if rover_age > DEVICE_TTL_SECONDS["rover"]:
                    rover = None
                else:
                    rover = {k: v for k, v in rover.items() if k != "received_epoch"}

            links = []
            for link_id in sorted(LINK_IDS):
                entry = self.links.get(link_id)
                if not entry:
                    links.append({"link_id": link_id, "status": "no_observation", "connected": None,
                                  "updated_at": None, "source_device_id": None, "error": None})
                    continue
                age = max(0.0, now - entry["timestamp_epoch"])
                fresh = age <= LINK_TTL_SECONDS[link_id]
                links.append({"link_id": link_id,
                              "status": ("connected" if entry["connected"] else "disconnected") if fresh else "stale",
                              "connected": entry["connected"] if fresh else None,
                              "updated_at": entry["timestamp"], "seconds_since_update": round(age, 1),
                              "source_device_id": entry["source_device_id"], "error": entry["error"]})

            online = sum(d["status"] == "online" for d in devices)
            known_nodes = sum(d["device_type"] == "sensor_node" for d in devices)
            return {
                "mode": "live_only", "generated_at": utc_iso(now),
                "ingestion_enabled": self.ingestion_enabled,
                "status": "LIVE" if online or latest_sensors or rover else "WAITING_FOR_LIVE_DATA",
                "device_summary": {"online": online, "known": len(devices), "sensor_nodes_discovered": known_nodes,
                                   "live_sensor_nodes": len(latest_sensors), "stale_sensor_nodes": stale_sensor_count},
                "devices": devices, "links": links,
                "rover_telemetry": rover, "sensor_readings": latest_sensors,
                "inspections": [self._public_report(x) for x in self.inspections],
                "events": [self._public_event(x) for x in self.events[:100]],
                "data_policy": "Only authenticated live device-bridge messages are accepted. No sample, demo, synthetic or default readings are generated.",
            }

    @staticmethod
    def _public_report(report: dict[str, Any]) -> dict[str, Any]:
        return {k: v for k, v in report.items() if k != "received_epoch"}

    @staticmethod
    def _public_event(event: dict[str, Any]) -> dict[str, Any]:
        return dict(event)
