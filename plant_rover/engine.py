"""Integrated PC software-in-the-loop simulation engine."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import threading
import time
import uuid
from typing import Any

from .fusion import DemoFusionEngine
from .models import (ASSESSMENT_NAMES, CUE_NAMES, AssessmentCode, HistorySummary,
                     PlantReport, RadioType, SensorReading)
from .protocol import RadioCodec, RadioFrame, SerialCodec, SerialFrame, SerialType
from .scenarios import SCENARIOS, cue_for
from .sensors import VirtualSensorNode, build_history_summary
from .storage import Repository, utc_iso
from .transports import (FaultInjectingLink, FaultProfile, SimulatedSerialSession,
                         reliable_radio_send)
from .vision import ImageQualityGate, SyntheticCamera


class SimulationEngine:
    PLANT_COUNT = 12
    ROW_INDEX = 1
    SENSOR_NODES = (1, 2, 3)
    SAMPLE_INTERVAL_SECONDS = 600
    MAX_RETAKES = 2

    def __init__(self, root_dir: str | Path, seed: int = 2026, plant_count: int = 12) -> None:
        self.root_dir = Path(root_dir)
        self.root_dir.mkdir(parents=True, exist_ok=True)
        self.seed = seed
        if not 1 <= int(plant_count) <= 200:
            raise ValueError("plant_count must be between 1 and 200")
        self.plant_count = int(plant_count)
        self.repository = Repository(self.root_dir / "rover.sqlite3")
        self._lock = threading.RLock()
        self._worker: threading.Thread | None = None
        self._stop_worker = threading.Event()
        self.speed = 1.0
        self.language = "en"
        self.reset("mixed_field")

    def reset(self, scenario: str = "mixed_field") -> None:
        if scenario not in SCENARIOS:
            raise ValueError(f"unknown scenario: {scenario}")
        self.pause()
        with self._lock:
            self.session_id = f"SIM-{uuid.uuid4().hex[:8].upper()}"
            self.scenario = scenario
            self.step_index = 0
            self.sim_epoch = int(time.time())
            self.runtime_ms = 0
            self.rover_battery_pct = 96 if scenario != "low_battery" else 16
            self.navigation_state = "IDLE"
            self.run_state = "READY"
            self.current_plant_id: str | None = None
            self.completed_local_inspections = 0
            self.failed_captures = 0
            self.last_quality: dict[str, Any] | None = None
            self.last_fixture_image: str | None = None
            self.rover_history: dict[int, dict[str, Any]] = {}
            self.pending_reports: dict[int, tuple[bytes, PlantReport]] = {}
            self._radio_seq = 0
            self._serial_seq = 0
            self._seen_sensor: set[tuple[int, int]] = set()
            self._seen_report: set[int] = set()
            self._summary_notification_created = False
            self.safety_stop_reason: str | None = None
            self.fusion = DemoFusionEngine()
            self.camera = SyntheticCamera(self.root_dir / "images", seed=self.seed)
            self.quality_gate = ImageQualityGate()
            serial_profile = FaultProfile(max_frame_bytes=260)
            espnow_profile = FaultProfile(max_frame_bytes=250)
            lora_profile = FaultProfile(max_frame_bytes=64)
            if scenario == "radio_loss":
                lora_profile.drop_probability = 0.34
                lora_profile.corrupt_probability = 0.10
                lora_profile.duplicate_probability = 0.07
            self.serial = SimulatedSerialSession(serial_profile, seed=self.seed + 11)
            self.espnow_link = FaultInjectingLink("virtual-esp-now", espnow_profile, seed=self.seed + 22)
            self.lora_link = FaultInjectingLink("virtual-lora", lora_profile, seed=self.seed + 33)
            self.nodes = {node_id: VirtualSensorNode(node_id, f"zone-{node_id}", self.seed + node_id * 101)
                          for node_id in self.SENSOR_NODES}
            self._last_node_dropout_logged = False
            self._seed_24h_history()
            self._event("simulation", "session_reset", details={"scenario": scenario, "seed": self.seed,
                "simulation_mode": True, "synthetic_readings": True})

    def _event(self, component: str, event_type: str, severity: str = "info", **kwargs: Any) -> None:
        self.repository.add_event(self.session_id, component, event_type, severity,
                                  timestamp=utc_iso(self.sim_epoch), **kwargs)

    def _seed_24h_history(self) -> None:
        """Warm the virtual field nodes through the real packet path for repeatable history tests."""
        now = self.sim_epoch
        for i in range(144):
            minutes = i * 10
            epoch = now - (143 - i) * 600
            for node_id, node in self.nodes.items():
                reading = node.sample(epoch, minutes, self.scenario)
                if self._transmit_sensor(reading, log_event=False):
                    pass
        # Base -> rover return traffic: compact 24-hour summaries are also serialized over LoRa.
        for node_id in self.SENSOR_NODES:
            history = self.repository.reading_objects(self.session_id, node_id=node_id, limit=144)
            summary = build_history_summary(history, node_id, now)
            seq = self._next_radio_seq()
            frame = RadioCodec.encode(RadioFrame(seq, RadioType.HISTORY_SUMMARY, summary.to_payload()))
            ok, attempts, reason = reliable_radio_send(self.lora_link, frame, seq,
                self._receive_history_summary, lambda raw, expected=seq: self._validate_radio_ack(raw, expected), retries=3)
            if not ok:
                self._event("lora", "history_summary_delivery_failed", "warning", node_id=node_id,
                            message_id=str(seq), outcome=reason, details={"attempts": attempts})

    def _next_radio_seq(self) -> int:
        self._radio_seq = (self._radio_seq + 1) & 0xFFFF
        return self._radio_seq

    def _transmit_sensor(self, reading: SensorReading, log_event: bool = True) -> bool:
        frame = RadioCodec.encode(RadioFrame(reading.sequence, RadioType.SENSOR_READING, reading.to_payload()))
        ok, attempts, reason = reliable_radio_send(self.espnow_link, frame, reading.sequence,
            self._receive_sensor_packet, lambda raw, expected=reading.sequence: self._validate_radio_ack(raw, expected), retries=3)
        if log_event:
            self._event("esp_now", "sensor_packet_received" if ok else "sensor_packet_delivery_failed",
                "info" if ok else "warning", node_id=reading.node_id, message_id=f"N{reading.node_id}:{reading.sequence}",
                outcome=reason if not ok else "acknowledged", details={"attempts": attempts,
                "synthetic": True, "soil_moisture_pct": reading.soil_moisture_pct})
        return ok

    def _receive_sensor_packet(self, raw: bytes) -> bytes | None:
        try:
            frame = RadioCodec.decode_exact(raw)
            if frame.message_type != RadioType.SENSOR_READING:
                return None
            reading = SensorReading.from_payload(frame.payload)
        except (ValueError, TypeError):
            return None
        key = (reading.node_id, reading.sequence)
        if key not in self._seen_sensor:
            self._seen_sensor.add(key)
            inserted = self.repository.add_reading(self.session_id, reading, received_at=utc_iso(self.sim_epoch))
            if inserted and reading.quality_flags != 0:
                self._event("base_station", "sensor_quality_flagged", "warning", node_id=reading.node_id,
                            message_id=f"N{reading.node_id}:{reading.sequence}", details={"quality_flags": reading.quality_flags})
        return self._radio_ack(frame.sequence)

    def _receive_lora_report(self, raw: bytes) -> bytes | None:
        try:
            frame = RadioCodec.decode_exact(raw)
            if frame.message_type != RadioType.PLANT_REPORT:
                return None
            report = PlantReport.from_payload(frame.payload)
        except (ValueError, TypeError):
            return None
        if frame.sequence not in self._seen_report:
            self._seen_report.add(frame.sequence)
            # This evidence is reconstructed by the base station from fields that actually crossed LoRa.
            evidence = {
                "transmitted_fixture_cue": CUE_NAMES[report.cue_code],
                "image_quality_passed": report.quality_passed,
                "image_quality_score": report.image_quality_score,
                "risk_flags": report.risk_flags,
                "sensor_summary_in_packet": {
                    "soil_moisture_pct": report.soil_moisture_pct,
                    "air_temperature_c": report.air_temperature_c,
                    "relative_humidity_pct": report.relative_humidity_pct,
                },
                "provenance": "compact_serialized_LoRa_report; fixture cue is not an ML prediction",
            }
            self.repository.add_report(self.session_id, frame.sequence, report, evidence)
            if report.assessment_code not in (AssessmentCode.HEALTHY, AssessmentCode.UNCERTAIN):
                self._event("base_station", "plant_risk_report_received", "warning", plant_id=f"R{report.row_index:02d}-P{report.plant_index:03d}",
                    message_id=str(frame.sequence), details={"assessment": ASSESSMENT_NAMES[report.assessment_code], "risk_flags": report.risk_flags})
        # Duplicate retransmissions must still receive an ACK, but are not stored twice.
        return self._radio_ack(frame.sequence)

    def _receive_history_summary(self, raw: bytes) -> bytes | None:
        try:
            frame = RadioCodec.decode_exact(raw)
            if frame.message_type != RadioType.HISTORY_SUMMARY:
                return None
            summary = HistorySummary.from_payload(frame.payload)
        except (ValueError, TypeError):
            return None
        self.rover_history[summary.node_id] = {
            "node_id": summary.node_id, "timestamp_epoch": summary.timestamp_epoch,
            "sample_count": summary.sample_count, "soil_moisture_min_pct": summary.soil_moisture_min_pct,
            "soil_moisture_max_pct": summary.soil_moisture_max_pct,
            "soil_moisture_trend_pct_per_hour": summary.soil_moisture_trend_pct_per_hour,
            "air_temp_min_c": summary.air_temp_min_c, "air_temp_max_c": summary.air_temp_max_c,
            "humidity_max_pct": summary.humidity_max_pct, "stale": summary.stale,
        }
        return self._radio_ack(frame.sequence)

    @staticmethod
    def _radio_ack(sequence: int) -> bytes:
        return RadioCodec.encode(RadioFrame(sequence, RadioType.ACK, sequence.to_bytes(2, "big")))

    @staticmethod
    def _validate_radio_ack(raw: bytes, expected_sequence: int) -> bytes | None:
        try:
            frame = RadioCodec.decode_exact(raw)
        except ValueError:
            return None
        if frame.message_type != RadioType.ACK or frame.sequence != expected_sequence:
            return None
        if frame.payload != expected_sequence.to_bytes(2, "big"):
            return None
        return b"ACK_ACCEPTED"

    def _send_serial_move(self) -> bool:
        self._serial_seq = (self._serial_seq + 1) & 0xFF
        if self.scenario == "serial_loss" and self.step_index >= 2:
            # First move succeeds, then the cable/serial path disappears while in motion.
            self.serial.link.profile.drop_probability = 1.0
        seq = self._serial_seq
        payload = f"MOVE:28".encode("ascii")
        raw = SerialCodec.encode(SerialFrame(seq, SerialType.COMMAND, payload))
        ok, attempts, reason = self.serial.send(raw, seq, self.runtime_ms, retries=3)
        if ok:
            self._event("serial", "movement_command_acked", details={"sequence": seq, "attempts": attempts, "speed_pct": 28})
        else:
            self._event("serial", "movement_command_failed", "warning", message_id=str(seq), outcome=reason,
                        details={"sequence": seq, "attempts": attempts})
        tripped = self.serial.esp32.check_watchdog(self.runtime_ms)
        if tripped:
            self.safety_stop_reason = "serial_heartbeat_timeout"
            self.navigation_state = "SAFETY_STOP"
            self.run_state = "SAFETY_STOP"
            self._event("safety_supervisor", "watchdog_stopped_virtual_motor", "critical",
                        details={"runtime_ms": self.runtime_ms, "timeout_ms": self.serial.esp32.WATCHDOG_TIMEOUT_MS,
                                 "important": "software simulation only; physical stopping unverified"})
        return ok

    def _sample_nodes(self) -> list[SensorReading]:
        readings: list[SensorReading] = []
        # Continue the 24-hour warm-up timeline rather than resetting trends at run start.
        minutes = (143 + self.step_index) * 10
        for node_id, node in self.nodes.items():
            if self.scenario == "node_dropout" and node_id == 2 and self.step_index >= 4:
                if node.online:
                    node.online = False
                    self._event("sensor_node", "node_disconnected", "warning", node_id=node_id,
                                details={"stale_after_seconds": 1800, "other_nodes_continue": True})
                continue
            node.online = True
            reading = node.sample(self.sim_epoch, minutes, self.scenario)
            if self._transmit_sensor(reading, log_event=True):
                readings.append(reading)
        return readings

    def _camera_condition(self) -> str:
        return {
            "blurry_capture": "blur",
            "capture_failure": "capture_failure",
            "no_leaf": "no_leaf",
        }.get(self.scenario, "normal")

    def _build_report(self, plant_number: int) -> tuple[PlantReport, dict[str, Any]]:
        plant_id = f"R{self.ROW_INDEX:02d}-P{plant_number:03d}"
        cue = cue_for(self.scenario, plant_number)
        quality_result = None
        capture_path: Path | None = None
        for attempt in range(1, self.MAX_RETAKES + 2):
            capture_path = self.camera.capture(self.session_id, plant_id, CUE_NAMES[cue], self._camera_condition(), attempt)
            if capture_path is None:
                self._event("camera", "capture_failed", "warning", plant_id=plant_id,
                            details={"attempt": attempt, "scenario": self.scenario})
                continue
            quality_result = self.quality_gate.evaluate(capture_path)
            if quality_result.accepted:
                break
            self._event("image_quality_gate", "image_rejected", "warning", plant_id=plant_id,
                        details={"attempt": attempt, **quality_result.as_dict()})
        accepted = bool(quality_result and quality_result.accepted)
        if not accepted:
            self.failed_captures += 1
        self.last_fixture_image = str(capture_path) if capture_path else None
        self.last_quality = quality_result.as_dict() if quality_result else {"accepted": False, "reason": "capture_failed", "score": 0}

        readings = self.repository.reading_objects(self.session_id, limit=600)
        decision = self.fusion.assess(cue, accepted, readings, self.sim_epoch)
        current = [r for r in readings if 0 <= self.sim_epoch-r.timestamp_epoch <= self.fusion.MAX_READING_AGE_SECONDS and r.quality_flags == 0]
        moisture = round(sum(r.soil_moisture_pct for r in current)/len(current)) if current else 0
        air_temp = round(max((r.air_temperature_c for r in current), default=0))
        humidity = round(sum(r.relative_humidity_pct for r in current)/len(current)) if current else 0
        qscore = int(quality_result.score) if quality_result else 0
        report = PlantReport(self.ROW_INDEX, plant_number, self.sim_epoch, cue, decision.assessment_code,
            decision.demo_confidence_pct, accepted, decision.risk_flags, max(0, min(100, moisture)),
            max(-128, min(127, air_temp)), max(0, min(100, humidity)), self.rover_battery_pct,
            max(0, min(100, qscore)), provenance_code=1)
        local_meta = {"plant_id": plant_id, "capture_path": str(capture_path) if capture_path else None,
            "quality": self.last_quality, "decision": decision.as_dict(), "scenario": self.scenario,
            "is_synthetic_image": True, "is_ml_prediction": False}
        self.completed_local_inspections += 1
        self._event("image_pipeline", "inspection_prepared", "info" if accepted else "warning", plant_id=plant_id,
            details={"assessment": ASSESSMENT_NAMES[decision.assessment_code], "quality_passed": accepted,
                     "provenance": "scenario fixture + demo rules; not ML"})
        return report, local_meta

    def _send_report(self, report: PlantReport) -> bool:
        seq = self._next_radio_seq()
        frame = RadioCodec.encode(RadioFrame(seq, RadioType.PLANT_REPORT, report.to_payload()))
        self.pending_reports[seq] = (frame, report)
        ok, attempts, reason = reliable_radio_send(self.lora_link, frame, seq, self._receive_lora_report,
            lambda raw, expected=seq: self._validate_radio_ack(raw, expected), retries=3)
        if ok:
            self.pending_reports.pop(seq, None)
            self._event("lora", "plant_report_acknowledged", details={"sequence": seq, "plant_index": report.plant_index, "attempts": attempts})
        else:
            self._event("lora", "plant_report_pending", "warning", plant_id=f"R{report.row_index:02d}-P{report.plant_index:03d}",
                        message_id=str(seq), outcome=reason, details={"sequence": seq, "attempts": attempts})
        return ok

    def _retry_pending(self) -> None:
        for seq, (frame, report) in list(self.pending_reports.items()):
            ok, attempts, reason = reliable_radio_send(self.lora_link, frame, seq, self._receive_lora_report,
                lambda raw, expected=seq: self._validate_radio_ack(raw, expected), retries=3)
            if ok:
                self.pending_reports.pop(seq, None)
                self._event("lora", "pending_report_delivered", details={"sequence": seq, "plant_index": report.plant_index, "attempts": attempts})
            else:
                self._event("lora", "pending_report_retry_failed", "warning", message_id=str(seq), outcome=reason,
                            details={"sequence": seq, "attempts": attempts})

    def step(self) -> dict[str, Any]:
        with self._lock:
            if self.run_state in ("COMPLETE", "EMERGENCY_STOP", "SAFETY_STOP", "LOW_BATTERY_STOP"):
                return self.state()
            if self.completed_local_inspections >= self.plant_count:
                self._retry_pending()
                if not self.pending_reports:
                    self._finish_run()
                else:
                    self.run_state = "AWAITING_REPORT_DELIVERY"
                return self.state()

            self.step_index += 1
            self.sim_epoch += self.SAMPLE_INTERVAL_SECONDS
            self.runtime_ms += 150
            self.navigation_state = "SENSING"
            self._sample_nodes()

            if not self._send_serial_move():
                if self.run_state == "SAFETY_STOP":
                    return self.state()
                self.navigation_state = "WAITING_FOR_SERIAL_ACK"
                self.run_state = "RUNNING"
                return self.state()
            if self.serial.esp32.emergency_stop:
                self.run_state = "EMERGENCY_STOP"
                return self.state()

            if self.rover_battery_pct <= 10:
                self.serial.esp32.motor_enabled = False
                self.serial.esp32.speed_pct = 0
                self.run_state = "LOW_BATTERY_STOP"
                self.navigation_state = "SAFETY_STOP"
                self.safety_stop_reason = "low_battery"
                self._event("safety_supervisor", "low_battery_safe_stop", "critical",
                    details={"battery_pct": self.rover_battery_pct, "important": "battery model is simulated"})
                return self.state()

            plant_number = self.completed_local_inspections + 1
            self.current_plant_id = f"R{self.ROW_INDEX:02d}-P{plant_number:03d}"
            self.navigation_state = "STOPPED_FOR_INSPECTION"
            self.run_state = "RUNNING"
            report, meta = self._build_report(plant_number)
            self._last_local_artifact = meta
            self.navigation_state = "TRANSMITTING_REPORT"
            self._send_report(report)
            self.rover_battery_pct = max(0, self.rover_battery_pct - (2 if self.scenario == "low_battery" else 1))
            self.navigation_state = "NAVIGATING"

            if self.completed_local_inspections >= self.plant_count:
                self._retry_pending()
                if not self.pending_reports:
                    self._finish_run()
                else:
                    self.run_state = "AWAITING_REPORT_DELIVERY"
            return self.state()

    def _finish_run(self) -> None:
        self.navigation_state = "IDLE"
        self.run_state = "COMPLETE"
        if not self._summary_notification_created:
            reports = self.repository.reports(self.session_id, limit=self.plant_count)
            at_risk = sum(1 for r in reports if r["assessment"] not in ("healthy", "uncertain"))
            uncertain = sum(1 for r in reports if r["assessment"] == "uncertain")
            if self.language == "ta":
                message = f"தாவர ஆய்வு முடிந்தது. பதிவுகள்: {len(reports)}; கவனம் தேவை: {at_risk}; உறுதியற்றவை: {uncertain}. இது கணினி சிமுலேஷன் அறிக்கை."
            else:
                message = f"Simulated plant inspection complete. Reports: {len(reports)}; attention items: {at_risk}; uncertain: {uncertain}. Synthetic PC-only run; not a diagnosis."
            self.repository.add_outbox(self.session_id, "sms_mock", "TEST_RECIPIENT", message, self.language, "queued_mock")
            self.repository.add_outbox(self.session_id, "farmer_dashboard", "LOCAL_DASHBOARD", message, self.language, "displayed_locally")
            self._summary_notification_created = True
        self._event("simulation", "daily_inspection_complete", details={"local_inspections": self.completed_local_inspections,
            "base_reports": self.repository.report_count(self.session_id), "failed_captures": self.failed_captures,
            "pending_reports": len(self.pending_reports), "mode": "simulation"})
        self._stop_worker.set()

    def start(self) -> None:
        with self._lock:
            if self.run_state in ("COMPLETE", "EMERGENCY_STOP", "SAFETY_STOP", "LOW_BATTERY_STOP"):
                return
            self.run_state = "RUNNING"
            self._stop_worker.clear()
            if self._worker and self._worker.is_alive():
                return
            self._worker = threading.Thread(target=self._worker_loop, name="plant-rover-sim", daemon=True)
            self._worker.start()

    def _worker_loop(self) -> None:
        while not self._stop_worker.is_set():
            with self._lock:
                if self.run_state in ("COMPLETE", "EMERGENCY_STOP", "SAFETY_STOP", "LOW_BATTERY_STOP"):
                    break
                current_speed = max(0.1, min(5.0, self.speed))
            self.step()
            time.sleep(max(0.05, 1.0 / current_speed))

    def pause(self) -> None:
        self._stop_worker.set()
        with self._lock:
            if hasattr(self, "run_state") and self.run_state == "RUNNING":
                self.run_state = "PAUSED"

    def resume(self) -> None:
        self.start()

    def activate_emergency_stop(self) -> dict[str, Any]:
        with self._lock:
            self.pause()
            self._serial_seq = (self._serial_seq + 1) & 0xFF
            frame = SerialCodec.encode(SerialFrame(self._serial_seq, SerialType.STOP, b"ESTOP"))
            ok, attempts, reason = self.serial.send(frame, self._serial_seq, self.runtime_ms, retries=2)
            # The physical e-stop is represented by a separate software latch; no real motor exists.
            self.serial.esp32.motor_enabled = False
            self.serial.esp32.speed_pct = 0
            self.serial.esp32.emergency_stop = True
            self.run_state = "EMERGENCY_STOP"
            self.navigation_state = "SAFETY_STOP"
            self.safety_stop_reason = "operator_estop"
            self._event("safety_supervisor", "operator_emergency_stop", "critical", outcome="latched",
                        details={"serial_stop_acknowledged": ok, "attempts": attempts, "reason": reason,
                                 "physical_estop_verified": False})
            return self.state()

    def clear_emergency_stop(self) -> dict[str, Any]:
        with self._lock:
            self.serial.esp32.clear_emergency_stop()
            self.serial.link.profile.drop_probability = 0
            self.safety_stop_reason = None
            self.navigation_state = "IDLE"
            self.run_state = "READY"
            self._event("safety_supervisor", "operator_cleared_simulated_estop", "warning",
                        details={"note": "simulation latch cleared; physical hardware not connected"})
            return self.state()

    def set_scenario(self, scenario: str) -> dict[str, Any]:
        self.reset(scenario)
        return self.state()

    def set_plant_count(self, plant_count: int) -> dict[str, Any]:
        count = int(plant_count)
        if not 1 <= count <= 200:
            raise ValueError("plant_count must be between 1 and 200")
        with self._lock:
            selected = self.scenario
            self.plant_count = count
        self.reset(selected)
        self._event("simulation", "plant_count_changed", details={"plant_count": count})
        return self.state()

    def set_fault(self, link_name: str, enabled: bool) -> dict[str, Any]:
        links = {"serial": self.serial.link, "espnow": self.espnow_link, "lora": self.lora_link}
        if link_name not in links:
            raise ValueError("link must be serial, espnow, or lora")
        links[link_name].profile.enabled = bool(enabled)
        self._event("fault_injection", "link_enabled" if enabled else "link_disconnected",
                    "warning" if not enabled else "info", details={"link": link_name, "enabled": bool(enabled)})
        return self.state()

    def set_speed(self, speed: float) -> dict[str, Any]:
        if not 0.1 <= float(speed) <= 5.0:
            raise ValueError("speed must be between 0.1x and 5x")
        self.speed = float(speed)
        self._event("simulation", "speed_changed", details={"speed": self.speed})
        return self.state()

    def set_language(self, language: str) -> dict[str, Any]:
        if language not in ("en", "ta"):
            raise ValueError("language must be en or ta")
        self.language = language
        return self.state()

    def state(self) -> dict[str, Any]:
        # Caller may already hold the re-entrant lock.
        with self._lock:
            reports = self.repository.reports(self.session_id, limit=self.plant_count)
            readings = self.repository.readings(self.session_id, limit=30)
            recent_by_node = {}
            for row in readings:
                if row["node_id"] not in recent_by_node:
                    age = max(0, self.sim_epoch - int(row["timestamp_epoch"]))
                    recent_by_node[row["node_id"]] = {
                        "node_id": row["node_id"], "zone": f"zone-{row['node_id']}",
                        "soil_moisture_pct": row["soil_moisture_pct"], "soil_temperature_c": row["soil_temperature_c"],
                        "air_temperature_c": row["air_temperature_c"], "relative_humidity_pct": row["relative_humidity_pct"],
                        "nitrogen_mg_kg": row["nitrogen_mg_kg"], "battery_pct": row["battery_pct"],
                        "last_seen_epoch": row["timestamp_epoch"], "age_seconds": age,
                        "status": "online" if age <= 1800 and self.nodes[row["node_id"]].online else "stale/offline",
                        "synthetic": True,
                    }
            for node_id in self.SENSOR_NODES:
                if node_id not in recent_by_node:
                    recent_by_node[node_id] = {"node_id": node_id, "zone": f"zone-{node_id}", "status": "never_seen", "synthetic": True}
            assessments = {}
            for report in reports:
                assessments[report["assessment"]] = assessments.get(report["assessment"], 0) + 1
            return {
                "session_id": self.session_id,
                "scenario": self.scenario,
                "scenario_title": SCENARIOS[self.scenario]["title"],
                "scenario_description": SCENARIOS[self.scenario]["description"],
                "simulation_mode": True,
                "run_state": self.run_state,
                "navigation_state": self.navigation_state,
                "current_plant_id": self.current_plant_id,
                "plants_total": self.plant_count,
                "plants_visited": self.completed_local_inspections,
                "base_reports_received": self.repository.report_count(self.session_id),
                "failed_captures": self.failed_captures,
                "pending_lora_reports": len(self.pending_reports),
                "rover_battery_pct": self.rover_battery_pct,
                "motor_enabled": self.serial.esp32.motor_enabled,
                "virtual_motor_speed_pct": self.serial.esp32.speed_pct,
                "watchdog_tripped": self.serial.esp32.watchdog_tripped,
                "safety_stop_reason": self.safety_stop_reason,
                "sensor_nodes": sorted(recent_by_node.values(), key=lambda item: item["node_id"]),
                "history_summaries": list(self.rover_history.values()),
                "assessment_counts": assessments,
                "recent_reports": reports[:8],
                "last_quality": self.last_quality,
                "last_fixture_image": self.last_fixture_image,
                "latest_sensor_readings": readings[:9],
                "links": {
                    "serial": {"connected": self.serial.link.profile.enabled, "stats": vars(self.serial.link.stats), "last_reason": self.serial.link.last_reason},
                    "espnow": {"connected": self.espnow_link.profile.enabled, "stats": vars(self.espnow_link.stats), "last_reason": self.espnow_link.last_reason},
                    "lora": {"connected": self.lora_link.profile.enabled, "stats": vars(self.lora_link.stats), "last_reason": self.lora_link.last_reason},
                },
                "demo_disclaimer": "No trained ML model is loaded. Image cues are scenario fixtures; all sensor readings and camera fixtures are synthetic. This is not an agronomic diagnosis or proof of hardware safety.",
            }

    def api_payload(self) -> dict[str, Any]:
        with self._lock:
            return {"state": self.state(), "events": self.repository.events(self.session_id, 18),
                "reports": self.repository.reports(self.session_id, 30),
                "readings": self.repository.readings(self.session_id, 18),
                "outbox": self.repository.outbox(self.session_id, 12),
                "scenarios": SCENARIOS}

    def export_replay_trace(self) -> dict[str, Any]:
        import hashlib
        with self._lock:
            reports = self.repository.reports(self.session_id, limit=self.plant_count)
            events = self.repository.events(self.session_id, limit=200)
            readings = self.repository.readings(self.session_id, limit=200)
            digest_input = f"{self.session_id}:{self.scenario}:{self.seed}:{len(reports)}:{len(events)}:{len(readings)}"
            state_digest = hashlib.sha256(digest_input.encode()).hexdigest()
            return {
                "session_id": self.session_id,
                "scenario": self.scenario,
                "seed": self.seed,
                "plant_count": self.plant_count,
                "completed_inspections": self.completed_local_inspections,
                "base_reports_received": len(reports),
                "total_events": len(events),
                "total_readings": len(readings),
                "state_digest": state_digest,
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "reports_summary": [{"plant_id": r["plant_id"], "assessment": r["assessment"], "confidence_pct": r["confidence_pct"]} for r in reports],
            }

