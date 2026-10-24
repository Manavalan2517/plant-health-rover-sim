"""Offline-first SQLite persistence for the local simulation/base station."""
from __future__ import annotations

from pathlib import Path
import json
import sqlite3
import threading
from datetime import datetime, timezone
from typing import Any

from .models import ASSESSMENT_NAMES, CUE_NAMES, AssessmentCode, CueCode, PlantReport, SensorReading


def utc_iso(epoch: int | float | None = None) -> str:
    dt = datetime.fromtimestamp(epoch, tz=timezone.utc) if epoch is not None else datetime.now(timezone.utc)
    return dt.isoformat(timespec="seconds")


class Repository:
    def __init__(self, database_path: str | Path) -> None:
        self.database_path = Path(database_path)
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(self.database_path, check_same_thread=False, timeout=10.0)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._init_schema()

    def _init_schema(self) -> None:
        with self._lock, self._conn:
            self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS sensor_readings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                node_id INTEGER NOT NULL,
                sequence INTEGER NOT NULL,
                timestamp_epoch INTEGER NOT NULL,
                received_at TEXT NOT NULL,
                soil_moisture_pct REAL NOT NULL,
                soil_temperature_c REAL NOT NULL,
                air_temperature_c REAL NOT NULL,
                relative_humidity_pct REAL NOT NULL,
                nitrogen_mg_kg INTEGER NOT NULL,
                phosphorus_mg_kg INTEGER NOT NULL,
                potassium_mg_kg INTEGER NOT NULL,
                ph REAL NOT NULL,
                ec_us_cm INTEGER NOT NULL,
                battery_pct INTEGER NOT NULL,
                quality_flags INTEGER NOT NULL,
                synthetic INTEGER NOT NULL DEFAULT 1,
                UNIQUE(session_id, node_id, sequence)
            );
            CREATE INDEX IF NOT EXISTS idx_readings_session_time ON sensor_readings(session_id, timestamp_epoch);
            CREATE TABLE IF NOT EXISTS plant_reports (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                radio_sequence INTEGER NOT NULL,
                row_index INTEGER NOT NULL,
                plant_index INTEGER NOT NULL,
                plant_id TEXT NOT NULL,
                timestamp_epoch INTEGER NOT NULL,
                received_at TEXT NOT NULL,
                cue_code INTEGER NOT NULL,
                cue_name TEXT NOT NULL,
                assessment_code INTEGER NOT NULL,
                assessment TEXT NOT NULL,
                confidence_pct INTEGER NOT NULL,
                quality_passed INTEGER NOT NULL,
                risk_flags INTEGER NOT NULL,
                soil_moisture_pct INTEGER NOT NULL,
                air_temperature_c INTEGER NOT NULL,
                relative_humidity_pct INTEGER NOT NULL,
                rover_battery_pct INTEGER NOT NULL,
                image_quality_score INTEGER NOT NULL,
                provenance_code INTEGER NOT NULL,
                evidence_json TEXT NOT NULL,
                UNIQUE(session_id, radio_sequence),
                UNIQUE(session_id, row_index, plant_index)
            );
            CREATE TABLE IF NOT EXISTS events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                component TEXT NOT NULL,
                event_type TEXT NOT NULL,
                severity TEXT NOT NULL,
                plant_id TEXT,
                node_id INTEGER,
                message_id TEXT,
                outcome TEXT,
                details_json TEXT NOT NULL DEFAULT '{}'
            );
            CREATE INDEX IF NOT EXISTS idx_events_session_id ON events(session_id, id DESC);
            CREATE TABLE IF NOT EXISTS outbox (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                channel TEXT NOT NULL,
                recipient TEXT NOT NULL,
                language TEXT NOT NULL,
                message TEXT NOT NULL,
                status TEXT NOT NULL
            );
            """)

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def add_reading(self, session_id: str, reading: SensorReading, received_at: str | None = None) -> bool:
        with self._lock, self._conn:
            cur = self._conn.execute("""
                INSERT OR IGNORE INTO sensor_readings
                (session_id,node_id,sequence,timestamp_epoch,received_at,soil_moisture_pct,soil_temperature_c,
                 air_temperature_c,relative_humidity_pct,nitrogen_mg_kg,phosphorus_mg_kg,potassium_mg_kg,
                 ph,ec_us_cm,battery_pct,quality_flags,synthetic)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,1)
            """, (session_id, reading.node_id, reading.sequence, reading.timestamp_epoch,
                  received_at or utc_iso(), reading.soil_moisture_pct, reading.soil_temperature_c,
                  reading.air_temperature_c, reading.relative_humidity_pct, reading.nitrogen_mg_kg,
                  reading.phosphorus_mg_kg, reading.potassium_mg_kg, reading.ph, reading.ec_us_cm,
                  reading.battery_pct, reading.quality_flags))
            return cur.rowcount == 1

    def readings(self, session_id: str, limit: int = 1000, node_id: int | None = None) -> list[dict[str, Any]]:
        with self._lock:
            if node_id is None:
                rows = self._conn.execute("SELECT * FROM sensor_readings WHERE session_id=? ORDER BY timestamp_epoch DESC,id DESC LIMIT ?", (session_id, limit)).fetchall()
            else:
                rows = self._conn.execute("SELECT * FROM sensor_readings WHERE session_id=? AND node_id=? ORDER BY timestamp_epoch DESC,id DESC LIMIT ?", (session_id, node_id, limit)).fetchall()
        return [dict(row) for row in rows]

    def reading_objects(self, session_id: str, node_id: int | None = None, limit: int = 2000) -> list[SensorReading]:
        records = self.readings(session_id, limit=limit, node_id=node_id)
        return [SensorReading(
            node_id=int(row["node_id"]), sequence=int(row["sequence"]), timestamp_epoch=int(row["timestamp_epoch"]),
            soil_moisture_pct=float(row["soil_moisture_pct"]), soil_temperature_c=float(row["soil_temperature_c"]),
            air_temperature_c=float(row["air_temperature_c"]), relative_humidity_pct=float(row["relative_humidity_pct"]),
            nitrogen_mg_kg=int(row["nitrogen_mg_kg"]), phosphorus_mg_kg=int(row["phosphorus_mg_kg"]),
            potassium_mg_kg=int(row["potassium_mg_kg"]), ph=float(row["ph"]), ec_us_cm=int(row["ec_us_cm"]),
            battery_pct=int(row["battery_pct"]), quality_flags=int(row["quality_flags"])
        ) for row in reversed(records)]

    def add_report(self, session_id: str, sequence: int, report: PlantReport, evidence: dict[str, Any]) -> bool:
        plant_id = f"R{report.row_index:02d}-P{report.plant_index:03d}"
        with self._lock, self._conn:
            cur = self._conn.execute("""
                INSERT OR IGNORE INTO plant_reports
                (session_id,radio_sequence,row_index,plant_index,plant_id,timestamp_epoch,received_at,
                 cue_code,cue_name,assessment_code,assessment,confidence_pct,quality_passed,risk_flags,
                 soil_moisture_pct,air_temperature_c,relative_humidity_pct,rover_battery_pct,
                 image_quality_score,provenance_code,evidence_json)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """, (session_id, sequence, report.row_index, report.plant_index, plant_id, report.timestamp_epoch,
                  utc_iso(), int(report.cue_code), CUE_NAMES[report.cue_code], int(report.assessment_code),
                  ASSESSMENT_NAMES[report.assessment_code], report.confidence_pct, int(report.quality_passed),
                  report.risk_flags, report.soil_moisture_pct, report.air_temperature_c,
                  report.relative_humidity_pct, report.rover_battery_pct, report.image_quality_score,
                  report.provenance_code, json.dumps(evidence, separators=(",", ":"), sort_keys=True)))
            return cur.rowcount == 1

    def reports(self, session_id: str, limit: int = 100) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute("SELECT * FROM plant_reports WHERE session_id=? ORDER BY id DESC LIMIT ?", (session_id, limit)).fetchall()
        result = []
        for row in rows:
            item = dict(row)
            item["evidence"] = json.loads(item.pop("evidence_json"))
            item["quality_passed"] = bool(item["quality_passed"])
            result.append(item)
        return result

    def report_count(self, session_id: str) -> int:
        with self._lock:
            return int(self._conn.execute("SELECT COUNT(*) FROM plant_reports WHERE session_id=?", (session_id,)).fetchone()[0])

    def add_event(self, session_id: str, component: str, event_type: str, severity: str = "info",
                  plant_id: str | None = None, node_id: int | None = None, message_id: str | None = None,
                  outcome: str | None = None, details: dict[str, Any] | None = None,
                  timestamp: str | None = None) -> None:
        with self._lock, self._conn:
            self._conn.execute("""
                INSERT INTO events(session_id,timestamp,component,event_type,severity,plant_id,node_id,message_id,outcome,details_json)
                VALUES (?,?,?,?,?,?,?,?,?,?)
            """, (session_id, timestamp or utc_iso(), component, event_type, severity, plant_id, node_id,
                  message_id, outcome, json.dumps(details or {}, separators=(",", ":"), sort_keys=True)))

    def events(self, session_id: str, limit: int = 100) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute("SELECT * FROM events WHERE session_id=? ORDER BY id DESC LIMIT ?", (session_id, limit)).fetchall()
        result = []
        for row in rows:
            item = dict(row)
            item["details"] = json.loads(item.pop("details_json"))
            result.append(item)
        return result

    def add_outbox(self, session_id: str, channel: str, recipient: str, message: str,
                   language: str = "en", status: str = "queued_mock") -> None:
        with self._lock, self._conn:
            self._conn.execute("INSERT INTO outbox(session_id,timestamp,channel,recipient,language,message,status) VALUES (?,?,?,?,?,?,?)",
                (session_id, utc_iso(), channel, recipient, language, message, status))

    def outbox(self, session_id: str, limit: int = 50) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute("SELECT * FROM outbox WHERE session_id=? ORDER BY id DESC LIMIT ?", (session_id, limit)).fetchall()
        return [dict(row) for row in rows]
