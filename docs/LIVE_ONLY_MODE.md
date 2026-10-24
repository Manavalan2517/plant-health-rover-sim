# Live-only mode and data provenance

## Runtime contract

`python -m plant_rover.cli serve` starts the live-only monitor. The runtime does not create sample devices/readings/reports, run the `SimulationEngine`, load the historical simulation SQLite database, generate synthetic images, or expose simulation controls. With no hardware/bridge connected, all live data panels remain empty.

## Data path

Physical device → trusted device bridge → authenticated `POST /api/live/*` endpoint → validated in-memory live state → current dashboard snapshot. The dashboard is read-only. No client-side code creates measurements or assessments.

## Authentication

The bridge must use `Authorization: Bearer $ROVER_INGEST_TOKEN`. The ingestion service stays disabled unless `ROVER_INGEST_TOKEN` contains at least 16 characters. Use a randomly generated secret and keep the server bound to loopback while developing.

## Timestamps and stale data

All measurements/reports need explicit timezone-aware source timestamps. The dashboard uses server receipt times for device heartbeats, source timestamps for measurement freshness, and reports both where applicable. Stale measurements are not rendered as current. Missing sensor fields remain `null` and render as an em dash.

## Current limitation

No hardware is attached and a physical polling/serial/radio bridge is not yet implemented. Therefore this app correctly has no live measurements. The APIs are only a validated integration boundary; they do not magically discover devices or make the PC behave like an ESP32/Raspberry Pi. Do not insert fabricated packets into the live interface.
