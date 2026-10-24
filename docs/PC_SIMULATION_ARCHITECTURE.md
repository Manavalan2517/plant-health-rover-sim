# PC simulation architecture

> **Live-only runtime update:** The default `python -m plant_rover.cli serve` command starts the live monitor only. It does not start the simulator or load generated fixtures. The simulation details below describe internal protocol/domain regression tests only. With no authenticated hardware bridge, the live dashboard intentionally remains empty. See `docs/LIVE_ONLY_MODE.md`.


## Purpose

Provide a practical software-in-the-loop foundation while no physical parts are connected. The simulation mirrors the planned data journey and protocol boundaries; it does not pretend to emulate electrical, optical, mechanical or RF physics accurately.

## Top-level flow

```text
Virtual sensor nodes
  └─ SensorReading payload → RadioCodec → ESP-NOW FaultInjectingLink → base receiver
                                                        └─ validate → SQLite readings/events
                                                             └─ 24 h summaries
                                                                  ↓ serialized RadioFrame
Virtual Raspberry Pi app ← LoRa FaultInjectingLink ← base station
  ├─ encoded serial movement command → simulated serial link → VirtualESP32
  │     ← ACK / encoder-style telemetry ← CRC parser / timeout / watchdog
  ├─ SyntheticCamera → ImageQualityGate → scenario image cue ┐
  ├─ sensor/history features from received summaries ────────┤→ DemoFusionEngine
  ├─ SQLite/local event log ← serialized report over LoRa ──┘
  └─ inspection result → LoRa retry/ACK path → base SQLite → dashboard + mock SMS outbox
```

## Runtime and boundaries

- `protocol.py`: binary frame encoders/decoders, CRC, incremental serial parser, payload codecs.
- `transports.py`: a seeded fault-injectable byte link, application ACK/retry logic and host-side `VirtualESP32`.
- `sensors.py`: deterministic sensor values, plausible time series and history summarisation.
- `vision.py`: synthetic image fixture creation and an actual image-quality gate.
- `fusion.py`: explicitly provisional rule-based demo assessments; no trainable model loaded.
- `engine.py`: orchestration, state machine, virtual communication, inspection runs and faults.
- `storage.py`: local SQLite tables and access methods.
- `server.py`: loopback HTTP API and dashboard static file server.
- `static/index.html`: dashboard display/control surface; it polls the same backend that owns the running simulation.
- `cli.py`: run server, list scenarios or execute a standalone scenario.

## Mode isolation

This baseline intentionally implements **simulation mode only**. It does not include a hidden hardware mode and does not instantiate `serial`, GPIO, Picamera2, ESP-NOW or LoRa driver objects at startup. That prevents a PC-only run from accidentally attempting to access nonexistent hardware. Future adapters should be registered behind typed interfaces and enabled explicitly with configuration; hardware mode must fail closed if a requested adapter or device is missing.

Proposed later configuration (not all implemented as environment variables yet):

```ini
ROVER_MODE=simulation
SENSOR_NODE_MODE=simulation
SERIAL_TRANSPORT=simulation
LORA_TRANSPORT=simulation
ESPNOW_TRANSPORT=simulation
CAMERA_BACKEND=synthetic
NOTIFICATION_BACKEND=mock
DATABASE_PATH=./data/plant_rover.sqlite3
SIMULATION_SEED=2026
```

## Determinism and fault injection

The seed initializes reproducible scenario fixture choices and transport random streams. Faults can drop, corrupt, duplicate, delay, disable or size-reject byte frames. Receivers parse the encoded bytes and can reject invalid CRC/length rather than receiving an in-memory Python object directly. The application sends serialized ACKs and applies retries. Tests should assert delivery counts and resulting states, not hard-code wall-clock times.

Determinism is bounded: OS thread scheduling, request timing and actual elapsed timestamps can differ. A strict replay mode should eventually persist the seed, scenario ID, configuration, and ordered external/control events.

## Data provenance

Every generated image is marked synthetic in its metadata. Virtual sensor readings returned by the API have `synthetic: true`; synthetic outbox messages are mock records rather than carrier delivery receipts. UI labels and the `demo_disclaimer` state that assessments are not a diagnosis. No synthetic image or reading should be exported into a “real-world” training dataset as ground truth.

## Persistence

SQLite is the only required storage service. Readings are inserted when valid encoded readings arrive at the base receiver, plant reports are inserted after radio packet receipt/validation, and events/outbox messages provide a local record of the run. The browser calls the HTTP API rather than mutating Python objects or reading SQLite directly. Use separate `--data-dir` paths for clean test sessions.

## Current limitations

- Single-process orchestration with a threaded HTTP server; no external message broker is required.
- Rule-based fixture cue fusion, not MobileNetV2 or a trained tabular classifier.
- Synthetic environment, camera and battery; no validated kinematics, robot geometry or optical model.
- Virtual links model packets and selected network faults, not modulation, RF propagation, collisions, channel occupancy, weather, or ESP-NOW Wi-Fi channel coexistence.
- Watchdog timing is an application-level virtual time check, not a real hardware timer or motor power cut.
- Dashboard speech uses browser capabilities where available; audio delivery is not guaranteed on all browsers/headless sessions.
