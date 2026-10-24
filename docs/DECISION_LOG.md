# Decision log

Status labels: **Accepted for prototype** means a local software decision; **Open** means do not freeze a hardware purchase/design based on it.

## D-001 — PC-only is the default execution mode
- **Status:** Accepted.
- **Decision:** initial build uses synthetic camera, virtual ESP32, virtual sensors and virtual serial/radio links only. No hardware APIs are imported to launch the app.
- **Reason:** no hardware is connected and software development must not wait for procurement.
- **Consequence:** simulated pass/fail does not validate physical electrical, optical, RF or mechanical behaviour.

## D-002 — Lightweight Python + SQLite local application
- **Status:** Accepted for prototype.
- **Decision:** use Python standard library for CLI, server and SQLite; Pillow for images; handwritten, unit-tested frame codecs; local HTML/JS dashboard.
- **Reason:** low setup burden, offline operation, immediate testability.
- **Trade-off:** this is not a hardened web service. Bind localhost by default; split into processes only after profiling or target deployment needs justify it.

## D-003 — Freeze protocol v1 in a separate specification
- **Status:** Accepted for software simulation, open for hardware compatibility.
- **Decision:** serial CRC-16/CCITT-FALSE, big-endian CRC; explicit sequence/type/length; radio frame has version/type/length/uint16 sequence/CRC.
- **Reason:** the supplied diagram names fields but does not define every wire-level detail.
- **Consequence:** physical firmware must implement this exact contract or a reviewed, versioned alternative. Do not assume that this prototype is already the team's firmware standard.

## D-004 — Application ACK/retries and deduplication
- **Status:** Accepted for prototype.
- **Decision:** radio ACK bytes are sent through same fault-injectable link; serial commands get framed ACKs; retransmissions are bounded; stale readings are explicit.
- **Reason:** MAC delivery and application storage are different events; loss and duplication need tests.
- **Open:** production retry budget, node IDs/boot IDs and idempotent database uniqueness need final specification.

## D-005 — Rule-based fusion only for baseline
- **Status:** Accepted as clearly labelled demo, not as a diagnostic.
- **Decision:** demonstrate a five-category workflow with provisional rules/fixtures and explicit `Uncertain` outcome before training an ML model.
- **Reason:** no project-specific labelled images paired with sensor observations were attached; it would be misleading to claim a trained classifier.
- **Consequence:** thresholds and confidence-like values have no scientific calibration. Train/evaluate image-only first, then sensor-only and fused approaches.

## D-006 — Sensor/image fixtures are synthetic and traceable
- **Status:** Accepted.
- **Decision:** synthetic readings and generated image metadata are explicitly labelled; no generated fixture is ground truth.
- **Reason:** supports integration/fault testing without external parts while protecting data validity.
- **Consequence:** field accuracy claims require real paired observations and an evaluation protocol.

## D-007 — Radio hardware/frequency discrepancy blocks selection
- **Status:** Open; high priority.
- **Problem:** source plan says SX1278/RA-02 at 865–867 MHz. RA-02/SX1278 is a 410–525 MHz-class configuration; incompatible with the specified band.
- **Decision:** do not purchase that exact combination for 865–867 MHz. Compare complete modules, antenna, front-end, output power and regional compliance before choosing another radio.
- **Evidence:** see `RESEARCH_REGISTER.md` (AI-Thinker RA-02 and Semtech radio tables).

## D-008 — Risk values remain provisional
- **Status:** Accepted with limitation.
- **Decision:** synthetic threshold values are used only to exercise workflows.
- **Reason:** sensor accuracy, tomato growth stage, soil type, irrigation regime and local cultivar-specific thresholds were not supplied or validated.
- **Consequence:** farmer UI must call these demo/risk fixtures, not an actionable agronomic recommendation.

## D-009 — Single local process until there is evidence to split it
- **Status:** Accepted for prototype.
- **Decision:** one Python application with modular protocol, transport, sensing, vision, fusion, storage and API layers; use a thread for dashboard server only.
- **Reason:** the first priority is end-to-end testing without hidden service dependencies.
- **Future:** evaluate process isolation and systemd/supervisor deployment after the pipeline is stable and target Pi performance is measured.

## D-010 — Dataset terms must be checked before reuse
- **Status:** Open before dataset packaging/redistribution.
- **Decision:** use datasets for research only after documenting source, version, labels, licence/terms and redistribution restrictions.
- **Reason:** “publicly downloadable” or “open access” does not automatically grant permission to redistribute data or trained weights for all uses.
