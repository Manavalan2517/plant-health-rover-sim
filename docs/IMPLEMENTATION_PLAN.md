# Implementation plan and milestone tracking

> **Live-only runtime update:** The default `python -m plant_rover.cli serve` command starts the live monitor only. It does not start the simulator or load generated fixtures. The simulation details below describe internal protocol/domain regression tests only. With no authenticated hardware bridge, the live dashboard intentionally remains empty. See `docs/LIVE_ONLY_MODE.md`.


This plan separates shipped in the PC baseline from future work. The source PDF (pp. 21–22) proposes a 16-week hardware-to-field roadmap; its dates and acceptance values are planning targets, not measured results. For PC-first work, reorder dependencies so simulated end-to-end interfaces are verified before any real hardware is available.

## M0 — Inputs and architecture review (documented)
- [x] Inspect supplied architecture diagram and HTML labels/components.
- [x] Use the 25-page plan as a primary reference, especially data flow, wire protocols, ML/AI approach, safety and proposed roadmap.
- [x] Identify the LoRa frequency / RA-02 incompatibility.
- [ ] Audit the full existing code repository. No existing source repository was present in the supplied files; repeat this step when one is available.

## M1 — End-to-end PC software-in-the-loop foundation (implemented; 34 tests passing at baseline)
- [x] Project scaffold, CLI, local dashboard and API.
- [x] Wire codecs, CRC, stream parser, packet limits.
- [x] Fault-injecting serial/ESP-NOW/LoRa simulator and encoded ACK/retry.
- [x] Virtual ESP32 controller with software stop/watchdog.
- [x] Three virtual field sensor nodes, time-series and 24-hour summaries.
- [x] Synthetic camera fixtures and tested image-quality checks/retake path.
- [x] Demonstration fusion rules and explicit uncertainty.
- [x] SQLite base persistence, event history and mock SMS outbox.
- [x] Scenario CLI/dashboard controls and automated tests.
- [ ] Add replay-export for deterministic event traces and tests for restart/session browsing.

**Historical simulation-harness criterion:** internal regression tests can run complete scenarios. This is not the default runtime. The default user-facing command is now live-only and starts with empty data until authenticated device packets arrive.

## M2 — Protocol and reliability hardening (next)
1. Freeze message payload schema and CRC/length coverage with golden byte-level test vectors shared across Python and eventual firmware.
2. Add `device_id + boot_id + sequence` addressing, application-level deduplication and database uniqueness constraints.
3. Add explicit stale/missing-data TTL per sensor type, monotonic/UTC timestamp rules and clock-reset simulation.
4. Test database locked/disk-full/permission failures and ensure safe failure modes.
5. Add API negative tests, concurrency tests, server shutdown cleanup and multiple session browsing.
6. Add deterministic replay trace export (seed, scenario version, commands, external faults and result digest).

## M3 — Real image dataset, baseline model and honest evaluation
1. Create a dataset manifest with dataset name/version, original path, licence, image checksum, class label, plant ID, capture session, source, split and annotation confidence.
2. Build data loader with reproducible class mapping and group-based split by plant/source/session; prevent augmented variants from crossing splits.
3. Train image-only MobileNetV2 baseline and compare at least with a transparent non-neural baseline where feasible.
4. Report per-class precision/recall, macro-F1, balanced accuracy, confusion matrix, calibration and failure cases. Include an explicit `uncertain`/reject policy.
5. Evaluate on real farm/camera-hood images; dataset-only evaluation is insufficient due to domain shift.
6. Measure sensor-only baseline independently. Train fusion on genuinely paired image/sensor/history observations, and compare A=image-only, B=sensor-only, C=fusion on held-out plants.
7. Keep “disease-like/suspected” separate from confirmed disease labels; obtain expert-reviewed labels where possible.

## M4 — Model export and PC deployment profiling
1. Build reproducible saved model, requirements/lockfile, class-index file and model card.
2. Convert to TFLite, evaluate float/quantized versions on an untouched representative validation set, and save conversion metadata.
3. Profile inference time, CPU/memory and peak memory on PC and separately on Raspberry Pi. Do not extrapolate results from PC.
4. Verify output index, preprocessing, RGB/BGR, normalization and version compatibility.
5. Add a model abstraction so the engine can use either `DemoFusionEngine` or a `ModelInferenceEngine`, selected explicitly and labelled in reports.

## M5 — Physical hardware interfaces, only after simulator contract is stable
1. Choose verified Pi/ESP32 revisions, camera, wheel encoders, motor driver, IMU, distance sensors, power rails and independent E-stop.
2. Resolve LoRa module and band discrepancy; verify Indian rules, antenna and module data before procurement.
3. Write ESP32 firmware around the exact serial schema; compile native/hardware-independent logic where possible.
4. Add Pi camera adapter via current Raspberry Pi camera software / Picamera2; camera capture should remain injectable for tests.
5. Add separate physical serial/ESP-NOW/LoRa adapters with no import side effects in simulation.
6. Commission each adapter with hardware-in-loop tests, starting with one device and motor-power disconnected.

## M6 — Field base station and notifications
1. Add ESP32 base firmware, DS3231 clock handling and local store/SD strategy as hardware requires.
2. Verify base-to-rover history summaries and rover-to-base report transfer with loss/reboot scenarios.
3. Implement real GSM modem adapter only after power-supply sizing, modem/network compatibility and SIM provisioning; never label mock outbox as sent SMS.
4. Test local display/Tamil font, voice playback, audible buzzer, acknowledgement workflow and farmer comprehension.
5. Maintain offline operation when SMS, radio or optional dashboard is unavailable.

## M7 — Field trials and measured acceptance
1. Start with wheels raised or drivetrain mechanically isolated; verify emergency stop/power cut and watchdog before testing motion.
2. Test track following, plant marker localisation and inspection pose with AI disabled.
3. Run a controlled tomato row; record completion %, navigation error, failures, battery/temperature, packet loss and recovery time.
4. Collect paired, dated image + sensor observations and later confirmed labels; keep synthetic data out of field evaluation.
5. Measure farmer comprehension and alert usefulness in Tamil/target language.
6. Report acceptance targets vs measured results and missed detections, not just successful cases.

## Blockers / unresolved design inputs

- No hardware or original source repository supplied yet.
- Exact LoRa module/frequency/Indian rule compliance unresolved.
- Specific sensor part numbers and units/calibration methods unresolved.
- Wire payload versioning needs a firmware-ready golden-byte specification.
- No labelled paired farm dataset available in these inputs.
- No validated disease thresholds, calibration, model metrics, farmer-study results or Pi profiling yet.
