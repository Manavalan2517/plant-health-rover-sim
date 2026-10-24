# Architecture review — supplied plan and current code

> **Live-only runtime update:** The default `python -m plant_rover.cli serve` command starts the live monitor only. It does not start the simulator or load generated fixtures. The simulation details below describe internal protocol/domain regression tests only. With no authenticated hardware bridge, the live dashboard intentionally remains empty. See `docs/LIVE_ONLY_MODE.md`.


**Inputs reviewed:** the supplied 1440×900 architecture diagram, interactive `plant-health-rover.html`, and 25-page `Plant_Health_Rover_Implementation_Plan.pdf` (especially pages 5–6 for topology/protocols, page 9 for base-station notification rules, pages 14–17 for ML design, pages 19–22 for stack/roadmap/targets). The starter code in this folder is new and does not claim to be an audit of a pre-existing user repository; no project source repository was attached.

## System model from the source artifacts

The design has an off-rover training workstation, rover, virtual/physical field nodes, and base station/farmer interface:

1. **Off-rover training:** tomato-leaf datasets → MobileNetV2 fine-tuning/experiments → model selection on unseen plants → quantised TFLite export.
2. **Rover high-level computer (Raspberry Pi 4, proposed 2 GB):** camera capture and quality gate, image branch, sensor/history branch, fusion head, plant history/database, navigation planner, safety supervisor, LoRa report link, optional sync/dashboard path.
3. **Rover real-time controller (ESP32):** framed USB/serial protocol, sensor acquisition, motor control/PID, encoder/IMU/distance processing, watchdog and stop-on-timeout.
4. **Field sensor nodes:** ESP32-based periodic moisture, temperature, humidity, NPK estimates and optional soil metrics, with node status/battery and ESP-NOW transmissions.
5. **Base station:** LoRa + ESP-NOW, RTC, local history, risk/alert rules, display, voice/audio, buzzer, simulated/physical SMS channel, farmer language.
6. **Connectivity/data path:** nodes → ESP-NOW → base → pre-run history summary over two-way LoRa → rover; plant inspection reports return over LoRa; the base validates and stores received records before notification.

The source design's key ML point is appropriate: image-only output should be compared with sensor-only and fused outputs, an uncertain outcome should exist, and disease-like findings should be considered suspected rather than confirmed. Its acceptance values on PDF page 22 are starting targets, not achieved results.

## Component inventory and current PC implementation status

The source drawing should be read as a target architecture, not a claim that each target component is already coded. The table below distinguishes runnable simulation code from future physical or ML work.

| Component | Responsibilities and inputs → outputs | Interface / dependencies | PC baseline status and failure handling | Still required for physical/field acceptance |
|---|---|---|---|---|
| Off-rover training workstation | Public/collected images + labels → experiment results, model file, class map and quantised export | Python ML training pipeline, dataset manifest, model/version files | Not implemented yet; no model supplied or loaded | Dataset licence/label audit, split design, repeatable training, metrics, model card and export tests |
| Plant row and plant IDs | Scenario/checkpoint → stable `R01-Pnnn` identity | Virtual row sequence; future ArUco/RFID/positioning | Deterministic virtual plant sequence implemented; does not detect real markers | Marker readability, tag-to-plant mapping, stopping distance and identity error measurement |
| Camera + LED hood | Plant ID + capture condition → image + file path + quality metadata | `SyntheticCamera`; future Pi camera adapter | Synthetic PNG fixture source implemented; supports no-image failure and visual condition fixtures | Real camera configuration, fixed distance/angle, light uniformity, colour card, capture latency and reconnect behaviour |
| Capture quality gate | Image bytes → sharpness/exposure/clipping score, accept/reject and retake decision | Pillow image I/O/statistics in `vision.py` | Implemented and tested against synthetic blur/exposure/no-leaf fixtures; thresholds are placeholders | Threshold tuning using true camera samples and measured failure-vs-rejection trade-off |
| Row detector (HSV/Excess-Green/ArUco) | Camera frames → row/marker detections and navigation hints | OpenCV proposed by plan; image stream / marker dictionary | Not implemented; simulation uses known virtual plant position | Domain testing outdoors in shadows, weeds, occlusion, plant overlap and lighting variation |
| Image branch (MobileNetV2/TFLite) | Accepted leaf crop → model class scores and image embedding/features | TensorFlow/Keras training and TFLite interpreter intended | Not implemented as trained ML; scenario fixture cues are used explicitly instead | Fair per-plant evaluation, quantisation comparison, class mapping and inference/RAM/CPU profiling on Pi |
| Field sensor node firmware/model | Clock + probe samples → timestamped environmental record and battery/status | Future ESP32 firmware, sensor drivers, calibration metadata | Three virtual sensor nodes produce synthetic time series, faults and battery levels | Choose exact probe models; validate accuracy, units, calibration, drift, environmental sealing and duty-cycle power |
| ESP-NOW node link | Encoded node packet → base application receive/ACK | `RadioCodec` + `FaultInjectingLink`; future Espressif ESP-NOW API/peer config | Serialized byte path, ACK/retry, faults and dedupe are exercised | SDK/version interoperability, fixed channel, peer address, application ACK semantics, packet-loss/range and outdoor testing |
| Base sensor receiver and stale-data monitor | Valid packets → local records, fresh/stale status, latest sensor view | Radio payload decoder + SQLite | Implemented. Stale/offline values are visible rather than silently represented as fresh; demo uses 30-minute timeout | Configure TTL for chosen node sampling cadence, reboot handling and clock sync |
| Sensor feature branch | Fresh readings + node history → aggregate trend/environment features | Sensor decoding, history summaries and `DemoFusionEngine` | Synthetic moisture/time-series and summaries implemented; only basic aggregates are used; VPD/model feature engineering is not complete | Validate units, aggregation windows, sensor outlier policy and crop-specific features |
| History-summary return link | Base node history → compact per-node 24-hour summary to rover | Encoded radio message over simulated LoRa link | Implemented with real serialized frames/ACK and delivery-failure events | Define stale-summary policy, missing-node behaviour, packet budget and timing for the actual radio |
| Image/sensor fusion head | Image result + environment/history evidence → assessment, risks, evidence and next-step wording | `DemoFusionEngine`; future learned Logistic Regression/LightGBM | Provisional transparent rules; explicit `Uncertain`; provenance states `DEMO_RULES_NOT_VALIDATED` | Train image-only, sensor-only and fused models on paired real observations; measure calibration and class errors |
| Plant DB / daily report store | Received report/readings/events → durable history and query results | `storage.py` + SQLite tables, unique keys and local file | Implemented; valid decoded base-received reports/readings are persisted; mock outbox stored locally | Disk-full/power-cut/backup/retention/migration soak tests on the target device |
| Pi process bus / queue | Capture, inference, navigation and reporting work → bounded messages between workers | Source plan proposes multiprocessing/ZeroMQ/systemd | Not implemented as a separate bus; Python modules are composed in one application, while wire links still serialise and parse packets | Split into isolated services only if target profiling/reliability justify it; add process supervision and queue backpressure tests |
| Pi ↔ ESP32 serial transport | Drive commands down; telemetry/heartbeat/ACK/fault frames up | `SerialCodec`, parser, `SimulatedSerialSession`; future USB/UART | Implemented encoded frames, CRC, parser recovery, ACK/retry and injected faults | Golden vectors shared with firmware, real USB reconnection/timeout tests, exact binary command schema |
| Serial frame handler / virtual ESP32 | Valid command → ack + motor state; missing valid traffic → stopped motor/watchdog event | Host-side `VirtualESP32` model | Implemented software state machine and 400 ms virtual watchdog timeout | Native/hardware firmware test, MCU watchdog configuration, real-time measurement and fail-safe boot behaviour |
| Motor control / PID / encoders | Speed/heading target + feedback → PWM/direction/motion telemetry | Future ESP32 PWM, encoder interrupt processing and PID | Only virtual speed/motor enabled state; no physical PWM/PID/encoder controller is implemented | Wheel calibration, PID tuning, encoder polarity, motor stall and stop-time tests on raised wheels |
| IMU / distance / obstacle sensing | IMU/range/bump readings → heading, obstacle/row-loss alerts | Virtual telemetry planned; real IMU/range sensors later | Some simulated status/navigation fields and fault states exist; real pose fusion and robust obstacle planner are not implemented | Driver units/calibration, IMU bias, range materials and avoidance behaviour on an indoor test course |
| Navigation planner | Position + plant checkpoints + obstacles → follow/stop/inspect/resume commands | Engine's virtual navigation states; future state estimator and controller link | Basic virtual row workflow and stop states implemented; no physical row-follow algorithm | Closed-loop tracking, checkpoint accuracy, row-loss recovery, manual recovery and route safety |
| Safety supervisor | Watchdog, comms, battery and E-stop conditions → safe-stop state/fault events | Engine + virtual ESP32 controller | Serial-loss and low-battery stop paths tested; dashboard E-stop models a latched software action | Layered independent firmware safety, motor disable line and physical E-stop that cuts power without the Pi/app |
| Battery + supplies / buck converter | Battery/load measurements → voltage/current state and brownout response | Simulated battery percentage only | Estimated battery values and low-battery scenario; not an electrical power model | Measure peak motor/GSM current, regulator margins, fuse, battery protection, brownout and real runtime |
| Rover LoRa radio link | Compact reports rover → base; 24 h summary base → rover | Serialized `RadioFrame`, 64-byte total-frame cap in virtual link, retries/ACK | Software transport/fault model implemented | Resolve incompatible RA-02/SX1278 frequency choice, module/antenna, Indian rules, airtime/duty-cycle, range and coexistence |
| RTC (DS3231) | Trusted offline time → node/report timestamps | Simulation uses epoch/time source; no DS3231 adapter | Logical timestamped events and seeded sensor history implemented | Driver, backup-battery retention, drift, time setting and reboot/offline tests |
| Base alert rules + mock SMS/audio/buzzer | Received report/risk → local messages/status/notification events | SQLite outbox + dashboard notifications; future GSM/audio/buzzer drivers | Simulated SMS outbox exists; dashboard provides browser speech/beep substitutes; not real network delivery | Real modem power and carrier tests, Tamil SMS encoding, delivery state, audio volume, duplicate alarms and acknowledgements |
| Farmer dashboard and team monitoring | Base/API state → farmer statuses, readings, reports, log and controls | Local HTML/JavaScript + local HTTP/JSON API | Implemented; API/root/fault/reset smoke-tested. Server binds to loopback by default | Browser/accessibility/usability test, daylight display, Tamil fonts, security hardening if served beyond localhost |
| Optional sync uploader / MQTT | Local summaries → optional remote monitoring | Offline-first optional path in diagram | Not implemented; not needed for offline core | Only add with a specific requirement and secure offline queue/reconnect semantics |
| Physical e-stop | Human action → independent removal of motor power | Target electrical circuit outside software | **Not present.** The UI control only tests simulated state transitions | Independent hardware wiring, accessible switch, latch/reset design and witnessed power-cut testing |

## Important discrepancies and decisions

### D1 — Radio module and frequency are incompatible as specified
The diagram/plan labels the radio as **LoRa SX1278 / RA-02** in **865–867 MHz**. AI-Thinker's RA-02 specification says **410–525 MHz**, and Semtech's SX1278 coverage is **137–525 MHz**. These cannot be paired for 865–867 MHz operation. Before buying parts, choose an actual module and antenna designed for the selected band, verify its front-end and regional variant, and review India's current short-range-device rules. A silicon chip's broad frequency range alone does not certify a third-party module or deployment. See `RESEARCH_REGISTER.md`.

### D2 — Serial wire framing needs exact definitions beyond the diagram
The PDF specifies `0xAA | LEN | SEQ | TYPE | PAYLOAD | CRC16` but does not explicitly define CRC parameters, coverage, byte order, sequence wrap/duplicate window or the ACK's exact framing. The prototype defines these explicitly in `PROTOCOL_SPEC.md`, with CRC-16/CCITT-FALSE and big-endian CRC. Treat this as a proposed v1 software contract, not as an established existing firmware contract; freeze/version it before firmware is written.

### D3 — ESP-NOW send callback is not an application-level proof
An ESP-NOW MAC-layer send status is not the same as the base application parsing and saving a reading. The software design therefore implements explicit message sequence, CRC, acknowledgements/retry, duplicate suppression, stale-reading visibility, and a link-level timeout. The real firmware should mirror this behaviour and be tested against the Espressif version actually selected.

### D4 — Diagram mentions INT8/TFLite, but no trained model was supplied
The current image cues are scenario fixtures. They must not be confused with a MobileNetV2 classifier or its probabilities. Model architecture, model versioning, the dataset split and evaluation report must be established before presenting any class score as model output.

### D5 — The plan’s “prediction before detection” needs measurable evidence
Prediction must mean a prospective alert emitted before later independently recorded visible symptoms, not a rule simply detecting a low current measurement. Log the prediction timestamp, definition of visible onset, reference label, warning lead time, and false alerts in held-out field runs.

### D6 — Sensor values and thresholds are synthetic placeholders
All demo moisture, climate, NPK, pH and EC values are simulated. Their ranges, units and risk rules are engineering test fixtures, not calibration or agronomic advice. In particular, generic `mg/kg` NPK estimates cannot be asserted valid for any specific probe without understanding the instrument's method and calibration.

### D7 — Scope of first software build
To keep setup lightweight, the prototype is a single Python process with threaded local HTTP server and SQLite, not a multi-process `ZeroMQ` deployment. This still exercises byte protocols and independent subsystem interfaces; processes can be split after measurement identifies a need. The UI is for localhost-only development, not a production network endpoint.

## Recommended near-term order

1. Keep simulation as the only selectable mode until core behaviours are stable.
2. Lock the protocol spec and add schema/versioning tests before implementing microcontroller firmware.
3. Expand the current test set for database transaction failures, stale history, timeouts, duplicate application messages and UI/API error paths.
4. Replace synthetic image cues with a reproducible image-only baseline; compare it on held-out plants and keep the uncertain threshold explicit.
5. Resolve the LoRa radio/band incompatibility and local compliance requirements before procuring radio hardware.
6. Integrate real devices one at a time; never mark a hardware-specific acceptance item passed because a software simulation passed.
