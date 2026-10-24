# Future physical-hardware integration plan

This plan is intentionally future work. There is no physical-device validation in the current PC simulation.

## Before buying or energising anything

- Resolve the RA-02/SX1278 vs 865–867 MHz mismatch documented in `ARCHITECTURE_REVIEW.md`.
- Confirm the exact ESP32 variants, motor driver, encoder, IMU, ranging sensor, camera module, environmental probes, battery chemistry/capacity, regulators, RTC, LoRa module/antenna, GSM modem and base display.
- Review the current Indian rules for the selected radio/device configuration and verify actual module frequency, antenna and power are consistent with the allowed conditions.
- Build a power budget for motor stall current and modem transmit bursts; don't power an ESP32 or Pi from a GPIO rail. Provide appropriate regulators, fuse/overcurrent and brownout handling.
- Design an independent hardware E-stop that physically interrupts motor-driver power regardless of Pi software, serial communication, MCU application state or radio. Validate it before wheels are allowed to touch ground.
- Document GPIO/UART/SPI/I²C pin assignments, voltage levels, common grounds, isolation where needed, connector pinouts and power rails.

## Adapter boundaries to retain

| Adapter | PC now | Physical implementation later | Acceptance needed |
|---|---|---|---|
| Camera | Synthetic fixture source; image-quality algorithm | Raspberry Pi libcamera/Picamera2 adapter | Capture/release lifecycle, timestamps, dimensions, colour/exposure, missing camera recovery |
| Serial | Byte-pair with fault injection | USB serial/UART to ESP32 | Golden test vectors, framing fragmentation, CRC rejection, ACK timing, reconnect and watchdog |
| Motion control | VirtualESP32 | ESP-IDF or Arduino-ESP32 C/C++ firmware | Speed caps, PWM/PID, encoder tick correctness, direction, stall and stop tests |
| IMU/encoder/range | Simulated telemetry/event | Real driver firmware + framed telemetry | Units, sampling rate, calibration, stale values, obstacle thresholds under test |
| Field sensors | VirtualSensorNode | Firmware drivers selected from exact datasheets | Range, accuracy, calibration, failure sentinel, electrical and thermal limits |
| ESP-NOW | FaultInjectingLink with serialised sensor payload | Espressif SDK adapter | Channel/peer configuration, app ACK/deduplication, loss rate, node identity and reconnect |
| LoRa | FaultInjectingLink with encoded report/summaries | Validated module-specific driver | Module/band/antenna first, range/packet error/latency/airtime/power, legal transmission settings |
| RTC | Deterministic simulation epoch/time | DS3231 I²C driver | Drift, time-setting, reboot and battery-backed time tests |
| Storage | SQLite local file | SQLite on Pi/SD or base storage as designed | Power-loss recovery, WAL behaviour, space exhaustion, backup/retention |
| SMS | Local outbox only | GSM modem/SIM/network adapter | Power transient, AT command parser, network registration, encoding, delivery status, no false “sent” state |
| Farmer display/audio/buzzer | Browser UI, browser speech/beep substitute | Actual display/speaker/driver/buzzer | Daylight visibility, Tamil fonts/voice, volume, duplicate alarms, acknowledgement and usability |
| Battery/power | Simulated percentages/brownout events | Battery monitor/regulators/protection | Worst-case runtime, motor stall load, modem burst, undervoltage and safe recovery |
| E-stop | Dashboard/software stop state | Independent physical interrupt/power cut | Hardware-in-loop stop-time measurement under Pi lockup, serial loss, MCU crash and reboot |

## Suggested bench order

1. Validate and build the controller firmware with motor-power output disconnected. Confirm serial parsing with shared golden byte vectors and host tests.
2. Power only controller and low-risk sensors; verify voltages, current, resets and watchdog.
3. Verify independent E-stop physically removes motor-driver enable/power and latches safe. Record tests, don't infer it from app state.
4. Test one motor at low speed with wheels raised and current limiting. Verify direction, stop time, encoder polarity and watchdog on unplugged serial cable.
5. Test camera independently on Pi OS using Picamera2/libcamera; replay captured images through same quality-gate API.
6. Verify every environmental probe against a reference instrument and documented calibration; don't combine uncalibrated readings into risk rules.
7. Bench-test ESP-NOW with two selected ESP32 boards and agreed SDK. Measure app-level delivered-and-stored messages, not merely MAC send callbacks.
8. Choose and test the compliant LoRa module/antenna configuration; measure packet delivery/retry/latency/power with the intended payloads and install location.
9. Integrate the base station and RTC, then GSM and farmer outputs. Label modem states “queued”, “sent to modem”, “network acknowledged” only when that level has actually been verified.
10. Test integrated system in a controlled area, then a short outdoor row, then longer field trials. Keep an accessible manual stop and a recovery plan.

## Hardware-in-loop sign-off record

For each test, store: test ID/date/operator, device/module revision, firmware commit, configuration, power supply, ambient conditions, test input, expected result, measured outcome, logs/packet capture, pass/fail, defect link and reviewer. Required sign-off groups: mechanical/electrical safety; serial+firmware; RF/regulatory; sensor calibration; Pi camera/inference performance; storage and notification resilience; end-user usability.
