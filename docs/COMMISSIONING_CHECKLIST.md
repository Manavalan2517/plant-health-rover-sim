# Physical Hardware Commissioning & Bench Testing Checklist

**Date:** 2026-10-22  
**Purpose:** Step-by-step physical commissioning guide to transition from PC simulation to physical Raspberry Pi 4 and ESP32 hardware without risk of motor runaway, brownout, or RF violation.

---

## Stage 0: Pre-Power Electrical Inspection
- [ ] Verify battery polarity (3S LiFePO4 / Li-ion pack, nominal 11.1V–12.8V).
- [ ] Measure buck regulator outputs before connecting microcontrollers:
  - Rail A (Raspberry Pi 4): $5.1\text{ V} \pm 0.1\text{ V}$, minimum $3.0\text{ A}$ continuous capacity.
  - Rail B (ESP32 & Sensors): $3.3\text{ V} \pm 0.05\text{ V}$.
  - Rail C (Motor Drivers): Unregulated battery bus ($12\text{ V}$).
- [ ] Verify independent physical Emergency Stop switch:
  - Must physically de-energize the motor driver $12\text{ V}$ power bus via contactor or high-side P-channel MOSFET, completely independent of Raspberry Pi or ESP32 GPIO.
  - Actuation of E-Stop must also pull ESP32 GPIO 4 LOW.

---

## Stage 1: Bench Safety & Motor Cutoff (Raised Wheels)
> [!CAUTION]
> The rover chassis must be raised on blocks with all wheels off the ground for all tests in this stage.

1. **Watchdog Verification Test:**
   - Power up ESP32 with firmware loaded.
   - Send drive command: `0xAA 0x06 0x01 0x01 [speed=50, steer=0] [CRC16]`.
   - Verify motor wheels begin rotating.
   - Disconnect USB/UART cable.
   - **Pass Criterion:** Motors must cease rotation within $\le 400\text{ ms}$. Status LED must turn off.
2. **Emergency Stop Interrupt Test:**
   - Command continuous motion.
   - Press physical E-Stop button.
   - **Pass Criterion:** Motor PWM cut to 0 within $< 10\text{ ms}$. Power relay opened.

---

## Stage 2: Microcontroller Serial Protocol Loopback
- [ ] Connect Raspberry Pi USB to ESP32 CP2102/CH340 USB-UART.
- [ ] Run Python verification script:
  ```bash
  python3 -c "from plant_rover.hardware_adapters import PhysicalSerialAdapter; from plant_rover.protocol import SerialFrame, SerialType; a = PhysicalSerialAdapter('/dev/ttyUSB0'); assert a.send_command(SerialFrame(1, SerialType.HEARTBEAT)); print('Physical Serial Verified!')"
  ```
- [ ] Verify CRC-16/CCITT-FALSE handshake and round-trip latency $< 5\text{ ms}$.

---

## Stage 3: Camera & Illumination Hood Calibration
- [ ] Connect Raspberry Pi Camera Module 3 (IMX708) via ribbon cable.
- [ ] Verify detected by `rpicam-hello` / `libcamera-hello`.
- [ ] Install camera hood with white LED array.
- [ ] Place Macbeth 24-patch color calibration chart under the hood at fixed working distance ($25\text{ cm}$).
- [ ] Capture image using `PhysicalPicamera2Adapter` and run `ImageQualityGate`:
  - Mean brightness must fall in $90\text{--}160$ range.
  - Sharpness variance $\sigma^2 \ge 180.0$.
  - Clipped pixel fraction $\le 2\%$.

---

## Stage 4: LoRa Radio & Spectrum Compliance (865–867 MHz)
- [ ] Verify installed radio module is Semtech SX1276 / RFM95W or SX1262 (RA-01SH). Confirm AI-Thinker RA-02 (433 MHz) is NOT installed.
- [ ] Connect tuned $868\text{ MHz}$ $1/4\lambda$ whip antenna or dipole.
- [ ] Inspect output on spectrum analyzer:
  - Center frequency: $865.5\text{ MHz}$.
  - Transmit power: $\le 20\text{ dBm}$ (well within the Indian 1 Watt / $30\text{ dBm}$ ERP limit under G.S.R. 564(E)).
  - Bandwidth: $125\text{ kHz}$.
- [ ] Verify base station receives `PlantReport` packets and returns `RadioType.ACK` frame.

---

## Stage 5: ESP-NOW Field Sensor Node Commissioning
- [ ] Power on 3 field sensor nodes.
- [ ] Verify Wi-Fi channel is fixed to Channel 1.
- [ ] Check base station SQLite database for incoming `SensorReading` rows.
- [ ] Cover sensor probes to test plausibility checks:
  - Moisture: $0\text{--}100\%$.
  - Nitrogen: $0\text{--}100\text{ mg/kg}$.
  - Air temp: $-10\text{--}60^\circ\text{C}$.

---

## Stage 6: Field Row Trial (Ground Contact)
- [ ] Lower rover wheels to ground inside a test row.
- [ ] Execute `mixed_field` or target navigation profile.
- [ ] Verify plant checkpoint alignment, capture gate, report persistence, and farmer SMS outbox generation.
