# Proposed software protocol specification v1

This document freezes the byte-level format used by the PC prototype. The original supplied PDF states the high-level field order but leaves details such as CRC parameters and ACK representation unspecified; this file makes those choices explicit for tests. If an existing firmware protocol is found later, compare both sides and record a versioned migration rather than silently changing bytes.

## 1. Pi ↔ ESP32 serial frame

```text
+-------+--------+-----+------+------------------+------------+
| START | LENGTH | SEQ | TYPE | PAYLOAD          | CRC16 (BE) |
+-------+--------+-----+------+------------------+------------+
| 0xAA  | u8     | u8  | u8   | LENGTH - 2 bytes | u16        |
+-------+--------+-----+------+------------------+------------+
```

- `START`: one byte, fixed `0xAA`.
- `LENGTH`: uint8 byte count of `SEQ + TYPE + PAYLOAD`; minimum 2, maximum 255.
- `SEQ`: uint8, wraps from 255 to 0. Receiver tracks a bounded recent window; firmware should not treat sequence number alone as a globally unique command across reboots.
- `TYPE`: uint8 enum: `0x01 COMMAND`, `0x02 TELEMETRY`, `0x03 HEARTBEAT`, `0x04 STOP`, `0x7F ACK`, `0x80 FAULT`.
- `PAYLOAD`: payload definition depends on the type. Current movement command text is a debugging contract (`MOVE:<integer percent>`, `STOP`); do not make a physical motor firmware parse this ad-hoc text forever. Define a versioned typed binary motor payload before hardware.
- `CRC16`: two-byte unsigned big-endian CRC of `LENGTH | SEQ | TYPE | PAYLOAD`. CRC bytes and START are excluded from the checksum input.

CRC algorithm: CRC-16/CCITT-FALSE, polynomial `0x1021`, initial value `0xFFFF`, input/output not reflected, final XOR `0x0000`. Check vector `CRC("123456789") = 0x29B1`.

### Serial acknowledgements and safety

The simulator replies with a serialized `ACK` containing the acknowledged sequence in the frame sequence and one-byte payload. The sender retries within a bounded retry budget. A corrupted or invalid frame is rejected and must not update the motor command. The virtual controller stops enabled motion when no valid control frame arrives for >400 ms of virtual runtime; the plan's 300–500 ms is a proposed range, not a hardware measurement.

For physical firmware, the protocol must separately define: command idempotency; exact typed payloads; heartbeat cadence; bounded ACK timeout/retries; wraparound-safe duplicate windows; failsafe behaviour after malformed traffic; safe boot state; watchdog feeds; and how an independent e-stop cuts motor power. An application ACK proves application parsing only when sent after validation, not that a motor actually moved.

## 2. Virtual radio frame (LoRa/ESP-NOW transport)

```text
+-------+---------+------+--------+------------+---------+------------+
| MAGIC | VERSION | TYPE | LENGTH | SEQUENCE   | PAYLOAD | CRC16 (BE) |
+-------+---------+------+--------+------------+---------+------------+
| 0xC3  | u8      | u8   | u8     | uint16 BE  | n bytes | u16        |
+-------+---------+------+--------+------------+---------+------------+
```

- Header struct: `!BBBBH` (6 bytes), followed by `LENGTH` payload bytes and two CRC bytes.
- Version 1 only. Unknown versions are rejected, not interpreted as v1.
- `SEQUENCE` is uint16 big-endian. Payload length is constrained by transport profile even though one-byte wire length permits 255 bytes.
- CRC16 covers `VERSION | TYPE | LENGTH | SEQUENCE | PAYLOAD`; `MAGIC` and CRC bytes are excluded.
- Base software protocol currently caps virtual LoRa frames at 64 bytes to keep packets compact. ESP-NOW simulation defaults to 250-byte payload/packet budget; actual ESP-NOW constraints depend on SDK version and framing overhead. These defaults model application policy, not an assertion about every physical radio or SDK.
- Radio ACK is a radio frame of type `ACK`, with sequence echoed in the header and two-byte sequence in payload. ACK frames pass through the same simulated fault-injectable link.

### Current message payloads

The application defines fixed-width typed payload codecs in `plant_rover/protocol.py`:

- `SENSOR_READING`: node ID, sample sequence, timestamp, moisture, soil temperature, air temperature, relative humidity, N/P/K estimates, pH, EC, battery and quality flags. Multi-byte values are big-endian. Fields use scaled integer encodings; inspect the corresponding codec dataclass for exact scales.
- `PLANT_REPORT`: plant ID, scenario assessment enum, confidence-like demonstration score, risk flags, image quality indicators, battery/measurement summary and sample timestamp; never interpret the demo score as calibrated probability.
- `HISTORY_SUMMARY`: node ID, timestamp, sample count, minimum/maximum/trend moisture, min/max air temperature, maximum humidity and stale flag.
- `ACK`: acknowledged radio sequence number.

The Python codec's exact fixed struct strings and validation are the source of truth for software tests. Before firmware integration, publish a tabular field-by-field schema with signedness, scale, invalid/sentinel values, allowed ranges and compatibility/version policy. Do not port Python struct formats by guesswork.

## 3. Link behaviours

- Serial parser supports arbitrary chunk boundaries, ignores noise until START, checks length/CRC and attempts resynchronisation.
- Virtual transports may inject packet loss, bit corruption, duplication, latency and link disconnection; over-limit frames are rejected before delivery.
- Receiver validation is the only path into base persistence. CRC/length validation fails closed.
- Sender retransmission is bounded (currently three total attempts in radio helper; the exact retry count is a software default, to be confirmed for radio airtime/battery budget).
- Repeated report packets are deduplicated at transport/application boundaries where implemented. Physical system should deduplicate using a stronger key such as `(device_id, boot_id, message_sequence)` and idempotent database constraints.
- Missing/stale sensor nodes remain visible; the latest old value is not silently represented as live. Current demo status threshold is 30 minutes and should be configurable against the sampling interval.

## 4. Not specified by this v1 frame

Cryptographic authentication/encryption, key rotation, anti-replay across reboots, radio collision avoidance, duty-cycle scheduling, network addressing for dozens of nodes, fragment/reassembly for large reports, firmware update protocol, and security claims are not implemented. CRC detects accidental corruption; it does not authenticate the sender or prevent malicious changes.
