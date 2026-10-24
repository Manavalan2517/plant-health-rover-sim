# Test plan and current evidence

> **Live-only runtime update:** The default `python -m plant_rover.cli serve` command starts the live monitor only. It does not start the simulator or load generated fixtures. The simulation details below describe internal protocol/domain regression tests only. With no authenticated hardware bridge, the live dashboard intentionally remains empty. See `docs/LIVE_ONLY_MODE.md`.


## Automated test command

```bash
python -m unittest discover -v
python -m compileall -q plant_rover tests
```

The current live-only revision has **72 automated tests** covering serial/radio frame validation, CRC known vector, fragmented serial parsing and resynchronisation, encoded ACK/retry paths, transport fault behaviour, sensor time series/history summaries, image quality conditions, disconnected nodes, scenario workflows, SQLite base receipt/mock notifications, low-battery stop and serial-watchdog stop. Rerun the commands against the checked-out version; a test count alone is not a substitute for inspecting assertions or testing on target devices.

## Test layers

### Layer 1 — wire protocol

- CRC-16/CCITT-FALSE known vector is correct.
- Serial/radio round trips preserve message values.
- Bad CRC, invalid length, unsupported version and over-size payloads are rejected.
- Incremental parser handles arbitrary chunks/noise and resynchronises after corrupted frames.
- Application ACK is serialized and passes through the simulated transport.

### Layer 2 — transport/fault handling

- Drop/corrupt/duplicate/disconnect/latency/size constraints are exercised deterministically under a seed.
- Repeated radio sends stop at the retry budget when no valid ACK returns.
- Receiver never inserts an undecoded/corrupt report as valid.
- Duplicate reception must not create duplicate reports or duplicate notification side effects (extend database uniqueness tests in the next sprint).

### Layer 3 — sensing and quality gate

- Drying-soil scenario produces a descending, timestamped moisture trend rather than unrelated random samples.
- History summary includes extrema/trend and identifies stale data.
- Clear fixture passes the quality gate; blurred, underexposed, overexposed and no-leaf fixtures are not treated as healthy.
- Capture failure returns an uncertain assessment with absent-image evidence instead of a fake healthy prediction.

### Layer 4 — integrated scenario tests

- Normal / mixed field completes and saves reports in base SQLite only after serialised LoRa receive/validation.
- Drought shows a synthetic context risk and retains the synthetic-data labels.
- Node dropout is visible while other nodes remain active.
- Serial loss results in a software safe stop/watchdog state.
- Low battery triggers simulated safe stop.
- Radio loss exercises retries; it must not be reported as physical-range evidence.
- Dashboard API reports simulation mode and `hardware_access: false`.

## Manual dashboard acceptance recipes

1. **Healthy baseline:** reset, choose healthy baseline, start. Expected: all configured plant checkpoints complete unless a fault is injected; reports arrive at base; mock outbox contains only local mock records.
2. **Drought:** reset to drought, inspect node graph/history and report evidence. Expected: moisture trends downward; all values remain labelled synthetic; rule-based outcome says “risk” or `Uncertain`, never a confirmed diagnosis.
3. **Blur/no leaf:** reset selected scenario. Expected: quality gate rejects capture, retries are logged and output is uncertain/not healthy by default.
4. **Serial loss:** disconnect serial while moving. Expected: virtual motion stops and watchdog/fault event becomes visible; application does not continue driving the row.
5. **Radio loss:** select packet-loss scenario or disconnect LoRa. Expected: pending report / failed delivery is visible; base must not show an undelivered report as received.
6. **Node dropout:** expected offline/stale node indicator, with other nodes still reporting; old data is not relabelled current.
7. **Emergency stop:** activate while stopped and while running. Expected: virtual motor state false and speed zero. Clearing E-stop merely clears the latched state; it should not automatically start movement. Verify this manually after the run.
8. **API robustness:** malformed JSON, unknown route, invalid scenario/plant count/speed and oversized request should return a bounded error, not crash the server. Some edge cases need additional automated tests.
9. **Data persistence:** stop and restart the server with same data directory. Check whether historical UI should preserve multiple sessions; the current UI emphasises the active session, so historical browsing/session selection is a known future requirement.

## What is not proven by these tests

No test here validates actual disease classification, agronomic threshold correctness, LoRa/Espressif RF range, antenna tuning, regulatory compliance, physical emergency-stop wiring, actual motor stopping time, battery runtime, sensor calibration, sunlight/weather robustness, Raspberry Pi 4 2 GB memory/latency, or Tamil usability with target users. Track these as open hardware/field acceptance items.
