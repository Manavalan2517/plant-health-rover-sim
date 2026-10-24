# Plant-Health Inspection Rover — Live-Only Monitor

**Default runtime: LIVE ONLY. Synthetic/demo readings and scenario generation are disabled in the user-facing application.**

This build is a read-only local dashboard for authenticated messages forwarded from real rover, ESP32, sensor-node, camera, base-station and link-status adapters. It does not simulate devices, generate leaf images, invent readings, assume a battery percentage, estimate a plant condition, or fill missing measurements with zero.

## Important: no hardware is connected yet

Until physical hardware and a device bridge are configured, the dashboard is expected to show:

- **Waiting for live data**
- No devices discovered / no live heartbeat
- No current sensor readings
- No rover telemetry
- No inspection reports
- No live event history

That is the correct result for the current setup. Empty fields mean *unknown or not received*, not zero and not healthy.

## Run on Ubuntu / Linux

```bash
cd plant-health-rover-sim
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m unittest discover -v
python -m plant_rover.cli serve --host 127.0.0.1 --port 8765
```

Open **http://127.0.0.1:8765**. Stop with `Ctrl+C`.

This command serves only the live-monitor page and its live-only API. It does not import/start the simulation engine, open serial/GPIO/radio/camera devices, connect to cloud services or load a prior simulation database. The dashboard refreshes every two seconds and shows only values received from the local live-data API.

## Ingestion is closed by default

Real device data can be forwarded by a trusted local bridge through authenticated endpoints. Until you configure a strong token, all ingestion requests are rejected. No authentication secret is included in this repository.

Generate and export a token before starting the dashboard:

```bash
export ROVER_INGEST_TOKEN="$(python3 -c 'import secrets; print(secrets.token_urlsafe(32))')"
python -m plant_rover.cli serve --host 127.0.0.1 --port 8765
```

Keep the token private. Bind to `127.0.0.1`; do not expose this development server to the LAN or internet. Your future trusted hardware bridge must pass the token as `Authorization: Bearer <token>` when posting genuine messages.

### Live bridge endpoints

All payloads must be derived from real device packets. Endpoints return `202 Accepted` only after validation.

- `POST /api/live/heartbeat` — `{ "device_id": "ROVER-PI", "device_type": "rover", "transport": "serial", "firmware_version": "..." }`
- `POST /api/live/link-status` — `{ "link_id": "serial", "source_device_id": "ROVER-PI", "connected": true, "timestamp": "2026-10-22T10:00:00Z" }`
- `POST /api/live/sensor-reading` — `{ "node_id": "FIELD-01", "timestamp": "2026-10-22T10:00:00Z", "air_temperature_c": 31.2, "relative_humidity_pct": 69.0 }` plus any actually measured optional fields
- `POST /api/live/rover-telemetry` — a real rover telemetry packet with `device_id` and timezone-aware `timestamp`, plus the values actually reported
- `POST /api/live/inspection-report` — a real device report with `report_id`, `device_id`, `plant_id`, `timestamp`, `assessment`, `confidence_pct`, `image_quality_passed`, and `model_version`

Example request after you have a genuine source message available:

```bash
curl -X POST http://127.0.0.1:8765/api/live/sensor-reading \
  -H "Authorization: Bearer $ROVER_INGEST_TOKEN" \
  -H 'Content-Type: application/json' \
  -d '{"node_id":"FIELD-01","timestamp":"2026-10-22T10:00:00Z","air_temperature_c":31.2,"relative_humidity_pct":69.0}'
```

**Do not run that example with invented values as a test.** It is a schema example only; posting it would add a live-looking record even if its values were not truly measured. For the current hardware-free setup, do not post any data.

## Live freshness policy

- Rover telemetry older than 12 seconds is excluded from the current rover panel.
- Sensor measurements older than 15 minutes are excluded from the current sensor panel. Stale node count is shown separately.
- Link status expires using per-link time limits; absent status is shown as **No observation**, never silently assumed connected.
- Incoming measurements must carry timezone-aware timestamps. Obvious invalid ranges, future timestamps and out-of-order sensor packets are rejected.
- Optional values are displayed as `—` until they are actually measured. They are never filled with zero.
- Device-reported model results are shown as reports, not treated as independently validated diagnoses.
- Values are kept in memory for this live monitor and are cleared when it restarts; this prevents old packets from appearing to be current after restart.

## What this build does not do yet

No physical hardware is connected, and this build does not yet implement a physical serial/radio polling bridge. It provides the authenticated live-message boundary and dashboard state handling only. To get actual live readings later, implement and validate the device bridge that reads real hardware packets and forwards them to the endpoints above. Do not hand-enter or generate test measurements in live mode.

The older simulation engine and synthetic fixtures remain in the codebase for automated software tests of protocol and domain logic only. The default CLI and live dashboard do not expose or start those scenarios. They are not real data and must not be used for field decisions or claimed as hardware validation.

## Tests

```bash
python -m unittest discover -v
python -m compileall -q plant_rover tests
```

Tests may intentionally create test packets inside isolated temporary memory/processes; those fixtures are test-only and are never loaded by the live dashboard.

## Manual Leaf Image Inspection (PC Only)

In addition to live monitoring, the application provides a **Manual Leaf Image Inspection** workflow that runs entirely on your PC without requiring connected rover hardware.

### Key Capabilities
- **Two Distinct Workflows**:
  - **Live Monitoring Tab**: Shows real-time device telemetry only. When no rover/sensors are connected, it remains in the `WAITING FOR LIVE DATA` state with empty/unavailable sensor readings.
  - **Manual Leaf Inspection Tab**: Upload an infected or potentially infected leaf image for local AI inference.
- **Strict Isolation**: Image uploads never populate the live telemetry store, never invent synthetic sensor readings (soil moisture, temperature, humidity), and never alter the rover connection state.
- **Image Quality Gate**:
  - File size: Up to 10 MB.
  - Dimension check: Minimum 64×64 px, maximum 4096×4096 px. Decompression bomb limits (`MAX_IMAGE_PIXELS = 17,895,697`).
  - Supported formats: JPEG, PNG, WebP. Magic bytes and PIL decoding validation (filenames/Content-Types are never blindly trusted).
  - EXIF orientation auto-transpose and RGB normalization.
  - Laplacian blur check (inner border cropped) and luminance exposure checking (underexposed/overexposed detection).
- **MobileNetV2 Classifier**:
  - Input resolution: 224×224 px.
  - Preprocessing: Strict MobileNetV2 scaling to $[-1.0, 1.0]$ (`(x / 127.5) - 1.0`).
  - Dual inference backend: Default high-speed ONNX Runtime (`plant_rover/models/plant_disease_mobilenetv2.onnx`) with automatic TensorFlow/Keras fallback (`plant_disease_mobilenetv2.keras`).
  - Supported classes (10 classes): Bacterial Spot, Early Blight, Late Blight, Leaf Mold, Septoria Leaf Spot, Spider Mites (Two-Spotted Spider Mite), Target Spot, Tomato Yellow Leaf Curl Virus, Tomato Mosaic Virus, Healthy.
- **Uncertainty & Out-of-Distribution Handling**:
  - Calibrated policy: Predictions with top-1 confidence $< 0.65$ or top-1/top-2 margin $< 0.15$ are flagged as `UNCERTAIN` instead of presenting weak guesses as facts.
  - Distinct statuses: `ASSESSED`, `UNCERTAIN`, `REJECTED_QUALITY`, `MODEL_UNAVAILABLE`.
  - Findings are explicitly identified as *suspected visual findings*, not confirmed agronomic diagnoses.
  - Prominent disclaimer: *"Environmental sensor data unavailable — this is an image-only assessment."*

### Image Inspection API Endpoints

- `POST /api/inspect/analyze`: Accepts `multipart/form-data` (`image` file field), `application/json` (with base64 `image_base64`), or raw image binary.
  Returns complete structured JSON assessment with quality gate metrics, top prediction, confidence, alternative plausible conditions, uncertainty analysis, and sensor disclaimer.
- `GET /api/inspect/model`: Returns model metadata, status (`READY` or `MODEL_UNAVAILABLE`), architecture, input shape, preprocessing formula, and supported classes list.
- `GET /api/inspect/history`: Returns in-memory inspection history for the current session.

### Training the Classifier

To train or retrain the MobileNetV2 classifier from an augmented tomato leaf dataset:

```bash
# Ensure training dependencies are installed (TensorFlow, Keras, scikit-learn, tf2onnx)
python -m plant_rover.ml.train_classifier \
    --data-dir ../dataset/augmented_dataset \
    --output-dir plant_rover/models \
    --epochs-frozen 10 \
    --epochs-fine-tune 5 \
    --batch-size 32 \
    --seed 42
```

The training script automatically:
1. Audits dataset structure and validates 10 class directories.
2. Performs group-based chunk splitting (groups of 10 adjacent augmented variants) to strictly prevent data leakage across training, validation, and test partitions.
3. Applies training-only data augmentation.
4. Trains a MobileNetV2 transfer learning backbone with frozen feature extractor, followed by controlled fine-tuning.
5. Evaluates the model honestly on 600 unseen test samples (producing accuracy, macro F1, per-class metrics, confusion matrix, and threshold evaluation).
6. Exports `plant_disease_mobilenetv2.keras`, `plant_disease_mobilenetv2.onnx`, `class_manifest.json`, and `model_metadata.json`.

## Architecture and safety notes

- Keep the web server local-only. It is a development server, not production-hardened.
- The dashboard is read-only and cannot issue motor commands or clear an emergency stop.
- Software status is not a replacement for a separate physical motor-power cutoff or tested firmware watchdog.
- The supplied architecture's RA-02/SX1278 radio selection appears incompatible with the proposed 865–867 MHz band; resolve the module, antenna and applicable Indian requirements before purchasing or transmitting.
- No claim is made that disease detection, model confidence, radio range, motor safety, sensor calibration or Raspberry Pi performance is validated. Real-world laboratory or field conditions may cause variations in disease appearance not represented in training datasets.

