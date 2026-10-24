"""Local, read-only live rover dashboard with authenticated device ingestion and PC-based manual leaf inspection."""
from __future__ import annotations

import base64
import email
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
from typing import Any
from urllib.parse import urlparse, parse_qs

from .live_mode import LiveDataHub
from .ml.image_classifier import PlantLeafClassifier


class LiveOnlyHTTPServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, address: tuple[str, int], hub: LiveDataHub, classifier: PlantLeafClassifier | None = None) -> None:
        self.hub = hub
        self.classifier = classifier if classifier is not None else PlantLeafClassifier()
        super().__init__(address, LiveOnlyRequestHandler)


class LiveOnlyRequestHandler(BaseHTTPRequestHandler):
    server: LiveOnlyHTTPServer
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt: str, *args: Any) -> None:
        if self.path.startswith("/api/") and self.path != "/api/health":
            return
        super().log_message(fmt, *args)

    def _json(self, payload: Any, status: int = 200) -> None:
        body = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self) -> dict[str, Any]:
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0 or length > 64 * 1024:
                raise ValueError("request body must be between 1 byte and 64 KiB")
            value = json.loads(self.rfile.read(length).decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise ValueError("request body must be valid UTF-8 JSON") from exc
        if not isinstance(value, dict):
            raise ValueError("request body must be a JSON object")
        return value

    def _authenticate(self) -> bool:
        hub = self.server.hub
        if not hub.ingestion_enabled:
            self._json({"error": "live ingestion is disabled; set ROVER_INGEST_TOKEN to a random secret of at least 16 characters before accepting device messages"}, 503)
            return False
        token = self.headers.get("Authorization", "")
        if token.startswith("Bearer "):
            token = token[7:].strip()
        else:
            token = self.headers.get("X-Rover-Token", "").strip()
        if not hub.authenticate(token):
            self._json({"error": "unauthorized live device bridge"}, 401)
            return False
        return True

    def do_HEAD(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        if path in ("/", "/index.html", "/inspect"):
            file_path = Path(__file__).parent / "static" / "live.html"
            body = file_path.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
        else:
            self.send_response(200)
            self.end_headers()

    def do_GET(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        hub = self.server.hub
        if path in ("/", "/index.html", "/inspect"):
            file_path = Path(__file__).parent / "static" / "live.html"
            body = file_path.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Security-Policy", "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; connect-src 'self'; img-src 'self' data: blob:")
            self.end_headers()
            self.wfile.write(body)
            return
        if path.startswith("/static/samples/") or path.startswith("/samples/"):
            filename = Path(path).name
            sample_path = Path(__file__).parent / "static" / "samples" / filename
            if sample_path.exists() and sample_path.is_file():
                body = sample_path.read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "image/jpeg")
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "public, max-age=3600")
                self.end_headers()
                self.wfile.write(body)
                return
        if path == "/api/health":
            self._json({
                "ok": True,
                "mode": "live_only",
                "hardware_opened_by_dashboard": False,
                "ingestion_enabled": hub.ingestion_enabled,
                "synthetic_data_enabled": False,
                "manual_inspection_enabled": True,
                "model_status": "ready" if self.server.classifier.is_ready else "not_ready",
            })
        elif path in ("/api/snapshot", "/api/state"):
            self._json(hub.snapshot())
        elif path == "/api/readings":
            self._json(hub.snapshot()["sensor_readings"])
        elif path == "/api/reports":
            self._json(hub.snapshot()["inspections"])
        elif path == "/api/events":
            try:
                limit = min(100, max(1, int(parse_qs(urlparse(self.path).query).get("limit", ["50"])[0])))
            except ValueError:
                limit = 50
            self._json(hub.snapshot()["events"][:limit])
        elif path == "/api/inspect/model":
            self._json(self.server.classifier.get_model_info())
        elif path == "/api/inspect/history":
            self._json(self.server.hub.manual_inspections)
        elif path == "/api/healthz":
            self._json({"ok": True})
        else:
            self._json({"error": "not found"}, 404)

    def do_POST(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        if path.startswith("/api/inspect/"):
            self._handle_manual_inspection(path)
            return

        if not path.startswith("/api/live/"):
            self._json({"error": "this dashboard is read-only; only live device-bridge ingestion endpoints are available"}, 405)
            return
        if not self._authenticate():
            return
        try:
            body = self._read_json()
            if path == "/api/live/heartbeat":
                result = self.server.hub.heartbeat(body)
            elif path == "/api/live/link-status":
                result = self.server.hub.link_status(body)
            elif path == "/api/live/sensor-reading":
                result = self.server.hub.sensor_reading(body)
            elif path == "/api/live/rover-telemetry":
                result = self.server.hub.rover_telemetry_packet(body)
            elif path == "/api/live/inspection-report":
                result = self.server.hub.inspection_report(body)
            else:
                self._json({"error": "not found"}, 404)
                return
            self._json(result, 202)
        except ValueError as exc:
            self._json({"error": str(exc)}, 400)
        except Exception as exc:
            self._json({"error": f"request failed: {type(exc).__name__}"}, 500)

    def _handle_manual_inspection(self, path: str) -> None:
        if path not in ("/api/inspect/analyze", "/api/inspect/upload"):
            self._json({"error": "not found"}, 404)
            return

        ctype = self.headers.get("Content-Type", "")
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except (ValueError, TypeError):
            length = 0

        if length <= 0:
            self._json({"status": "IMAGE_REJECTED", "error": "Request body cannot be empty"}, 400)
            return
        if length > 10 * 1024 * 1024:
            remaining = length
            while remaining > 0:
                chunk = self.rfile.read(min(remaining, 64 * 1024))
                if not chunk:
                    break
                remaining -= len(chunk)
            self._json({"status": "IMAGE_REJECTED", "error": "File size exceeds 10 MB limit"}, 413)
            return

        body_bytes = self.rfile.read(length)
        image_bytes = None
        filename = "upload.jpg"

        if "multipart/form-data" in ctype:
            msg = email.message_from_bytes(b"Content-Type: " + ctype.encode("utf-8") + b"\r\n\r\n" + body_bytes)
            if msg.is_multipart():
                for part in msg.get_payload():
                    fname = part.get_filename()
                    if fname:
                        image_bytes = part.get_payload(decode=True)
                        filename = fname
                        break
                    elif part.get_content_type() and part.get_content_type().startswith("image/"):
                        image_bytes = part.get_payload(decode=True)
                        break
            else:
                image_bytes = msg.get_payload(decode=True)

        elif "application/json" in ctype:
            try:
                data = json.loads(body_bytes.decode("utf-8"))
                if not isinstance(data, dict):
                    raise ValueError
            except Exception:
                self._json({"status": "IMAGE_REJECTED", "error": "Request body must be valid JSON object"}, 400)
                return

            raw_b64 = data.get("image", "")
            if not raw_b64 or not isinstance(raw_b64, str):
                self._json({"status": "IMAGE_REJECTED", "error": "JSON payload must contain 'image' as base64 string"}, 400)
                return
            if "base64," in raw_b64:
                raw_b64 = raw_b64.split("base64,")[-1]
            try:
                image_bytes = base64.b64decode(raw_b64)
            except Exception:
                self._json({"status": "IMAGE_REJECTED", "error": "Invalid base64 encoding for image data"}, 400)
                return
            filename = data.get("filename", "upload.jpg")

        elif ctype.startswith("image/") or ctype == "application/octet-stream" or not ctype:
            image_bytes = body_bytes
            ext = ctype.split("/")[-1] if "/" in ctype else "jpg"
            filename = f"upload.{ext}"
        else:
            self._json({"status": "IMAGE_REJECTED", "error": f"Unsupported Content-Type '{ctype}'. Send multipart/form-data, application/json (base64), or image/jpeg."}, 415)
            return

        if not image_bytes:
            self._json({"status": "IMAGE_REJECTED", "error": "No image data found in request"}, 400)
            return

        classifier = self.server.classifier
        # If model is not ready
        if not classifier.is_ready:
            result = classifier.inspect_image(image_bytes, filename)
            self._json(result.to_dict(), 503)
            return

        result = classifier.inspect_image(image_bytes, filename)

        # Handle corrupt or unsupported format
        if result.status == "IMAGE_REJECTED":
            reason = result.quality_checks.get("rejection_reason") or ""
            if "Corrupted" in reason or "Unsupported format" in reason or "too small" in reason:
                self._json(result.to_dict(), 400)
                return
            if "exceeds maximum allowed" in reason and "MB limit" in reason:
                self._json(result.to_dict(), 413)
                return

        # Record manual inspection in hub session history
        self.server.hub.record_manual_inspection(result.to_dict())
        self._json(result.to_dict(), 200)


def serve(host: str = "127.0.0.1", port: int = 8765) -> None:
    if host not in ("127.0.0.1", "localhost", "::1"):
        print("WARNING: the live dashboard is not hardened for exposure to a LAN or the internet.")
    token = os.environ.get("ROVER_INGEST_TOKEN", "")
    hub = LiveDataHub(token)
    classifier = PlantLeafClassifier()
    server = LiveOnlyHTTPServer((host, port), hub, classifier)
    print("Plant-Health Inspection Rover — LIVE-ONLY MONITOR + IMAGE INSPECTION")
    print(f"Dashboard: http://{host}:{port}")
    print("Synthetic/demo data: DISABLED")
    print(f"Leaf Classifier: {'READY (' + classifier.backend + ')' if classifier.is_ready else 'NOT READY (Model unavailable)'}")
    print("Hardware connection: waiting for a real device bridge; the dashboard itself does not open hardware ports.")
    if not hub.ingestion_enabled:
        print("Device ingestion: DISABLED until ROVER_INGEST_TOKEN is set (minimum 16 characters).")
    else:
        print("Device ingestion: ENABLED for authenticated device-bridge messages.")
    try:
        server.serve_forever(poll_interval=0.25)
    except KeyboardInterrupt:
        print("\nStopping live-only dashboard...")
    finally:
        server.server_close()
