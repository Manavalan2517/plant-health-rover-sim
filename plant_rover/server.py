"""Local-only HTTP server for the PC simulation dashboard; no cloud is required."""
from __future__ import annotations

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import json
import mimetypes
from urllib.parse import urlparse, parse_qs
from typing import Any

from .engine import SimulationEngine


class RoverHTTPServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, address: tuple[str, int], engine: SimulationEngine) -> None:
        self.engine = engine
        super().__init__(address, RoverRequestHandler)


class RoverRequestHandler(BaseHTTPRequestHandler):
    server: RoverHTTPServer
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt: str, *args: Any) -> None:
        # Keep normal dashboard polling from flooding the terminal.
        if self.path.startswith("/api/") and not self.path.startswith("/api/health"):
            return
        super().log_message(fmt, *args)

    def _json(self, payload: Any, status: int = 200) -> None:
        body = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self) -> dict[str, Any]:
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length < 0 or length > 64 * 1024:
                raise ValueError("request body too large")
            raw = self.rfile.read(length) if length else b"{}"
            value = json.loads(raw.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise ValueError("request body must be valid UTF-8 JSON") from exc
        if not isinstance(value, dict):
            raise ValueError("request body must be a JSON object")
        return value

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        path = parsed.path
        engine = self.server.engine
        if path == "/" or path == "/index.html":
            file_path = Path(__file__).parent / "static" / "index.html"
            body = file_path.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)
            return
        if path == "/api/health":
            self._json({"ok": True, "mode": "simulation", "hardware_access": False})
        elif path == "/api/state":
            self._json(engine.state())
        elif path == "/api/snapshot":
            self._json(engine.api_payload())
        elif path == "/api/events":
            limit = min(500, max(1, int(parse_qs(parsed.query).get("limit", ["100"])[0])))
            self._json(engine.repository.events(engine.session_id, limit))
        elif path == "/api/reports":
            limit = min(200, max(1, int(parse_qs(parsed.query).get("limit", ["100"])[0])))
            self._json(engine.repository.reports(engine.session_id, limit))
        elif path == "/api/readings":
            limit = min(1000, max(1, int(parse_qs(parsed.query).get("limit", ["100"])[0])))
            self._json(engine.repository.readings(engine.session_id, limit))
        elif path == "/api/outbox":
            self._json(engine.repository.outbox(engine.session_id, 100))
        elif path == "/api/trace":
            self._json(engine.export_replay_trace())
        elif path.startswith("/images/"):
            # Only serve files within the simulator's generated-image directory.
            candidate = (engine.root_dir / "images" / path.removeprefix("/images/")).resolve()
            image_root = (engine.root_dir / "images").resolve()
            if image_root not in candidate.parents or not candidate.is_file():
                self._json({"error": "image not found"}, 404)
                return
            body = candidate.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", mimetypes.guess_type(candidate.name)[0] or "application/octet-stream")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)
        else:
            self._json({"error": "not found"}, 404)

    def do_POST(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        engine = self.server.engine
        try:
            body = self._read_json()
            if path == "/api/control":
                action = body.get("action")
                if action == "start":
                    engine.start()
                elif action == "pause":
                    engine.pause()
                elif action == "resume":
                    engine.resume()
                elif action == "step":
                    engine.step()
                elif action == "reset":
                    engine.reset(body.get("scenario", engine.scenario))
                elif action == "estop":
                    engine.activate_emergency_stop()
                elif action == "clear_estop":
                    engine.clear_emergency_stop()
                elif action == "plants":
                    engine.set_plant_count(int(body.get("count", engine.plant_count)))
                elif action == "speed":
                    engine.set_speed(float(body.get("speed", 1.0)))
                elif action == "language":
                    engine.set_language(str(body.get("language", "en")))
                else:
                    raise ValueError("action must be start, pause, resume, step, reset, estop, clear_estop, plants, speed, or language")
                self._json({"ok": True, "state": engine.state()})
            elif path == "/api/fault":
                result = engine.set_fault(str(body.get("link", "")), bool(body.get("enabled", False)))
                self._json({"ok": True, "state": result})
            else:
                self._json({"error": "not found"}, 404)
        except ValueError as exc:
            self._json({"error": str(exc)}, 400)
        except Exception as exc:
            self._json({"error": f"request failed: {type(exc).__name__}: {exc}"}, 500)


def serve(root_dir: str | Path, host: str = "127.0.0.1", port: int = 8765, seed: int = 2026, plant_count: int = 12) -> None:
    engine = SimulationEngine(root_dir=root_dir, seed=seed, plant_count=plant_count)
    server = RoverHTTPServer((host, port), engine)
    print("Plant-Health Inspection Rover — PC Simulation")
    print(f"Dashboard: http://{host}:{port}")
    print(f"Data: {Path(root_dir).resolve()}")
    print("Simulation-only mode: no serial ports, GPIO, radios or hardware drivers are opened.")
    try:
        server.serve_forever(poll_interval=0.25)
    except KeyboardInterrupt:
        print("\nStopping simulation server...")
    finally:
        engine.pause()
        server.server_close()
        engine.repository.close()
