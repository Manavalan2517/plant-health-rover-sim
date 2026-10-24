"""HTTP smoke tests for the actual local dashboard API."""
from __future__ import annotations

import json
import tempfile
import threading
import unittest
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from plant_rover.engine import SimulationEngine
from plant_rover.server import RoverHTTPServer


class DashboardHTTPTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.engine = SimulationEngine(self.temp.name, seed=7, plant_count=2)
        self.server = RoverHTTPServer(("127.0.0.1", 0), self.engine)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.base = f"http://127.0.0.1:{self.server.server_address[1]}"

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)
        self.engine.repository.close()
        self.temp.cleanup()

    def request_json(self, path, method="GET", payload=None):
        body = None if payload is None else json.dumps(payload).encode("utf-8")
        req = Request(self.base + path, data=body, method=method,
                      headers={"Content-Type": "application/json"} if body is not None else {})
        with urlopen(req, timeout=3) as response:
            raw = response.read()
            return response.status, response.headers.get("Content-Type", ""), raw

    def test_health_endpoint_explicitly_reports_simulation_only(self):
        status, content_type, raw = self.request_json("/api/health")
        body = json.loads(raw)
        self.assertEqual(status, 200)
        self.assertIn("application/json", content_type)
        self.assertEqual(body["mode"], "simulation")
        self.assertFalse(body["hardware_access"])

    def test_dashboard_serves_real_html(self):
        with urlopen(self.base + "/", timeout=3) as response:
            html = response.read().decode("utf-8")
            self.assertEqual(response.status, 200)
        self.assertIn("Plant-Health Inspection Rover", html)
        self.assertIn("/api/snapshot", html)
        self.assertIn("Disconnect ESP-NOW", html)

    def test_reset_scenario_uses_engine_not_hardcoded_sample(self):
        status, _, raw = self.request_json("/api/control", "POST", {"action": "reset", "scenario": "drought"})
        state = json.loads(raw)["state"]
        self.assertEqual(status, 200)
        self.assertEqual(state["scenario"], "drought")
        self.assertTrue(state["simulation_mode"])
        self.assertEqual(state["base_reports_received"], 0)

    def test_invalid_control_returns_bounded_client_error(self):
        with self.assertRaises(HTTPError) as ctx:
            self.request_json("/api/control", "POST", {"action": "run_motors_for_real"})
        self.assertEqual(ctx.exception.code, 400)
        payload = json.loads(ctx.exception.read())
        self.assertIn("error", payload)

    def test_fault_endpoint_changes_simulated_link_state(self):
        status, _, raw = self.request_json("/api/fault", "POST", {"link": "serial", "enabled": False})
        body = json.loads(raw)
        self.assertEqual(status, 200)
        self.assertFalse(body["state"]["links"]["serial"]["connected"])

    def test_trace_endpoint_returns_valid_digest(self):
        status, content_type, raw = self.request_json("/api/trace")
        self.assertEqual(status, 200)
        self.assertIn("application/json", content_type)
        body = json.loads(raw)
        self.assertIn("session_id", body)
        self.assertIn("state_digest", body)
        self.assertEqual(len(body["state_digest"]), 64)  # SHA-256 hex length
        self.assertIn("timestamp_utc", body)


if __name__ == "__main__":
    unittest.main()
