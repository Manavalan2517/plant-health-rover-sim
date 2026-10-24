"""Tests for the live-only monitor: empty by default, live packets only."""
from __future__ import annotations

import json
import threading
import unittest
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from plant_rover.live_mode import LiveDataHub
from plant_rover.live_server import LiveOnlyHTTPServer


TOKEN = "test-only-token-that-is-long-enough"


class LiveModeHubTests(unittest.TestCase):
    def test_ingestion_is_disabled_without_a_configured_secret(self):
        hub = LiveDataHub()
        self.assertFalse(hub.ingestion_enabled)
        self.assertFalse(hub.authenticate(TOKEN))
        self.assertEqual(hub.snapshot()["sensor_readings"], [])

    def test_initial_snapshot_contains_no_seeded_or_demo_values(self):
        snapshot = LiveDataHub(TOKEN).snapshot()
        self.assertEqual(snapshot["mode"], "live_only")
        self.assertEqual(snapshot["status"], "WAITING_FOR_LIVE_DATA")
        self.assertEqual(snapshot["devices"], [])
        self.assertEqual(snapshot["sensor_readings"], [])
        self.assertEqual(snapshot["inspections"], [])
        self.assertEqual(snapshot["events"], [])
        self.assertFalse(snapshot["device_summary"]["online"])

    def test_live_readings_appear_and_missing_fields_stay_missing(self):
        hub = LiveDataHub(TOKEN)
        self.assertTrue(hub.authenticate(TOKEN))
        from datetime import datetime, timezone
        hub.sensor_reading({
            "node_id": "FIELD-01", "sequence": 7, "timestamp": datetime.now(timezone.utc).isoformat(),
            "soil_moisture_pct": 42.5, "soil_temperature_c": 25.3,
            "air_temperature_c": 31.2, "relative_humidity_pct": 69.0,
            "battery_pct": 87, "transport": "esp_now",
        })
        snapshot = hub.snapshot()
        self.assertEqual(snapshot["status"], "LIVE")
        self.assertEqual(len(snapshot["sensor_readings"]), 1)
        reading = snapshot["sensor_readings"][0]
        self.assertEqual(reading["node_id"], "FIELD-01")
        self.assertEqual(reading["soil_moisture_pct"], 42.5)
        self.assertIsNone(reading["nitrogen_mg_kg"])
        self.assertEqual(reading["source"], "authenticated_live_hardware_bridge")

    def test_stale_sensor_values_are_not_shown_as_live(self):
        hub = LiveDataHub(TOKEN)
        from datetime import datetime, timedelta, timezone
        old = (datetime.now(timezone.utc) - timedelta(hours=2)).isoformat()
        hub.sensor_reading({"node_id": "FIELD-STALE", "timestamp": old,
            "air_temperature_c": 29, "relative_humidity_pct": 60})
        snapshot = hub.snapshot()
        self.assertEqual(snapshot["sensor_readings"], [])
        self.assertEqual(snapshot["device_summary"]["stale_sensor_nodes"], 1)

    def test_rejects_bad_ranges_and_out_of_order_packets(self):
        hub = LiveDataHub(TOKEN)
        from datetime import datetime, timezone
        now = datetime.now(timezone.utc).isoformat()
        with self.assertRaises(ValueError):
            hub.sensor_reading({"node_id": "FIELD-01", "timestamp": now,
                                "soil_moisture_pct": 120, "air_temperature_c": 20,
                                "relative_humidity_pct": 60})
        hub.sensor_reading({"node_id": "FIELD-01", "sequence": 2, "timestamp": now,
                            "air_temperature_c": 20, "relative_humidity_pct": 60})
        with self.assertRaises(ValueError):
            hub.sensor_reading({"node_id": "FIELD-01", "sequence": 1,
                                "timestamp": "2020-01-01T00:00:00Z",
                                "air_temperature_c": 20, "relative_humidity_pct": 60})


class LiveOnlyHTTPTests(unittest.TestCase):
    def setUp(self):
        self.hub = LiveDataHub(TOKEN)
        self.server = LiveOnlyHTTPServer(("127.0.0.1", 0), self.hub)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.base = f"http://127.0.0.1:{self.server.server_address[1]}"

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)

    def call(self, path, method="GET", body=None, token=None):
        data = None if body is None else json.dumps(body).encode()
        headers = {"Content-Type": "application/json"} if data is not None else {}
        if token:
            headers["Authorization"] = f"Bearer {token}"
        request = Request(self.base + path, data=data, method=method, headers=headers)
        with urlopen(request, timeout=3) as response:
            return response.status, json.loads(response.read().decode())

    def test_health_and_dashboard_are_live_only(self):
        status, body = self.call("/api/health")
        self.assertEqual(status, 200)
        self.assertEqual(body["mode"], "live_only")
        self.assertFalse(body["synthetic_data_enabled"])
        _, snapshot = self.call("/api/snapshot")
        self.assertEqual(snapshot["sensor_readings"], [])

    def test_html_contains_no_simulation_controls(self):
        with urlopen(self.base + "/", timeout=3) as response:
            html = response.read().decode()
        self.assertIn("Real device data only", html)
        self.assertIn("No live sensor readings", html)
        self.assertNotIn("scenarioSelect", html)
        self.assertNotIn("Reset scenario", html)

    def test_ingest_requires_bearer_token(self):
        request = Request(self.base + "/api/live/heartbeat", data=json.dumps({
            "device_id": "ROVER-PI", "device_type": "rover", "transport": "serial"
        }).encode(), headers={"Content-Type": "application/json"}, method="POST")
        with self.assertRaises(HTTPError) as ctx:
            urlopen(request, timeout=3)
        self.assertEqual(ctx.exception.code, 401)

    def test_ingest_updates_actual_live_state(self):
        status, body = self.call("/api/live/heartbeat", "POST", {
            "device_id": "ROVER-PI", "device_type": "rover", "transport": "serial", "firmware_version": "0.2"
        }, token=TOKEN)
        self.assertEqual(status, 202)
        self.assertTrue(body["accepted"])
        _, snapshot = self.call("/api/snapshot")
        self.assertEqual(snapshot["device_summary"]["online"], 1)
        self.assertEqual(snapshot["devices"][0]["device_id"], "ROVER-PI")

    def test_mutating_non_ingest_routes_is_not_allowed(self):
        request = Request(self.base + "/api/control", data=b'{"action":"start"}',
                          headers={"Content-Type": "application/json", "Authorization": f"Bearer {TOKEN}"}, method="POST")
        with self.assertRaises(HTTPError) as ctx:
            urlopen(request, timeout=3)
        self.assertEqual(ctx.exception.code, 405)


if __name__ == "__main__":
    unittest.main()
