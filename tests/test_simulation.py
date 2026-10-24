import tempfile
import unittest

from plant_rover.engine import SimulationEngine


class SimulationIntegrationTests(unittest.TestCase):
    def new_engine(self, scenario="healthy", plants=12):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        engine = SimulationEngine(tmp.name, seed=42, plant_count=plants)
        self.addCleanup(engine.repository.close)
        engine.reset(scenario)
        return engine

    def run_until_terminal(self, engine, limit=60):
        terminal = {"COMPLETE", "SAFETY_STOP", "EMERGENCY_STOP", "LOW_BATTERY_STOP"}
        for _ in range(limit):
            state = engine.step()
            if state["run_state"] in terminal:
                return state
        self.fail(f"simulation did not terminate within {limit} steps; state={engine.state()['run_state']}")

    def test_bootstrap_uses_espnow_and_lora_serialized_frames(self):
        engine = self.new_engine("healthy")
        state = engine.state()
        self.assertEqual(len(engine.repository.readings(engine.session_id, 1000)), 432)
        self.assertEqual(len(state["history_summaries"]), 3)
        self.assertGreater(engine.espnow_link.stats.attempts, 432)
        self.assertGreater(engine.lora_link.stats.accepted, 0)

    def test_full_daily_workflow_persists_base_reports_and_mock_sms(self):
        engine = self.new_engine("healthy", plants=5)
        state = self.run_until_terminal(engine)
        self.assertEqual(state["run_state"], "COMPLETE")
        self.assertEqual(state["plants_visited"], 5)
        self.assertEqual(state["base_reports_received"], 5)
        reports = engine.repository.reports(engine.session_id, 20)
        self.assertEqual(len(reports), 5)
        self.assertTrue(all(r["quality_passed"] for r in reports))
        outbox = engine.repository.outbox(engine.session_id, 10)
        self.assertTrue(any(item["channel"] == "sms_mock" for item in outbox))

    def test_drought_scenario_triggers_water_stress_risk_from_sensor_data(self):
        engine = self.new_engine("drought", plants=4)
        state = self.run_until_terminal(engine)
        self.assertEqual(state["run_state"], "COMPLETE")
        reports = engine.repository.reports(engine.session_id, 20)
        self.assertTrue(all(r["assessment"] == "water_stress_risk" for r in reports))
        self.assertLess(min(r["soil_moisture_pct"] for r in reports), 27)

    def test_blurry_capture_is_not_marked_healthy(self):
        engine = self.new_engine("blurry_capture", plants=3)
        state = self.run_until_terminal(engine)
        reports = engine.repository.reports(engine.session_id, 20)
        self.assertEqual(state["failed_captures"], 3)
        self.assertTrue(all(not r["quality_passed"] for r in reports))
        self.assertTrue(all(r["assessment"] == "uncertain" for r in reports))

    def test_serial_loss_triggers_watchdog_and_stops_virtual_motor(self):
        engine = self.new_engine("serial_loss", plants=10)
        state = self.run_until_terminal(engine)
        self.assertEqual(state["run_state"], "SAFETY_STOP")
        self.assertTrue(state["watchdog_tripped"])
        self.assertFalse(state["motor_enabled"])
        self.assertLess(state["plants_visited"], 10)

    def test_low_battery_causes_safe_stop(self):
        engine = self.new_engine("low_battery", plants=10)
        state = self.run_until_terminal(engine)
        self.assertEqual(state["run_state"], "LOW_BATTERY_STOP")
        self.assertLessEqual(state["rover_battery_pct"], 10)
        self.assertFalse(state["motor_enabled"])

    def test_capture_failure_records_uncertain_reports(self):
        engine = self.new_engine("capture_failure", plants=2)
        state = self.run_until_terminal(engine)
        reports = engine.repository.reports(engine.session_id, 10)
        self.assertEqual(state["base_reports_received"], 2)
        self.assertTrue(all(r["assessment"] == "uncertain" for r in reports))

    def test_disconnected_field_node_is_visible_but_other_nodes_continue(self):
        engine = self.new_engine("node_dropout", plants=6)
        for _ in range(4):
            engine.step()
        state = engine.state()
        node2 = next(node for node in state["sensor_nodes"] if node["node_id"] == 2)
        self.assertEqual(node2["status"], "stale/offline")
        self.assertTrue(any(node["status"] == "online" for node in state["sensor_nodes"] if node["node_id"] != 2))

    def test_configurable_plant_count_resets_session(self):
        engine = self.new_engine("healthy", plants=3)
        old_session = engine.session_id
        state = engine.set_plant_count(7)
        self.assertNotEqual(old_session, state["session_id"])
        self.assertEqual(state["plants_total"], 7)


if __name__ == "__main__":
    unittest.main()
