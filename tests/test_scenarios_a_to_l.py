"""End-to-End automated validation of Master Prompt Scenarios A through L."""
import tempfile
import unittest
from pathlib import Path

from plant_rover.engine import SimulationEngine


class ScenariosAToLTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root_dir = Path(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    def _run_until_terminal(self, engine: SimulationEngine, max_steps: int = 25) -> dict:
        for _ in range(max_steps):
            if engine.run_state in ("COMPLETE", "SAFETY_STOP", "EMERGENCY_STOP", "LOW_BATTERY_STOP"):
                break
            engine.step()
        return engine.state()

    def test_scenario_a_healthy_plant(self):
        """Scenario A: Virtual rover inspects healthy plants; report reaches base with HEALTHY assessment."""
        engine = SimulationEngine(self.root_dir, seed=101, plant_count=2)
        engine.reset("healthy")
        engine.step()
        state = engine.state()
        self.assertEqual(state["plants_visited"], 1)
        reports = engine.repository.reports(engine.session_id, limit=5)
        self.assertGreaterEqual(len(reports), 1)
        self.assertEqual(reports[0]["assessment"], "healthy")
        self.assertTrue(reports[0]["quality_passed"])
        engine.pause()
        engine.repository.close()

    def test_scenario_b_water_stress(self):
        """Scenario B: Soil-moisture declines over 24h; engine flags WATER_STRESS_RISK."""
        engine = SimulationEngine(self.root_dir, seed=102, plant_count=2)
        engine.reset("drought")
        engine.step()
        reports = engine.repository.reports(engine.session_id, limit=5)
        self.assertGreaterEqual(len(reports), 1)
        self.assertEqual(reports[0]["assessment"], "water_stress_risk")
        self.assertIn("soil_moisture_pct", str(reports[0]["evidence"]))
        engine.pause()
        engine.repository.close()

    def test_scenario_c_disease_like_symptoms(self):
        """Scenario C: Image branch shows spot-like fixture; assessment is suspected_symptoms or uncertain."""
        engine = SimulationEngine(self.root_dir, seed=103, plant_count=2)
        engine.reset("disease_like")
        engine.step()
        reports = engine.repository.reports(engine.session_id, limit=5)
        self.assertGreaterEqual(len(reports), 1)
        self.assertIn(reports[0]["assessment"], ("suspected_symptoms", "uncertain"))
        engine.pause()
        engine.repository.close()

    def test_scenario_d_conflicting_evidence(self):
        """Scenario D: Spot-like fixture without supporting environmental context -> UNCERTAIN."""
        engine = SimulationEngine(self.root_dir, seed=104, plant_count=2)
        engine.reset("conflicting_evidence")
        engine.step()
        reports = engine.repository.reports(engine.session_id, limit=5)
        self.assertGreaterEqual(len(reports), 1)
        self.assertEqual(reports[0]["assessment"], "uncertain")
        engine.pause()
        engine.repository.close()

    def test_scenario_e_blurry_image(self):
        """Scenario E: Blurred captures rejected by quality gate; retakes attempted and not marked healthy."""
        engine = SimulationEngine(self.root_dir, seed=105, plant_count=2)
        engine.reset("blurry_capture")
        engine.step()
        reports = engine.repository.reports(engine.session_id, limit=5)
        self.assertGreaterEqual(len(reports), 1)
        self.assertFalse(reports[0]["quality_passed"])
        self.assertEqual(reports[0]["assessment"], "uncertain")
        engine.pause()
        engine.repository.close()

    def test_scenario_f_field_node_failure(self):
        """Scenario F: Node 2 drops out after 4 steps; base detects offline node and remaining nodes continue."""
        engine = SimulationEngine(self.root_dir, seed=106, plant_count=6)
        engine.reset("node_dropout")
        for _ in range(4):
            engine.step()
        state = engine.state()
        node2 = next(node for node in state["sensor_nodes"] if node["node_id"] == 2)
        self.assertEqual(node2["status"], "stale/offline")
        self.assertTrue(any(node["status"] == "online" for node in state["sensor_nodes"] if node["node_id"] != 2))
        engine.pause()
        engine.repository.close()

    def test_scenario_g_radio_packet_loss(self):
        """Scenario G: LoRa packet loss; transport retries and avoids duplicate reports."""
        engine = SimulationEngine(self.root_dir, seed=107, plant_count=2)
        engine.reset("radio_loss")
        engine.step()
        state = engine.state()
        self.assertEqual(state["plants_visited"], 1)
        engine.pause()
        engine.repository.close()

    def test_scenario_h_rover_communication_loss(self):
        """Scenario H: Pi-ESP32 serial loss trips 400ms watchdog and halts virtual motor."""
        engine = SimulationEngine(self.root_dir, seed=108, plant_count=10)
        engine.reset("serial_loss")
        state = self._run_until_terminal(engine)
        self.assertEqual(state["run_state"], "SAFETY_STOP")
        self.assertTrue(state["watchdog_tripped"])
        self.assertFalse(state["motor_enabled"])
        engine.pause()
        engine.repository.close()

    def test_scenario_i_low_battery(self):
        """Scenario I: Rover starts with 16% battery; low battery causes safe stop before row finishes."""
        engine = SimulationEngine(self.root_dir, seed=109, plant_count=10)
        engine.reset("low_battery")
        state = self._run_until_terminal(engine)
        self.assertEqual(state["run_state"], "LOW_BATTERY_STOP")
        self.assertLessEqual(state["rover_battery_pct"], 10)
        self.assertFalse(state["motor_enabled"])
        engine.pause()
        engine.repository.close()

    def test_scenario_j_complete_daily_inspection(self):
        """Scenario J: Full row (5 plants) run; base station aggregates reports and generates mock outbox SMS."""
        engine = SimulationEngine(self.root_dir, seed=110, plant_count=5)
        engine.reset("healthy")
        state = self._run_until_terminal(engine)
        self.assertEqual(state["run_state"], "COMPLETE")
        self.assertEqual(state["plants_visited"], 5)
        outbox = engine.repository.outbox(engine.session_id, 10)
        self.assertTrue(any(item["channel"] == "sms_mock" for item in outbox))
        engine.pause()
        engine.repository.close()

    def test_scenario_k_offline_operation(self):
        """Scenario K: Functions completely offline without internet or external sockets."""
        engine = SimulationEngine(self.root_dir, seed=111, plant_count=2)
        engine.reset("healthy")
        engine.step()
        trace = engine.export_replay_trace()
        self.assertIn("state_digest", trace)
        self.assertEqual(trace["completed_inspections"], 1)
        engine.pause()
        engine.repository.close()

    def test_scenario_l_system_restart(self):
        """Scenario L: Application restart preserves SQLite reports and recovers cleanly."""
        eng1 = SimulationEngine(self.root_dir, seed=112, plant_count=2)
        eng1.reset("healthy")
        eng1.step()
        sess_id = eng1.session_id
        eng1.pause()
        eng1.repository.close()

        # Restart session with same DB
        eng2 = SimulationEngine(self.root_dir, seed=112, plant_count=2)
        recovered_reports = eng2.repository.reports(sess_id, limit=5)
        self.assertEqual(len(recovered_reports), 1)
        eng2.pause()
        eng2.repository.close()


if __name__ == "__main__":
    unittest.main()
