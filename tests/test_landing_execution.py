"""Regression for the LAND_powered terminal-braking crash."""

import unittest

import numpy as np

from guidance.scheduling import GuidanceSchedule
from scenarios.descent.LAND_powered import Scenario
from sim.bodies import Earth
from sim.forces import Force
from sim.state import State


class LandingExecutionTests(unittest.TestCase):
    def test_powered_scenario_lands_with_thrust_reserved_for_tracking(self):
        scene = Scenario()
        landing = scene.guidance_schedule.maneuvers[-1]
        landing.log = False
        # Captured just after the orbital braking/landing handoff. Include the
        # actual attitude, slew, pulse modulation, and held-force simulation:
        # validating only the ideal continuous-thrust plan missed this crash.
        state = State(
            [-428100.1097965819, 5962961.516346798, 2259597.706724197],
            [-2.305610083265785, -83.72439742905024, -35.12158534895929],
            [0.2395533954841939, 0.4446046324184773,
             -0.450346974761183, -0.7362937551463239],
            [-0.00039507765563426, -0.00039507971338552, 0.00096774409102775],
        )
        landing.arrival_time = 373.6
        control_dt = scene.dt * scene.dt_between_gnc_updates
        schedule = GuidanceSchedule([landing], state, 0, log=False,
                                    control_dt=control_dt)
        completed = False
        contact_seen = False
        command = Force()
        for k in range(int(400 / scene.dt)):
            if k % scene.dt_between_gnc_updates == 0:
                command = sum(schedule.update(state, k*scene.dt).values(), start=Force())
                if schedule.curr_maneuver is None:
                    completed = True
            scene.control_force = command
            scene.step(state)  # Impact guard checks the velocity at contact.
            if np.linalg.norm(state.pos()-Earth.pos_I) <= Earth.radius + 1e-7:
                contact_seen = True
            if completed and contact_seen:
                break

        self.assertTrue(completed, landing.last_failure)
        self.assertTrue(contact_seen)
        self.assertLess(np.linalg.norm(state.vel()), 10.0)
        self.assertLessEqual(np.max(np.linalg.norm(landing.active_plan.U, axis=0)),
                             0.9 + 1e-6)
        self.assertTrue(landing.optimizer.validate(landing.active_plan))
        self.assertAlmostEqual(landing.active_plan.tf, 373.6)


if __name__ == "__main__":
    unittest.main()
