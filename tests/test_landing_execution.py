"""Regression for the LAND_powered terminal-braking crash."""

import unittest

import numpy as np

from guidance.scheduling import GuidanceSchedule
from scenarios.descent.LAND_powered import Scenario
from sim.bodies import Earth
from sim.forces import Force, gravity, normal_force, torque_free_rotation
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
        contact_speed = None
        command = Force()
        for k in range(int(400 / scene.dt)):
            if k % scene.dt_between_gnc_updates == 0:
                command = sum(schedule.update(state, k*scene.dt).values(), start=Force())
                if schedule.curr_maneuver is None:
                    completed = True
            contact = normal_force(state, Earth, scene.spacecraft)
            if contact.normal_force_active and contact_speed is None:
                contact_speed = np.linalg.norm(state.vel())
            force = (gravity(state, Earth, scene.spacecraft) + contact
                     + torque_free_rotation(state, scene.spacecraft) + command)
            state.update(scene.dt, scene.spacecraft.mass, force)
            if completed and contact_speed is not None:
                break

        self.assertTrue(completed, landing.last_failure)
        self.assertIsNotNone(contact_speed)
        self.assertLess(contact_speed, 10.0)
        self.assertLessEqual(np.max(np.linalg.norm(landing.active_plan.U, axis=0)),
                             0.9 + 1e-6)
        self.assertTrue(landing.optimizer.validate(landing.active_plan))
        self.assertAlmostEqual(landing.active_plan.tf, 373.6)


if __name__ == "__main__":
    unittest.main()
