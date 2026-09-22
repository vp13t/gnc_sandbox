import unittest
from unittest.mock import Mock

import numpy as np

from guidance.descent.brake_maneuver import BrakeManeuver
from guidance.descent.descent_optimizer import DescentLimits, DescentOptimizer
from guidance.descent.land_maneuver import LandManeuver
from guidance.descent.powered_descent_maneuver import DescentFailure
from guidance.descent.trajectory_plan import TrajectoryPlan
from sim.bodies import Earth
from sim.state import State
from spacecraft.sounding_rocket import SoundingRocket


class TrajectoryTests(unittest.TestCase):
    def test_control_integral_across_nodes_and_plan_boundary(self):
        plan = TrajectoryPlan(10, 2, 10, np.zeros((6, 3)), np.array([[1, 0], [0, 1], [0, 0]]))
        np.testing.assert_allclose(plan.average_control(11, 14), [1/3, 2/3, 0])
        np.testing.assert_allclose(plan.average_control(9, 11), [0.5, 0, 0])
        np.testing.assert_allclose(plan.average_control(14, 15), 0)

    def test_plan_owns_immutable_arrays_and_interpolates(self):
        x = np.zeros((6, 3))
        x[:, 1], x[:, 2] = 2, 4
        plan = TrajectoryPlan(0, 2, 0, x, np.zeros((3, 2)))
        x[:] = 100
        np.testing.assert_allclose(plan.state_at(1), 1)
        with self.assertRaises(ValueError):
            plan.X[0, 0] = 5

    def test_coast_looks_ahead_without_following_pulse_residual(self):
        plan = TrajectoryPlan(0, 1, 1, np.zeros((6, 4)), np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]]))
        np.testing.assert_allclose(plan.pointing_direction(0, [1, 0, 0]), [0, 1, 0])


class ExecutionTests(unittest.TestCase):
    def setUp(self):
        self.craft = SoundingRocket()
        self.m = BrakeManeuver(np.array([1., 0, 0]), 20000, Earth, self.craft, 1200, log=False)
        self.state = State(self.m.target + [500, 0, 0], [0, 0, 0], [0, 0, 0, 1], [0, 0, 0])

    def fake_plan(self):
        x = np.repeat(np.r_[self.state.pos(), [1000, 0, 0]][:, None], 101, axis=1)
        return TrajectoryPlan(0, 1, 0, x, np.zeros((3, 100)))

    def test_rejected_replans_preserve_plan_deadline_and_rate_limit(self):
        original = self.fake_plan()
        self.m._accept_plan(original)
        self.m.optimize = Mock(return_value=None)
        for t in [0, 1, 2, 9, 10]:
            self.m.act(self.state, t)
        self.assertEqual(self.m.optimize.call_count, 2)
        self.assertIs(self.m.active_plan, original)
        self.assertEqual(self.m.arrival_time, 100)
        self.assertEqual(self.m.last_replan_attempt, 10)

    def test_acceptance_resets_debt_and_disallows_deadline_drift(self):
        original = self.fake_plan()
        self.m._accept_plan(original)
        self.m.pending_dv[:] = 99
        replacement = TrajectoryPlan(10, 1, 10, original.X[:, :91], original.U[:, :90])
        self.m._accept_plan(replacement)
        self.assertIs(self.m.active_plan, replacement)
        np.testing.assert_allclose(self.m.pending_dv, 0)
        with self.assertRaises(ValueError):
            self.m._accept_plan(TrajectoryPlan(11, 1, 11, original.X[:, :91], original.U[:, :90]))

    def test_pulses_conserve_delta_v_at_nonunit_control_period(self):
        self.m.control_dt = 0.25
        accel = self.m.thruster.force / self.m.spacecraft.mass
        requested = np.zeros(3)
        delivered = np.zeros(3)
        for _ in range(20):
            u = np.array([0.3, 0, 0])
            requested += accel*u*self.m.control_dt
            force = self.m._pulse(self.state, np.array([1, 0, 0]), u, True)
            delivered += np.array([force.xddot, force.yddot, force.zddot])*self.m.control_dt
        np.testing.assert_allclose(self.m.pending_dv, requested-delivered, atol=1e-12)
        self.assertLessEqual(np.linalg.norm(self.m.pending_dv), 0.5*accel*self.m.control_dt+1e-12)

    def test_pulse_subtracts_actual_body_direction(self):
        self.m.control_dt = 0.25
        accel = self.m.thruster.force / self.m.spacecraft.mass
        self.m._aligned = Mock(return_value=True)
        u = np.array([0.8, 0.6, 0])
        self.m._pulse(self.state, u, u, True)
        expected = accel*self.m.control_dt*(u-np.array([1, 0, 0]))
        np.testing.assert_allclose(self.m.pending_dv, expected, atol=1e-12)

    def test_no_thrust_before_ignition(self):
        force = self.m._pulse(self.state, np.array([1, 0, 0]), np.ones(3), False)
        self.assertEqual(force.xddot, 0)

    def test_deadline_is_not_completion(self):
        self.m._accept_plan(self.fake_plan())
        self.assertFalse(self.m.check_complete(self.state, 100))
        with self.assertRaises(DescentFailure):
            self.m.act(self.state, 100)

    def test_handoff_requires_actual_state_and_successor_plan(self):
        successor = Mock()
        successor.prepare_handoff.return_value = False
        self.m.landing_maneuver = successor
        state = State(self.m.target + [10, 0, 0], [-5, 0, 0], [0, 0, 0, 1], [0, 0, 0])
        self.assertFalse(self.m.check_complete(state, 10))
        successor.prepare_handoff.return_value = True
        self.assertTrue(self.m.check_complete(state, 20))

    def test_landing_uses_the_same_executor(self):
        landing = LandManeuver([1, 0, 0], np.pi/2, Earth, self.craft, 60, log=False)
        self.assertEqual(landing.limits.speed, 1.0)
        self.assertEqual(type(landing)._pulse, type(self.m)._pulse)

    def test_landing_completion_accepts_surface_below_approach_cone_apex(self):
        landing = LandManeuver([1, 0, 0], np.pi/3, Earth, self.craft, 60, log=False)
        surface = State(Earth.pos_I + [Earth.radius, 0, 0], [-0.2, 0, 0],
                        [0, 0, 0, 1], [0, 0, 0])
        self.assertFalse(landing._inside_corridor(surface.pos()))
        self.assertTrue(landing.check_complete(surface, 60))
        too_fast = State(surface.pos(), [-2, 0, 0], [0, 0, 0, 1], [0, 0, 0])
        self.assertFalse(landing.check_complete(too_fast, 60))
        too_far = State(surface.pos() + [0, 2, 0], [0, 0, 0],
                        [0, 0, 0, 1], [0, 0, 0])
        self.assertFalse(landing.check_complete(too_far, 60))

    def test_candidate_generation_does_not_replace_active_state(self):
        candidate = self.fake_plan()
        self.m.optimizer.solve = Mock(return_value=candidate)
        self.m._slew_time = Mock(return_value=0)
        self.assertIs(self.m.optimize(self.state, 0), candidate)
        self.assertIsNone(self.m.active_plan)
        self.assertIsNone(self.m.arrival_time)

    def test_slew_delay_is_added_to_candidate_before_acceptance(self):
        first = self.fake_plan()
        second = TrajectoryPlan(0, 1, 11, first.X, first.U)
        self.m.optimizer.solve = Mock(side_effect=[first, second])
        self.m._slew_time = Mock(return_value=10)
        result = self.m.optimize(self.state, 0)
        self.assertIs(result, second)
        self.assertEqual(self.m.optimizer.solve.call_args_list[1].args[3], 11)
        self.assertIsNone(self.m.active_plan)


class OptimizerTests(unittest.TestCase):
    def setUp(self):
        normal = np.array([1., 0, 0])
        self.target = Earth.pos_I + normal*(Earth.radius+20000)
        self.optimizer = DescentOptimizer(self.target, normal, Earth, SoundingRocket(),
                                           DescentLimits(100, 10, 100), dt=3)

    def test_feasible_plan_passes_nonlinear_validation(self):
        x0 = np.r_[self.target+[500, 0, 0], [-10, 0, 0]]
        plan = self.optimizer.solve(x0, 0, 90, 0)
        self.assertIsNotNone(plan, self.optimizer.last_failure)
        self.assertTrue(self.optimizer.validate(plan, x0))
        self.assertEqual(plan.tf, 90)
        self.assertLessEqual(np.linalg.norm(plan.X[:3, -1]-self.target), 100)

    def test_infeasible_initial_halfspace_is_rejected(self):
        x0 = np.r_[self.target- np.array([500, 0, 0]), np.zeros(3)]
        self.assertIsNone(self.optimizer.solve(x0, 0, 30, 0))

    def test_fake_hover_is_rejected_by_nonlinear_validation(self):
        x0 = np.r_[self.target+[10, 0, 0], np.zeros(3)]
        fake = TrajectoryPlan(0, 1, 0, np.repeat(x0[:, None], 3, axis=1), np.zeros((3, 2)))
        self.assertFalse(self.optimizer.validate(fake))

    def test_coast_is_propagated_without_thrust(self):
        # Leave enough height to arrest the speed gained during six seconds of coast.
        x0 = np.r_[self.target+[2000, 0, 0], [-10, 0, 0]]
        plan = self.optimizer.solve(x0, 5, 155, 11)
        self.assertIsNotNone(plan, self.optimizer.last_failure)
        self.assertEqual(plan.ignition_time, 11)
        np.testing.assert_allclose(plan.U[:, :2], 0, atol=1e-12)
        self.assertLess(plan.X[3, 2], x0[3])
        self.assertTrue(self.optimizer.validate(plan, x0))

    def test_terminal_landing_plan_has_physical_tolerances(self):
        landing = LandManeuver([1, 0, 0], np.pi/3, Earth, SoundingRocket(), 60, log=False)
        x0 = np.r_[landing.target+[100, 0, 0], [-5, 0, 0]]
        plan = landing.optimizer.solve(x0, 0, 25, 0)
        self.assertIsNotNone(plan, landing.optimizer.last_failure)
        self.assertTrue(landing.optimizer.validate(plan, x0))
        self.assertLessEqual(np.linalg.norm(plan.X[3:, -1]), landing.limits.speed)


if __name__ == "__main__":
    unittest.main()
