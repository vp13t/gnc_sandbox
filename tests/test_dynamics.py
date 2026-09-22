"""Accuracy, control-hold, and contact regressions for the integrator."""

import importlib
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from sim.bodies import CelestialBody
from sim.forces import Force, control_torque, gravity, normal_force, thrust
from sim.frames import QuaternionFrame
from sim.state import State, dynamics
from spacecraft.spacecraft import Thruster


class DynamicsTests(unittest.TestCase):
    def setUp(self):
        self.body = CelestialBody(1.0, 0.1, np.array([10., -5., 3.]), "gray", "test")
        self.craft = SimpleNamespace(mass=2.0, inertia=np.eye(3))

    def test_applied_force_is_in_newtons_and_acceleration_scales_with_mass(self):
        light = State([0, 0, 0], [0, 0, 0], [0, 0, 0, 1], [0, 0, 0])
        heavy = State([0, 0, 0], [0, 0, 0], [0, 0, 0, 1], [0, 0, 0])
        load = Force(fx=4.0, fy=-2.0)
        light.update(1, self.craft, load)
        heavy.update(1, SimpleNamespace(mass=4.0, inertia=np.eye(3)), load)
        np.testing.assert_allclose(light.vel(), [2, -1, 0], atol=1e-12)
        np.testing.assert_allclose(heavy.vel(), [1, -0.5, 0], atol=1e-12)

    def test_gravity_force_scales_with_mass_and_torque_uses_physical_units(self):
        state = State(self.body.pos_I + [1, 0, 0], [0, 0, 0],
                      [0.2, -0.1, 0.3, 0.9], [0, 0, 0])
        self.craft.inertia = np.diag([2., 3., 5.])
        load = gravity(state, self.body, self.craft)
        np.testing.assert_allclose(load.force_I, [-2, 0, 0], atol=1e-12)
        rotation = QuaternionFrame(state)
        radial_body = rotation.T @ np.array([1., 0, 0])
        expected_torque = rotation @ (3*self.body.mu*np.cross(
            radial_body, self.craft.inertia @ radial_body))
        np.testing.assert_allclose(load.torque_I, expected_torque, atol=1e-12)
        self.craft.mass *= 3
        heavier = gravity(state, self.body, self.craft)
        np.testing.assert_allclose(heavier.force_I, 3*load.force_I, atol=1e-12)
        np.testing.assert_allclose(heavier.torque_I, load.torque_I, atol=1e-12)

    def test_constant_inertial_torque_changes_angular_momentum_by_its_impulse(self):
        self.craft.inertia = np.array([[2., 0.2, 0.1], [0.2, 3., 0.3], [0.1, 0.3, 4.]])
        state = State([0, 0, 0], [0, 0, 0], [0.2, -0.1, 0.3, 0.9], [0.4, -0.7, 1.1])
        rotation = QuaternionFrame(state)
        initial = rotation @ self.craft.inertia @ rotation.T @ state.omega()
        torque = np.array([0.5, -0.2, 0.3])
        held = control_torque(state, torque, self.craft)
        np.testing.assert_allclose(held.torque_I, torque, atol=1e-12)
        state.update(2, self.craft, held, max_step=0.005)
        rotation = QuaternionFrame(state)
        final = rotation @ self.craft.inertia @ rotation.T @ state.omega()
        np.testing.assert_allclose(final, initial + 2*torque, atol=1e-8)

    def test_tracking_controller_torque_produces_requested_angular_acceleration(self):
        from controllers.pointing_tracking import pointing_tracking
        self.craft.inertia = np.diag([2., 3., 5.])
        self.craft.thrusters = {"X_body": Thruster(4, np.array([1., 0, 0]))}
        state = State([0, 0, 0], [0, 0, 0], [0.2, -0.1, 0.3, 0.9], [0.4, -0.7, 1.1])
        axis = QuaternionFrame(state)[:, 0]
        alpha = np.array([0.2, -0.1, 0.3])
        torque = pointing_tracking(state, axis, self.craft, state.omega(), alpha)
        derivative = dynamics(0, state.vec(), self.craft, control_torque(state, torque, self.craft))
        np.testing.assert_allclose(derivative[10:13], alpha, atol=1e-12)

    def orbit(self, max_step):
        state = State(self.body.pos_I + [1, 0, 0], [0, 1, 0], [0, 0, 0, 1], [0, 0, 0])
        initial = state.vec()
        state.update(2*np.pi, self.craft,
                     state_forces=lambda x: gravity(x, self.body, self.craft),
                     max_step=max_step)
        return state, np.linalg.norm(state.vec()[:6] - initial[:6])

    def test_circular_orbit_accuracy_and_fourth_order_convergence(self):
        _, coarse = self.orbit(2*np.pi/100)
        state, fine = self.orbit(2*np.pi/200)
        self.assertLess(fine, 1e-5)
        self.assertGreater(coarse/fine, 12)
        radius = np.linalg.norm(state.pos()-self.body.pos_I)
        energy = np.dot(state.vel(), state.vel())/2 - self.body.mu/radius
        self.assertAlmostEqual(energy, -0.5, delta=1e-8)

    def test_stage_states_are_independent_and_state_dependent(self):
        state = State([1, 0, 0], [0, 0, 0], [0, 0, 0, 1], [0, 0, 0])
        original = state.vec()
        positions = []
        def spring(stage):
            self.assertIsNot(stage, state)
            np.testing.assert_array_equal(state.vec(), original)
            positions.append(stage.x)
            f = Force()
            f.fx = -self.craft.mass*stage.x
            return f
        state.update(1.0, self.craft, state_forces=spring, max_step=0.1)
        self.assertGreater(np.ptp(positions), 0.4)
        self.assertAlmostEqual(state.x, np.cos(1.0), delta=1e-6)
        self.assertAlmostEqual(state.xdot, -np.sin(1.0), delta=1e-6)

    def test_thrust_and_control_torque_stay_held_while_rotating(self):
        state = State([0, 0, 0], [0, 0, 0], [0, 0, 0, 1], [0, 0, np.pi/2])
        thruster = Thruster(4.0, np.array([1., 0, 0]))
        held = thrust(state, self.craft, thruster) + control_torque(state, [0, 0, 0.3], self.craft)
        state.update(1, self.craft, held, max_step=0.05)
        np.testing.assert_allclose(state.pos(), [1, 0, 0], atol=1e-12)
        np.testing.assert_allclose(state.vel(), [2, 0, 0], atol=1e-12)
        np.testing.assert_allclose(state.omega(), [0, 0, np.pi/2+0.3], atol=1e-12)
        self.assertGreater((QuaternionFrame(state) @ thruster.direction)[1], 0.98)

    def test_free_rotation_conserves_momentum_energy_and_unit_quaternion(self):
        self.craft.inertia = np.array([[2., 0.2, 0.1], [0.2, 3., 0.3], [0.1, 0.3, 4.]])
        state = State([0, 0, 0], [0, 0, 0], [0.2, -0.1, 0.3, 0.9], [0.4, -0.7, 1.1])
        rotation = QuaternionFrame(state)
        momentum = rotation @ self.craft.inertia @ rotation.T @ state.omega()
        energy = state.omega() @ momentum / 2
        state.update(2, self.craft, max_step=0.005)
        rotation = QuaternionFrame(state)
        final = rotation @ self.craft.inertia @ rotation.T @ state.omega()
        np.testing.assert_allclose(final, momentum, atol=1e-8)
        self.assertAlmostEqual(state.omega() @ final/2, energy, delta=1e-8)
        self.assertAlmostEqual(np.linalg.norm(state.rotvec()), 1.0, delta=1e-14)


class ContactTests(unittest.TestCase):
    def setUp(self):
        self.body = CelestialBody(100., 10., np.array([0., 20., 0.]), "gray", "surface")
        self.craft = SimpleNamespace(mass=1., inertia=np.eye(3))

    def environment(self, state):
        return gravity(state, self.body, self.craft) + normal_force(state, self.body, self.craft)

    def state(self, height=0, speed=0):
        return State(self.body.pos_I + [self.body.radius+height, 0, 0], [speed, 0, 0],
                     [0, 0, 0, 1], [0, 0, 0])

    def test_surface_support_and_liftoff_use_net_force(self):
        state = self.state()
        control = Force()
        control.fx = 0.5  # Less than the 1 m/s² gravity at the surface.
        state.update(1, self.craft, control, state_forces=self.environment)
        np.testing.assert_allclose(state.pos(), self.body.pos_I + [10, 0, 0], atol=1e-12)
        np.testing.assert_allclose(state.vel(), 0, atol=1e-12)
        control.fx = 2.0
        state.update(0.1, self.craft, control, state_forces=self.environment)
        self.assertAlmostEqual(state.x, 10.005, delta=1e-6)
        self.assertAlmostEqual(state.xdot, 0.1, delta=4e-5)

    def test_impact_inside_step_stops_at_surface_and_can_lift_off_again(self):
        state = self.state(height=0.1, speed=-2.)
        state.update(0.2, self.craft, state_forces=self.environment)
        np.testing.assert_allclose(state.pos(), self.body.pos_I + [10, 0, 0], atol=1e-10)
        np.testing.assert_allclose(state.vel(), 0, atol=1e-10)
        control = Force()
        control.fx = 2
        state.update(0.1, self.craft, control, state_forces=self.environment)
        self.assertGreater(state.x, 10)
        self.assertGreater(state.xdot, 0)

    def test_crash_guard_uses_impact_velocity(self):
        state = self.state(height=1, speed=-9.99)
        with self.assertRaisesRegex(RuntimeError, "Crashed at 10"):
            state.update(0.2, self.craft, state_forces=self.environment)

    def test_cannot_tunnel_through_a_body_between_step_endpoints(self):
        state = self.state(height=1, speed=-100.)
        with self.assertRaisesRegex(RuntimeError, "Crashed at 100"):
            state.update(1, self.craft, state_forces=lambda x: normal_force(x, self.body, self.craft))

    def test_resting_on_tilted_surface_does_not_repeat_impact_searches(self):
        from sim.bodies import Earth
        normal = np.array([-0.0669872981077805, 0.9330127018922193, 0.3535533905932737])
        state = State(Earth.pos_I + Earth.radius*normal, [0, 0, 0],
                      [0, 0, 0, 1], [0, 0, 0])
        def environment(x):
            return gravity(x, Earth, self.craft) + normal_force(x, Earth, self.craft)
        with patch("sim.state._impact_time", side_effect=AssertionError("Repeated impact at rest")):
            for _ in range(100):
                state.update(0.1, self.craft, state_forces=environment)
        self.assertAlmostEqual(np.linalg.norm(state.pos()-Earth.pos_I), Earth.radius, delta=1e-8)
        np.testing.assert_allclose(state.vel(), 0, atol=1e-12)


class ScenarioTests(unittest.TestCase):
    def test_noise_is_sampled_once_per_step_with_integration_substeps(self):
        from scenarios.ascent.LAUNCH_karman import Scenario
        scene = Scenario()
        scene.max_integration_step = 0.025
        with patch("scenarios.ascent.LAUNCH_karman.forces.perturbations", return_value=Force()) as noise:
            scene.step(scene.X0)
            self.assertEqual(noise.call_count, 1)
            scene.step(scene.X0)
            self.assertEqual(noise.call_count, 2)

    def test_all_scenarios_use_the_force_callback_api(self):
        root = Path(__file__).resolve().parents[1]
        for path in sorted((root / "scenarios").glob("*/*.py")):
            if path.name.startswith("__"):
                continue
            name = ".".join(path.relative_to(root).with_suffix("").parts)
            with self.subTest(scenario=name):
                scene = importlib.import_module(name).Scenario()
                scene.update_gnc(scene.X0, scene.t0)
                scene.step(scene.X0)
                self.assertTrue(np.isfinite(scene.X0.vec()).all())


if __name__ == "__main__":
    unittest.main()
