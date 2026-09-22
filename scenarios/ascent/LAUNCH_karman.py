import numpy as np
import quaternion
import guidance.oe as oe
import sim.state as state
from scenarios.base_scenario import BaseScenario
from sim.angles import vec2quat
from sim.frames import IZ, QuaternionFrame
from sim.bodies import Earth, Sun, Moon
from spacecraft.sounding_rocket import SoundingRocket
from visualization.camera_mode import CameraMode
import sim.forces as forces

from guidance.scheduling import GuidanceSchedule
from guidance.maneuver import IdlePeriod
from guidance.ascent.rise_maneuver import RiseManeuver

class Scenario(BaseScenario):
    def __init__(self):
        self.name = "LAUNCH_karman"

        r0 = [0, 0, Earth.radius]
        v0 = [0, 0, 0]
        q0 = np.roll(quaternion.as_float_array(vec2quat(IZ)), -1)
        w0 = [0, 0, 0]    # Initial angular velocity (1 rotation per orbit)

        self.X0 = state.State(
            r=r0,
            v=v0,
            q=q0,
            w=w0
        )
        self.last_X = self.X0

        self.cam_target = CameraMode.NORMAL_FACING
        self.spacecraft = SoundingRocket()

        rnoise = 1e1
        wnoise = 1e-12
        self.rng = np.random.default_rng()
        # Inertial covariance of [force (N), torque (N m)]. Convert the
        # previous acceleration-noise tuning at the launch attitude; this
        # force/torque distribution then remains fixed in inertial axes.
        rotation = QuaternionFrame(self.X0)
        inertia_I = rotation @ self.spacecraft.inertia @ rotation.T
        self.perturbations_Q = np.zeros((6, 6))
        self.perturbations_Q[:3, :3] = rnoise * self.spacecraft.mass**2 * np.eye(3)
        self.perturbations_Q[3:, 3:] = wnoise * inertia_I @ inertia_I.T

        t0 = 0.0
        karman_alt = 100000
        self.guidance_schedule = GuidanceSchedule(
            [IdlePeriod(30),
            RiseManeuver(karman_alt, Earth, self.spacecraft)],
            self.X0,
            t0,
            control_dt=self.dt * self.dt_between_gnc_updates,
        )
        self.control_force = forces.Force()

        self.duration = 700.0
        self.dt = 0.1

    def update_gnc(self, X, t) -> dict[str, forces.Force]:
        control_inputs = self.guidance_schedule.update(X, t)
        self.control_force = sum(control_inputs.values(), start=forces.Force())
        self.last_X = X
        return control_inputs

    def held_forces(self):
        return super().held_forces() + forces.perturbations(self.perturbations_Q, self.rng)

    def forces(self, X) -> forces.Force:
        return (
            forces.gravity(X, Earth, self.spacecraft)
            + forces.normal_force(X, Earth, self.spacecraft)
        )

scene = Scenario()
