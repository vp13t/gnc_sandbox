import numpy as np
import quaternion
import guidance.oe as oe
import sim.state as state
from scenarios.base_scenario import BaseScenario
from sim.angles import vec2quat
from sim.frames import IZ
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
        self.pertubations_Q = np.diag([rnoise, rnoise, rnoise, wnoise, wnoise, wnoise])

        t0 = 0.0
        karman_alt = 100000
        self.guidance_schedule = GuidanceSchedule(
            [IdlePeriod(30),
            RiseManeuver(karman_alt, Earth, self.spacecraft)],
            self.X0,
            t0
        )
        self.control_force = forces.Force()

        self.duration = 700.0

    def update_gnc(self, X, t) -> dict[str, forces.Force]:
        control_inputs = self.guidance_schedule.update(X, t)
        self.control_force = sum(control_inputs.values(), start=forces.Force())
        self.last_X = X
        return control_inputs

    def forces(self, X) -> forces.Force:
        return (
            forces.gravity(X, Earth, self.spacecraft)
            + forces.normal_force(X, Earth, self.spacecraft)
            + forces.perturbations(self.pertubations_Q)
            + forces.torque_free_rotation(X, self.spacecraft)
            + self.control_force
        )

scene = Scenario()