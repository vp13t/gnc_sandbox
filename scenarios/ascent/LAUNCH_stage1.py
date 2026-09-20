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
from guidance.ascent.pitch_maneuver import PitchManeuver
from guidance.ascent.gravity_turn_maneuver import GravityTurnManeuver

def desired_oe():
    min_alt = 400000.0  # Altitude above Earth's surface in meters
    max_alt = 10000000.0  # Altitude above Earth's surface in meters
    r_p = Earth.radius + min_alt  # Perigee distance from Earth's center
    r_a = Earth.radius + max_alt  # Apogee distance from Earth's center
    a = (r_p+r_a)/2
    e = (r_a-r_p)/(r_a+r_p)

    OE = oe.OrbitalElements(
        a=a,         # Semi-major axis
        e=e, # Eccentricity
        i=np.pi/4,      # Inclination
        Omega=0,  # Right ascension of ascending node
        omega=-np.pi/2,  # Argument of periapsis
        theta=0   # True anomaly
    )
    return OE

class Scenario(BaseScenario):
    def __init__(self):
        self.name = "LAUNCH_stage1"

        rhat0 = np.array([0, 1/np.sqrt(2), 1/np.sqrt(2)])
        r0 = (Earth.radius) * rhat0
        v0 = [0, 0, 0]
        q0 = np.roll(quaternion.as_float_array(vec2quat(rhat0)), -1)
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

        t0 = 0.0
        alt = 40000
        self.guidance_schedule = GuidanceSchedule(
            [
                IdlePeriod(60),
                RiseManeuver(alt, Earth, self.spacecraft),
                PitchManeuver(desired_oe(), Earth, self.spacecraft, np.deg2rad(5), np.deg2rad(2)),
                GravityTurnManeuver(900, Earth, self.spacecraft),
                IdlePeriod(60)
            ],
            self.X0,
            t0
        )
        self.control_force = forces.Force()

        self.duration = 12000.0

    def update_gnc(self, X, t) -> dict[str, forces.Force]:
        control_inputs = self.guidance_schedule.update(X, t)
        self.control_force = sum(control_inputs.values(), start=forces.Force())
        self.last_X = X
        return control_inputs

    def forces(self, X) -> forces.Force:
        return (
            forces.gravity(X, Earth, self.spacecraft)
            + forces.normal_force(X, Earth, self.spacecraft)
            + forces.torque_free_rotation(X, self.spacecraft)
            + self.control_force
        )

scene = Scenario()