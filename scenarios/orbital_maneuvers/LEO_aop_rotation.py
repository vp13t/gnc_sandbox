import numpy as np
import guidance.oe as oe
import sim.state as state
from scenarios.base_scenario import BaseScenario
from sim.bodies import Earth, Sun, Moon
from spacecraft.cubesat import CubeSat
from visualization.camera_mode import CameraMode
import sim.forces as forces

from guidance.scheduling import GuidanceSchedule
from guidance.maneuver import IdlePeriod
from guidance.orbits.aop_maneuver import SetArgumentOfPeriapsisManeuver

class Scenario(BaseScenario):
    def __init__(self):
        self.name = "LEO_aop_rotation"

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
        r0, v0 = oe.oe_to_rv(OE, mu=Earth.mu)

        h = np.cross(r0, v0)
        hhat = h / np.linalg.norm(h)

        self.period = OE.period(mu=Earth.mu)  # Orbital period in seconds
        spin = 2 * np.pi / self.period  # Angular velocity in rad/s

        q0 = [0, 0, 0, 1]  # Initial quaternion (no rotation)
        w0 = spin * hhat    # Initial angular velocity (1 rotation per orbit)

        self.X0 = state.State(
            r=r0,
            v=v0,
            q=q0,
            w=w0
        )
        self.last_X = self.X0

        self.cam_target = CameraMode.VELOCITY_FOLLOWING
        self.spacecraft = CubeSat()

        t0 = 0.0
        omega_new = np.pi/2
        self.guidance_schedule = GuidanceSchedule(
            [
                IdlePeriod(self.period),
                SetArgumentOfPeriapsisManeuver(omega_new, Earth, self.spacecraft),
            ],
            self.X0,
            t0,
            control_dt=self.dt * self.dt_between_gnc_updates,
        )
        self.control_force = forces.Force()

        self.duration = self.period * 4

    def update_gnc(self, X, t) -> dict[str, forces.Force]:
        control_inputs = self.guidance_schedule.update(X, t)
        self.control_force = sum(control_inputs.values(), start=forces.Force())
        self.last_X = X
        return control_inputs

    def forces(self, X) -> forces.Force:
        return (
            forces.gravity(X, Earth, self.spacecraft)
            + forces.torque_free_rotation(X, self.spacecraft)
            + self.control_force
        )

scene = Scenario()