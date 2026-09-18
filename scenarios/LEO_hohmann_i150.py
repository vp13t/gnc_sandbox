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
from guidance.orbits.apse_maneuver import HohmannTransferOut

class Scenario(BaseScenario):
    def __init__(self):
        self.name = "LEO_hohmann_i150"

        alt = 400000.0  # Altitude above Earth's surface in meters
        r_p = Earth.radius + alt  # Perigee distance from Earth's center
        OE = oe.OrbitalElements(
            a=r_p,         # Semi-major axis
            e=0, # Eccentricity
            i=5*np.pi/6,      # Inclination
            Omega=3*np.pi/4,  # Right ascension of ascending node
            omega=-np.pi/4,  # Argument of periapsis
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
        self.alt_f_target = 2500000.0
        r_f = Earth.radius + self.alt_f_target
        self.guidance_schedule = GuidanceSchedule(
            [IdlePeriod(0.25 * self.period), *HohmannTransferOut(r_f, Earth, self.spacecraft)],
            self.X0,
            t0
        )
        self.control_force = forces.Force()

        self.duration = self.period * 3.5

    def __del__(self):
        alt_f = np.linalg.norm(self.last_X.pos()) - Earth.radius
        print(f"Hohmann Transfer Targeting Altitude {self.alt_f_target:.2f}m. Final Altitude {alt_f:.2f}m")

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