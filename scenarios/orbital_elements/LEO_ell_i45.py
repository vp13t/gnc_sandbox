import numpy as np
import guidance.oe as oe
import sim.state as state
from scenarios.base_scenario import BaseScenario
from sim.bodies import Earth, Sun, Moon
from spacecraft.cubesat import CubeSat
from visualization.camera_mode import CameraMode
import sim.forces as forces

class Scenario(BaseScenario):
    def __init__(self):
        self.name = "LEO_ell_i45"

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
        r0, v0 = oe.oe_to_rv(OE)

        h = np.cross(r0, v0)
        hhat = h / np.linalg.norm(h)

        # 1 revolution per period
        self.period = OE.period()  # Orbital period in seconds
        spin = 2 * np.pi / self.period  # Angular velocity in rad/s

        q0 = [0, 0, 0, 1]  # Initial quaternion (no rotation)
        w0 = spin * hhat   # Initial angular velocity (1 rotation per orbit)

        self.X0 = state.State(
            r=r0,
            v=v0,
            q=q0,
            w=w0
        )

        self.cam_target = CameraMode.VELOCITY_FOLLOWING
        self.spacecraft = CubeSat()

        self.duration = self.period

    def forces(self, X):
        return (
            forces.gravity(X, Earth, self.spacecraft)
        )

scene = Scenario()