import numpy as np
import guidance.oe as oe
import sim.state as state
from scenarios.base_scenario import BaseScenario
from sim.bodies import Earth, Sun, Moon, GEO_alt
from spacecraft.cubesat import CubeSat
from visualization.camera_mode import CameraMode
import sim.forces as forces

class Scenario(BaseScenario):
    def __init__(self):
        self.name = "GEO_synchronous"

        OE = oe.OrbitalElements(
            a=Earth.radius+GEO_alt,   # Semi-major axis
            e=0, # Eccentricity
            i=np.deg2rad(45),      # Inclination
            Omega=-np.pi/2,  # Right ascension of ascending node
            omega=np.pi/4,  # Argument of periapsis
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
        self.dt = 1.0

    def forces(self, X):
        return (
            forces.gravity(X, Earth, self.spacecraft)
        )

scene = Scenario()