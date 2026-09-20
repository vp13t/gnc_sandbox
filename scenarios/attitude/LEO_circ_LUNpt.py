import numpy as np
import guidance.oe as oe
import sim.state as state
from scenarios.base_scenario import BaseScenario
from sim.bodies import Earth, Sun, Moon
from spacecraft.cubesat import CubeSat
from visualization.camera_mode import CameraMode
import sim.forces as forces
from controllers.pointing_lyapunov import pointing_lyapunov

class Scenario(BaseScenario):
    def __init__(self):
        self.name = "LEO_circ_LUNpt"

        alt = 400000.0  # Altitude above Earth's surface in meters
        r_p = Earth.radius + alt  # Perigee distance from Earth's center
        OE = oe.OrbitalElements(
            a=r_p,      # Semi-major axis
            e=0.0,      # Eccentricity
            i=0.0,      # Inclination
            Omega=0.0,  # Right ascension of ascending node
            omega=0.0,  # Argument of periapsis
            theta=np.pi/2   # True anomaly
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

        self.cam_target = CameraMode.VELOCITY_FOLLOWING
        self.spacecraft = CubeSat()

        self.control_force = forces.Force()

        self.duration = self.period
    
    def update_gnc(self, X, t):
        """Updates maneuver forces, and returns commanded inputs"""
        L_I, _ = pointing_lyapunov(X, Moon, self.spacecraft)
        self.control_force = forces.control_torque(X, L_I, self.spacecraft)
        return {"X_body": 0.0, "torque": L_I}

    def forces(self, X):
        return (
            forces.gravity(X, Earth, self.spacecraft)
            + forces.torque_free_rotation(X, self.spacecraft)
            + self.control_force
        )

scene = Scenario()