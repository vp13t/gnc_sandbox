import numpy as np
from scipy.linalg import expm
from spacecraft.spacecraft import Spacecraft
from sim.bodies import CelestialBody

def linearized_discretized_translational_gravity(xbar: np.ndarray[float], dt:float, body: CelestialBody, spacecraft: Spacecraft):
        r = xbar[:3] - body.pos_I
        rho = np.linalg.norm(r)

        g = -body.mu * r / rho**3
        J = body.mu * (
            3 * np.outer(r, r) / rho**5 - np.eye(3) / rho**3
        )

        A = np.zeros((6, 6))
        A[:3, 3:] = np.eye(3)
        A[3:, :3] = J

        B = np.zeros((6, 3))
        B[3:, :] = (
            spacecraft.thrusters["X_body"].force / spacecraft.mass
        ) * np.eye(3)

        c = np.r_[np.zeros(3), g - J @ xbar[:3]]

        # Augment with three held control components and a constant 1.
        M = np.zeros((10, 10))
        M[:6, :6] = A
        M[:6, 6:9] = B
        M[:6, 9] = c

        E = expm(M * dt)
        Ad = E[:6, :6]
        Bd = E[:6, 6:9]
        cd = E[:6, 9]
        return Ad, Bd, cd