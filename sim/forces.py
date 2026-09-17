import numpy as np
from sim.bodies import CelestialBody
from sim.frames import DCM, InertialFrame, QuaternionFrame
from spacecraft.spacecraft import Spacecraft, Thruster

class Force:
    xddot: float = 0.0
    yddot: float = 0.0
    zddot: float = 0.0
    w1dot: float = 0.0
    w2dot: float = 0.0
    w3dot: float = 0.0

    def __add__(self, other):
        result = Force()
        result.xddot = self.xddot + other.xddot
        result.yddot = self.yddot + other.yddot
        result.zddot = self.zddot + other.zddot
        result.w1dot = self.w1dot + other.w1dot
        result.w2dot = self.w2dot + other.w2dot
        result.w3dot = self.w3dot + other.w3dot
        return result
    
    def __radd__(self, other):
        if other == 0:
            return self
        return self.__add__(other)

def gravity(state: "State", body: CelestialBody, spacecraft: Spacecraft):
    # Gravity Acceleration
    r_vec = state.pos() - body.pos_I
    r_mag = np.linalg.norm(r_vec)
    r_hat = r_vec / r_mag
    if r_mag == 0:
        raise ValueError("Distance between bodies cannot be zero.")
    accel = -body.mu * r_hat / r_mag**2

    # Gravity Gradient Torque
    R = DCM(InertialFrame(), QuaternionFrame(state))
    r_b = R @ r_hat
    torque_b = np.cross(3 * body.mu * r_b / r_mag**3, spacecraft.inertia @ r_b)
    wdot_I = R.T @ np.linalg.solve(spacecraft.inertia, torque_b)

    force = Force()
    force.xddot, force.yddot, force.zddot = accel
    force.w1dot, force.w2dot, force.w3dot = wdot_I
    return force

def thrust(state: "State", spacecraft: Spacecraft, thruster: Thruster):
    R = DCM(QuaternionFrame(state), InertialFrame())
    thrust_dir_I = R @ thruster.direction
    accel = thruster.force * thrust_dir_I / spacecraft.mass

    force = Force()
    force.xddot, force.yddot, force.zddot = accel
    return force

def torque_free_rotation(state, spacecraft: Spacecraft):
    w = state.omega()
    R = DCM(QuaternionFrame(state), InertialFrame())
    I = R @ spacecraft.inertia @ R.T

    wdot = np.linalg.solve(I, -np.cross(w, I @ w))

    force = Force()
    force.w1dot, force.w2dot, force.w3dot = wdot
    return force

def control_torque(state, L, spacecraft: Spacecraft):
    w = state.omega()
    R = DCM(QuaternionFrame(state), InertialFrame())
    I = R @ spacecraft.inertia @ R.T

    wdot = np.linalg.solve(I, L)

    force = Force()
    force.w1dot, force.w2dot, force.w3dot = wdot
    return force
