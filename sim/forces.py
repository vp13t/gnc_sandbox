from dataclasses import dataclass

import numpy as np
from sim.bodies import CelestialBody
from sim.frames import QuaternionFrame
from spacecraft.spacecraft import Spacecraft, Thruster


def _cross3(a, b):
    # These force models always use single three-vectors. Avoid numpy.cross's
    # batch/axis dispatch overhead in the four evaluations per RK4 step.
    return np.array([a[1]*b[2] - a[2]*b[1],
                     a[2]*b[0] - a[0]*b[2],
                     a[0]*b[1] - a[1]*b[0]])

@dataclass(slots=True)
class Force:
    """External load in inertial coordinates: force (N) and torque (N m).

    Torque is about the spacecraft's center of mass. Acceleration and
    gyroscopic dynamics are computed by State.update, not by force models.
    """
    fx: float = 0.0
    fy: float = 0.0
    fz: float = 0.0
    tx: float = 0.0
    ty: float = 0.0
    tz: float = 0.0
    normal_force_active: bool = False
    contact_bodies: tuple = ()

    @classmethod
    def from_vectors(cls, force_I=(0., 0., 0.), torque_I=(0., 0., 0.)):
        if len(force_I) != 3 or len(torque_I) != 3:
            raise ValueError("Force and torque must each have three components")
        return cls(*force_I, *torque_I)

    @property
    def force_I(self):
        return np.array([self.fx, self.fy, self.fz])

    @property
    def torque_I(self):
        return np.array([self.tx, self.ty, self.tz])

    def __add__(self, other):
        if not isinstance(other, Force):
            return NotImplemented
        return Force(self.fx + other.fx, self.fy + other.fy, self.fz + other.fz,
                     self.tx + other.tx, self.ty + other.ty, self.tz + other.tz,
                     self.normal_force_active or other.normal_force_active,
                     self.contact_bodies + other.contact_bodies)

    def __sub__(self, other):
        if not isinstance(other, Force):
            return NotImplemented
        return Force(self.fx - other.fx, self.fy - other.fy, self.fz - other.fz,
                     self.tx - other.tx, self.ty - other.ty, self.tz - other.tz,
                     self.normal_force_active or other.normal_force_active,
                     self.contact_bodies + other.contact_bodies)

    def __radd__(self, other):
        if other == 0:
            return self
        return self.__add__(other)

def gravity(state: "State", body: CelestialBody, spacecraft: Spacecraft):
    # Gravitational force at the center of mass
    r_vec = state.pos() - body.pos_I
    r_mag = np.linalg.norm(r_vec)
    if r_mag == 0:
        raise ValueError("Distance between bodies cannot be zero.")
    r_hat = r_vec / r_mag
    force_I = -spacecraft.mass * body.mu * r_hat / r_mag**2

    # Gravity Gradient Torque
    R = QuaternionFrame(state).T
    r_b = R @ r_hat
    torque_b = _cross3(3 * body.mu * r_b / r_mag**3, spacecraft.inertia @ r_b)
    return Force.from_vectors(force_I, R.T @ torque_b)

def normal_force(state: "State", body: CelestialBody, spacecraft: Spacecraft):
    """Register a spherical contact surface, including while above it.

    State.update resolves impacts and supplies the outward reaction to the
    *total* inward acceleration. Canceling gravity here would also cancel it
    during liftoff and would not account for other applied forces.
    """
    f = Force()
    f.contact_bodies = (body,)
    f.normal_force_active = np.linalg.norm(state.pos() - body.pos_I) <= body.radius
    return f

def thrust(state: "State", spacecraft: Spacecraft, thruster: Thruster):
    """Sample the thruster's inertial force in newtons."""
    thrust_dir_I = QuaternionFrame(state) @ thruster.direction
    return Force.from_vectors(force_I=thruster.force * thrust_dir_I)


def control_torque(state, L, spacecraft: Spacecraft):
    """Package an inertial controller torque (N m), without dividing by inertia."""
    return Force.from_vectors(torque_I=L)


def perturbations(Q, rng=None):
    """Sample [Fx, Fy, Fz, Tx, Ty, Tz] with force/torque covariance Q.

    Sample once per hold interval, never inside a state-force callback.
    """
    rng = np.random.default_rng() if rng is None else rng
    sample = rng.multivariate_normal(np.zeros(6), Q)
    return Force.from_vectors(sample[:3], sample[3:])
