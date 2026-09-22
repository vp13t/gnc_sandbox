"""Rigid-body integration with stage-evaluated environment and held controls."""

import math
from copy import copy

import numpy as np
import quaternion
from sim.forces import Force, _cross3
from sim.frames import QuaternionFrame


def dynamics(t, x, spacecraft, force, state_forces=None, supported=(), inverse_inertia=None):
    stage = State(x[:3], x[3:6], x[6:10], x[10:13])
    load = force
    if state_forces is not None:
        load = load + state_forces(stage)
    dx = np.empty(13)
    dx[:3] = x[3:6]
    dx[3:6] = load.force_I / spacecraft.mass
    for body in supported:
        r = x[:3] - body.pos_I
        radius = np.linalg.norm(r)
        n = r / radius
        if radius <= body.radius + 1e-7 and n @ x[3:6] <= 1e-7:
            dx[3:6] -= min(0.0, n @ dx[3:6]) * n
    # Inertial angular velocity, scalar-last quaternion. Avoid constructing
    # the 4x4 Omega matrix at every integration stage.
    a, b, c, d = x[6:10]
    w1, w2, w3 = x[10:13]
    dx[6:10] = (0.5*(-w3*b + w2*c + w1*d),
                0.5*(w3*a - w1*c + w2*d),
                0.5*(-w2*a + w1*b + w3*d),
                -0.5*(w1*a + w2*b + w3*c))
    # Euler's equation in body coordinates, then rotate alpha back to the
    # inertial frame used by the state. Even a held torque has a changing
    # angular-acceleration response as an asymmetric spacecraft rotates.
    rotation = QuaternionFrame(stage)
    omega_body = rotation.T @ x[10:13]
    rhs = rotation.T @ load.torque_I - _cross3(omega_body, spacecraft.inertia @ omega_body)
    if inverse_inertia is None:
        alpha_body = np.linalg.solve(spacecraft.inertia, rhs)
    else:
        alpha_body = inverse_inertia @ rhs
    dx[10:13] = rotation @ alpha_body
    return dx


def _rk4(x, dt, derivative, k1=None):
    k1 = derivative(x) if k1 is None else k1
    k2 = derivative(x + (dt/2)*k1)
    k3 = derivative(x + (dt/2)*k2)
    k4 = derivative(x + dt*k3)
    return x + (dt/6)*(k1 + 2*k2 + 2*k3 + k4)


def _stop_at_surface(x, body):
    speed = np.linalg.norm(x[3:6])
    if speed >= 10.0:
        raise RuntimeError(f"\nCrashed at {speed:.2f} m/s!")
    radial = x[:3] - body.pos_I
    x[:3] = body.pos_I + body.radius * radial / np.linalg.norm(radial)
    x[3:6] = 0.0  # Preserve the simulator's inelastic, no-slip impact model.


def _impact_time(x, end, dt, derivative, body):
    """Bracket contact, including a step whose endpoints straddle the body."""
    hi = dt
    if np.linalg.norm(end[:3] - body.pos_I) >= body.radius:
        r = x[:3] - body.pos_I
        travel = end[:3] - x[:3]
        distance2 = travel @ travel
        if distance2 == 0:
            return None
        fraction = np.clip(-(r @ travel)/distance2, 0, 1)
        if np.linalg.norm(r + fraction*travel) >= body.radius:
            return None
        hi = fraction*dt
        trial = _rk4(x, hi, derivative)
        if np.linalg.norm(trial[:3] - body.pos_I) >= body.radius:
            return None
    # Root refinement is only paid for on a potential impact step.
    lo = 0.0
    for _ in range(30):
        mid = (lo + hi) / 2
        trial = _rk4(x, mid, derivative)
        if np.linalg.norm(trial[:3] - body.pos_I) > body.radius:
            lo = mid
        else:
            hi = mid
    return hi


class State:
    x: float
    y: float
    z: float
    xdot: float
    ydot: float
    zdot: float
    q1: float
    q2: float
    q3: float
    q4: float # Scalar last
    w1: float
    w2: float
    w3: float
    
    def __init__(self, r, v, q, w):
        self.x, self.y, self.z = r
        self.xdot, self.ydot, self.zdot = v
        self.q1, self.q2, self.q3, self.q4 = q
        self.w1, self.w2, self.w3 = w

    def pos(self):
        return np.array([self.x, self.y, self.z])
    
    def rotvec(self):
        return np.array([self.q1, self.q2, self.q3, self.q4])

    def rot(self):
        self.normalize_quat()
        return np.quaternion(self.q4, self.q1, self.q2, self.q3)
    
    def vel(self):
        return np.array([self.xdot, self.ydot, self.zdot])
    
    def omega(self):
        return np.array([self.w1, self.w2, self.w3])
    
    def vec(self):
        return np.concatenate((self.pos(), self.vel(), self.rotvec(), self.omega()))
    
    def normalize_quat(self):
        norm = np.linalg.norm(self.rotvec())
        if norm > 0:
            self.q1 /= norm
            self.q2 /= norm
            self.q3 /= norm
            self.q4 /= norm

    def update(self, dt, spacecraft, force=None, *, state_forces=None, max_step=1.0):
        """Advance with RK4; reevaluate state_forces(stage_state) at each stage.

        spacecraft supplies mass (kg) and body-frame inertia (kg m²).
        `force` contains forces (N) and torques (N m) about the center of mass,
        held in the inertial frame for the entire dt, even when
        max_step subdivides it. Callbacks must be deterministic and must not
        change guidance, sample noise, or mutate the real simulation state.
        max_step bounds the fixed integration step; there is no adaptive error
        estimator. Decrease it for fast rotation or rapidly varying forces.
        """
        if not np.isfinite(dt) or dt <= 0 or not np.isfinite(max_step) or max_step <= 0:
            raise ValueError("dt and max_step must be finite and positive")
        if not np.isfinite(spacecraft.mass) or spacecraft.mass <= 0:
            raise ValueError("Spacecraft mass must be finite and positive")
        # Mass and body inertia are constant over this update. Invert once,
        # rather than solving the same body-frame matrix at every RK stage.
        inverse_inertia = np.linalg.inv(spacecraft.inertia)
        held = Force() if force is None else copy(force)
        x = self.vec()
        initial = held if state_forces is None else held + state_forces(
            State(x[:3], x[3:6], x[6:10], x[10:13]))
        bodies = tuple(dict.fromkeys(initial.contact_bodies))
        # Support legacy sampled contact flags, which have no surface geometry.
        if initial.normal_force_active and not bodies:
            if np.linalg.norm(x[3:6]) >= 10:
                raise RuntimeError(f"\nCrashed at {np.linalg.norm(x[3:6]):.2f} m/s!")
            x[3:6] = 0
        steps = max(1, math.ceil(dt / max_step))
        step = dt / steps
        for substep in range(steps):
            remaining = step
            # At most one impact per configured surface in this substep. The
            # common airborne path takes just four force evaluations.
            supported = []
            for body in bodies:
                r = x[:3] - body.pos_I
                radius = np.linalg.norm(r)
                # Roundoff in gravity minus the reaction can leave a tiny
                # outward velocity. Treat it as rest, rather than repeatedly
                # losing support and root-finding a new impact every step.
                if radius <= body.radius + 1e-7 and r @ x[3:6] <= radius*1e-7:
                    _stop_at_surface(x, body)
                    supported.append(body)
            # Reuse the initial evaluation that registered contact surfaces.
            # After a contact reset the state has changed, so evaluate afresh.
            k1 = (dynamics(0, x, spacecraft, initial, inverse_inertia=inverse_inertia)
                  if substep == 0 and not supported and not initial.normal_force_active else None)
            while remaining > 0:
                derivative = lambda y: dynamics(0, y, spacecraft, held, state_forces,
                                                supported, inverse_inertia)
                end = _rk4(x, remaining, derivative, k1)
                k1 = None
                impacts = []
                for body in bodies:
                    if body in supported:
                        continue
                    hit_time = _impact_time(x, end, remaining, derivative, body)
                    if hit_time is not None:
                        impacts.append((hit_time, body))
                if not impacts:
                    x = end
                    break
                hit_time, body = min(impacts, key=lambda item: item[0])
                x = _rk4(x, hit_time, derivative)
                _stop_at_surface(x, body)
                supported.append(body)
                remaining -= hit_time
            # Bound quaternion drift between substeps as well as at output.
            x[6:10] /= np.linalg.norm(x[6:10])
        self.x, self.y, self.z = x[0:3]
        self.xdot, self.ydot, self.zdot = x[3:6]
        self.q1, self.q2, self.q3, self.q4 = x[6:10]
        self.w1, self.w2, self.w3 = x[10:13]
        return self.vec()
