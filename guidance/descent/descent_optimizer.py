"""Bounded successive convexification for a translational descent segment.

Virtual control is permitted only while constructing a reference trajectory.
Accepted plans must satisfy the physical constraints on a nonlinear rollout.
"""

from dataclasses import dataclass
import math

import cvxpy as cp
import numpy as np
from scipy.sparse import block_diag, csc_matrix

from guidance.descent.linearized_dynamics import linearized_discretized_translational_gravity
from guidance.descent.trajectory_plan import TrajectoryPlan
from guidance.descent.translational_prediction import rollout


@dataclass(frozen=True)
class DescentLimits:
    position: float
    lateral_speed: float
    speed: float
    glide_angle: float | None = None


class DescentOptimizer:
    def __init__(self, target, normal, body, spacecraft, limits, dt=1.0,
                 max_iterations=12, solver_time_limit=5.0, throttle_limit=1.0,
                 terminal_fraction=0.9):
        self.target = np.asarray(target)
        self.normal = np.asarray(normal)
        self.P = np.eye(3) - np.outer(normal, normal)
        self.body = body
        self.spacecraft = spacecraft
        self.limits = limits
        self.dt = dt
        self.max_iterations = max_iterations
        self.solver_time_limit = solver_time_limit
        if not 0 < throttle_limit <= 1:
            raise ValueError("Planning throttle limit must be in (0, 1]")
        self.throttle_limit = throttle_limit
        if not 0 < terminal_fraction <= 1:
            raise ValueError("Terminal constraint fraction must be in (0, 1]")
        self.terminal_fraction = terminal_fraction
        self.acceleration = spacecraft.thrusters["X_body"].force / spacecraft.mass
        self.scale = np.array([1e6]*3 + [1e4]*3)
        self.origin = np.r_[target, np.zeros(3)]
        self.last_failure = None
        self.last_iterations = 0

    def _reference(self, x0, n, dt, coast_steps, warm_start, t):
        if warm_start is not None:
            reference = np.column_stack([warm_start.state_at(t + i*dt) for i in range(n+1)])
            reference[:, 0] = x0
            return reference
        coast, _ = rollout(x0, np.zeros((3, coast_steps)), dt, self.body, self.acceleration)
        # A kinematic seed; virtual control allows the first subproblem to repair it.
        start = coast[:, -1]
        tau = np.linspace(0, 1, n - coast_steps + 1)
        duration = (n - coast_steps) * dt
        positions = ((2*tau**3 - 3*tau**2 + 1)[None, :] * start[:3, None]
                     + (tau**3 - 2*tau**2 + tau)[None, :] * duration * start[3:, None]
                     + (-2*tau**3 + 3*tau**2)[None, :] * self.target[:, None])
        velocities = (((6*tau**2 - 6*tau)/duration)[None, :] * start[:3, None]
                      + (3*tau**2 - 4*tau + 1)[None, :] * start[3:, None]
                      + ((-6*tau**2 + 6*tau)/duration)[None, :] * self.target[:, None])
        # Keep the linearization points away from the gravity singularity.
        radial = positions - self.body.pos_I[:, None]
        radii = np.linalg.norm(radial, axis=0)
        radial *= np.maximum(1, (self.body.radius + 1) / np.maximum(radii, 1))[None, :]
        positions = radial + self.body.pos_I[:, None]
        reference = np.column_stack((coast[:, :-1], np.vstack((positions, velocities))))
        reference[:, 0] = x0
        return reference

    def validate(self, plan, x0=None):
        """Physical tolerances, including dense samples between optimizer nodes."""
        x0 = plan.X[:, 0] if x0 is None else np.asarray(x0)
        if np.max(np.abs(plan.X[:, 0] - x0)) > 1e-4:
            return False
        if np.max(np.linalg.norm(plan.U, axis=0)) > 1 + 1e-7:
            return False
        coast_steps = int(round((plan.ignition_time - plan.t0) / plan.dt))
        if coast_steps and np.max(np.abs(plan.U[:, :coast_steps])) > 1e-8:
            return False
        try:
            nodes, dense = rollout(x0, plan.U, plan.dt, self.body, self.acceleration)
        except (ValueError, FloatingPointError):
            return False
        if not np.isfinite(dense).all():
            return False
        position_error = np.max(np.linalg.norm(nodes[:3] - plan.X[:3], axis=0))
        velocity_error = np.max(np.linalg.norm(nodes[3:] - plan.X[3:], axis=0))
        if position_error > min(10.0, self.limits.position * 0.2):
            return False
        if velocity_error > min(1.0, self.limits.speed * 0.2):
            return False
        if np.min(np.linalg.norm(dense[:3] - self.body.pos_I[:, None], axis=0)) < self.body.radius:
            return False
        subdivisions = max(1, int(np.ceil(plan.dt)))
        powered = dense[:, coast_steps*subdivisions:]
        error = powered[:3] - self.target[:, None]
        height = self.normal @ error
        if np.min(height) < -0.01:
            return False
        if self.limits.glide_angle is not None:
            a = self.limits.glide_angle
            violation = np.cos(a)*np.linalg.norm(self.P @ error, axis=0) - np.sin(a)*height
            if np.max(violation) > 0.01:
                return False
        final = nodes[:, -1]
        return bool(
            np.linalg.norm(final[:3] - self.target) <= self.limits.position
            and np.linalg.norm(self.P @ final[3:]) <= self.limits.lateral_speed
            and np.linalg.norm(final[3:]) <= self.limits.speed
        )

    def solve(self, x0, t, arrival_time, ignition_time, warm_start=None):
        self.last_failure = None
        self.last_iterations = 0
        remaining = arrival_time - t
        if remaining <= 0:
            self.last_failure = "Arrival deadline has expired"
            return None
        n = max(1, math.ceil(remaining / self.dt))
        dt = remaining / n
        coast_steps = max(0, math.ceil((ignition_time - t) / dt - 1e-9))
        if coast_steps >= n:
            self.last_failure = "No powered time remains after the slew/coast interval"
            return None
        ignition_time = t + coast_steps*dt
        reference = self._reference(x0, n, dt, coast_steps, warm_start, t)
        radius, previous_error = 1.0, np.inf

        for iteration in range(self.max_iterations):
            self.last_iterations = iteration + 1
            a_nodes, b_nodes, c_nodes = [], [], []
            for i in range(n):
                a, b, c = linearized_discretized_translational_gravity(
                    reference[:, i], dt, self.body, self.spacecraft)
                a_nodes.append(csc_matrix(a * self.scale[None, :] / self.scale[:, None]))
                b_nodes.append(csc_matrix(b / self.scale[:, None]))
                c_nodes.append((a @ self.origin + c - self.origin) / self.scale)

            y = cp.Variable((6, n+1))
            u = cp.Variable((3, n))
            sigma = cp.Variable(n, nonneg=True)
            virtual = cp.Variable((6, n))
            ybar = (reference - self.origin[:, None]) / self.scale[:, None]
            rhs = (block_diag(a_nodes, format="csc") @ cp.vec(y[:, :-1], order="F")
                   + block_diag(b_nodes, format="csc") @ cp.vec(u, order="F")
                   + np.concatenate(c_nodes) + cp.vec(virtual, order="F"))
            constraints = [
                y[:, 0] == (x0 - self.origin) / self.scale,
                cp.vec(y[:, 1:], order="F") == rhs,
                cp.norm(y - ybar, axis=0) <= radius,
                cp.norm(u, axis=0) <= sigma, sigma <= self.throttle_limit,
                cp.norm(y[:3, -1]) <= self.terminal_fraction*self.limits.position / self.scale[0],
                cp.norm(self.P @ y[3:, -1]) <= self.terminal_fraction*self.limits.lateral_speed / self.scale[3],
                cp.norm(y[3:, -1]) <= self.terminal_fraction*self.limits.speed / self.scale[3],
                self.normal @ y[:3, coast_steps:] >= 0,
            ]
            if coast_steps:
                constraints.append(u[:, :coast_steps] == 0)
                # Fix coast states to nonlinear physics, independent of virtual control.
                coast, _ = rollout(x0, np.zeros((3, coast_steps)), dt, self.body, self.acceleration)
                constraints.append(y[:, :coast_steps+1] == (coast - self.origin[:, None])/self.scale[:, None])
            if self.limits.glide_angle is not None:
                angle = self.limits.glide_angle
                constraints.append(
                    np.cos(angle)*cp.norm(self.P @ y[:3, coast_steps:], axis=0)
                    <= np.sin(angle)*(self.normal @ y[:3, coast_steps:]))
            # Conservative tangent planes keep even the pre-ignition coast outside the body.
            normals = reference[:3] - self.body.pos_I[:, None]
            normals /= np.maximum(np.linalg.norm(normals, axis=0), 1)[None, :]
            offset = (self.target - self.body.pos_I) / self.scale[0]
            constraints.append(cp.sum(cp.multiply(normals, y[:3] + offset[:, None]), axis=0)
                               >= self.body.radius / self.scale[0])
            objective = cp.Minimize(cp.sum(sigma)/n + 1e4*cp.sum(cp.abs(virtual))
                                    + 1e-3*cp.sum_squares(y-ybar)/n)
            problem = cp.Problem(objective, constraints)
            try:
                problem.solve(solver=cp.CLARABEL, time_limit=self.solver_time_limit)
            except cp.error.SolverError as exc:
                self.last_failure = f"CLARABEL failed: {exc}"
                return None
            if problem.status != cp.OPTIMAL:
                self.last_failure = f"Convex subproblem: {problem.status}"
                return None
            predicted = self.scale[:, None]*y.value + self.origin[:, None]
            controls = u.value.copy()
            # Remove only solver-level violations, then validate the resulting controls.
            controls /= np.maximum(1, np.linalg.norm(controls, axis=0))[None, :]
            controls[:, :coast_steps] = 0
            candidate = TrajectoryPlan(t, dt, ignition_time, predicted, controls)
            virtual_size = np.max(np.abs(virtual.value))
            if virtual_size <= 1e-7 and self.validate(candidate, x0):
                actual, _ = rollout(x0, controls, dt, self.body, self.acceleration)
                self.last_failure = None
                return TrajectoryPlan(t, dt, ignition_time, actual, controls)
            try:
                actual, _ = rollout(x0, controls, dt, self.body, self.acceleration)
                mismatch = np.max(np.linalg.norm((actual - predicted)/self.scale[:, None], axis=0))
            except (ValueError, FloatingPointError):
                mismatch = np.inf
            error = mismatch + np.sum(np.abs(virtual.value))
            if error > previous_error*1.5:
                radius *= 0.5
            else:
                reference = predicted
                previous_error = error
                radius = min(2.0, radius*1.2)
            self.last_failure = (f"Nonlinear validation did not converge: "
                                 f"scaled prediction error={mismatch:.3g}, virtual control={virtual_size:.3g}")
        return None
