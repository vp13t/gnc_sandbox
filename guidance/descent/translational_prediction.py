"""Nonlinear translational prediction with inertial, held thrust commands."""

import numpy as np


def step_translational(x, u, dt, body, acceleration):
    """RK4 step; unlike the optimizer this uses inverse-square gravity."""
    def derivative(state):
        r = state[:3] - body.pos_I
        radius = np.linalg.norm(r)
        if radius < 1.0 or not np.isfinite(radius):
            raise ValueError("Invalid radius in translational prediction")
        return np.r_[state[3:], -body.mu * r / radius**3 + acceleration * u]

    k1 = derivative(x)
    k2 = derivative(x + dt * k1 / 2)
    k3 = derivative(x + dt * k2 / 2)
    k4 = derivative(x + dt * k3)
    return x + dt * (k1 + 2*k2 + 2*k3 + k4) / 6


def rollout(x0, controls, dt, body, acceleration, max_step=1.0):
    """Return node states and dense samples for clearance/constraint checks."""
    subdivisions = max(1, int(np.ceil(dt / max_step)))
    step = dt / subdivisions
    x = np.array(x0, dtype=float, copy=True)
    nodes, samples = [x.copy()], [x.copy()]
    for u in controls.T:
        for _ in range(subdivisions):
            x = step_translational(x, u, step, body, acceleration)
            samples.append(x.copy())
        nodes.append(x.copy())
    return np.column_stack(nodes), np.column_stack(samples)
