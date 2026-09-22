"""A solved trajectory, independent of CVXPY and the next solve attempt."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TrajectoryPlan:
    t0: float
    dt: float
    ignition_time: float
    X: np.ndarray                 # Physical states, (6, N + 1).
    U: np.ndarray                 # Normalized inertial thrust, (3, N).

    def __post_init__(self):
        x, u = np.array(self.X, dtype=float, copy=True), np.array(self.U, dtype=float, copy=True)
        if self.dt <= 0 or u.ndim != 2 or u.shape[0] != 3 or u.shape[1] == 0:
            raise ValueError("A trajectory needs a positive timestep and nonempty (3, N) controls")
        if x.shape != (6, u.shape[1] + 1) or not np.all(np.isfinite(x)) or not np.all(np.isfinite(u)):
            raise ValueError("Trajectory states and controls must be finite and have matching shapes")
        if not np.isfinite([self.t0, self.dt, self.ignition_time]).all():
            raise ValueError("Trajectory timing must be finite")
        if not self.t0 <= self.ignition_time <= self.t0 + self.dt * u.shape[1]:
            raise ValueError("Ignition must lie within the trajectory")
        x.setflags(write=False)
        u.setflags(write=False)
        object.__setattr__(self, "X", x)
        object.__setattr__(self, "U", u)

    @property
    def tf(self):
        return self.t0 + self.dt * self.U.shape[1]

    def state_at(self, t):
        """Interpolate for tracking; an expired plan must be handled by the caller."""
        times = self.t0 + self.dt * np.arange(self.X.shape[1])
        return np.array([np.interp(t, times, row) for row in self.X])

    def average_control(self, start, end):
        """Integrate held controls exactly, with zero thrust outside the plan."""
        if end <= start:
            raise ValueError("Control interval must have positive duration")
        edges = self.t0 + self.dt * np.arange(self.U.shape[1] + 1)
        overlap = np.maximum(0, np.minimum(edges[1:], end) - np.maximum(edges[:-1], start))
        return self.U @ overlap / (end - start)

    def pointing_direction(self, t, fallback):
        """Look ahead through coast intervals; never chase a small pulse residual."""
        k = max(0, int(np.floor((t - self.t0) / self.dt)))
        magnitudes = np.linalg.norm(self.U, axis=0)
        indices = np.flatnonzero((np.arange(self.U.shape[1]) >= k) & (magnitudes > 1e-3))
        if indices.size:
            i = indices[0]
            return self.U[:, i] / magnitudes[i]
        return np.asarray(fallback)

    def smooth_control(self, t):
        """Interpolate interval-center controls for attitude reference derivatives.

        Pulse accounting still uses average_control(), the exact held-command
        integral. Small coast controls are excluded by the reference generator.
        """
        centers = self.t0 + self.dt*(np.arange(self.U.shape[1]) + 0.5)
        return np.array([np.interp(t, centers, row) for row in self.U])
