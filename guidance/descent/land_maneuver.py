"""Terminal powered descent using the shared trajectory executor."""

import numpy as np

from guidance.descent.descent_optimizer import DescentLimits
from guidance.descent.powered_descent_maneuver import PoweredDescentManeuver


class LandManeuver(PoweredDescentManeuver):
    def __init__(self, landing_site_uvec, glide_slope_angle, body, spacecraft,
                 duration, dt=1.0, log=True, **options):
        if not 0 <= glide_slope_angle <= np.pi/2:
            raise ValueError("Glideslope angle must lie between zero and pi/2")
        self.glide_slope_angle = glide_slope_angle
        # Aim inside the actual handoff box. At the scenario's 0.1 s cadence,
        # pulse quantization alone can leave about 0.7 m/s of velocity error;
        # a nominal 0.9 m/s terminal speed leaves insufficient tracking margin.
        options.setdefault("terminal_fraction", 0.2)
        # The remaining distance after completion is an unpowered drop, so
        # preserve the tight physical position/speed limits at handoff.
        super().__init__(
            "LandManeuver", landing_site_uvec, 0.1, body, spacecraft, duration,
            DescentLimits(position=1.5, lateral_speed=0.8, speed=1.0,
                          glide_angle=glide_slope_angle),
            dt, log, **options,
        )

    def at_handoff(self, state):
        # The approach cone ends 0.1 m above the surface. An actual touchdown
        # can pass below that apex while staying inside the landing box; do
        # not keep commanding thrust solely to re-enter the approach cone.
        return bool(
            np.linalg.norm(state.pos()-self.target) <= self.limits.position
            and np.linalg.norm(self.P @ state.vel()) <= self.limits.lateral_speed
            and np.linalg.norm(state.vel()) <= self.limits.speed
        )
