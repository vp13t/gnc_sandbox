"""Powered braking into a landing handoff region."""

from guidance.descent.descent_optimizer import DescentLimits
from guidance.descent.powered_descent_maneuver import PoweredDescentManeuver


class BrakeManeuver(PoweredDescentManeuver):
    def __init__(self, landing_site_uvec, target_alt, body, spacecraft, duration,
                 dt=1.0, log=True, *, landing_maneuver=None, **options):
        super().__init__(
            "BrakeManeuver", landing_site_uvec, target_alt, body, spacecraft,
            duration, DescentLimits(position=100.0, lateral_speed=10.0, speed=100.0),
            dt, log, **options,
        )
        self.landing_maneuver = landing_maneuver

    def check_complete(self, state, t):
        if not self.at_handoff(state):
            return False
        # Validate a landing plan at the actual handoff state, then reuse it.
        if self.landing_maneuver is not None:
            self.landing_maneuver.control_dt = self.control_dt
            if not self.landing_maneuver.prepare_handoff(state, t):
                return False
        self.burn_ended = True
        return True
