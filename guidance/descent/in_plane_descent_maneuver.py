import numpy as np
from copy import copy
from sim.state import State
from sim.bodies import CelestialBody
import guidance.oe as OE
from guidance.maneuver import Maneuver
from sim.forces import Force, thrust, control_torque
from sim.angles import wrap_pi, wrap_2pi
from spacecraft.spacecraft import Spacecraft
from guidance.kepler import tpp_eccentric_anomaly
from controllers.pointing_lyapunov import pointing_lyapunov
from enum import Enum

from guidance.orbits.apse_maneuver import Apse

class InPlaneDescentManeuver(Maneuver):
    def __init__(
        self,
        theta_target: float,
        alt_target: float,
        body: CelestialBody,
        spacecraft: Spacecraft,
        fixed_initial_oe = None
    ):
        self.name = "InPlaneDescentManeuver"
        self.burn_apse = Apse.PERIAPSIS if wrap_pi(theta_target) > 0 else Apse.APOAPSIS
        self.theta_target = theta_target
        self.alt_target = alt_target
        self.body = body
        self.spacecraft = spacecraft
        self.thruster = self.spacecraft.thrusters["X_body"]
        self.fixed_initial_oe = fixed_initial_oe

        self.burn_ready = False
        self.burn_end_time = None
        self.burn_ended = False

    def plan(self, state: State, t: float):
        mu = self.body.mu
        if self.fixed_initial_oe:
            initial_oe = self.fixed_initial_oe
        else:
            initial_oe = OE.rv_to_oe(state.pos(), state.vel(), mu)
        r_theta_tgt_curr_orbit, _ = OE.projected_rv_at_anomaly(initial_oe, self.body.mu, self.theta_target)
        rhat_theta_tgt_curr_orbit = r_theta_tgt_curr_orbit / np.linalg.norm(r_theta_tgt_curr_orbit)
        r_t = self.body.radius + self.alt_target
        self.target = self.body.pos_I + rhat_theta_tgt_curr_orbit * r_t

        ra_vec, va_vec = OE.projected_rv_at_anomaly(initial_oe, mu, self.burn_apse.value)
        ra = np.linalg.norm(ra_vec)
        va = np.linalg.norm(va_vec)
        self.DeltaV_hat = -va_vec / va
        ct = np.cos(self.theta_target)

        if self.burn_apse == Apse.PERIAPSIS:
            # Retrograde at old periapsis: it becomes the new apoapsis.
            rp = r_t * ra * (1 - ct) / (2 * ra - r_t * (1 + ct))
        else:
            # Retrograde at apoapsis: it remains the new apoapsis.
            rp = r_t * ra * (1 + ct) / (2 * ra - r_t * (1 - ct))
        a2 = (ra + rp)/2
        e2 = (ra-rp)/(ra+rp)

        new_oe = copy(initial_oe)
        new_oe.a = a2
        new_oe.e = e2
        if self.burn_apse == Apse.PERIAPSIS:
            # Periapsis becomes apoapsis
            new_oe.omega = wrap_pi(new_oe.omega + np.pi)

        _, va2_vec = OE.projected_rv_at_anomaly(new_oe, mu, Apse.APOAPSIS.value)
        va2 = np.linalg.norm(va2_vec)

        self.DeltaV_mag = va - va2
        if self.DeltaV_mag < 0:
            raise ValueError("Invalid descent trajectory, not descending!")
        if self.DeltaV_mag < 1e-10:
            self.burn_ended = True

        accel = self.thruster.force / self.spacecraft.mass
        self.burn_duration = self.DeltaV_mag / accel
        
        burn_pt_E = self.burn_apse.value
        burn_pt_M = burn_pt_E - initial_oe.e * np.sin(burn_pt_E)
        self.period = initial_oe.period(mu)
        burn_pt_tpp = self.period * burn_pt_M / (2*np.pi)
    
        start_tpp = burn_pt_tpp - self.burn_duration/2
        start_E = tpp_eccentric_anomaly(start_tpp, initial_oe.a, initial_oe.e, mu)
        self.burn_time = OE.time_until_eccentric_anomaly(initial_oe, self.body.mu, start_E) + t
    
    def act(self, state: State, t: float):
        L, V = pointing_lyapunov(state, self.DeltaV_hat, self.spacecraft)
        torque = control_torque(state, L, self.spacecraft)

        time_until_burn_start = ((self.burn_time - t + self.period/2) % self.period) - self.period/2

        # Stabilize pointing before starting burn
        if np.sum(V) < 1e-6 and time_until_burn_start > 0:
            self.burn_ready = True
        
        thrust_force = Force()
        if self.burn_ready and time_until_burn_start <= 0 and not self.burn_end_time:
            if np.sum(V) > 1e-6:
                # Abort readied burn
                self.burn_ready = False
            else:
                # Start readied burn
                self.burn_end_time = t + self.burn_duration
        if self.burn_end_time:
            if t >= self.burn_end_time:
                # End burn
                self.burn_ended = True
            else:
                # Burn
                thrust_force = thrust(state, self.spacecraft, self.thruster)

        return {"torque": torque, "X_body": thrust_force}
    
    def check_complete(self, state: State, t: float):
        return self.burn_ended
