import numpy as np
from copy import copy
from sim.angles import rotate_vec_about_axis
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


class SetArgumentOfPeriapsisManeuver(Maneuver):
    def __init__(self, omega_target: float, body: CelestialBody, spacecraft: Spacecraft):
        self.name = "SetArgumentOfPeriapsisManeuver"
        self.omega_target = omega_target
        self.body = body
        self.spacecraft = spacecraft
        self.thruster = self.spacecraft.thrusters["X_body"]

        self.burn_ready = False
        self.burn_end_time = None
        self.burn_ended = False

    def plan(self, state: State, t: float):
        mu = self.body.mu
        initial_oe = OE.rv_to_oe(state.pos(), state.vel(), body=self.body)
        if initial_oe.e < 1e-6:
            raise ValueError("Periapsis undefined for circular orbits. Cannot rotate argument of periapsis.")

        ra_vec, va_vec = OE.projected_rv_at_anomaly(initial_oe, Apse.APOAPSIS.value)
        ra = np.linalg.norm(ra_vec)
        va = np.linalg.norm(va_vec)
        self.DeltaV_hat = va_vec / va

        a1 = initial_oe.a
        a2 = ra

        # Calculate difference between circular and current apoapsis velocity
        self.DeltaV_mag = np.sqrt(mu/ra) - va
        if self.DeltaV_mag < 0:
            self.DeltaV_mag = -self.DeltaV_mag
            self.DeltaV_hat = -self.DeltaV_hat
        if self.DeltaV_mag < 1e-10:
            self.burn_ended = True

        accel = self.thruster.force / self.spacecraft.mass
        self.burn_duration = self.DeltaV_mag / accel
        
        new_oe = copy(initial_oe)
        new_oe.omega = self.omega_target
        ra2_vec, va2_vec = OE.projected_rv_at_anomaly(new_oe, Apse.APOAPSIS.value)
        self.DeltaV_hat2 = -va2_vec/(np.linalg.norm(va2_vec))

        burn_pt_E = Apse.APOAPSIS.value
        burn_pt_M = burn_pt_E - initial_oe.e * np.sin(burn_pt_E)
        self.period = initial_oe.period()
        burn_pt_tpp = self.period * burn_pt_M / (2*np.pi)
    
        start_tpp = burn_pt_tpp - self.burn_duration/2
        start_E = tpp_eccentric_anomaly(start_tpp, initial_oe.a, initial_oe.e, mu)
        self.burn_time = OE.time_until_eccentric_anomaly(initial_oe, start_E) + t

        sec_per_rad = np.sqrt(ra**3/self.body.mu)
        self.transfer_period = sec_per_rad * 2*np.pi
        delta_omega = wrap_2pi(self.omega_target - initial_oe.omega)
        self.burn_2_time = self.burn_time + delta_omega * sec_per_rad

        self.burn_idx = 1

    def reset(self):
        self.burn_ready = False
        self.burn_end_time = None
        self.burn_ended = False

    def act(self, state: State, t: float):
        if self.burn_ended and self.burn_idx == 1:
            self.DeltaV_hat = self.DeltaV_hat2
            self.burn_time = self.burn_2_time
            self.period = self.transfer_period
            self.reset()
            self.burn_idx = 2

        L, V = pointing_lyapunov(state, self.DeltaV_hat, self.spacecraft)
        torque = control_torque(state, L, self.spacecraft)

        time_until_burn_start = self.burn_time - t

        # Stabilize pointing before starting burn
        if np.sum(V) < 1e-6 and time_until_burn_start > 0:
            self.burn_ready = True
        
        thrust_force = Force()
        if self.burn_ready and time_until_burn_start <= 0 and not self.burn_end_time:
            if np.sum(V) > 1e-6:
                # Abort readied burn
                self.burn_ready = False
                self.burn_time += self.period
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
        return self.burn_ended and self.burn_idx >= 2
