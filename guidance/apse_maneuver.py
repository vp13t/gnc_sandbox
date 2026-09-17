import numpy as np
from sim.state import State
from sim.bodies import CelestialBody
import guidance.oe as OE
from guidance.maneuver import Maneuver
from sim.forces import Force, thrust, control_torque
from sim.angles import wrap_pi, wrap_2pi
from collections.abc import Callable
from spacecraft.spacecraft import Spacecraft
from guidance.kepler import tpp_eccentric_anomaly
from controllers.pointing_lyapunov import pointing_lyapunov

class SetPeriapsisDistManeuver(Maneuver):
    def __init__(self, rp: float, body: CelestialBody, spacecraft: Spacecraft):
        self.rp = rp
        self.body = body
        self.spacecraft = spacecraft
        self.thruster = self.spacecraft.thrusters["X_body"]

        self.burn_ready = False
        self.burn_end_time = None
        self.burn_ended = False

    def plan(self, state: State, t: float):
        mu = self.body.mu
        initial_oe = OE.rv_to_oe(state.pos(), state.vel(), mu)
        ra_vec, va_vec = OE.projected_rv_apoapsis(initial_oe, mu)
        ra = np.linalg.norm(ra_vec)
        va = np.linalg.norm(va_vec)
        self.DeltaV_hat = va_vec / va

        a1 = initial_oe.a
        a2 = (ra + self.rp)/2

        self.DeltaV_mag = np.sqrt(mu*(2/ra - 1/a2)) - va
        if self.DeltaV_mag < 0:
            self.DeltaV_mag = -self.DeltaV_mag
            self.DeltaV_hat = -self.DeltaV_hat
        if self.DeltaV_mag < 1e-10:
            self.burn_ended = True

        accel = self.thruster.force / self.spacecraft.mass
        self.burn_duration = self.DeltaV_mag / accel
        
        apoapsis_E = np.pi
        period = initial_oe.period(mu)
        apoapsis_tpp = period / 2
        start_tpp = apoapsis_tpp - self.burn_duration/2

        self.start_E = tpp_eccentric_anomaly(start_tpp, initial_oe.a, initial_oe.e, mu)
    
    def act(self, state: State, t: float):
        L, V = pointing_lyapunov(state, self.DeltaV_hat, self.spacecraft)
        torque = control_torque(state, L, self.spacecraft)
        curr_oe = OE.rv_to_oe(state.pos(), state.vel(), self.body.mu)
        # Positive eccentric anomaly near apoapsis
        E = OE.eccentric_anomaly(curr_oe)
        phase_from_burn_start = wrap_pi(E - self.start_E)

        # Stabilize pointing before starting burn
        if np.sum(V) < 1e-6 and phase_from_burn_start < 0:
            self.burn_ready = True
        
        thrust_force = Force()
        if self.burn_ready and phase_from_burn_start >= 0 and not self.burn_end_time:
            if np.sum(V) > 1e-6:
                # Abort readied burn
                self.burn_ready = False
            else:
                # Start readied burn
                self.burn_end_time = t + self.burn_duration
        if self.burn_end_time:
            # Burn
            thrust_force = thrust(state, self.spacecraft, self.thruster)
            if t > self.burn_end_time:
                # End burn
                self.burn_ended = True

        return {"torque": torque, "X_body": thrust_force}
    
    def check_complete(self, state: State, t: float):
        return self.burn_ended


class SetApoapsisDistManeuver(Maneuver):
    def __init__(self, ra: float, body: CelestialBody, spacecraft: Spacecraft):
        self.ra = ra
        self.body = body
        self.spacecraft = spacecraft
        self.thruster = self.spacecraft.thrusters["X_body"]

        self.burn_ready = False
        self.burn_end_time = None
        self.burn_ended = False

    def plan(self, state: State, t: float):
        mu = self.body.mu
        initial_oe = OE.rv_to_oe(state.pos(), state.vel(), mu)
        rp_vec, vp_vec = OE.projected_rv_periapsis(initial_oe, mu)
        rp = np.linalg.norm(rp_vec)
        vp = np.linalg.norm(vp_vec)
        self.DeltaV_hat = vp_vec / vp

        a1 = initial_oe.a
        a2 = (self.ra + rp)/2

        self.DeltaV_mag = np.sqrt(mu*(2/rp - 1/a2)) - vp
        if self.DeltaV_mag < 0:
            self.DeltaV_mag = -self.DeltaV_mag
            self.DeltaV_hat = -self.DeltaV_hat
        if self.DeltaV_mag < 1e-10:
            self.burn_ended = True

        accel = self.thruster.force / self.spacecraft.mass
        self.burn_duration = self.DeltaV_mag / accel
        
        periapsis_E = 0.0
        period = initial_oe.period(mu)
        periapsis_tpp = 0.0
        start_tpp = periapsis_tpp - self.burn_duration/2

        # Symmetric eccentric anomalies near periapsis
        self.start_E = tpp_eccentric_anomaly(start_tpp, initial_oe.a, initial_oe.e, mu)
    
    def act(self, state: State, t: float):
        L, V = pointing_lyapunov(state, self.DeltaV_hat, self.spacecraft)
        torque = control_torque(state, L, self.spacecraft)
        curr_oe = OE.rv_to_oe(state.pos(), state.vel(), self.body.mu)
        # Symmetric eccentric anomaly near periapsis
        E = OE.eccentric_anomaly(curr_oe)
        phase_from_burn_start = wrap_pi(E - self.start_E)

        # Stabilize pointing before starting burn
        if np.sum(V) < 1e-6 and phase_from_burn_start < 0:
            self.burn_ready = True
        
        thrust_force = Force()
        if self.burn_ready and phase_from_burn_start >= 0 and not self.burn_end_time:
            if np.sum(V) > 1e-6:
                # Abort readied burn
                self.burn_ready = False
            else:
                # Start readied burn
                self.burn_end_time = t + self.burn_duration
        if self.burn_end_time:
            # Burn
            thrust_force = thrust(state, self.spacecraft, self.thruster)
            if t > self.burn_end_time:
                # End burn
                self.burn_ended = True

        return {"torque": torque, "X_body": thrust_force}
    
    def check_complete(self, state: State, t: float):
        return self.burn_ended


def HohmannTransferIn(r: float, body: CelestialBody, spacecraft: Spacecraft):
    return [
        SetPeriapsisDistManeuver(r, body, spacecraft),
        SetApoapsisDistManeuver(r, body, spacecraft)
    ]

def HohmannTransferOut(r: float, body: CelestialBody, spacecraft: Spacecraft):
    return [
        SetApoapsisDistManeuver(r, body, spacecraft),
        SetPeriapsisDistManeuver(r, body, spacecraft)
    ]

def BiellipticTransfer(rt: float, rf: float, body: CelestialBody, spacecraft: Spacecraft):
    return [
        SetApoapsisDistManeuver(rt, body, spacecraft),
        SetPeriapsisDistManeuver(rf, body, spacecraft),
        SetApoapsisDistManeuver(rf, body, spacecraft)
    ]
