import numpy as np
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

class Apse(Enum):
    PERIAPSIS = 0
    APOAPSIS = np.pi

    def __neg__(self):
        return self.APOAPSIS if self == self.PERIAPSIS else self.PERIAPSIS


class SetApseDistManeuver(Maneuver):
    def __init__(self, r_target_apse: float, target_apse: Apse, body: CelestialBody, spacecraft: Spacecraft):
        self.name = "SetApseDistManeuver"
        self.r_target = r_target_apse
        self.targest_apse = target_apse
        self.burn_apse = -target_apse
        self.body = body
        self.spacecraft = spacecraft
        self.thruster = self.spacecraft.thrusters["X_body"]

        self.burn_ready = False
        self.burn_end_time = None
        self.burn_ended = False

    def plan(self, state: State, t: float):
        mu = self.body.mu
        initial_oe = OE.rv_to_oe(state.pos(), state.vel(), mu)

        # Calculate rv for the apse opposite the one we're trying to change.
        ra_vec, va_vec = OE.projected_rv_at_anomaly(initial_oe, mu, self.burn_apse.value)
        ra = np.linalg.norm(ra_vec)
        va = np.linalg.norm(va_vec)
        self.DeltaV_hat = va_vec / va

        if (self.targest_apse == Apse.APOAPSIS and self.r_target < ra):
            raise ValueError(f"Cannot set apoapsis radius {self.r_target:.2f}m below periapsis radius {ra:.2f}m.")
        elif (self.targest_apse == Apse.PERIAPSIS and self.r_target > ra):
            raise ValueError(f"Cannot set periapsis radius {self.r_target:.2f}m above apoapsis radius {ra:.2f}m.")

        a1 = initial_oe.a
        a2 = (ra + self.r_target)/2

        self.DeltaV_mag = np.sqrt(mu*(2/ra - 1/a2)) - va
        if self.DeltaV_mag < 0:
            self.DeltaV_mag = -self.DeltaV_mag
            self.DeltaV_hat = -self.DeltaV_hat
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

def SetPeriapsisDistManeuver(r: float, body: CelestialBody, spacecraft: Spacecraft):
    return SetApseDistManeuver(r, Apse.PERIAPSIS, body, spacecraft)

def SetApoapsisDistManeuver(r: float, body: CelestialBody, spacecraft: Spacecraft):
    return SetApseDistManeuver(r, Apse.APOAPSIS, body, spacecraft)

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
