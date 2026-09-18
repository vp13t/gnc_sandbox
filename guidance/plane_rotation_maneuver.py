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
from guidance.apse_maneuver import Apse

class PlaneRotationManeuver(Maneuver):
    def __init__(self, rotation_angle: float, body: CelestialBody, spacecraft: Spacecraft):
        self.rot = rotation_angle
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
        h_vec = initial_oe.h(self.body.mu)

        n = np.cross(ra_vec, va_vec)
        t = np.cross(h_vec, ra_vec)
        DCM_ItoApoRTN = np.column_stack((
            ra_vec / np.linalg.norm(ra),
            t / np.linalg.norm(t),
            n / np.linalg.norm(n)
        ))
        DCM_R = np.array([
            [1, 0, 0],
            [0, np.cos(self.rot), np.sin(self.rot)],
            [0, -np.sin(self.rot), np.cos(self.rot)]
        ])
        vrot = DCM_ItoApoRTN @ DCM_R @ DCM_ItoApoRTN.T @ va_vec

        deltaV = vrot - va_vec
        self.DeltaV_mag = np.linalg.norm(deltaV)
        if self.DeltaV_mag < 1e-10:
            self.burn_ended = True
        else:
            self.DeltaV_hat = deltaV / self.DeltaV_mag

        accel = self.thruster.force / self.spacecraft.mass
        self.burn_duration = self.DeltaV_mag / accel
        
        burn_pt_E = Apse.APOAPSIS.value
        burn_pt_M = burn_pt_E - initial_oe.e * np.sin(burn_pt_E)
        period = initial_oe.period(mu)
        burn_pt_tpp = period * burn_pt_M / (2*np.pi)
        start_tpp = burn_pt_tpp - self.burn_duration/2

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
            if t >= self.burn_end_time:
                # End burn
                self.burn_ended = True
            else:
                # Burn
                thrust_force = thrust(state, self.spacecraft, self.thruster)

        return {"torque": torque, "X_body": thrust_force}
    
    def check_complete(self, state: State, t: float):
        return self.burn_ended