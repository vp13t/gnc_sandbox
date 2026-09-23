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

class PlaneRotationManeuver(Maneuver):
    def __init__(self, rotation_angle: float, body: CelestialBody, spacecraft: Spacecraft):
        self.name = "PlaneRotationManeuver"
        self.rot = rotation_angle
        self.body = body
        self.spacecraft = spacecraft
        self.thruster = self.spacecraft.thrusters["X_body"]

        self.burn_ready = False
        self.burn_end_time = None
        self.burn_ended = False

    def plan(self, state: State, t: float):
        mu = self.body.mu
        initial_oe = OE.rv_to_oe(state.pos(), state.vel(), body=self.body)

        ra_vec, va_vec = OE.projected_rv_apoapsis(initial_oe)
        ra = np.linalg.norm(ra_vec)
        va = np.linalg.norm(va_vec)
        h_vec = initial_oe.h()

        n = np.cross(ra_vec, va_vec)
        th = np.cross(h_vec, ra_vec)
        DCM_ItoApoRTN = np.column_stack((
            ra_vec / np.linalg.norm(ra),
            th / np.linalg.norm(th),
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
        self.period = initial_oe.period()
        burn_pt_tpp = self.period * burn_pt_M / (2*np.pi)
    
        start_tpp = burn_pt_tpp - self.burn_duration/2
        start_E = tpp_eccentric_anomaly(start_tpp, initial_oe.a, initial_oe.e, mu)
        self.burn_time = OE.time_until_eccentric_anomaly(initial_oe, start_E) + t
    
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