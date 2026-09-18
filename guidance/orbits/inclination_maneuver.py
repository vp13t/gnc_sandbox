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

class SetInclinationManeuver(Maneuver):
    def __init__(self, inclination_target: float, body: CelestialBody, spacecraft: Spacecraft):
        self.name = "SetInclinationManeuver"
        self.i_target = inclination_target
        self.body = body
        self.spacecraft = spacecraft
        self.thruster = self.spacecraft.thrusters["X_body"]

        self.burn_ready = False
        self.burn_end_time = None
        self.burn_ended = False

    def plan(self, state: State, t: float):
        mu = self.body.mu
        initial_oe = OE.rv_to_oe(state.pos(), state.vel(), mu)

        # Check ascending and descending nodes, use whichever has lower velocity.
        r_an, v_an = OE.projected_rv_at_anomaly(initial_oe, mu, -initial_oe.omega)
        r_dn, v_dn = OE.projected_rv_at_anomaly(initial_oe, mu, np.pi-initial_oe.omega)
        ascending_node = 1
        if np.linalg.norm(v_an) <= np.linalg.norm(v_dn):
            rn_vec = r_an
            vn_vec = v_an

            oe_an = copy(initial_oe)
            oe_an.theta = -oe_an.omega
            burn_pt_E = OE.eccentric_anomaly(oe_an)
        else:
            rn_vec = r_dn
            vn_vec = v_dn
            ascending_node = -1

            oe_an = copy(initial_oe)
            oe_an.theta = np.pi-oe_an.omega
            burn_pt_E = OE.eccentric_anomaly(oe_an)

        rn = np.linalg.norm(rn_vec)
        rhat = rn_vec / rn
        h = initial_oe.h(self.body.mu)
        hhat = h / np.linalg.norm(h)
        that = np.cross(hhat, rhat)
        vt = np.dot(vn_vec, that)
        deltai = ascending_node * wrap_pi(self.i_target - initial_oe.i)

        DeltaV = vt * ((np.cos(deltai) - 1) * that + np.sin(deltai) * hhat)
        self.DeltaV_mag = np.linalg.norm(DeltaV)
        if self.DeltaV_mag < 1e-10:
            self.burn_ended = True
        else:
            self.DeltaV_hat = DeltaV / self.DeltaV_mag

        accel = self.thruster.force / self.spacecraft.mass
        self.burn_duration = self.DeltaV_mag / accel
        
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