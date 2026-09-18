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

class RiseManeuver(Maneuver):
    def __init__(self, altitude: float, body: CelestialBody, spacecraft: Spacecraft):
        self.name = "RiseManeuver"
        self.des_r = altitude + body.radius
        self.body = body
        self.spacecraft = spacecraft
        self.thruster = self.spacecraft.thrusters["X_body"]

        self.burn_ready = False
        self.burn_on = False
        self.burn_ended = False

    def plan(self, state: State, t: float):
        rel_pos = state.pos() - self.body.pos_I
        self.Vhat = rel_pos / np.linalg.norm(rel_pos)
    
    def act(self, state: State, t: float):
        L, V = pointing_lyapunov(state, self.Vhat, self.spacecraft)
        torque = control_torque(state, L, self.spacecraft)

        # Stabilize pointing before starting burn
        if np.sum(V) < 1e-6:
            self.burn_ready = True
        
        thrust_force = Force()
        if self.burn_ready and not self.burn_on:
            if np.sum(V) > 1e-6:
                # Abort readied burn
                self.burn_ready = False
            else:
                # Start readied burn
                self.burn_on = True
        if self.burn_on:
            rel_pos = state.pos() - self.body.pos_I
            if np.linalg.norm(rel_pos) >= self.des_r:
                # End burn
                self.burn_ended = True
            else:
                # Burn
                thrust_force = thrust(state, self.spacecraft, self.thruster)

        return {"torque": torque, "X_body": thrust_force}
    
    def check_complete(self, state: State, t: float):
        return self.burn_ended