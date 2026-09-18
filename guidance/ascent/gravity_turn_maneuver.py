import numpy as np
from copy import copy
from sim.state import State
from sim.bodies import CelestialBody
import guidance.oe as OE
from guidance.maneuver import Maneuver
from sim.angles import vec2quat, angle_between_quats, rotate_vec_about_axis, rotate_vec_towards_vec
from sim.forces import Force, thrust, control_torque
from sim.angles import wrap_pi, wrap_2pi
from spacecraft.spacecraft import Spacecraft
from guidance.kepler import tpp_eccentric_anomaly
from controllers.pointing_lyapunov import pointing_lyapunov
from enum import Enum

class GravityTurnManeuver(Maneuver):
    def __init__(
        self,
        duration: float,
        body: CelestialBody,
        spacecraft: Spacecraft,
    ):
        self.name = "GravityTurnManeuver"
        self.duration = duration
        self.body = body
        self.spacecraft = spacecraft
        self.thruster = self.spacecraft.thrusters["X_body"]

        self.burn_ended = False

    def plan(self, state: State, t: float):
        self.end_t = t + self.duration
    
    def act(self, state: State, t: float):
        vhat = state.vel() / np.linalg.norm(state.vel())
        L, V = pointing_lyapunov(state, vhat, self.spacecraft)
        torque = control_torque(state, L, self.spacecraft)

        thrust_force = Force()
        if t >= self.end_t:
            self.burn_ended = True
        else:
            thrust_force = thrust(state, self.spacecraft, self.thruster)

        return {"torque": torque, "X_body": thrust_force}
    
    def check_complete(self, state: State, t: float):
        return self.burn_ended