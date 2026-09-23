import numpy as np
from copy import copy
from sim.state import State
from sim.bodies import CelestialBody
import guidance.oe as OE
from guidance.maneuver import Maneuver
from sim.angles import quat2vec, rotate_vec_towards_vec
from sim.forces import Force, thrust, control_torque
from sim.angles import wrap_pi, wrap_2pi
from spacecraft.spacecraft import Spacecraft
from controllers.pointing_lyapunov import pointing_lyapunov
from enum import Enum

class PitchManeuver(Maneuver):
    def __init__(
        self,
        des_oe: OE.OrbitalElements,
        body: CelestialBody,
        spacecraft: Spacecraft,
        des_pitch = np.deg2rad(5),
        angle_of_attack_constraint = np.deg2rad(3)
    ):
        self.name = "PitchManeuver"
        self.des_oe = des_oe
        self.des_pitch = des_pitch
        self.aoa_lim = angle_of_attack_constraint
        self.body = body
        self.spacecraft = spacecraft
        self.thruster = self.spacecraft.thrusters["X_body"]

        self.burn_ready = False
        self.burn_on = False
        self.burn_ended = False

    def plan(self, state: State, t: float):
        self.h = self.des_oe.h()
        self.rhat0 = state.pos()/np.linalg.norm(state.pos())
    
    def act(self, state: State, t: float):
        vhat = state.vel() / np.linalg.norm(state.vel())
        pitch = np.arccos(np.clip(np.dot(self.rhat0, vhat), -1.0, 1.0))
        body_x = quat2vec(state.rot())
        angle_of_attack = np.arccos(np.clip(np.dot(body_x, vhat), -1.0, 1.0))

        pitch_outstanding = self.des_pitch - pitch
        aoa_budget = self.aoa_lim - angle_of_attack
        turn_angle = np.clip(pitch_outstanding, -aoa_budget, aoa_budget)

        d = np.cross(self.h, state.pos())
        side = d - np.dot(d, vhat) * vhat
        side_norm = np.linalg.norm(side)
        pointing = np.cos(turn_angle)*vhat + np.sin(turn_angle)*side/side_norm

        L, V = pointing_lyapunov(state, pointing, self.spacecraft)
        torque = control_torque(state, L, self.spacecraft)

        thrust_force = Force()
        if abs(pitch_outstanding) <= np.deg2rad(1e-1):
            self.burn_ended = True
        else:
            thrust_force = thrust(state, self.spacecraft, self.thruster)

        return {"torque": torque, "X_body": thrust_force}
    
    def check_complete(self, state: State, t: float):
        return self.burn_ended