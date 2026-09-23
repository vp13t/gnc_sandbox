import numpy as np
from copy import copy
from sim.state import State
from sim.bodies import CelestialBody
import guidance.oe as OE
from guidance.maneuver import Maneuver
from sim.forces import Force, thrust, control_torque
from sim.angles import wrap_pi, wrap_2pi
from spacecraft.spacecraft import Spacecraft
from controllers.pointing_lyapunov import pointing_lyapunov
from enum import Enum

class GravityTurnManeuver(Maneuver):
    def __init__(
        self,
        v_target: float,
        body: CelestialBody,
        spacecraft: Spacecraft,
        ignition_angle: float = np.pi/2,
    ):
        self.name = "GravityTurnManeuver"
        self.v_target = v_target
        self.body = body
        self.spacecraft = spacecraft
        self.thruster = self.spacecraft.thrusters["X_body"]
        # Angle between the radius and velocity vectors (0 = still climbing
        # straight up, pi/2 = apoapsis, purely tangential) at which the burn
        # is allowed to ignite. Defaults to apoapsis.
        self.ignition_angle = ignition_angle

        self.burn_ready = False
        self.burn_ended = False

    def plan(self, state: State, t: float):
        pass

    def act(self, state: State, t: float):
        Vmag = np.linalg.norm(state.vel())
        vhat = state.vel() / Vmag
        L, V = pointing_lyapunov(state, vhat, self.spacecraft)
        torque = control_torque(state, L, self.spacecraft)

        r = state.pos() - self.body.pos_I
        rhat = r / np.linalg.norm(r)
        radial_speed = np.dot(state.vel(), rhat)

        thrust_force = Force()
        # Coast (no thrust) until the flight path has rotated at least
        # ignition_angle away from vertical: thrusting prograde any earlier
        # reinforces the still-mostly-radial velocity (thrust adds energy
        # along whatever direction the vehicle already happens to be
        # moving), which can run away into an escape trajectory at this
        # thrust/weight ratio instead of letting gravity curve the
        # trajectory over first, as an actual gravity turn relies on.
        if not self.burn_ready and radial_speed <= Vmag*np.cos(self.ignition_angle):
            self.burn_ready = True
        if self.burn_ready:
            if Vmag >= self.v_target:
                self.burn_ended = True
            else:
                thrust_force = thrust(state, self.spacecraft, self.thruster)

        return {"torque": torque, "X_body": thrust_force}

    def check_complete(self, state: State, t: float):
        return self.burn_ended
