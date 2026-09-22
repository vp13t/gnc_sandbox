"""Shared plan lifecycle and pulse execution for braking and landing."""

from copy import deepcopy
import math

import numpy as np

from controllers.pointing_tracking import pointing_tracking
from guidance.descent.descent_optimizer import DescentOptimizer
from guidance.descent.translational_prediction import step_translational
from guidance.maneuver import Maneuver
from sim.forces import Force, control_torque, gravity, thrust, torque_free_rotation


class DescentFailure(RuntimeError):
    """The planned deadline was reached without a valid handoff."""


class PoweredDescentManeuver(Maneuver):
    def __init__(self, name, landing_site_uvec, target_alt, body, spacecraft,
                 duration, limits, dt=1.0, log=True, *, replan_interval=10.0,
                 preview_lead=300.0, max_slew_time=300.0,
                 position_tracking_limit=1e4, velocity_tracking_limit=100.0,
                 attitude_frequency=0.2, tracking_frequency=0.02,
                 planning_throttle_limit=0.9, terminal_fraction=0.9):
        normal = np.asarray(landing_site_uvec, dtype=float)
        if normal.shape != (3,) or not np.isfinite(normal).all() or np.linalg.norm(normal) == 0:
            raise ValueError("Landing normal must be a finite nonzero three-vector")
        if min(dt, duration, replan_interval, preview_lead, max_slew_time) <= 0:
            raise ValueError("Planning and control time settings must be positive")
        self.name = name
        self.landing_site_uvec = normal / np.linalg.norm(normal)
        self.P = np.eye(3) - np.outer(self.landing_site_uvec, self.landing_site_uvec)
        self.target = body.pos_I + self.landing_site_uvec*(body.radius + target_alt)
        self.body, self.spacecraft = body, spacecraft
        self.thruster = spacecraft.thrusters["X_body"]
        self.duration, self.dt, self.log = duration, dt, log
        self.control_dt = 1.0  # Set by GuidanceSchedule from the actual update period.
        self.limits = limits
        self.optimizer = DescentOptimizer(self.target, self.landing_site_uvec,
                                          body, spacecraft, limits, dt,
                                          throttle_limit=planning_throttle_limit,
                                          terminal_fraction=terminal_fraction)
        self.replan_interval = replan_interval
        self.preview_lead = preview_lead
        self.max_slew_time = max_slew_time
        self.position_tracking_limit = position_tracking_limit
        self.velocity_tracking_limit = velocity_tracking_limit
        self.attitude_frequency = attitude_frequency
        self.tracking_frequency = tracking_frequency
        self.pointing_tolerance = np.deg2rad(3.0)
        self.rate_tolerance = np.deg2rad(0.5)
        self.active_plan = None
        self.arrival_time = None
        self.pending_dv = np.zeros(3)
        self.last_replan_attempt = -np.inf
        self.next_preview_time = -np.inf
        self.last_failure = None
        self.solve_attempts = 0
        self.accepted_plans = 0
        self.burn_ready = self.burn_on = self.burn_ended = False

    def plan(self, state, t):
        # A prepared handoff plan may already exist; do not reset its epoch here.
        pass

    def _fallback_direction(self, state):
        speed = np.linalg.norm(state.vel())
        return -state.vel()/speed if speed > 1e-6 else self.landing_site_uvec

    def _inside_corridor(self, position):
        error = position - self.target
        height = self.landing_site_uvec @ error
        if height < 0:
            return False
        if self.limits.glide_angle is None:
            return True
        angle = self.limits.glide_angle
        return np.cos(angle)*np.linalg.norm(self.P @ error) <= np.sin(angle)*height

    def _coast_to_corridor(self, state):
        """Find a future entry so attitude preparation can start before entry."""
        x = np.r_[state.pos(), state.vel()]
        for seconds in range(math.ceil(self.duration + self.preview_lead) + 1):
            if np.linalg.norm(x[:3] - self.body.pos_I) < self.body.radius:
                break
            if self._inside_corridor(x[:3]):
                return float(seconds)
            x = step_translational(x, np.zeros(3), 1.0, self.body, 0.0)
        return None

    def _aligned(self, state, direction, angular_rate=None):
        force = thrust(state, self.spacecraft, self.thruster)
        actual = np.array([force.xddot, force.yddot, force.zddot])
        actual /= np.linalg.norm(actual)
        reference_rate = np.zeros(3) if angular_rate is None else angular_rate
        return (actual @ direction >= np.cos(self.pointing_tolerance)
                and np.linalg.norm(state.omega()-reference_rate) <= self.rate_tolerance)

    def _slew_time(self, state, direction):
        """Predict the same attitude controller, without firing the main engine."""
        predicted = deepcopy(state)
        step = min(self.control_dt, 1.0)
        for i in range(math.ceil(self.max_slew_time/step) + 1):
            if self._aligned(predicted, direction):
                return i*step
            torque = pointing_tracking(predicted, direction, self.spacecraft,
                                       frequency=self.attitude_frequency)
            forces = (gravity(predicted, self.body, self.spacecraft)
                      + torque_free_rotation(predicted, self.spacecraft)
                      + control_torque(predicted, torque, self.spacecraft))
            predicted.update(step, self.spacecraft.mass, forces)
        return None

    def optimize(self, state, t):
        """Return a validated candidate; never modify the active trajectory."""
        self.last_failure = None
        entry_delay = self._coast_to_corridor(state)
        if entry_delay is None:
            self.last_failure = "Ballistic preview does not enter the descent corridor before impact"
            return None
        if self.active_plan is None and entry_delay > self.preview_lead:
            self.next_preview_time = t + entry_delay - self.preview_lead
            self.last_failure = "Waiting for the pre-ignition planning window"
            return None
        ignition_time = t + entry_delay
        if self.active_plan is not None and t < self.active_plan.ignition_time:
            ignition_time = max(ignition_time, self.active_plan.ignition_time)
        arrival = self.arrival_time
        if arrival is None:
            travel_time = np.linalg.norm(state.pos()-self.target)/max(np.linalg.norm(state.vel()), 1.0)
            horizon = ignition_time-t + travel_time + self.duration
            arrival = t + math.ceil(horizon/self.control_dt)*self.control_dt
        x0 = np.r_[state.pos(), state.vel()]
        for _ in range(3):
            candidate = self.optimizer.solve(x0, t, arrival, ignition_time, self.active_plan)
            if candidate is None:
                self.last_failure = self.optimizer.last_failure
                return None
            direction = candidate.pointing_direction(t, self._fallback_direction(state))
            slew_time = self._slew_time(state, direction)
            if slew_time is None:
                self.last_failure = "Pointing controller cannot align within the configured slew time"
                return None
            if t + slew_time <= candidate.ignition_time + 1e-8:
                return candidate
            # Include the missed preparation time in the next candidate's coast.
            ignition_time = t + math.ceil((slew_time + self.control_dt)/self.control_dt)*self.control_dt
        self.last_failure = "Coast duration and first thrust direction did not settle in three attempts"
        return None

    def validate_plan(self, candidate, state=None):
        x0 = None if state is None else np.r_[state.pos(), state.vel()]
        return self.optimizer.validate(candidate, x0)

    def _accept_plan(self, candidate):
        if self.arrival_time is not None and not np.isclose(candidate.tf, self.arrival_time, atol=1e-6, rtol=0):
            raise ValueError("A replacement plan must preserve the arrival deadline")
        self.active_plan = candidate
        self.arrival_time = candidate.tf
        self.pending_dv[:] = 0
        self.burn_ready = True
        self.accepted_plans += 1

    def _attempt_plan(self, state, t):
        # A failed attempt must also consume the retry interval.
        self.last_replan_attempt = t
        self.solve_attempts += 1
        candidate = self.optimize(state, t)
        if candidate is not None:
            self._accept_plan(candidate)
        if self.log:
            result = "accepted" if candidate is not None else f"rejected: {self.last_failure}"
            print(f"{self.name} plan {result} at t={t:.1f}")
        return candidate is not None

    def prepare_handoff(self, state, t):
        """Called by the previous maneuver before switching the schedule."""
        if t - self.last_replan_attempt < self.replan_interval:
            return False
        return self._attempt_plan(state, t)

    def at_handoff(self, state):
        return bool(
            self._inside_corridor(state.pos())
            and np.linalg.norm(state.pos()-self.target) <= self.limits.position
            and np.linalg.norm(self.P @ state.vel()) <= self.limits.lateral_speed
            and np.linalg.norm(state.vel()) <= self.limits.speed
        )

    def check_complete(self, state, t):
        self.burn_ended = self.at_handoff(state)
        return self.burn_ended

    def _pulse(self, state, direction, desired_u, allow_fire, angular_rate=None):
        """Requested minus commanded delta-v over the upcoming held-force interval."""
        acceleration = self.thruster.force/self.spacecraft.mass
        self.pending_dv += acceleration * desired_u * self.control_dt
        pulse = thrust(state, self.spacecraft, self.thruster)
        actual_dv = np.array([pulse.xddot, pulse.yddot, pulse.zddot])*self.control_dt
        beneficial = self.pending_dv @ actual_dv > 0.5*(actual_dv @ actual_dv)
        if allow_fire and beneficial and self._aligned(state, direction, angular_rate):
            self.pending_dv -= actual_dv
            self.burn_on = True
            return pulse
        self.burn_on = False
        return Force()

    def _tracking_reference(self, state, t):
        """Feedforward thrust plus physical position/velocity feedback."""
        plan = self.active_plan
        fallback = self._fallback_direction(state)
        if plan is None:
            return fallback, np.zeros(3), np.zeros(3), np.zeros(3)
        if t < plan.ignition_time:
            return plan.pointing_direction(t, fallback), np.zeros(3), np.zeros(3), np.zeros(3)
        reference = plan.state_at(t)
        radial_actual = state.pos()-self.body.pos_I
        radial_reference = reference[:3]-self.body.pos_I
        gravity_actual = -self.body.mu*radial_actual/np.linalg.norm(radial_actual)**3
        gravity_reference = -self.body.mu*radial_reference/np.linalg.norm(radial_reference)**3
        frequency = self.tracking_frequency
        correction = (gravity_reference-gravity_actual
                      + frequency**2*(reference[:3]-state.pos())
                      + 2*frequency*(reference[3:]-state.vel()))/self.optimizer.acceleration
        desired = plan.average_control(t, t+self.control_dt) + correction
        desired /= max(1.0, np.linalg.norm(desired))

        def direction_at(sample_time):
            nominal = plan.smooth_control(sample_time)
            # During a fuel-saving coast, prepare for the next burn instead of
            # chasing the direction of a tiny feedback/quantization residual.
            if np.linalg.norm(nominal) < 0.05:
                return plan.pointing_direction(sample_time, fallback)
            command = nominal + correction
            magnitude = np.linalg.norm(command)
            if magnitude < 1e-3:
                return plan.pointing_direction(sample_time, fallback)
            return command/magnitude

        # Evaluate the attitude target at the center of the upcoming command hold.
        center = t + self.control_dt/2
        difference_step = max(plan.dt, self.control_dt)
        before = direction_at(center-difference_step)
        direction = direction_at(center)
        after = direction_at(center+difference_step)
        derivative = (after-before)/(2*difference_step)
        second_derivative = (after-2*direction+before)/difference_step**2
        rate = np.cross(direction, derivative)
        acceleration = np.cross(direction, second_derivative)
        return direction, rate, acceleration, desired

    def act(self, state, t):
        if self.control_dt <= 0:
            raise ValueError("Control interval must be positive")
        if self.burn_ended:
            return {"torque": Force(), "X_body": Force()}
        if self.active_plan is not None and t >= self.active_plan.tf:
            raise DescentFailure(f"{self.name} missed its handoff at t={t:.1f}; "
                                 f"position error={np.linalg.norm(state.pos()-self.target):.1f} m, "
                                 f"speed={np.linalg.norm(state.vel()):.1f} m/s")
        outside = self.active_plan is None
        if self.active_plan is not None:
            predicted = self.active_plan.state_at(t)
            outside = (np.linalg.norm(state.pos()-predicted[:3]) > self.position_tracking_limit
                       or np.linalg.norm(state.vel()-predicted[3:]) > self.velocity_tracking_limit)
        if outside and t >= self.next_preview_time and t-self.last_replan_attempt >= self.replan_interval:
            self._attempt_plan(state, t)

        direction, rate, angular_acceleration, desired_u = self._tracking_reference(state, t)
        allow_fire = False
        if self.active_plan is not None:
            allow_fire = (t >= self.active_plan.ignition_time
                          and t+self.control_dt <= self.active_plan.tf + 1e-8)
        torque = pointing_tracking(state, direction, self.spacecraft, rate,
                                   angular_acceleration, self.attitude_frequency)
        return {"torque": control_torque(state, torque, self.spacecraft),
                "X_body": self._pulse(state, direction, desired_u, allow_fire, rate)}
