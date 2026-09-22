# Powered descent

`BrakeManeuver` and `LandManeuver` share `PoweredDescentManeuver`. The optimizer
returns a `TrajectoryPlan` containing copied, read-only physical states,
normalized inertial thrust commands, and its own timing. A failed solve never
replaces the active plan. Accepted replans preserve the original arrival time.

## Planning and attitude preparation

Before the braking half-space is reached, a ballistic preview predicts corridor
entry. Planning starts `preview_lead` seconds before that entry. The pre-ignition
coast has zero thrust and must stay outside the central body; the braking
half-space (and landing cone, when applicable) applies from ignition onward.

The first significant planned thrust direction is the attitude target during
coast. The actual pointing controller is propagated without main-engine thrust
to estimate slew time. Candidate generation allows up to three coast/direction
updates. A candidate requiring more than `max_slew_time` is rejected. The current
orbital scenario uses a 600-second preview and slew allowance: its initial
attitude prediction needs approximately 502 seconds.

The optimizer uses scaled target-relative states, separate gravity
linearizations at each node, sparse block dynamics, a trust region, and penalized
virtual dynamics control. It runs at most 12 convexification iterations, with a
five-second CLARABEL limit per subproblem. Virtual control must become negligible.
Acceptance additionally requires physical terminal/control constraints and a
nonlinear RK4 rollout, with clearance/corridor checks at samples no more than one
second apart. This sampled validation is not a continuous collision certificate.

The nonlinear rollout assumes ideal continuous inertial thrust. Attitude and
on/off pulse tracking are evaluated separately in the simulation. Nonlinear
prediction currently uses only the same static central body's gravity and
constant spacecraft mass; add any new physical forces to the predictor as well.

## Execution and replanning

`GuidanceSchedule.control_dt` is the duration for which a command is held.
`main.py` sets it to the simulation timestep times the guidance update stride.
Other entry points should set it explicitly. This is independent of the
optimization timestep. Control integration accounts for partial plan intervals.

`pending_dv` is requested minus commanded velocity change, in m/s. A pulse is
selected only when it reduces that vector error and the spacecraft is aligned.
The subtraction uses the actual body thrust direction and control hold duration.
It is booked for the upcoming interval, matching `State.update`'s held-control
convention. Environmental forces are reevaluated at RK4 stages; the thrust
vector (N) and control torque (N m) remain held. Pulse delta-v uses `F*dt/m`;
the integrator applies the current inertia and Euler's gyroscopic term to
compute angular acceleration. Attitude slew prediction uses the same split
as `scene.step(X)`. Do not reuse this accounting unchanged with an actuator that can
reject commands or an integrator that changes thrust direction within the hold.
Attitude points along planned thrust, not a small pulse residual.

Tracking checks compare actual state to interpolated physical plan states.
Replanning is limited to one attempt per `replan_interval` (default 10 seconds),
including failed attempts. Acceptance clears old pulse debt because the new
trajectory starts at the actual state. `solve_attempts`, `accepted_plans`,
`last_failure`, and `optimizer.last_iterations` provide diagnostics.

Completion uses actual position and velocity, never just elapsed time. Braking
requires position error <= 100 m, lateral speed <= 10 m/s and total speed <=
100 m/s at the elevated target. The configured successor landing maneuver must
also prepare a validated plan from that actual state before the schedule switches.
Landing uses position error <= 1.5 m, lateral speed <= 0.8 m/s and total speed
<= 1.0 m/s near the surface target. The landing optimizer aims inside 20% of
that box (`terminal_fraction=0.2`), leaving room for execution error. Braking
retains a 90% planning fraction. Landing completion checks the physical box
without requiring re-entry into the approach cone: the cone's apex is 0.1 m
above the surface, so a valid touchdown can be below it. The optimizer still
enforces and validates the approach cone along the planned trajectory.
Missing an accepted plan's deadline raises
`DescentFailure` instead of silently declaring success or holding its last
control indefinitely.

The thrust impulse resolution is `force / mass * control_dt`. For this rocket,
that is 14 m/s per pulse at `control_dt = 1.0` s (a coarse rate used only to
speed up interactive testing) and 1.4 m/s per pulse at the canonical
`control_dt = 0.1` s. The boxes above assume the canonical rate; at the coarse
rate a single pulse can overshoot them, so both maneuvers' tracking checks and
terminal boxes should be treated as validated against `dt = 0.1`, not the
faster 1.0 s setting. Landing is also the last maneuver in the schedule: once
it reports complete, control drops to zero, so any leftover position error at
that instant becomes an uncontrolled fall the rest of the way to the surface,
which is part of why its box stays tight even though Braking's is
comparatively loose.

SoundingRocket's 14 kN thrust (thrust/weight ~1.43 at Earth gravity) is sized
for `LandManeuver`'s near-hover terminal descent margin, not for ascent;
`GravityTurnManeuver`'s `ignition_angle` parameter (coast until the flight
path has rotated far enough from vertical before igniting prograde) is what
keeps the ascent gravity turn from running away into an eccentric orbit at
this higher thrust -- see `guidance/ascent/gravity_turn_maneuver.py`.

`LAND_powered` uses `planning_throttle_limit=0.9` for landing. Execution can
use full thrust, leaving 1.4 m/s² of acceleration to correct tracking errors.
At a planning limit of 1.0, the nominal final braking burn saturates the engine.
A small downward velocity error then accumulates into a position deficit that
feedback cannot recover, and subsequent convex subproblems need nonzero virtual
control to reach the target. Nonlinear validation correctly rejects those
plans: increasing solver iterations or loosening validation does not restore
the missing physical braking authority. Keep the reserve when changing the
landing duration or tracking thresholds.

Thrust reserve and terminal margin solve different problems. Reserve permits
feedback to arrest accumulated trajectory error during powered flight. The
tighter terminal target accounts for pulse quantization: at a 0.1 s control
period, a nominal terminal speed of 0.2 m/s leaves 0.8 m/s for tracking error,
whereas the former 0.9 m/s target left only 0.1 m/s. Coarser control periods
need a corresponding review of impulse resolution and landing tolerances.

## Verification

Run the regression suite without rendering:

```sh
.venv/bin/python -m unittest discover -s tests -v
```

It covers immutable plans, control integration, pulse delta-v conservation,
actual thrust direction, plan rejection/acceptance, retry timing, slew scheduling,
deadline failure, successor handoff, and nonlinear braking/landing validation.
It also replays the captured `LAND_powered` landing entry through attitude
tracking, on/off thrust, schedule completion, and physical surface contact.
