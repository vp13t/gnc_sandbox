# Visualization

Set these independent flags in `main.py`:

- `plot_pos`: save the position trajectory to `plots/pos/<scenario>.png`.
- `plot_att`: save attitude history to `plots/att/<scenario>.png`.
- `plot_ground`: save the ground track to `plots/ground/<scenario>_<body>.png`.
  Select the body with `ground_body` (Earth by default).
- `animate`: render the existing spacecraft animation.

When any plotting flag is enabled, the initial state and every simulation
timestep are recorded with their timestamps. All selected figures are saved before the interactive
Matplotlib windows open. Output directories are created automatically.

`plot_attitude` transforms each scalar-last quaternion into the spacecraft's
body X, Y, and Z unit vectors expressed in inertial coordinates. Three lines
trace the tips of these vectors over time, with endpoint dots to keep stationary
axes visible. X is red, Y is blue, and Z is green, matching `animation.py` through
the shared `BODY_AXIS_COLORS` palette. The origin is a white point. Axes have
equal scales and dimensionless coordinates from -1 to 1.

For direct use, both `plot_trajectory(Xhist, path, show=False)` and
`plot_attitude(Xhist, path, show=False)` save a figure and return `(fig, ax)`.
Set `show=True` to display it immediately. Attitude plotting normalizes copies
of the quaternions and leaves the supplied state history unchanged.

## Ground tracks

`plot_ground_track(Xhist, times, body, path, show=False, rotation_epoch=0.0)`
radially projects each position onto the selected body's radius, then applies
the inverse of the body's accumulated rotation at that timestamp. The red
trace is expressed in a body-fixed frame centered on `body.pos_I`, rendered
with its center at the plot origin. The translucent sphere uses `body.color`
and lets the complete track remain visible. Coordinates are in metres.

`times` must contain one timestamp per state, in simulation seconds. The
body-fixed axes coincide with the inertial axes at `rotation_epoch`; `main.py`
uses `scene.t0` as this epoch. Rotation thereafter uses the body's constant
`omega_I`, including a tilted axis or a zero spin rate. This reference does
not define a real-world prime meridian or calendar epoch.

For surface points without rendering, use
`project_ground_track(Xhist, times, body, rotation_epoch=...)`, which returns
an `(N, 3)` array. A spacecraft that co-rotates with the body has a stationary
ground track; an inertially stationary spacecraft moves backward relative
to the rotating surface. This visualization does not change simulation forces
or contact dynamics.
