from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import quaternion
from scipy.spatial.transform import Rotation

from sim.bodies import Bodies, Earth
from visualization.style import BODY_AXIS_COLORS


def _style_axes(ax, labels, *, legend_location="best"):
    for axis, label in zip((ax.xaxis, ax.yaxis, ax.zaxis), labels):
        axis.set_pane_color((0, 0, 0, 1))
        axis.line.set_color("white")
        axis.set_label_text(label, color="white")
        axis.get_offset_text().set_color("white")
    ax.tick_params(colors="white")
    ax.grid(False)
    ax.legend(loc=legend_location, facecolor="black", edgecolor="white", labelcolor="white")


def _save_figure(fig, name):
    path = Path(name)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight")


def plot_trajectory(Xhist, name, show=True):
    """Plot State snapshots in an interactive 3D figure and return (fig, ax).

    Limits depend only on the trajectory, with equal scales and a small margin.
    Drag to rotate; use Matplotlib's controls to zoom. Pass show=False to
    customize or save the figure without opening a window.
    """
    positions = np.asarray([state.pos() for state in Xhist], dtype=float)
    if positions.size == 0:
        raise ValueError("Xhist must contain at least one state.")
    if positions.ndim != 2 or positions.shape[1] != 3 or not np.isfinite(positions).all():
        raise ValueError("State positions must be finite three-dimensional vectors.")

    fig = plt.figure(facecolor="black")
    ax = fig.add_subplot(111, projection="3d", facecolor="black")
    ax.plot(*positions.T, color="red", label="Trajectory")
    for body in Bodies:
        ax.scatter(
            *body.pos_I, color=body.color, s=40, label=body.name,
            depthshade=False, axlim_clip=True,
        )

    longitude, colatitude = np.meshgrid(
        np.linspace(0, 2 * np.pi, 61), np.linspace(0, np.pi, 31)
    )
    ax.plot_surface(
        Earth.pos_I[0] + Earth.radius * np.sin(colatitude) * np.cos(longitude),
        Earth.pos_I[1] + Earth.radius * np.sin(colatitude) * np.sin(longitude),
        Earth.pos_I[2] + Earth.radius * np.cos(colatitude),
        color=Earth.color, alpha=0.25, linewidth=0,
        rcount=31, ccount=61, axlim_clip=True,
    )

    lower = positions.min(axis=0)
    upper = positions.max(axis=0)
    center = (lower + upper) / 2
    # A one-meter minimum span handles a single state or stationary history.
    half_width = 0.525 * max(float(np.max(upper - lower)), 1.0)
    ax.set_xlim(center[0] - half_width, center[0] + half_width)
    ax.set_ylim(center[1] - half_width, center[1] + half_width)
    ax.set_zlim(center[2] - half_width, center[2] + half_width)
    ax.set_box_aspect((1, 1, 1))
    ax.set_autoscale_on(False)

    _style_axes(ax, ("X (m)", "Y (m)", "Z (m)"))
    fig.tight_layout()

    _save_figure(fig, name)

    if show:
        plt.show()

    return fig, ax


def plot_attitude(Xhist, name, show=True):
    """Plot body X/Y/Z unit-vector tip histories in inertial 3D coordinates.

    Each line joins consecutive samples of one body axis on the unit sphere.
    Convert scalar-last State quaternions in a batch without changing history.
    Quaternion sign changes do not affect the plotted directions.
    """
    attitudes = np.asarray([state.rotvec() for state in Xhist], dtype=float)
    if attitudes.size == 0:
        raise ValueError("Xhist must contain at least one state.")
    if attitudes.ndim != 2 or attitudes.shape[1] != 4 or not np.isfinite(attitudes).all():
        raise ValueError("State attitudes must be finite four-component quaternions.")
    # Rescale before normalizing to accommodate nonunit quaternions without
    # overflowing their norm. Work on the copied array, not State snapshots.
    magnitudes = np.max(np.abs(attitudes), axis=1)
    if np.any(magnitudes == 0):
        raise ValueError("State attitudes must have nonzero quaternions.")
    attitudes /= magnitudes[:, None]
    attitudes /= np.linalg.norm(attitudes, axis=1)[:, None]
    rotations = quaternion.as_rotation_matrix(
        quaternion.as_quat_array(attitudes[:, [3, 0, 1, 2]]))

    fig = plt.figure(figsize=(7, 7), facecolor="black")
    ax = fig.add_subplot(111, projection="3d", facecolor="black")
    for column, axis in enumerate("XYZ"):
        directions = rotations[:, :, column]
        ax.plot(*directions.T, color=BODY_AXIS_COLORS[axis], label=f"Body {axis}", linewidth=1,
                marker=".", markevery=[0, len(directions)-1], markersize=4)
    ax.scatter(0, 0, 0, color="white", s=25, depthshade=False, label="Origin")
    ax.set_xlim(-1.05, 1.05)
    ax.set_ylim(-1.05, 1.05)
    ax.set_zlim(-1.05, 1.05)
    ticks = [-1, -0.5, 0, 0.5, 1]
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_zticks(ticks)
    ax.set_box_aspect((1, 1, 1))
    ax.set_autoscale_on(False)
    ax.set_title("Body-axis attitude history", color="white")
    _style_axes(ax, ("Inertial X", "Inertial Y", "Inertial Z"), legend_location="upper left")
    fig.tight_layout()
    _save_figure(fig, name)

    if show:
        plt.show()

    return fig, ax


def project_ground_track(Xhist, times, body, *, rotation_epoch=0.0):
    """Return surface points (m) in the selected body's rotating frame.

    Body-fixed and inertial axes coincide at rotation_epoch. Subtract the
    body's center, project radially onto its sphere, then undo its rotation
    at each timestamp. This supports arbitrary spin axes and offset centers.
    """
    positions = np.asarray([state.pos() for state in Xhist], dtype=float)
    times = np.asarray(times, dtype=float)
    if positions.size == 0:
        raise ValueError("Xhist must contain at least one state.")
    if positions.ndim != 2 or positions.shape[1] != 3 or not np.isfinite(positions).all():
        raise ValueError("State positions must be finite three-dimensional vectors.")
    if times.shape != (len(positions),) or not np.isfinite(times).all():
        raise ValueError("Provide one finite timestamp per state.")
    if np.any(np.diff(times) < 0) or not np.isfinite(rotation_epoch):
        raise ValueError("Timestamps must be nondecreasing and rotation_epoch finite.")
    center = np.asarray(body.pos_I, dtype=float)
    omega = np.asarray(body.omega_I, dtype=float)
    if (center.shape != (3,) or omega.shape != (3,)
            or not np.isfinite(center).all() or not np.isfinite(omega).all()
            or not np.isfinite(body.radius) or body.radius <= 0):
        raise ValueError("The body needs a finite center/spin and a positive radius.")
    radial = positions - center
    distances = np.linalg.norm(radial, axis=1)
    if np.any(distances == 0) or not np.isfinite(distances).all():
        raise ValueError("Ground track is undefined at the body's center.")
    surface = body.radius * (radial / distances[:, None])
    inverse_spin = Rotation.from_rotvec(-(times-rotation_epoch)[:, None]*omega)
    return inverse_spin.apply(surface)


def plot_ground_track(Xhist, times, body, name, show=True, *, rotation_epoch=0.0):
    """Plot the rotating-body ground track and a sphere in body.color.

    The sphere is centered at the origin in body-fixed coordinates. A
    translucent surface keeps both near- and far-side track segments visible.
    """
    ground = project_ground_track(Xhist, times, body, rotation_epoch=rotation_epoch)
    fig = plt.figure(figsize=(7, 7), facecolor="black")
    ax = fig.add_subplot(111, projection="3d", facecolor="black")
    longitude, colatitude = np.meshgrid(
        np.linspace(0, 2*np.pi, 81), np.linspace(0, np.pi, 41))
    radius = body.radius
    ax.plot_surface(
        radius*np.sin(colatitude)*np.cos(longitude),
        radius*np.sin(colatitude)*np.sin(longitude),
        radius*np.cos(colatitude),
        color=body.color, alpha=0.3, shade=False, linewidth=0,
        rcount=41, ccount=81, label=body.name,
    )
    ax.plot(*ground.T, color="red", linewidth=1, label="Ground track",
            marker=".", markevery=[0, len(ground)-1], markersize=4)
    ax.scatter(0, 0, 0, color="white", s=15, depthshade=False)
    for set_limits in (ax.set_xlim, ax.set_ylim, ax.set_zlim):
        set_limits(-1.05*radius, 1.05*radius)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.set_major_locator(plt.MaxNLocator(5))
    ax.set_box_aspect((1, 1, 1))
    ax.set_autoscale_on(False)
    ax.set_title(f"{body.name} ground track", color="white")
    _style_axes(ax, ("Body-fixed X (m)", "Body-fixed Y (m)", "Body-fixed Z (m)"),
                legend_location="upper left")
    fig.tight_layout()
    _save_figure(fig, name)
    if show:
        plt.show()
    return fig, ax
