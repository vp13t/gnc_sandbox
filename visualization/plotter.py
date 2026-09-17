import matplotlib.pyplot as plt
import numpy as np

from sim.bodies import Bodies, Earth


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

    for axis, label in ((ax.xaxis, "X (m)"), (ax.yaxis, "Y (m)"), (ax.zaxis, "Z (m)")):
        axis.set_pane_color((0, 0, 0, 1))
        axis.line.set_color("white")
        axis.set_label_text(label, color="white")
        axis.get_offset_text().set_color("white")
    ax.tick_params(colors="white")
    ax.grid(False)
    ax.legend(facecolor="black", edgecolor="white", labelcolor="white")
    fig.tight_layout()

    plt.savefig(name, dpi=300, bbox_inches="tight")

    if show:
        plt.show()

    return fig, ax
