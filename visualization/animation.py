import numpy as np
import pyvista as pv
from scipy.spatial.transform import Rotation
from weakref import WeakKeyDictionary

from sim.bodies import Bodies, CelestialBody
from sim.frames import IX, IY, IZ, QuaternionFrame
from sim.forces import Force
from sim.state import State
from visualization.camera_mode import CameraMode

_plumes = WeakKeyDictionary()


class _ThrusterPlume:
    """A reusable visual effect, animated in movie time rather than orbital time."""

    def __init__(self, plotter, framerate):
        self.framerate = framerate
        self.time = 0.0
        rng = np.random.default_rng(42)
        self.phase = rng.random(320)
        self.speed = rng.uniform(0.8, 1.2, self.phase.size)
        angle = rng.uniform(0, 2 * np.pi, self.phase.size)
        radius = np.sqrt(rng.random(self.phase.size))
        self.radial = radius[:, None] * np.column_stack((np.cos(angle), np.sin(angle)))
        self.mesh = pv.PolyData(np.zeros((self.phase.size, 3)))
        self.actors = []
        self.color_layers = []
        for name, size, opacity in (
            ("thruster_plume_glow", 0.14, 0.16),
            ("thruster_plume_core", 0.035, 0.85),
        ):
            self.mesh[name] = np.zeros((self.phase.size, 4), dtype=np.uint8)
            self.color_layers.append((name, opacity))
            actor = plotter.add_points(
                self.mesh, name=name, style="points_gaussian",
                scalars=name, rgba=True, emissive=True,
                render_points_as_spheres=False, lighting=False,
                show_scalar_bar=False, pickable=False,
                reset_camera=False, render=False,
            )
            actor.mapper.scale_factor = size
            actor.visibility = False
            self.actors.append(actor)


def update_thruster_plume(plotter, state: State, u=None):
    """Show glowing exhaust along body -X when u['X_body'] is nonzero.

    Missing commands mean no thrust. Particles travel outward and fade over
    movie frames; this is a local visual effect, not simulated exhaust dynamics.
    Geometry stays near the origin to preserve precision at orbital distances.
    """
    plume = _plumes.get(plotter)
    if plume is None:
        plume = _ThrusterPlume(plotter, 60)
        _plumes[plotter] = plume

    firing = False
    if u is not None and isinstance((thrust := u.get("X_body", None)), Force):
        firing = np.any(thrust.force_I)
    for actor in plume.actors:
        actor.visibility = bool(firing)
    if not firing:
        plume.time = 0.0
        return

    age = (plume.phase + plume.time * plume.speed / 0.65) % 1.0
    points = np.empty((age.size, 3))
    points[:, 0] = -0.56 - 3.0 * age
    points[:, 1:] = plume.radial * (0.025 + 0.32 * age)[:, None]
    plume.mesh.points = points
    # Hot white-blue near the nozzle, fading to blue downstream.
    rgba = np.empty((age.size, 4), dtype=np.uint8)
    rgba[:, 0] = 220 * (1 - age) ** 2 + 25
    rgba[:, 1] = 190 * (1 - age) + 50
    rgba[:, 2] = 255
    rgba[:, 3] = 255 * (1 - age) ** 1.5
    # Keep actor opacity at 1: emissive splats use their own additive blending.
    for name, opacity in plume.color_layers:
        layer = rgba.copy()
        layer[:, 3] = rgba[:, 3] * opacity
        plume.mesh[name] = layer

    transform = np.eye(4)
    transform[:3, :3] = Rotation.from_quat(state.rotvec()).as_matrix()
    transform[:3, 3] = state.pos()
    for actor in plume.actors:
        actor.user_matrix = transform
    plume.time += 1.0 / plume.framerate


def init(filename, framerate=60, *, off_screen=True):
    """Open a movie writer and create the scene in meters, with a 1 m cube.

    Requires imageio-ffmpeg. Call save_frame for each state and close to finish.
    """
    plotter = pv.Plotter(off_screen=off_screen, window_size=(1280, 720))
    try:
        plotter.set_background("black")
        for body in Bodies:
            # Local mesh coordinates preserve precision at large distances.
            actor = plotter.add_mesh(
                pv.Sphere(radius=body.radius, theta_resolution=64, phi_resolution=64),
                name=body.name,
                color=body.color,
                lighting=False,
                reset_camera=False,
            )
            actor.position = body.pos_I
        plotter.add_mesh(
            pv.Cube(), name="spacecraft", color="orange", reset_camera=False
        )
        plotter.add_mesh(
            pv.Arrow(start=(0.55, 0, 0), direction=(1, 0, 0), scale=1.0),
            name="spacecraft_Xface",
            color="red",
            reset_camera=False
        )
        plotter.add_mesh(
            pv.Arrow(start=(0, 0.55, 0), direction=(0, 1, 0), scale=1.0),
            name="spacecraft_Yface",
            color="blue",
            reset_camera=False
        )
        plotter.add_mesh(
            pv.Arrow(start=(0, 0, 0.55), direction=(0, 0, 1), scale=1.0),
            name="spacecraft_Zface",
            color="green",
            reset_camera=False
        )
        _plumes[plotter] = _ThrusterPlume(plotter, framerate)
        plotter.open_movie(str(filename), framerate=framerate)
    except Exception:
        _plumes.pop(plotter, None)
        plotter.close()
        raise
    return plotter


def save_frame(plotter, state: State, cam_target: CelestialBody | CameraMode, u=None):
    """Look through the spacecraft along the target direction, offset 5 m up."""
    position = state.pos().astype(float)
    if isinstance(cam_target, CelestialBody):
        tgt_direction = cam_target.pos_I - position
    else:
        match cam_target:
            case CameraMode.VELOCITY_FACING:
                tgt_direction = -state.vel()
                if np.linalg.norm(tgt_direction) == 0:
                    tgt_direction = -IX
            case CameraMode.VELOCITY_FOLLOWING:
                tgt_direction = state.vel()
                if np.linalg.norm(tgt_direction) == 0:
                    tgt_direction = IX
            case CameraMode.NORMAL_FACING:
                tgt_direction = QuaternionFrame(state) @ IY

    distance = np.linalg.norm(tgt_direction)
    if distance == 0:
        raise ValueError("Cannot aim at the target from its center.")
    tgt_direction /= distance

    transform = np.eye(4)
    R = Rotation.from_quat(state.rotvec()).as_matrix()
    transform[:3, :3] = R
    transform[:3, 3] = position
    plotter.actors["spacecraft"].user_matrix = transform
    plotter.actors["spacecraft_Xface"].user_matrix = transform
    plotter.actors["spacecraft_Yface"].user_matrix = transform
    plotter.actors["spacecraft_Zface"].user_matrix = transform
    update_thruster_plume(plotter, state, u)
    
    for body in Bodies:
        plotter.actors[body.name].position = body.pos_I

    up = np.array([0.0, 0.0, 1.0])
    if abs(np.dot(up, tgt_direction)) > 0.99:
        up = np.array([0.0, 1.0, 0.0])
    camera_position = position - 50.0 * tgt_direction + up
    plotter.camera_position = [camera_position, position, up]
    # Perspective depth precision depends strongly on the near plane. A 0.1 m
    # near plane makes Earth and the Sun round to the same depth, letting the
    # Sun bleed through Earth. Put it just in front of the 1 m cube's bounding
    # sphere, including the exhaust when visible, while preserving precision.
    visible_radius = 4.0 if u is not None and u.get("X_body", 0.0) != 0.0 else np.sqrt(3.0) / 2.0
    near = 0.9 * (np.linalg.norm(position - camera_position) - visible_radius)
    # Automatic clipping based on the Sun's distance would hide the nearby cube.
    far = max(
        np.linalg.norm(body.pos_I - camera_position) + body.radius
        for body in Bodies
    )
    plotter.camera.clipping_range = (near, 1.1 * far)
    plotter.write_frame()


def close(plotter):
    """Finalize the video writer and release the rendering window."""
    _plumes.pop(plotter, None)
    plotter.close()
