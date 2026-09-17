import numpy as np
import quaternion
from sim.bodies import CelestialBody
from sim.frames import IX, IY

def pointing_error(state, target: CelestialBody | np.ndarray):
    q = state.rot()

    if isinstance(target, CelestialBody):
        pointing_vec = target.pos_I - state.pos()
    else:
        pointing_vec = target
    pointing_vec = pointing_vec / np.linalg.norm(pointing_vec)
    axis = np.cross(IX, pointing_vec)
    axis_norm = np.linalg.norm(axis)
    if axis_norm < 1e-6:
        if np.dot(IX, pointing_vec) < 0:
            qref = quaternion.from_rotation_vector(np.pi * IY)
        else:
            qref = np.quaternion(1, 0, 0, 0)
    else:
        axis = axis / axis_norm
        angle = np.arccos(np.clip(np.dot(IX, pointing_vec), -1.0, 1.0))
        qref = quaternion.from_rotation_vector(axis * angle)

    eq = (1/qref) * q
    return eq

def ang_vel_error(state, wref):
    w = state.omega()
    return w - wref