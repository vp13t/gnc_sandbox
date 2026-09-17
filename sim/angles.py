import numpy as np
import quaternion
from sim.frames import IX, IY

def oriented_angle(u, v, normal):
    return np.arctan2(
        np.dot(normal, np.cross(u, v)),
        np.dot(u, v),
    ) % (2 * np.pi)

def wrap_pi(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi

def wrap_2pi(angle):
    return angle % (2 * np.pi)

def vec2quat(pointing_vec):
    pointing_vec = pointing_vec / np.linalg.norm(pointing_vec)
    axis = np.cross(IX, pointing_vec)
    axis_norm = np.linalg.norm(axis)
    if axis_norm < 1e-6:
        if np.dot(IX, pointing_vec) < 0:
            quat = quaternion.from_rotation_vector(np.pi * IY)
        else:
            quat = np.quaternion(1, 0, 0, 0)
    else:
        axis = axis / axis_norm
        angle = np.arccos(np.clip(np.dot(IX, pointing_vec), -1.0, 1.0))
        quat = quaternion.from_rotation_vector(axis * angle)
    return quat

def quat2vec(quat):
    return quaternion.as_rotation_matrix(quat) @ IX

def angle_between_quats(q1, q2):
    return 2*np.arccos(abs(quaternion.as_float_array(q1) @ quaternion.as_float_array(q2)))
