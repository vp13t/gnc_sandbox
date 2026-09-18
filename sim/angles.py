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

def rotate_vec_about_axis(vec, axis, angle):
    cross = np.cross(axis, vec)
    dot = np.dot(axis, vec)
    return vec*np.cos(angle) + cross*np.sin(angle) + axis*dot*(1-np.cos(angle))

def rotate_vec_towards_vec(vec_moving, vec_target, angle):
    target_prime = np.cross(np.cross(vec_moving, vec_target), vec_moving)
    target_prime = target_prime / np.linalg.norm(target_prime)
    vec_new = np.cos(angle)*vec_moving + np.sin(angle)*target_prime
    return vec_new