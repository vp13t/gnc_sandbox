import numpy as np

def oriented_angle(u, v, normal):
    return np.arctan2(
        np.dot(normal, np.cross(u, v)),
        np.dot(u, v),
    ) % (2 * np.pi)

def wrap_pi(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi

def wrap_2pi(angle):
    return angle % (2 * np.pi)