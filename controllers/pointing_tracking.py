"""Track a moving inertial thrust axis, including its reference angular rate."""

import numpy as np
import quaternion


def pointing_tracking(state, direction, spacecraft, angular_rate=None,
                      angular_acceleration=None, frequency=0.2):
    """Return inertial torque for a critically damped axis-tracking controller.

    Frequency is in rad/s. As elsewhere in this simulator, control torque has
    no modeled actuator limit. Roll about the thrust axis is damped, not targeted.
    """
    rotation = quaternion.as_rotation_matrix(state.rot())
    axis = rotation @ spacecraft.thrusters["X_body"].direction
    axis /= np.linalg.norm(axis)
    direction = np.asarray(direction)/np.linalg.norm(direction)
    cross = np.cross(axis, direction)
    sine = np.linalg.norm(cross)
    cosine = np.clip(axis @ direction, -1, 1)
    if sine > 1e-8:
        error = np.arctan2(sine, cosine)*cross/sine
    elif cosine < 0:
        # A deterministic escape axis for a 180-degree pointing error.
        orthogonal = np.eye(3)[np.argmin(np.abs(axis))]
        cross = np.cross(axis, orthogonal)
        error = np.pi*cross/np.linalg.norm(cross)
    else:
        error = np.zeros(3)
    omega = state.omega()
    reference_rate = np.zeros(3) if angular_rate is None else angular_rate
    reference_acceleration = np.zeros(3) if angular_acceleration is None else angular_acceleration
    alpha = (reference_acceleration + frequency**2*error
             + 2*frequency*(reference_rate-omega))
    inertia = rotation @ spacecraft.inertia @ rotation.T
    return inertia @ alpha + np.cross(omega, inertia @ omega)
