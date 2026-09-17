import numpy as np
import quaternion
from sim.bodies import CelestialBody
from sim.angles import angle_between_quats, vec2quat
from sim.frames import IX
from controllers.error import pointing_error, ang_vel_error
from spacecraft.spacecraft import Spacecraft

def pointing_lyapunov(state, target: CelestialBody | np.ndarray, spacecraft: Spacecraft, debug=False):
    """
    V = (1/2 w^T I w) + (Kr 1/2 tr(I_3 - err_rot))
    Vdot = w^T L + Kr er^T w
    """    
    err_rot = quaternion.as_rotation_matrix(pointing_error(state, target))
    er_skew_sym = (err_rot - err_rot.T)/2
    er = np.array([er_skew_sym[2,1], er_skew_sym[0,2], er_skew_sym[1,0]])

    R = quaternion.as_rotation_matrix(state.rot())
    ew = ang_vel_error(state, np.zeros(3))
    ew_b = R.T @ ew

    Kr = spacecraft.gains["Kr"]
    Kw = spacecraft.gains["Kw"]
    L_b = -Kr*er - Kw*ew_b
    L_I = R @ L_b

    V = 1/2 * (ew_b.T @ spacecraft.inertia @ ew_b) + (Kr/2 * np.trace(np.eye(3) - err_rot))
    if debug:
        angle_between = angle_between_quats(pointing_error(state, target), vec2quat(IX))
        print(f"V: {np.sum(V)} Angular Error: {angle_between}")

    return L_I, V