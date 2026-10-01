import numpy as np
import quaternion
from sim.bodies import CelestialBody
from sim.angles import angle_between_quats, vec2quat
from sim.frames import IX
from sim.forces import thrust, Force
from controllers.error import pointing_error, ang_vel_error
from spacecraft.spacecraft import Spacecraft

def pointing_lyapunov(state, target: CelestialBody | np.ndarray, spacecraft: Spacecraft, debug=False):
    """
    Return raw inertial control torque in N m and the Lyapunov value.
    Kr has units N m; Kw has units N m s (angles are in radians).

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


def pointing_lyapunov_thruster(state, target: CelestialBody | np.ndarray, spacecraft: Spacecraft, pending_aa=np.zeros(3), debug=False):
    """
    Return RCS thruster forces and the pending angular acceleration integral in inertial axes.
    Kr has units N m; Kw has units N m s (angles are in radians).

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

    V = 1/2 * (ew_b.T @ spacecraft.inertia @ ew_b) + (Kr/2 * np.trace(np.eye(3) - err_rot))
    if debug:
        angle_between = angle_between_quats(pointing_error(state, target), vec2quat(IX))
        print(f"V: {np.sum(V)} Angular Error: {angle_between}")
    
    pending_aa_b = R.T @ pending_aa + L_b / spacecraft.inertia.diagonal()

    thrust_dict = {}
    thrf_Xp = spacecraft.thrusters["X+@Y+Z+"].force
    thrf_Xm = spacecraft.thrusters["X-@Y+Z-"].force
    if pending_aa_b[0] >= thrf_Xp / spacecraft.mass:
        thrust_dict["X+@Y+Z+"] = thrust(state, spacecraft, spacecraft.thrusters["X+@Y+Z+"])
        thrust_dict["X+@Y-Z-"] = thrust(state, spacecraft, spacecraft.thrusters["X+@Y-Z-"])
        pending_aa_b[0] -= (2 * thrf_Xp / spacecraft.mass)
    elif pending_aa_b[0] <= -thrf_Xm / spacecraft.mass:
        thrust_dict["X-@Y+Z-"] = thrust(state, spacecraft, spacecraft.thrusters["X-@Y+Z-"])
        thrust_dict["X-@Y-Z+"] = thrust(state, spacecraft, spacecraft.thrusters["X-@Y-Z+"])
        pending_aa_b[0] += (2 * thrf_Xm / spacecraft.mass)

    thrf_Yp = spacecraft.thrusters["Y+@X+Z+"].force
    thrf_Ym = spacecraft.thrusters["Y-@X+Z-"].force
    if pending_aa_b[1] >= thrf_Yp / spacecraft.mass:
        thrust_dict["Y+@X+Z+"] = thrust(state, spacecraft, spacecraft.thrusters["Y+@X+Z+"])
        thrust_dict["Y+@X-Z-"] = thrust(state, spacecraft, spacecraft.thrusters["Y+@X-Z-"])
        pending_aa_b[1] -= (2 * thrf_Yp / spacecraft.mass)
    elif pending_aa_b[1] <= -thrf_Ym / spacecraft.mass:
        thrust_dict["Y-@X+Z-"] = thrust(state, spacecraft, spacecraft.thrusters["Y-@X+Z-"])
        thrust_dict["Y-@X-Z+"] = thrust(state, spacecraft, spacecraft.thrusters["Y-@X-Z+"])
        pending_aa_b[1] += (2 * thrf_Ym / spacecraft.mass)

    thrf_Zp = spacecraft.thrusters["Z+@X+Y+"].force
    thrf_Zm = spacecraft.thrusters["Z-@X+Y-"].force
    if pending_aa_b[2] >= thrf_Zp / spacecraft.mass:
        thrust_dict["Z+@X+Y+"] = thrust(state, spacecraft, spacecraft.thrusters["Z+@X+Y+"])
        thrust_dict["Z+@X-Y-"] = thrust(state, spacecraft, spacecraft.thrusters["Z+@X-Y-"])
        pending_aa_b[2] -= (2 * thrf_Zp / spacecraft.mass)
    elif pending_aa_b[2] <= -thrf_Zm / spacecraft.mass:
        thrust_dict["Z-@X+Y-"] = thrust(state, spacecraft, spacecraft.thrusters["Z-@X+Y-"])
        thrust_dict["Z-@X-Y+"] = thrust(state, spacecraft, spacecraft.thrusters["Z-@X-Y+"])
        pending_aa_b[2] += (2 * thrf_Zm / spacecraft.mass)
    
    pending_aa_I = R @ pending_aa_b

    return thrust_dict, pending_aa_I
