import numpy as np
import quaternion

IX = np.array([1.0, 0.0, 0.0])
IY = np.array([0.0, 1.0, 0.0])
IZ = np.array([0.0, 0.0, 1.0])

def InertialFrame():
    return np.eye(3)

def RTNFrame(state: "State"):
    r = state.pos()
    v = state.vel()

    n = np.cross(r, v)
    t = np.cross(h, r)

    return np.column_stack((
        r / np.linalg.norm(r),
        t / np.linalg.norm(t),
        n / np.linalg.norm(n)
    ))

def QuaternionFrame(state: "State"):
    return quaternion.as_rotation_matrix(state.rot())

def DCM(frame1, frame2):
    return frame2.T @ frame1

def DCM_oe_BtoI(oe):
    a, e, i, Omega, omega, theta = oe.vec()

    R_Omega = np.array([
        [np.cos(Omega), -np.sin(Omega), 0],
        [np.sin(Omega),  np.cos(Omega), 0],
        [0,              0,             1]
    ])

    R_i = np.array([
        [1, 0, 0],
        [0, np.cos(i), -np.sin(i)],
        [0, np.sin(i),  np.cos(i)]
    ])

    R_omega = np.array([
        [np.cos(omega), -np.sin(omega), 0],
        [np.sin(omega),  np.cos(omega), 0],
        [0,              0,             1]
    ])

    R_theta = np.array([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta),  np.cos(theta), 0],
        [0,              0,             1]
    ])

    DCM_total = R_Omega @ R_i @ R_omega @ R_theta
    return DCM_total
