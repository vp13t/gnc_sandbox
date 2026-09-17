import numpy as np
import quaternion
from scipy.integrate import solve_ivp
from sim.forces import Force

def dynamics(t, x, mass, force: Force):
    xdot, ydot, zdot = x[3:6]
    w1, w2, w3 = x[10:13]
    Omega = np.array([
        [0, -w3, w2, w1],
        [w3, 0, -w1, w2],
        [-w2, w1, 0, w3],
        [-w1, -w2, -w3, 0]
    ])
    Qdot = 0.5 * Omega @ x[6:10]  # Quaternion derivative
    return np.array([
        xdot,    # dx/dt
        ydot,    # dy/dt
        zdot,    # dz/dt
        force.xddot,  # dxdot/dt
        force.yddot,  # dydot/dt
        force.zddot,  # dzdot/dt
        Qdot[0],      # dq1/dt
        Qdot[1],      # dq2/dt
        Qdot[2],      # dq3/dt
        Qdot[3],      # dq4/dt
        force.w1dot,  # dw1/dt
        force.w2dot,  # dw2/dt
        force.w3dot   # dw3/dt
    ])

class State:
    x: float
    y: float
    z: float
    xdot: float
    ydot: float
    zdot: float
    q1: float
    q2: float
    q3: float
    q4: float # Scalar last
    w1: float
    w2: float
    w3: float
    
    def __init__(self, r, v, q, w):
        self.x, self.y, self.z = r
        self.xdot, self.ydot, self.zdot = v
        self.q1, self.q2, self.q3, self.q4 = q
        self.w1, self.w2, self.w3 = w

    def pos(self):
        return np.array([self.x, self.y, self.z])
    
    def rotvec(self):
        return np.array([self.q1, self.q2, self.q3, self.q4])

    def rot(self):
        self.normalize_quat()
        return np.quaternion(self.q4, self.q1, self.q2, self.q3)
    
    def vel(self):
        return np.array([self.xdot, self.ydot, self.zdot])
    
    def omega(self):
        return np.array([self.w1, self.w2, self.w3])
    
    def vec(self):
        return np.concatenate((self.pos(), self.vel(), self.rotvec(), self.omega()))
    
    def normalize_quat(self):
        norm = np.linalg.norm(self.rotvec())
        if norm > 0:
            self.q1 /= norm
            self.q2 /= norm
            self.q3 /= norm
            self.q4 /= norm

    def update(self, dt, mass, force=Force()):
        ic = self.vec()
        sol = solve_ivp(
            fun=dynamics, 
            t_span=(0,dt),
            y0=ic,
            args=(mass, force,),
        )
        vec = sol.y[:, -1]
        self.x, self.y, self.z = vec[0:3]
        self.xdot, self.ydot, self.zdot = vec[3:6]
        self.q1, self.q2, self.q3, self.q4 = vec[6:10]
        self.w1, self.w2, self.w3 = vec[10:13]
        self.normalize_quat()
        return self.vec()
