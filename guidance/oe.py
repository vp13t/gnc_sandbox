import numpy as np
from sim.frames import IX, IY, IZ, DCM_oe_BtoI
from sim.angles import oriented_angle
from copy import copy

class OrbitalElements:
    def __init__(self, a, e, i, Omega, omega, theta):
        self.a = a
        self.e = e
        self.i = i
        self.Omega = Omega
        self.omega = omega
        self.theta = theta
    
    def vec(self):
        return np.array([self.a, self.e, self.i, self.Omega, self.omega, self.theta])

    def __str__(self):
        return f"OrbitalElements(a={self.a}, e={self.e}, i={self.i}, Omega={self.Omega}, omega={self.omega}, theta={self.theta})"
    
    def period(self, mu):
        return 2 * np.pi * np.sqrt(self.a**3 / mu)
    
    def p(self):
        """Semi-latus rectum"""
        return self.a * (1 - self.e**2)
    
    def h(self, mu):
        hmag = np.sqrt(mu * self.p())
        hhat = np.array([
            np.sin(self.i)*np.sin(self.Omega),
            -np.sin(self.i)*np.cos(self.Omega),
            np.cos(self.i)
        ])
        return hmag * hhat

def rv_to_oe(r, v, mu):
    h = np.cross(r, v)

    rnorm = np.linalg.norm(r)
    vnorm = np.linalg.norm(v)
    hnorm = np.linalg.norm(h)

    rhat = r / rnorm
    vhat = v / vnorm
    hhat = h / hnorm

    specific_energy = vnorm**2 / 2 - mu / rnorm

    # a
    a = -mu / (2 * specific_energy)

    # e
    evec = (np.cross(v, h) / mu) - rhat
    e = np.linalg.norm(evec)

    # i
    i = np.arccos(np.clip(hhat[2], -1.0, 1.0))

    n = np.cross(IZ, hhat)
    n_norm = np.linalg.norm(n)
    if n_norm > 1e-10:
        nhat = n / n_norm
        # Omega
        Omega = np.arctan2(n[1], n[0]) % (2 * np.pi)
    else:
        nhat = IX
        # Omega
        Omega = 0.0
    
    if e > 1e-10:
        ehat = evec / e
        omega = oriented_angle(nhat, ehat, hhat)
        theta = oriented_angle(ehat, rhat, hhat)
    else:
        e = 0.0
        omega = 0.0
        theta = oriented_angle(nhat, rhat, hhat)

    return OrbitalElements(a, e, i, Omega, omega, theta)

def oe_to_rv(oe: OrbitalElements, mu):
    a, e, i, Omega, omega, theta = oe.vec()

    specific_energy = -mu / (2 * a)
    p = a * (1 - e**2)
    hnorm = np.sqrt(mu * p)

    dcm = DCM_oe_BtoI(oe)

    rnorm = p / (1 + e * np.cos(theta))
    r_B = np.array([rnorm, 0, 0])
    r_I = dcm @ r_B

    vr = (mu / hnorm) * e * np.sin(theta)
    vt = hnorm / rnorm
    v_B = np.array([vr, vt, 0])
    v_I = dcm @ v_B

    return r_I, v_I

def projected_rv_at_anomaly(oe, mu, theta):
    projected_oe = copy(oe)
    projected_oe.theta = theta
    projected_r, projected_v = oe_to_rv(projected_oe, mu)
    return projected_r, projected_v

def projected_rv_periapsis(oe, mu):
    return projected_rv_at_anomaly(oe, mu, 0)

def projected_rv_apoapsis(oe, mu):
    return projected_rv_at_anomaly(oe, mu, np.pi)

def eccentric_anomaly(oe):
    E = np.arctan2(
        np.sqrt(1 - oe.e**2) * np.sin(oe.theta),
        oe.e + np.cos(oe.theta),
    )
    return E % (2 * np.pi)

def time_until_anomaly(oe, mu, target_theta, k_revs=0):
    T = oe.period(mu)
    E1 = eccentric_anomaly(oe)
    
    oe2 = copy(oe)
    oe2.theta = target_theta
    E2 = eccentric_anomaly(oe2)

    rev = 2 * np.pi
    rads = (((E2 - oe.e*np.sin(E2)) - (E1 - oe.e*np.sin(E1))) % rev) + (rev * k_revs)
    return T * rads / rev