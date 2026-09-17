import numpy as np

def tpp_eccentric_anomaly(tpp, a, e, mu, max_iter=100, log=False):
    """
    tpp: Time past periapsis.
    a: Semi-major axis.
    e: Eccentricity.
    mu: Gravitational parameter.
    """
    mean_motion = np.sqrt(mu / a**3)
    M = mean_motion * tpp
    tol = 1e-3/a

    # Use a second-order Taylor expansion for the initial guess of E.
    E_i = M + e * np.sin(M) + e**2 * np.sin(2*M) /2
    if log:
        print(f"Initial guess for Eccentric anomaly: {E_i}")

    for i in range(max_iter):
        g = E_i - e * np.sin(E_i) - M
        g_prime = 1 - e * np.cos(E_i)
        step = g / g_prime
        E_i = E_i - step

        if log:
            print(f"Iteration {i}: E = {E_i}, g = {g}, step = {step}")
        if abs(g) < tol and abs(step) < tol:
            break
    if log:
        print(f"Ran for {i+1} iterations. Eccentric anomaly: {E_i}")
    return E_i