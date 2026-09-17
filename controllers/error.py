import numpy as np
from sim.angles import vec2quat
from sim.bodies import CelestialBody

def pointing_error(state, target: CelestialBody | np.ndarray):
    q = state.rot()

    if isinstance(target, CelestialBody):
        pointing_vec = target.pos_I - state.pos()
    else:
        pointing_vec = target
    qref = vec2quat(pointing_vec)

    eq = (1/qref) * q
    return eq

def ang_vel_error(state, wref):
    w = state.omega()
    return w - wref