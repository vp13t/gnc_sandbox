import numpy as np

class Thruster:
    def __init__(self, force: float, direction: np.ndarray[float]):
        self.force = force # N
        self.direction = direction # Unit vector

class Spacecraft:
    mass: float # kg
    inertia: np.ndarray[float]
    thrusters: dict[str, Thruster]
    gains: dict[str, float]
