import numpy as np

class Thruster:
    def __init__(self, force: float, direction: np.ndarray[float], position: np.ndarray[float] = np.zeros(3)):
        self.force = force # N
        self.direction = direction # Unit vector
        self.position = position # m, position of thruster relative to spacecraft center of mass

class Spacecraft:
    mass: float # kg
    inertia: np.ndarray[float]
    thrusters: dict[str, Thruster]
    gains: dict[str, float]
