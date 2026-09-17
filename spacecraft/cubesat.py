import numpy as np
from spacecraft.spacecraft import Spacecraft, Thruster
from sim.frames import IX

class CubeSat(Spacecraft):
    def __init__(self):
        self.mass = 100.0  # kg

        side_length = 1.0  # meters
        self.inertia = np.diag([
            self.mass * side_length**2 / 6,
            self.mass * side_length**2 / 6,
            self.mass * side_length**2 / 6])  # kg*m^2
        
        self.gains = {
            "Kr": np.ones(3) * 0.1,
            "Kw": np.ones(3) * 10.0
        }
        self.thrusters = {
            "X_body": Thruster(
                4000.0, # N
                IX
            )
        }