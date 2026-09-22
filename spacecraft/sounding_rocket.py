import numpy as np
from spacecraft.spacecraft import Spacecraft, Thruster
from sim.frames import IX

class SoundingRocket(Spacecraft):
    def __init__(self):
        self.mass = 1000.0  # kg

        h = 3.0
        r = 0.15
        self.inertia = np.diag([
            r**2 * self.mass/2,
            r**2 * self.mass/4 + h**2 * self.mass/12,
            r**2 * self.mass/4 + h**2 * self.mass/12])  # kg*m^2
        
        self.gains = {
            "Kr": np.ones(3) * 1.0,
            "Kw": np.ones(3) * 10.0
        }
        self.thrusters = {
            "X_body": Thruster(
                14000.0, # N; thrust/weight ~1.43 at Earth gravity. Sized for
                         # LandManeuver's near-hover terminal descent margin
                         # (see guidance/descent/README.md); ascent's gravity
                         # turn compensates for this by coasting to apoapsis
                         # before circularizing (see GravityTurnManeuver).
                IX
            )
        }