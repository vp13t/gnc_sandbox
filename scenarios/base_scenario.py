from abc import ABC, abstractmethod
from sim.state import State
from sim.bodies import CelestialBody
from spacecraft.spacecraft import Spacecraft
from visualization.camera_mode import CameraMode

class BaseScenario(ABC):
    name: str
    X0: State
    cam_target: CelestialBody | CameraMode
    spacecraft: Spacecraft
    duration: float
    t0 = 0.0

    @abstractmethod
    def forces(self, X):
        pass

    def update_gnc(self, X, t):
        pass
