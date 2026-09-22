from abc import ABC, abstractmethod
from sim.state import State
from sim.forces import Force
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
    dt = 0.1
    dt_between_gnc_updates = 1
    dt_between_frames = 10
    max_integration_step = 1.0

    @abstractmethod
    def forces(self, X):
        """Deterministic state-dependent loads, reevaluated at RK stages."""
        pass

    def held_forces(self):
        """Controls and disturbances sampled once per simulation step."""
        return getattr(self, "control_force", Force())

    def step(self, X, dt=None):
        """Integrate the configured environment while holding sampled loads."""
        return X.update(self.dt if dt is None else dt, self.spacecraft,
                        self.held_forces(), state_forces=self.forces,
                        max_step=self.max_integration_step)

    def update_gnc(self, X, t):
        pass
