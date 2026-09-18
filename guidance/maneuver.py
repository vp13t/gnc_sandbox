import numpy as np
from abc import ABC, abstractmethod
from sim.bodies import CelestialBody
from sim.forces import Force
from sim.state import State
import guidance.oe as OE

class Maneuver(ABC):

    @abstractmethod
    def plan(self, state):
        pass

    @abstractmethod
    def act(self, state):
        pass

    @abstractmethod
    def check_complete(self, state):
        pass


class IdlePeriod(Maneuver):
    def __init__(self, duration: float):
        self.name = "IdlePeriod"
        self.duration = duration
    
    def plan(self, state: State, t: float):
        self.end_t = t + self.duration
    
    def act(self, state: State, t: float):
        return {"": Force()}
    
    def check_complete(self, state: State, t: float):
        return t >= self.end_t
