import numpy as np
from guidance.maneuver import Maneuver
from sim.state import State
from sim.forces import Force

class GuidanceSchedule:
    def __init__(self, maneuvers: list[Maneuver], X0: State, t0: float, log=True):
        self.maneuvers = maneuvers
        self.curr_maneuver = self.maneuvers.pop(0)
        self.curr_maneuver.plan(X0, t0)
        self.log = log
    
    def update(self, state: State, t: float) -> dict[str, Force]:
        action = {}
        if self.curr_maneuver and self.curr_maneuver.check_complete(state, t):
            if self.log:
                print(f"\nCompleted {self.curr_maneuver.name}")
            if self.maneuvers:
                self.curr_maneuver = self.maneuvers.pop(0)
                self.curr_maneuver.plan(state, t)
            else:
                self.curr_maneuver = None
        if self.curr_maneuver:
            action = self.curr_maneuver.act(state, t)
        
        return action