import numpy as np
import guidance.oe as oe
import sim.state as state
from scenarios.base_scenario import BaseScenario
from sim.bodies import Earth, Sun, Moon, lunar_distance_from_earth, GEO_alt
from spacecraft.cubesat import CubeSat
from visualization.camera_mode import CameraMode
import sim.forces as forces

from guidance.scheduling import GuidanceSchedule
from guidance.orbits.apse_maneuver import SetApoapsisDistManeuver

class Scenario(BaseScenario):
    def __init__(self):
        self.name = "LUN_return"

        alt = GEO_alt  # Altitude above Earth's surface in meters
        r_p = Earth.radius + alt  # Perigee distance from Earth's center
        r_a = lunar_distance_from_earth + Moon.radius + alt  # Apogee distance from Earth's center
        a = (r_p+r_a)/2
        e = (r_a-r_p)/(r_a+r_p)

        OE = oe.OrbitalElements(
            a=a,         # Semi-major axis
            e=e, # Eccentricity
            i=np.pi/4,      # Inclination
            Omega=0,  # Right ascension of ascending node
            omega=-np.pi/2,  # Argument of periapsis
            theta=0   # True anomaly
        )
        r0, v0 = oe.oe_to_rv(OE, mu=Earth.mu)

        h = np.cross(r0, v0)
        hhat = h / np.linalg.norm(h)

        # 1 revolution per period
        period = OE.period(mu=Earth.mu)  # Orbital period in seconds
        spin = 2 * np.pi / period  # Angular velocity in rad/s

        q0 = [0, 0, 0, 1]  # Initial quaternion (no rotation)
        w0 = spin * hhat   # Initial angular velocity (1 rotation per orbit)

        self.X0 = state.State(
            r=r0,
            v=v0,
            q=q0,
            w=w0
        )

        self.cam_target = CameraMode.VELOCITY_FOLLOWING
        self.spacecraft = CubeSat()

        t0 = 0.0
        self.alt_f_target = 2500000.0
        r_f = Earth.radius + self.alt_f_target
        self.guidance_schedule = GuidanceSchedule(
            [SetApoapsisDistManeuver(r_p, Earth, self.spacecraft)],
            self.X0,
            t0
        )
        self.control_force = forces.Force()

        self.duration = period * 1.2

    def update_gnc(self, X, t) -> dict[str, forces.Force]:
        control_inputs = self.guidance_schedule.update(X, t)
        self.control_force = sum(control_inputs.values(), start=forces.Force())
        self.last_X = X
        return control_inputs

    def forces(self, X):
        return (
            forces.gravity(X, Earth, self.spacecraft)
            + forces.gravity(X, Moon, self.spacecraft)
            + forces.torque_free_rotation(X, self.spacecraft)
            + self.control_force
        )

scene = Scenario()