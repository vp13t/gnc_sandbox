import numpy as np
import guidance.oe as oe
import sim.state as state
from scenarios.base_scenario import BaseScenario
from sim.bodies import Earth, Sun, Moon
from spacecraft.cubesat import CubeSat
from visualization.camera_mode import CameraMode
import sim.forces as forces

from guidance.scheduling import GuidanceSchedule
from guidance.maneuver import IdlePeriod
from guidance.orbits.inclination_maneuver import SetInclinationManeuver

class Scenario(BaseScenario):
    def __init__(self):
        self.name = "MEO_delta_i"

        min_alt = 4000000.0  # Altitude above Earth's surface in meters
        max_alt = 20000000.0  # Altitude above Earth's surface in meters
        r_p = Earth.radius + min_alt  # Perigee distance from Earth's center
        r_a = Earth.radius + max_alt
        a = (r_p+r_a)/2
        e = (r_a-r_p)/(r_a+r_p)

        i_i = 0
        self.Omega = 0
        self.omega = -np.pi/3
        OE = oe.OrbitalElements(
            a=a,         # Semi-major axis
            e=e, # Eccentricity
            i=i_i,      # Inclination
            Omega=self.Omega,  # Right ascension of ascending node
            omega=self.omega,  # Argument of periapsis
            theta=0   # True anomaly
        )
        r0, v0 = oe.oe_to_rv(OE)

        h = np.cross(r0, v0)
        hhat = h / np.linalg.norm(h)

        # 1 revolution per period
        self.period = OE.period()  # Orbital period in seconds
        spin = 2 * np.pi / self.period  # Angular velocity in rad/s

        q0 = [0, 0, 0, 1]  # Initial quaternion (no rotation)
        w0 = spin * hhat   # Initial angular velocity (1 rotation per orbit)

        self.X0 = state.State(
            r=r0,
            v=v0,
            q=q0,
            w=w0
        )
        self.last_X = self.X0

        self.cam_target = CameraMode.VELOCITY_FOLLOWING
        self.spacecraft = CubeSat()

        t0 = 0.0
        self.i_f = i_i + 2*np.pi/48
        self.guidance_schedule = GuidanceSchedule(
            [
                IdlePeriod(self.period),
                # Decrementing i below 0 flips the acending node
                SetInclinationManeuver(i_i + 1*np.pi/48, Earth, self.spacecraft),
                SetInclinationManeuver(i_i + 2*np.pi/48, Earth, self.spacecraft)
            ],
            self.X0,
            t0,
            control_dt=self.dt * self.dt_between_gnc_updates,
        )
        self.control_force = forces.Force()

        self.duration = self.period * 4
        self.dt = 1.0

    def __del__(self):
        last_oe = oe.rv_to_oe(self.last_X.pos(), self.last_X.vel(), body=Earth)
        print(f"Plane Change Targeting Inclination {np.rad2deg(self.i_f):.2f} deg. Final Inclination {np.rad2deg(last_oe.i):.2f} deg")
        print(f"Initial RAAN: {np.rad2deg(self.Omega):.2f} deg. Final RAAN: {np.rad2deg(last_oe.Omega):.2f} deg")
        print(f"Initial AoP: {np.rad2deg(self.omega):.2f} deg. Final AoP: {np.rad2deg(last_oe.omega):.2f} deg")

    def update_gnc(self, X, t) -> dict[str, forces.Force]:
        control_inputs = self.guidance_schedule.update(X, t)
        self.control_force = sum(control_inputs.values(), start=forces.Force())
        self.last_X = X
        return control_inputs

    def forces(self, X) -> forces.Force:
        return (
            forces.gravity(X, Earth, self.spacecraft)
        )

scene = Scenario()