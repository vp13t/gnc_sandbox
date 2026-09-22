import numpy as np
import guidance.oe as oe
import sim.state as state
from scenarios.base_scenario import BaseScenario
from sim.bodies import Earth, Sun, Moon
from spacecraft.sounding_rocket import SoundingRocket
from visualization.camera_mode import CameraMode
import sim.forces as forces

from guidance.scheduling import GuidanceSchedule
from guidance.maneuver import IdlePeriod
from guidance.descent.in_plane_descent_maneuver import InPlaneDescentManeuver
from guidance.descent.brake_maneuver import BrakeManeuver
from guidance.descent.land_maneuver import LandManeuver

class Scenario(BaseScenario):
    def __init__(self):
        self.name = "LAND_powered"

        alt = 400000.0  # Altitude above Earth's surface in meters
        r_p = Earth.radius + alt  # Perigee distance from Earth's center
        OE = oe.OrbitalElements(
            a=r_p,         # Semi-major axis
            e=0, # Eccentricity
            i=5*np.pi/6,      # Inclination
            Omega=3*np.pi/4,  # Right ascension of ascending node
            omega=-np.pi/4,  # Argument of periapsis
            theta=-np.pi/2   # True anomaly
        )
        r0, v0 = oe.oe_to_rv(OE, mu=Earth.mu)

        h = np.cross(r0, v0)
        hhat = h / np.linalg.norm(h)

        self.period = OE.period(mu=Earth.mu)  # Orbital period in seconds
        spin = 2 * np.pi / self.period  # Angular velocity in rad/s

        q0 = [0, 0, 0, 1]  # Initial quaternion (no rotation)
        w0 = spin * hhat    # Initial angular velocity (1 rotation per orbit)

        self.X0 = state.State(
            r=r0,
            v=v0,
            q=q0,
            w=w0
        )
        self.last_X = self.X0

        self.cam_target = CameraMode.NORMAL_FACING
        self.spacecraft = SoundingRocket()

        t0 = 0.0
        theta_target = np.pi/2
        glide_slope_angle = 0.9*(np.pi/2)
        landing_duration = self.period/2

        r_theta_tgt_curr_orbit, _ = oe.projected_rv_at_anomaly(OE, Earth.mu, theta_target)
        rhat_theta_tgt = r_theta_tgt_curr_orbit / np.linalg.norm(r_theta_tgt_curr_orbit)

        # Reserve thrust for attitude/pulse tracking errors during final braking.
        # A nominal full-throttle descent has no authority to correct even a
        # small downward velocity error; later replans can then be infeasible.
        # Execution may still use 100% thrust to track this 90% nominal plan.
        landing = LandManeuver(rhat_theta_tgt, glide_slope_angle, Earth, self.spacecraft, 150,
                                planning_throttle_limit=0.9,
                                position_tracking_limit=20.0, velocity_tracking_limit=2.0)
        self.guidance_schedule = GuidanceSchedule(
            [
                InPlaneDescentManeuver(theta_target, 200000, Earth, self.spacecraft, fixed_initial_oe=OE),
                BrakeManeuver(rhat_theta_tgt, 20000, Earth, self.spacecraft, 1200,
                              landing_maneuver=landing, preview_lead=600,
                              max_slew_time=600),
                landing,
            ],
            self.X0,
            t0,
            control_dt=self.dt * self.dt_between_gnc_updates,
        )
        self.control_force = forces.Force()

        self.duration = self.period
        self.dt = 0.1

    def update_gnc(self, X, t) -> dict[str, forces.Force]:
        control_inputs = self.guidance_schedule.update(X, t)
        self.control_force = sum(control_inputs.values(), start=forces.Force())
        self.last_X = X
        return control_inputs

    def forces(self, X) -> forces.Force:
        return (
            forces.gravity(X, Earth, self.spacecraft)
            + forces.normal_force(X, Earth, self.spacecraft)
            + forces.torque_free_rotation(X, self.spacecraft)
            + self.control_force
        )

scene = Scenario()
