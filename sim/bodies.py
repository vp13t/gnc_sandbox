import numpy as np

class CelestialBody:
    def __init__(self, mu, radius, pos_I, color, name, *, omega_I=None):
        # Each body needs its own actor name; all share the same Python class.
        self.name = name
        self.mu = mu
        self.radius = radius
        self.pos_I = pos_I
        self.color = color
        # Constant axial angular velocity, in inertial coordinates (rad/s).
        self.omega_I = (np.zeros(3) if omega_I is None
                        else np.array(omega_I, dtype=float, copy=True))
        if self.omega_I.shape != (3,) or not np.isfinite(self.omega_I).all():
            raise ValueError("omega_I must be a finite three-component angular velocity")


def _axial_spin(pole_ra_deg, pole_dec_deg, rate_deg_per_day):
    """Convert a mean J2000 equatorial pole and spin rate into rad/s."""
    ra, dec = np.deg2rad([pole_ra_deg, pole_dec_deg])
    axis = np.array([np.cos(dec)*np.cos(ra), np.cos(dec)*np.sin(ra), np.sin(dec)])
    return axis * np.deg2rad(rate_deg_per_day) / 86400.0


# Mean pole constants and linear prime-meridian rates from NASA/JPL's PCK:
# https://naif.jpl.nasa.gov/pub/naif/generic_kernels/pck/pck00011.tpc
# Freeze the mean J2000 poles; omit precession, nutation and lunar libration.

GEO_alt = 35786000.0
Earth = CelestialBody(
    name = "Earth",
    mu = 3.986004418e14,  # Earth's gravitational parameter in m^3/s^2
    radius = 6371000.0,  # Earth's radius in meters
    pos_I = np.array([0, 0, 0]),
    color = "lightblue",
    omega_I = np.array([0., 0., np.deg2rad(360.9856235) / 86400.0]),
)

AU = 149597870700.0  # Average distance from Earth to Sun in meters
Sun = CelestialBody(
    name = "Sun",
    mu = 1.32712440018e20,  # Sun's gravitational parameter in m^3/s^2
    radius = 696340000.0,  # Sun's radius in meters
    pos_I = np.array([AU, 0, 0]),  # Sun's position in inertial frame
    color = "yellow",
    # Conventional mean solar rotation, not latitude-dependent surface flow.
    omega_I = _axial_spin(286.13, 63.87, 14.18440),
)

lunar_distance_from_earth = 384400000.0  # Average distance from Earth to Moon in meters
Moon = CelestialBody(
    name = "Moon",
    mu = 4.9048695e12,  # Moon's gravitational parameter in m^3/s^2
    radius = 1737100.0,  # Moon's radius in meters
    pos_I = np.array([
        0,
        lunar_distance_from_earth/np.sqrt(2),
        lunar_distance_from_earth/np.sqrt(2)]),  # Moon's position in inertial frame
    color = "gray",
    omega_I = _axial_spin(269.9949, 66.5392, 13.17635815),
)

Bodies = [Earth, Sun, Moon]
