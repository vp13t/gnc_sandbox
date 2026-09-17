import numpy as np

class CelestialBody:
    def __init__(self, mu, radius, pos_I, color, name):
        # Each body needs its own actor name; all share the same Python class.
        self.name = name
        self.mu = mu
        self.radius = radius
        self.pos_I = pos_I
        self.color = color

Earth = CelestialBody(
    name = "Earth",
    mu = 3.986004418e14,  # Earth's gravitational parameter in m^3/s^2
    radius = 6371000.0,  # Earth's radius in meters
    pos_I = np.array([0, 0, 0]),
    color = "lightblue"
)

AU = 149597870700.0  # Average distance from Earth to Sun in meters
Sun = CelestialBody(
    name = "Sun",
    mu = 1.32712440018e20,  # Sun's gravitational parameter in m^3/s^2
    radius = 696340000.0,  # Sun's radius in meters
    pos_I = np.array([AU, 0, 0]),  # Sun's position in inertial frame
    color = "yellow"
)

lunar_distance_from_earth = 384400000.0  # Average distance from Earth to Moon in meters
Moon = CelestialBody(
    name = "Moon",
    mu = 4.9048695e12,  # Moon's gravitational parameter in m^3/s^2
    radius = 1737100.0,  # Moon's radius in meters
    pos_I = np.array([
        0,
        lunar_distance_from_earth/np.sqrt(2),
        lunar_distance_from_earth]/np.sqrt(2)),  # Moon's position in inertial frame
    color = "gray"
)

Bodies = [Earth, Sun, Moon]
