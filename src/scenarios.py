"""Starting arrangements, built from real data.

The solar system is not hand-placed: each body is given its actual J2000 orbital
elements and the state vector is derived from them, so the orbital periods that come out
of the simulation are correct because the physics says so.  That is the whole premise of
the project, and ``selftest.py`` checks it.

Masses are in solar masses, radii in km, distances in AU, angles in degrees.
"""

import math

import numpy as np

from . import edits, materials, orbits, units
from .body import KIND_STAR
from .world import World

# name, a(AU), e, lon_peri(deg), mean_lon(deg), mass(Msun), radius(km), material, look
PLANETS = (
    ("Mercury", 0.38709927, 0.20563593, 77.45779628, 252.25032350,
     1.66013e-7, 2439.7, "iron", "mercury"),
    ("Venus", 0.72333566, 0.00677672, 131.60246718, 181.97909950,
     2.44784e-6, 6051.8, "rock", "venus"),
    ("Earth", 1.00000261, 0.01671123, 102.93768193, 100.46457166,
     3.0034896e-6, 6371.0, "rock", "terrestrial"),
    ("Mars", 1.52371034, 0.09339410, -23.94362959, -4.55343205,
     3.22716e-7, 3389.5, "rock", "mars"),
    ("Jupiter", 5.20288700, 0.04838624, 14.72847983, 34.39644051,
     9.5457937e-4, 69911.0, "gas", "jupiter"),
    ("Saturn", 9.53667594, 0.05386179, 92.59887831, 49.95424423,
     2.85885e-4, 58232.0, "gas", "saturn"),
    ("Uranus", 19.18916464, 0.04725744, 170.95427630, 313.23810451,
     4.36625e-5, 25362.0, "ice", "uranus"),
    ("Neptune", 30.06992276, 0.00859048, 44.96476227, -55.12002969,
     5.15139e-5, 24622.0, "ice", "neptune"),
)

# parent, name, a(AU), e, phase(deg), mass(Msun), radius(km), material, look
MOONS = (
    ("Earth", "Moon", 0.00256955, 0.0549, 210.0, 3.6923e-8, 1737.4, "rock", "moon"),
    ("Jupiter", "Io", 0.00281869, 0.0041, 30.0, 4.4919e-8, 1821.6, "rock", None),
    ("Jupiter", "Europa", 0.00448595, 0.0094, 150.0, 2.4139e-8, 1560.8, "rock_ice", None),
    ("Jupiter", "Ganymede", 0.00715525, 0.0013, 260.0, 7.4525e-8, 2634.1, "rock_ice", None),
    ("Jupiter", "Callisto", 0.01258520, 0.0074, 20.0, 5.4110e-8, 2410.3, "rock_ice", None),
    ("Saturn", "Titan", 0.00816766, 0.0288, 95.0, 6.7649e-8, 2574.7, "rock_ice", None),
)


def _km(radius_km):
    return radius_km / units.AU_KM


def solar_system(world=None, moons=True):
    """The real solar system: the Sun, eight planets and six major moons.

    ``moons=False`` leaves them out.  Moons force the timestep down by two orders of
    magnitude -- Io's dynamical time is a quarter of a day against Mercury's fourteen --
    so tests that need to run Neptune round its 165-year orbit use the planets alone.
    """
    world = world or World()
    world.clear()

    sun = world.add(
        "Sun", materials.get("hydrogen"), 1.0, _km(695700.0),
        np.zeros(2), np.zeros(2), kind=KIND_STAR,
    )
    del sun

    index_by_name = {"Sun": 0}

    for name, a, e, lon_peri, mean_lon, mass, radius_km, mat, look in PLANETS:
        mu = units.G * (1.0 + mass)
        mean_anom = math.radians(mean_lon - lon_peri)
        nu = orbits.true_anomaly_from_mean(mean_anom, e)
        pos, vel = orbits.state_from_elements(mu, a, e, math.radians(lon_peri), nu)
        world.add(name, materials.get(mat), mass, _km(radius_km), pos, vel, look=look)
        index_by_name[name] = world.n - 1

    for parent, name, a, e, phase, mass, radius_km, mat, look in (MOONS if moons else ()):
        pi = index_by_name[parent]
        mu = units.G * (world.mass[pi] + mass)
        nu = orbits.true_anomaly_from_mean(math.radians(phase), e)
        rel_pos, rel_vel = orbits.state_from_elements(mu, a, e, 0.0, nu)
        world.add(name, materials.get(mat), mass, _km(radius_km),
                  world.pos[pi] + rel_pos, world.vel[pi] + rel_vel, look=look)
        index_by_name[name] = world.n - 1

    world.bodies[index_by_name["Saturn"]].rings = True

    # Without this the whole system slowly drifts off screen, because giving the Sun a
    # velocity of exactly zero leaves the barycentre moving.
    world.zero_net_momentum()
    return world


def two_body(separation=1.0, primary_mass=1.0, secondary_mass=units.EARTH_MASS,
             eccentricity=0.0, world=None):
    """A clean two-body system, used by the tests that need an exact answer."""
    world = world or World()
    world.clear()
    mu = units.G * (primary_mass + secondary_mass)
    a = separation
    pos, vel = orbits.state_from_elements(mu, a, eccentricity, 0.0, 0.0)

    world.add("Primary", materials.get("hydrogen"), primary_mass,
              _km(695700.0), np.zeros(2), np.zeros(2), kind=KIND_STAR)
    world.add("Secondary", materials.get("rock"), secondary_mass,
              _km(6371.0), pos, vel)
    world.zero_net_momentum()
    return world


def black_hole(world=None, hole_mass=10.0):
    """A stellar-mass black hole with three bodies around it, one doomed.

    The doomed one starts on a steeply eccentric orbit whose periapsis falls
    well inside its tidal radius, so the disruption happens within the first
    minute rather than needing the child to arrange it.  Earth's tidal radius
    from a ten-solar-mass hole is 0.00636 AU -- some 32,000 horizon radii out
    -- so the shredding is large, slow and nowhere near the dot at the centre,
    which is exactly what makes it readable.
    """
    world = world or World()
    world.clear()

    world.add("Black hole", materials.get("hydrogen"), hole_mass,
              units.SUN_RADIUS, np.zeros(2), np.zeros(2), kind=KIND_STAR)
    edits.collapse(world, 0)
    mu = units.G * hole_mass

    # name, a (AU), e, phase (deg), mass, radius (km), material
    entries = (
        ("Doomed", 0.30, 0.985, 175.0, units.EARTH_MASS, 6371.0, "rock"),
        ("Wanderer", 0.60, 0.35, 40.0, units.EARTH_MASS * 0.6, 4800.0, "rock_ice"),
        ("Far one", 1.40, 0.10, 250.0, units.EARTH_MASS * 2.0, 9000.0, "ice"),
    )
    for name, a, e, phase, mass, radius_km, mat in entries:
        nu = orbits.true_anomaly_from_mean(math.radians(phase), e)
        pos, vel = orbits.state_from_elements(mu, a, e, 0.0, nu)
        world.add(name, materials.get(mat), mass, _km(radius_km), pos, vel)

    world.zero_net_momentum()
    return world


REGISTRY = {
    "solar_system": solar_system,
    "two_body": two_body,
    "black_hole": black_hole,
}
