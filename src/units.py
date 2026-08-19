"""Physical constants and unit conversions.

Internal simulation units are **AU, solar masses, and days**.  In these units the
gravitational constant is the square of the Gaussian gravitational constant, and every
quantity in the solar system sits within a few orders of magnitude of 1 -- masses run
3e-6 .. 1, distances 0.4 .. 30.  That keeps the O(n^2) force sums well conditioned and,
more importantly, avoids catastrophic cancellation when subtracting the positions of two
bodies that orbit each other (Earth and the Moon are 1.0 and 1.00257, not 1.4959787e11
and 1.4963627e11).

This module is the single source of truth for G and for every conversion factor.  No
other module may hard-code one.  It imports nothing but ``math`` so that the physics
layer stays testable without pygame.
"""

import math

# --- The one constant everything else rests on -------------------------------------

#: Gravitational constant in AU^3 / (solar mass * day^2).
G = 2.959122082855911e-4

# --- Length -------------------------------------------------------------------------

AU_KM = 1.495978707e8
AU_M = AU_KM * 1000.0

EARTH_RADIUS = 6371.0 / AU_KM        # AU, mean radius
JUPITER_RADIUS = 69911.0 / AU_KM     # AU
SUN_RADIUS = 695700.0 / AU_KM        # AU

# --- Mass ---------------------------------------------------------------------------

MSUN_KG = 1.98847e30

EARTH_MASS = 3.0034896e-6            # solar masses
JUPITER_MASS = 9.5457937e-4          # solar masses
MOON_MASS = 3.6923e-8                # solar masses

#: Below this mass hydrogen fusion never ignites: a brown dwarf, not a star.
STAR_MASS_THRESHOLD = 0.08
#: Below this, not even deuterium burns: it is a planet.
BROWN_DWARF_MASS_THRESHOLD = 0.013

# --- Time ---------------------------------------------------------------------------

SEC_PER_DAY = 86400.0
DAYS_PER_YEAR = 365.25

# --- Derived conversion factors -----------------------------------------------------

#: 1 kg/m^3 expressed in solar masses per AU^3.
KGM3_TO_SIM = AU_M ** 3 / MSUN_KG            # 1683.66
#: 1 g/cm^3 expressed in solar masses per AU^3.
GCM3_TO_SIM = 1000.0 * KGM3_TO_SIM           # 1.68366e6

#: 1 AU/day expressed in km/s.
AUD_TO_KMS = AU_KM / SEC_PER_DAY             # 1731.46
#: 1 AU/day^2 expressed in m/s^2.
AUD2_TO_MS2 = AU_M / SEC_PER_DAY ** 2        # 20.04

STANDARD_GRAVITY = 9.80665                   # m/s^2

FOUR_THIRDS_PI = 4.0 / 3.0 * math.pi


# --- Sphere geometry ----------------------------------------------------------------

def radius_from_mass_density(mass, density):
    """Radius of a uniform sphere, all arguments in simulation units."""
    return (mass / (FOUR_THIRDS_PI * density)) ** (1.0 / 3.0)


def density_from_mass_radius(mass, radius):
    """Mean density of a uniform sphere, all arguments in simulation units."""
    return mass / (FOUR_THIRDS_PI * radius ** 3)


def surface_gravity(mass, radius):
    """Surface gravity in simulation units (AU/day^2)."""
    return G * mass / (radius * radius)


def escape_velocity(mass, radius):
    """Escape velocity in simulation units (AU/day)."""
    return math.sqrt(2.0 * G * mass / radius)


def binding_energy(mass, radius):
    """Gravitational binding energy 3GM^2/5R of a uniform sphere, simulation units.

    This is the energy an impact must exceed to shatter a body rather than merge with it.
    """
    return 0.6 * G * mass * mass / radius


def hill_radius(mass, primary_mass, distance):
    """Radius of the region where ``mass`` dominates ``primary_mass`` gravitationally."""
    return distance * (mass / (3.0 * primary_mass)) ** (1.0 / 3.0)


def circular_speed(mu, radius):
    """Speed of a circular orbit of radius ``radius`` about gravitational parameter mu."""
    return math.sqrt(mu / radius)


# --- Conversions to human-facing units ----------------------------------------------

def to_earth_masses(mass):
    return mass / EARTH_MASS


def to_jupiter_masses(mass):
    return mass / JUPITER_MASS


def to_earth_radii(radius):
    return radius / EARTH_RADIUS


def to_km(length):
    return length * AU_KM


def to_years(days):
    return days / DAYS_PER_YEAR


def to_kms(speed):
    return speed * AUD_TO_KMS


def to_gcm3(density):
    return density / GCM3_TO_SIM


def from_gcm3(density_gcm3):
    return density_gcm3 * GCM3_TO_SIM


def to_g(accel):
    """Convert an acceleration in AU/day^2 to multiples of Earth's surface gravity."""
    return accel * AUD2_TO_MS2 / STANDARD_GRAVITY


def surface_gravity_g(mass, radius):
    """Surface gravity directly in Earth g's -- the number the inspector shows."""
    return to_g(surface_gravity(mass, radius))
