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

#: 1 (AU/day)^2 expressed in J/kg (== m^2/s^2).  Specific energies -- the
#: energy per unit mass an impact delivers -- are quoted in J/kg in every
#: paper, and in AU^2/day^2 in here.
SPECIFIC_ENERGY_SI = (AU_M / SEC_PER_DAY) ** 2

#: Benz & Asphaug (1999) basalt fit for the catastrophic-disruption specific
#: energy, converted from erg/g and centimetres to J/kg and metres.  The first
#: term is material strength and falls with size; the second is self-gravity
#: and rises with it.  They cross near 200 m, which is why a boulder is harder
#: to break apart than a mountain is.
QD_STRENGTH_SI = 608.0
QD_STRENGTH_EXP = -0.38
QD_GRAVITY_SI = 1.574e-2
QD_GRAVITY_EXP = 1.36


def disruption_threshold(radius, density, toughness=1.0):
    """Specific energy needed to disperse half of a body, in simulation units.

    ``radius`` and ``density`` are in simulation units; ``toughness`` scales
    the strength branch only, because how hard a material is to crack is
    chemistry, while its self-gravity is not.
    """
    r_m = max(radius * AU_M, 1.0e-3)
    strength = toughness * QD_STRENGTH_SI * r_m ** QD_STRENGTH_EXP
    gravity = QD_GRAVITY_SI * to_gcm3(density) * r_m ** QD_GRAVITY_EXP
    return (strength + gravity) / SPECIFIC_ENERGY_SI


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


def strength_acceleration(radius, toughness=1.0):
    """Acceleration a material's own strength can resist, in simulation units.

    Reuses the strength branch of the disruption threshold (a specific
    energy, J/kg) spread over the body's own radius to get an acceleration --
    the same quantity that resists being blown apart by an impact also
    resists being pulled apart by a tide, since both ask whether cohesion can
    outrun the disturbance.
    """
    r_m = max(radius * AU_M, 1.0e-3)
    specific_si = toughness * QD_STRENGTH_SI * r_m ** QD_STRENGTH_EXP  # J/kg
    accel_si = specific_si / r_m                                       # m/s^2
    return accel_si / AUD2_TO_MS2


#: The Sun's effective surface temperature, kelvin -- the anchor point for
#: the mass-temperature scaling below.
SUN_TEMPERATURE_K = 5772.0


def star_temperature(mass):
    """Effective surface temperature of a star, in kelvin.

    A serviceable main-sequence approximation (T ~ M^0.5) anchored at the
    Sun: good enough to make a red dwarf noticeably cooler and a blue giant
    scorching, without pulling in a full stellar-structure model to decide
    whether a comet boils.
    """
    return SUN_TEMPERATURE_K * math.sqrt(max(mass, 0.0))


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
