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

#: The speed of light.  Every other module asks for it here; the whole point
#: of this file is that a constant is written down once.
C_KMS = 299792.458
#: ... and in simulation units, AU per day: 173.1446 AU/day.
C = C_KMS * SEC_PER_DAY / AU_KM

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


# --- Black holes --------------------------------------------------------------------
#
# A black hole is completely described by mass, angular momentum and charge (the
# "no-hair" theorem).  Astrophysical holes carry no net charge -- the surrounding
# plasma neutralises any imbalance almost at once -- so there are two real
# parameters here, and everything below is derived from them.

#: Dimensionless spin ceiling.  Not 1: radiation from the accretion disk torques
#: a hole back down before it can reach the extremal limit (Thorne 1974).
MAX_SPIN = 0.998

#: Tolman-Oppenheimer-Volkoff limit: the heaviest a neutron star can be before it
#: collapses.  Below this nature makes neutron stars, not black holes.  Reported
#: in the inspector, deliberately not enforced -- see the design doc.
TOV_LIMIT_MASS = 3.0

#: Past this mass a star does not exist; the core collapses directly to a black
#: hole with no supernova at all (above the pair-instability window).
COLLAPSE_MASS_THRESHOLD = 150.0

#: The dark disc a black hole casts is not the horizon -- light bending magnifies
#: it to sqrt(27)/2 horizon radii, about 2.6x.
SHADOW_FACTOR = math.sqrt(27.0) / 2.0
#: Where photons orbit in circles, at 1.5 horizon radii.
PHOTON_SPHERE_FACTOR = 1.5

#: Hawking temperature of a one-solar-mass hole, kelvin.  Sixty nanokelvin --
#: far below the 2.7 K microwave background, which is why a stellar-mass hole
#: absorbs more than it radiates and grows instead of evaporating.
HAWKING_T_SUN_K = 6.169e-8
#: Evaporation time of a one-solar-mass hole, years.  Scales as M^3.
EVAPORATION_SUN_YEARS = 2.1e67

#: Main-sequence mass-radius exponent, used only to give a black hole a drawable
#: apparent size: the disk is roughly as big as the star that made it.
APPARENT_RADIUS_EXP = 0.8


def schwarzschild_radius(mass):
    """Event horizon radius 2GM/c^2, in AU.  2.953 km per solar mass."""
    return 2.0 * G * max(float(mass), 0.0) / (C * C)


def shadow_radius(mass):
    """Radius of the dark disc an observer actually sees, in AU."""
    return SHADOW_FACTOR * schwarzschild_radius(mass)


def photon_sphere_radius(mass):
    """Radius at which light orbits in a circle, in AU."""
    return PHOTON_SPHERE_FACTOR * schwarzschild_radius(mass)


def _kerr_isco_gm_c2(spin):
    """Prograde Kerr ISCO radius in units of GM/c^2 (Bardeen, Press &
    Teukolsky 1972).  Shared by isco_radius and accretion_efficiency so the
    orbit one reports and the binding energy the other computes can never
    drift apart.
    """
    a = min(abs(float(spin)), MAX_SPIN)
    z1 = 1.0 + (1.0 - a * a) ** (1.0 / 3.0) * (
        (1.0 + a) ** (1.0 / 3.0) + (1.0 - a) ** (1.0 / 3.0))
    z2 = math.sqrt(3.0 * a * a + z1 * z1)
    return 3.0 + z2 - math.sqrt((3.0 - z1) * (3.0 + z1 + 2.0 * z2))


def isco_radius(mass, spin=0.0):
    """Innermost stable circular orbit (prograde), in AU.

    Bardeen, Press & Teukolsky (1972).  This is the inner edge of an accretion
    disk: inside it there is no stable orbit and matter plunges.  Spin drags it
    from 3 horizon radii down to 0.62, which is why a fast-spinning hole
    converts 32% of what it eats into light against a still one's 5.7%.
    """
    r_gm = _kerr_isco_gm_c2(spin)
    # r_gm is in units of GM/c^2, which is half a Schwarzschild radius.
    return 0.5 * r_gm * schwarzschild_radius(mass)


def accretion_efficiency(spin=0.0):
    """Fraction of infalling rest mass radiated away, 0..1.

    One minus the specific binding energy at the ISCO.  5.7% for a still hole,
    32% at the Thorne limit -- against 0.7% for hydrogen fusion, which is why
    accreting black holes outshine the galaxies they sit in.
    """
    r_gm = _kerr_isco_gm_c2(spin)
    return 1.0 - math.sqrt(1.0 - 2.0 / (3.0 * r_gm))


def hawking_temperature_k(mass):
    """Temperature of the hole's own radiation, kelvin.  Scales as 1/M."""
    mass = float(mass)
    return HAWKING_T_SUN_K / mass if mass > 0.0 else math.inf


def evaporation_years(mass):
    """How long the hole takes to evaporate, years.  Scales as M^3."""
    return EVAPORATION_SUN_YEARS * max(float(mass), 0.0) ** 3


def apparent_radius(mass):
    """The size a black hole is *drawn* from, in AU -- not a physical radius.

    The horizon is seven orders of magnitude below anything else in the sim, so
    a hole would draw as a single pixel and, worse, be impossible to aim at
    (see render/scale.py).  This stands in for the accretion disk, using the
    main-sequence mass-radius relation on the grounds that the disk is about as
    big as the star that made the hole.  **Physics must never read this.**
    """
    return SUN_RADIUS * max(float(mass), 0.0) ** APPARENT_RADIUS_EXP


def circular_speed(mu, radius):
    """Speed of a circular orbit of radius ``radius`` about gravitational parameter mu."""
    return math.sqrt(mu / radius)


# --- Radiation and temperature ------------------------------------------------------
#
# Temperatures are in **kelvin**, not a sandbox-scaled unit.  Every source in the
# literature is in kelvin, there is no natural AU/solar-mass/day temperature to
# scale to, and the readout has to print a number a person recognises.  So this
# is the one SI quantity that crosses into the physics layer, and the conversions
# below are what keep it honest.

#: Stefan-Boltzmann constant, W m^-2 K^-4.  Radiated power goes as T^4, which is
#: why a glowing body cools in hours and a frozen one takes centuries.
STEFAN_BOLTZMANN = 5.670374419e-8

#: The cosmic microwave background, kelvin.  Nothing gets colder than this:
#: however far a body runs from every star, it still sits in a 2.725 K bath and
#: comes to equilibrium with it.  Sunlight does not drop this low until 10,400 AU.
CMB_TEMPERATURE_K = 2.725

#: How deep a layer takes part in a body's temperature changes, metres.
#:
#: Heating a *whole* planet gives Earth a thermal time constant near a hundred
#: thousand years -- entirely real, and entirely invisible in a sandbox where a
#: frame is a day.  The reason real surfaces equilibrate in days is that only a
#: thin skin follows the changing sunlight beneath it.  This is that skin, and it
#: is the one free parameter in the whole thermal model.
THERMAL_SKIN_DEPTH_M = 1.0

#: A body's temperature may not be scrubbed below the background it sits in, nor
#: above this -- hotter than any star's surface, and well clear of anything the
#: T^4 arithmetic could overflow on.
MIN_TEMPERATURE_K = CMB_TEMPERATURE_K
MAX_TEMPERATURE_K = 1.0e5

#: The freezing point of water in kelvin -- the offset between the unit the
#: physics is in and the unit the audience has.
ZERO_CELSIUS_K = 273.15


def to_celsius(kelvin):
    """Kelvin to degrees Celsius."""
    return kelvin - ZERO_CELSIUS_K


def to_kgm3(density):
    """Density in simulation units to kg/m^3, which is what the thermal
    arithmetic wants: specific heats are quoted per kilogram."""
    return density / KGM3_TO_SIM


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
