"""Editing a body that already exists.

**The invariant of this module: mass and density are what you set; radius is
always derived.**  ``materials.py`` was written for exactly this -- squeeze a
body denser and it really does shrink -- and honouring it means there is no code
path that can produce an inconsistent mass/radius/density triple, whatever the
child does to Jupiter.

Kind follows mass: cross ``units.STAR_MASS_THRESHOLD`` and the body ignites;
drop back below and it goes out.

Imports numpy and nothing from pygame -- ``tests/test_layering.py`` asserts it.
"""

from . import materials, units
from .body import KIND_PLANET, KIND_STAR

#: A body may not be lighter than a small asteroid or heavier than a large star.
MIN_MASS = 1e-16
MAX_MASS = 50.0

#: Ratio per scrub notch for mass, and g/cm^3 per notch for density.
MASS_NOTCH = 1.06
DENSITY_NOTCH = 0.15

EDIT_FIELDS = ("mass", "density", "material")


# --- scrubbing --------------------------------------------------------------------

def scrub_mass(mass, notches):
    """Logarithmic scrub: every notch is the same *ratio*, not the same amount.

    Mass spans twenty orders of magnitude between a pebble and a star, so a
    linear scrub would be unusable at one end and useless at the other.
    """
    scaled = max(float(mass), MIN_MASS) * (MASS_NOTCH ** notches)
    return float(min(max(scaled, MIN_MASS), MAX_MASS))


def scrub_density_gcm3(gcm3, notches):
    """Linear scrub, clamped to the range real matter occupies."""
    value = float(gcm3) + notches * DENSITY_NOTCH
    return float(min(max(value, materials.MIN_DENSITY_GCM3),
                     materials.MAX_DENSITY_GCM3))


# --- reading ----------------------------------------------------------------------

def body_density(world, index):
    return units.density_from_mass_radius(float(world.mass[index]),
                                          float(world.radius[index]))


def ignition_ratio(world, index):
    """How many times heavier this body must become before it lights up."""
    mass = float(world.mass[index])
    if mass <= 0.0:
        return float("inf")
    return units.STAR_MASS_THRESHOLD / mass


# --- writing ----------------------------------------------------------------------

def _refresh_kind(world, index):
    body = world.bodies[index]
    if float(world.mass[index]) >= units.STAR_MASS_THRESHOLD:
        body.kind = KIND_STAR
    elif body.kind == KIND_STAR:
        body.kind = KIND_PLANET


def set_mass(world, index, mass):
    """Set mass, holding density constant, so the radius follows."""
    density = body_density(world, index)
    mass = float(min(max(float(mass), MIN_MASS), MAX_MASS))
    world.mass[index] = mass
    world.radius[index] = units.radius_from_mass_density(mass, density)
    _refresh_kind(world, index)


def set_density(world, index, density):
    """Set density, holding mass constant, so the radius follows."""
    lo = units.from_gcm3(materials.MIN_DENSITY_GCM3)
    hi = units.from_gcm3(materials.MAX_DENSITY_GCM3)
    density = float(min(max(float(density), lo), hi))
    world.radius[index] = units.radius_from_mass_density(
        float(world.mass[index]), density)
    # Keep the composition label honest about what it now is.
    world.bodies[index].material = materials.nearest_by_density(density)


def set_material(world, index, material):
    """Adopt a substance, snapping density to that substance's real value."""
    world.bodies[index].material = material
    world.radius[index] = units.radius_from_mass_density(
        float(world.mass[index]), material.density)


def cycle_material(world, index, step=1):
    keys = [m.key for m in materials.MATERIALS]
    current = keys.index(world.bodies[index].material.key)
    set_material(world, index, materials.MATERIALS[(current + step) % len(keys)])
