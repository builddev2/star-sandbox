"""Editing a body that already exists.

**The invariant of this module: mass and density are what you set; radius is
always derived.**  ``materials.py`` was written for exactly this -- squeeze a
body denser and it really does shrink -- and honouring it means there is no code
path that can produce an inconsistent mass/radius/density triple, whatever the
child does to Jupiter.

Kind follows mass: cross ``units.STAR_MASS_THRESHOLD`` and the body ignites;
drop back below and it goes out.

Any field may be **locked**, per body: a locked field's setter returns without
writing anything.  The check is here rather than in the panel because the panel
is not the only caller -- the same argument ``set_temperature`` already makes
about stars.

Imports numpy and nothing from pygame -- ``tests/test_layering.py`` asserts it.
"""

from . import materials, units
from .body import (DEFAULT_BLACK_HOLE_DISK, DEFAULT_BLACK_HOLE_SPIN,
                   KIND_BLACK_HOLE, KIND_CRAFT, KIND_PLANET, KIND_STAR)

#: A body may not be lighter than a small asteroid, nor heavier than the point
#: where a star stops being possible -- past units.COLLAPSE_MASS_THRESHOLD the
#: core collapses instead, so the scrub tops out exactly where nature does.
MIN_MASS = 1e-16
MAX_MASS = units.COLLAPSE_MASS_THRESHOLD

#: Ratio per scrub notch for mass, and g/cm^3 per notch for density.
MASS_NOTCH = 1.06
DENSITY_NOTCH = 0.15
#: Linear per-notch steps for the two black hole controls.  Spin's range is
#: under 1.0 and disk's is exactly 1.0, so neither wants a ratio scrub.
SPIN_NOTCH = 0.02
DISK_NOTCH = 0.05
#: Ratio per notch for temperature.  The range runs from the microwave
#: background to 100,000 K -- four and a half orders of magnitude, which is the
#: same argument that makes the mass scrub logarithmic.
TEMPERATURE_NOTCH = 1.06

PLANET_EDIT_FIELDS = ("mass", "density", "temp", "material")
#: A star is offered no temperature: its surface temperature is set by fusion
#: and follows its mass, so a scrub that appeared to change it would be a lie
#: the very next frame, when thermal.advance pins it straight back.
STAR_EDIT_FIELDS = ("mass", "density", "material")
BLACK_HOLE_EDIT_FIELDS = ("mass", "spin", "disk")
#: Retained for callers that predate black holes and only ever meant the
#: ordinary case.
EDIT_FIELDS = PLANET_EDIT_FIELDS


def edit_fields(body):
    """The editable fields for this body -- a black hole has no density or
    composition to speak of, so it is offered spin and accretion instead, and
    a star has no temperature of its own to set.

    A craft gets none at all.  ``SPACECRAFT.density_gcm3`` is 2.70, but a 37 m
    hull at any realistic craft mass derives 0.003-0.2 g/cm3 -- three edit
    paths misbehave on that mismatch: the inspector's density readout is
    nonsense, ``set_mass`` holds the derived density constant and re-derives
    radius (silently inflating the hull past 37 m), and ``set_density`` calls
    ``nearest_by_density``, which relabels the craft "Rock" and loses
    SPACECRAFT entirely.  A craft's real editable properties are fuel and
    flight plan, and both live in the Fly panel instead.
    """
    if body.kind == KIND_CRAFT:
        return ()
    if body.kind == KIND_BLACK_HOLE:
        return BLACK_HOLE_EDIT_FIELDS
    if body.kind == KIND_STAR:
        return STAR_EDIT_FIELDS
    return PLANET_EDIT_FIELDS


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


def scrub_spin(spin, notches):
    """Linear scrub, clamped to the Thorne limit."""
    value = float(spin) + notches * SPIN_NOTCH
    return float(min(max(value, 0.0), units.MAX_SPIN))


def scrub_disk(disk, notches):
    """Linear scrub over the whole feeding range, 0 (starved) to 1."""
    value = float(disk) + notches * DISK_NOTCH
    return float(min(max(value, 0.0), 1.0))


def scrub_temperature(kelvin, notches):
    """Logarithmic scrub, clamped to the range a body may occupy."""
    scaled = max(float(kelvin), units.MIN_TEMPERATURE_K) \
        * (TEMPERATURE_NOTCH ** notches)
    return float(min(max(scaled, units.MIN_TEMPERATURE_K),
                     units.MAX_TEMPERATURE_K))


# --- reading ----------------------------------------------------------------------

def body_density(world, index):
    return units.density_from_mass_radius(float(world.mass[index]),
                                          float(world.radius[index]))


def ignition_ratio(world, index):
    """How many times heavier this body must become before it lights up.

    Zero for a black hole: it is never going to ignite, and the caller shows a
    different line entirely.
    """
    if world.bodies[index].kind == KIND_BLACK_HOLE:
        return 0.0
    mass = float(world.mass[index])
    if mass <= 0.0:
        return float("inf")
    return units.STAR_MASS_THRESHOLD / mass


# --- writing ----------------------------------------------------------------------

def _refresh_kind(world, index):
    """Kind follows mass -- except that collapse is one-way, and so is a craft."""
    body = world.bodies[index]
    if body.kind in (KIND_BLACK_HOLE, KIND_CRAFT):
        # Nothing un-collapses, and no amount of mass turns a shuttle into a
        # star.  A craft is what it is.
        return
    mass = float(world.mass[index])
    if mass >= units.COLLAPSE_MASS_THRESHOLD:
        collapse(world, index)
    elif mass >= units.STAR_MASS_THRESHOLD:
        body.kind = KIND_STAR
        # Pinned immediately, not on the next thermal step, so the readout can
        # never show a star at a planet's temperature.
        world.temperature[index] = units.star_temperature(mass)
    elif body.kind == KIND_STAR:
        body.kind = KIND_PLANET


def set_mass(world, index, mass):
    """Set mass so the radius follows.

    For an ordinary body density is held constant and the radius derived from
    it; for a black hole there is no density to hold -- the radius *is* the
    Schwarzschild radius, and it is set by ``_refresh_kind`` reaching
    ``collapse`` or by the branch below.
    """
    body = world.bodies[index]
    if is_locked(body, "mass"):
        return
    mass = float(min(max(float(mass), MIN_MASS), MAX_MASS))

    if body.kind == KIND_BLACK_HOLE:
        world.mass[index] = mass
        world.radius[index] = units.schwarzschild_radius(mass)
        world.contact_radius[index] = world.radius[index]
        world.temperature[index] = units.hawking_temperature_k(mass)
        return

    density = body_density(world, index)
    world.mass[index] = mass
    world.radius[index] = units.radius_from_mass_density(mass, density)
    _refresh_kind(world, index)


def set_temperature(world, index, kelvin):
    """Set a body's temperature.  It is not pinned there.

    Heat a world to 3000 K and it glows, boils away whatever is above its
    vaporisation point, and cools back to where the sunlight on it puts it.
    That relaxation is the whole lesson -- pinning an edited temperature would
    turn a physical quantity back into a dial.

    Refuses on a star or a black hole, whose temperatures their own physics
    fixes.  Guarded here as well as by the missing button, since the panel is
    not the only caller.

    Refuses a locked temperature for the same reason, and in the same place:
    the panel greys the row's buttons out, and this is what makes that mean
    something to every other caller.
    """
    body = world.bodies[index]
    if body.kind in (KIND_STAR, KIND_BLACK_HOLE) or is_locked(body, "temp"):
        return
    world.temperature[index] = float(
        min(max(float(kelvin), units.MIN_TEMPERATURE_K), units.MAX_TEMPERATURE_K))


def set_spin(world, index, spin):
    body = world.bodies[index]
    if is_locked(body, "spin"):
        return
    body.spin = float(min(max(float(spin), 0.0), units.MAX_SPIN))


def set_disk(world, index, disk):
    body = world.bodies[index]
    if is_locked(body, "disk"):
        return
    body.disk = float(min(max(float(disk), 0.0), 1.0))


def collapse(world, index):
    """Collapse a body into a black hole.  One-way, and mass is conserved.

    Collapsing the Sun leaves a one-solar-mass hole, which is below the
    Tolman-Oppenheimer-Volkoff limit and so something nature does not make.
    Allowing it is the lesser evil: refusing removes the most interesting
    thing the child can do here, and rounding the mass up to 3 suns invents
    matter out of nothing.
    """
    body = world.bodies[index]
    if body.kind == KIND_BLACK_HOLE:
        return
    body.kind = KIND_BLACK_HOLE
    body.material = materials.SINGULARITY
    body.rings = False
    body.spin = DEFAULT_BLACK_HOLE_SPIN
    body.disk = DEFAULT_BLACK_HOLE_DISK
    # The trail was drawn in the old material's colour and at the old size.
    body.trail.clear()
    world.radius[index] = units.schwarzschild_radius(float(world.mass[index]))
    world.contact_radius[index] = world.radius[index]
    world.temperature[index] = units.hawking_temperature_k(float(world.mass[index]))


def set_density(world, index, density):
    """Set density, holding mass constant, so the radius follows."""
    if is_locked(world.bodies[index], "density"):
        return
    lo = units.from_gcm3(materials.MIN_DENSITY_GCM3)
    hi = units.from_gcm3(materials.MAX_DENSITY_GCM3)
    density = float(min(max(float(density), lo), hi))
    world.radius[index] = units.radius_from_mass_density(
        float(world.mass[index]), density)
    # Keep the composition label honest about what it now is.
    world.bodies[index].material = materials.nearest_by_density(density)


def set_material(world, index, material):
    """Adopt a substance, snapping density to that substance's real value."""
    body = world.bodies[index]
    if is_locked(body, "material"):
        return
    # The material is now what the body is made of; a pinned look is a claim
    # about a body that no longer exists.
    body.look = None
    body.material = material
    world.radius[index] = units.radius_from_mass_density(
        float(world.mass[index]), material.density)


def cycle_material(world, index, step=1):
    body = world.bodies[index]
    if body.kind == KIND_BLACK_HOLE:
        return          # a hole is not made of anything to cycle through
    if is_locked(body, "material"):
        return
    keys = [m.key for m in materials.MATERIALS]
    current = keys.index(body.material.key)
    set_material(world, index, materials.MATERIALS[(current + step) % len(keys)])


# --- locking ----------------------------------------------------------------------
#
# A lock is a standing preference for *this* body rather than a session flag,
# which is why it lives on the Body and not in this module: select away and back,
# or leave the Edit tool and return, and it is still there.
#
# It guards its **own** field and nothing else.  Locking density does not stop a
# material swap from moving density, because ``set_material`` is editing
# material, not density.  Chasing a value through every other field that touches
# it is bookkeeping nobody asked for, and it would make "locked" mean something
# slightly different in every row.


def is_locked(body, field):
    """True if ``field`` on this body refuses to be edited."""
    return field in body.locked_fields


def toggle_lock(body, field):
    """Lock ``field`` if it is free; free it if it is locked."""
    if field in body.locked_fields:
        body.locked_fields = body.locked_fields - {field}
    else:
        body.locked_fields = body.locked_fields | {field}


def unlock_all(body):
    """Free every field at once -- one press to undo a panel full of padlocks."""
    body.locked_fields = frozenset()
