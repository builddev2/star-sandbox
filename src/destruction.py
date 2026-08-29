"""Destruction that needs no contact: a tide, and a star's light.

A body can be torn apart by a tide -- raised by a star or a black hole -- or
boiled away by a star's light, without ever touching what does it. The tidal
check runs over every attractor; the thermal check runs over stars only,
since a black hole radiates nothing. Both are O(n * n_attractors) --
negligible next to the O(n^2) force kernel.

Tidal disruption is instantaneous and, like ``CollisionResolver.resolve()``,
handles only the single worst violation per call: indices shift when a body
is removed, so anything else that also crossed the line this substep is
re-detected, invisibly, on the next one. Thermal ablation is continuous --
every body above its vaporisation point loses a little mass every call --
and is vectorised freely since that never changes an index; only the moment
a body's mass reaches the floor does removal happen, and that too is capped
at one per call for the same reason.
"""

import math

import numpy as np

from . import units
from .body import KIND_BLACK_HOLE, KIND_STAR
from .collisions import ImpactEvent

#: Below this mass, an ablating body is "gone" rather than a vanishingly
#: small remnant still worth tracking -- about 2e11 kg, a few hundred
#: metres of rock.
VAPORISE_FLOOR_MASS = 1e-19
#: Fractional mass lost per day at exactly the vaporisation point, growing
#: linearly with how far past it the body runs.
ABLATION_RATE_PER_DAY = 0.08
#: How far past the vaporisation temperature the ablation rate is allowed to
#: scale before it saturates -- a sungrazer boils away over a dramatic
#: handful of days, not in one substep.
ABLATION_MAX_MULTIPLE = 6.0


def tidal_stress_ratio(world, i, k):
    """How far past its own strength the tide from star ``k`` stretches body
    ``i``.  1.0 or more means the tide wins.
    """
    rel = world.pos[i] - world.pos[k]
    d = float(np.hypot(rel[0], rel[1]))
    if d <= 0.0:
        return math.inf
    r = float(world.radius[i])
    if r <= 0.0:
        return 0.0
    tide = 2.0 * units.G * float(world.mass[k]) * r / d ** 3
    material = world.bodies[i].material
    g_self = units.surface_gravity(float(world.mass[i]), r)
    strength = units.strength_acceleration(r, material.toughness)
    holding = g_self + strength
    return tide / holding if holding > 0.0 else math.inf


def equilibrium_temperature(world, i, k):
    """Equilibrium surface temperature of body ``i`` facing star ``k``, kelvin."""
    rel = world.pos[i] - world.pos[k]
    d = float(np.hypot(rel[0], rel[1]))
    if d <= 0.0:
        return math.inf
    r_star = float(world.radius[k])
    t_star = units.star_temperature(float(world.mass[k]))
    return t_star * math.sqrt(r_star / (2.0 * d))


def _bodies_and_attractors(world):
    """(victims, stars, attractors) for this world.

    **Attractors** raise tides: stars and black holes both. **Stars** alone
    radiate -- a stellar-mass hole's Hawking temperature is 6 nanokelvin,
    colder than the microwave background, so it boils nothing.

    A black hole is never a victim: nothing tears a horizon apart. A star may
    be a victim, but only of a hole -- a tidal disruption event is the most
    spectacular thing in this simulation, while two ordinary stars would have
    to be nearly touching, and letting them shred each other would wreck every
    binary arrangement.
    """
    n = world.n
    kinds = [world.bodies[k].kind for k in range(n)]
    stars = [k for k in range(n) if kinds[k] == KIND_STAR]
    attractors = [k for k in range(n)
                  if kinds[k] in (KIND_STAR, KIND_BLACK_HOLE)]
    victims = [i for i in range(n) if kinds[i] != KIND_BLACK_HOLE]
    return victims, stars, attractors


def _event(world, i, outcome):
    """Snapshot body ``i`` as an :class:`ImpactEvent`, before it changes."""
    pos = world.pos[i]
    vel = world.vel[i]
    body = world.bodies[i]
    return ImpactEvent(
        pos=(float(pos[0]), float(pos[1])),
        vel=(float(vel[0]), float(vel[1])),
        energy=0.0,          # no second body to measure a contact energy against
        outcome=outcome,
        target_radius=float(world.radius[i]),
        projectile_radius=0.0,
        tones=body.material.tones,
        on_star=False,
    )


def apply(world, resolver, dt):
    """Tidal disruption and thermal ablation near a star or a black hole.

    Returns ``(count, event)`` -- the number of bodies destroyed (0 or 1)
    this call, and an :class:`ImpactEvent` describing it, or ``None``.
    """
    victims, stars, attractors = _bodies_and_attractors(world)
    if not victims or not attractors:
        return 0, None

    worst_ratio, worst_i, worst_k = 0.0, None, None
    for i in victims:
        # Fragments are born a body-radius from where their parent died, which
        # is deep inside the tidal radius that killed it -- without this they
        # are re-shredded on the very next substep and the world fills with
        # debris. The immunity tidal_disrupt already grants them is exactly
        # long enough to carry them clear.
        if world.time < world.bodies[i].no_collide_until:
            continue
        star_victim = world.bodies[i].kind == KIND_STAR
        for k in attractors:
            if k == i:
                continue
            # Only a black hole may shred a star.
            if star_victim and world.bodies[k].kind != KIND_BLACK_HOLE:
                continue
            ratio = tidal_stress_ratio(world, i, k)
            if ratio > worst_ratio:
                worst_ratio, worst_i, worst_k = ratio, i, k
    if worst_ratio >= 1.0:
        event = _event(world, worst_i, "tidal")
        resolver.tidal_disrupt(world, worst_i, worst_k)
        return 1, event

    vaporised = None
    for i in victims:
        if world.bodies[i].kind == KIND_STAR:
            continue                    # a star does not boil in its own light
        hottest = max((equilibrium_temperature(world, i, k) for k in stars),
                     default=0.0)
        vapour = world.bodies[i].material.vapour_temperature_k
        if hottest <= vapour:
            continue
        excess = min(hottest / vapour - 1.0, ABLATION_MAX_MULTIPLE)
        mass = float(world.mass[i])
        density = units.density_from_mass_radius(mass, float(world.radius[i]))
        new_mass = mass * max(0.0, 1.0 - ABLATION_RATE_PER_DAY * excess * dt)
        if new_mass <= VAPORISE_FLOOR_MASS:
            vaporised = i
            break
        world.mass[i] = new_mass
        world.radius[i] = units.radius_from_mass_density(new_mass, density)
        world.contact_radius[i] = world.radius[i]

    if vaporised is not None:
        event = _event(world, vaporised, "vaporise")
        world.remove(vaporised)
        return 1, event
    return 0, None
