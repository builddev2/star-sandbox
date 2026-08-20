"""Destruction that needs no contact: a star's tide, and its light.

A body can be torn apart or boiled away by a star it never touches. Both
checks run over stars only, so the cost is O(n * n_stars) -- negligible next
to the O(n^2) force kernel.

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
from .body import KIND_STAR
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


def _bodies_and_stars(world):
    n = world.n
    stars = [k for k in range(n) if world.bodies[k].kind == KIND_STAR]
    bodies = [i for i in range(n) if world.bodies[i].kind != KIND_STAR]
    return bodies, stars


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
    """Tidal disruption and thermal ablation near a star.

    Returns ``(count, event)`` -- the number of bodies destroyed (0 or 1)
    this call, and an :class:`ImpactEvent` describing it, or ``None``.
    """
    bodies, stars = _bodies_and_stars(world)
    if not bodies or not stars:
        return 0, None

    worst_ratio, worst_i, worst_k = 0.0, None, None
    for i in bodies:
        for k in stars:
            ratio = tidal_stress_ratio(world, i, k)
            if ratio > worst_ratio:
                worst_ratio, worst_i, worst_k = ratio, i, k
    if worst_ratio >= 1.0:
        event = _event(world, worst_i, "tidal")
        resolver.tidal_disrupt(world, worst_i, worst_k)
        return 1, event

    vaporised = None
    for i in bodies:
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
