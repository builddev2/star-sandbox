"""Collision detection and resolution.

**Detection is swept, not point-in-time.**  A body thrown at 200 Earth radii per
step is outside the planet at the start of the step and outside it at the end;
only a test along the path it travelled sees the hit.  For each pair we minimise
|r0 + s*d| over s in [0, 1], where r0 is the relative position at the start of
the step and d the relative displacement during it.  Exact for linear motion,
one vectorised O(n^2) pass, and it cannot be fooled by speed.

Fragments born of the same shattering share an ``event_id`` and ignore each
other until ``no_collide_until``; without that they re-merge on the next substep
and the planet appears to flicker and reassemble.
"""

from dataclasses import dataclass

import numpy as np

from . import units
from .body import KIND_DEBRIS, KIND_PLANET, KIND_STAR


@dataclass
class Collision:
    #: Index of the more massive body.  Resolution always treats this as the target.
    i: int
    #: Index of the less massive body.
    j: int
    #: Fraction of the step at closest approach, 0..1.  Used only for ordering.
    s: float
    #: Relative speed at the end of the step, AU/day.
    v_rel: float


def find_collisions(world, prev_pos):
    """Every pair whose swept paths came within the sum of their radii.

    ``prev_pos`` is the position array as it was before the step.  Returns a list
    of :class:`Collision` sorted earliest first, with the more massive body in
    ``i``.  Pairs sharing a live ``event_id`` are excluded.
    """
    n = world.n
    if n < 2:
        return []

    p0 = prev_pos[:n]
    p1 = world.pos[:n]
    step = p1 - p0

    # r0[i, j] = p0[j] - p0[i];  d[i, j] = step[j] - step[i]
    r0 = p0[None, :, :] - p0[:, None, :]
    d = step[None, :, :] - step[:, None, :]

    dd = np.einsum("ijk,ijk->ij", d, d)
    rd = np.einsum("ijk,ijk->ij", r0, d)

    with np.errstate(divide="ignore", invalid="ignore"):
        s = np.where(dd > 0.0, -rd / dd, 0.0)
    np.clip(s, 0.0, 1.0, out=s)

    closest = r0 + s[:, :, None] * d
    dist2 = np.einsum("ijk,ijk->ij", closest, closest)

    radii = world.radius[:n]
    touch = radii[:, None] + radii[None, :]
    hit = dist2 < touch * touch
    np.fill_diagonal(hit, False)
    hit = np.triu(hit)

    rows, cols = np.nonzero(hit)
    if rows.size == 0:
        return []

    mass = world.mass[:n]
    vel = world.vel[:n]
    bodies = world.bodies
    now = world.time
    out = []
    for a, b in zip(rows.tolist(), cols.tolist()):
        ba, bb = bodies[a], bodies[b]
        if (ba.event_id and ba.event_id == bb.event_id
                and now < ba.no_collide_until):
            continue
        big, small = (a, b) if mass[a] >= mass[b] else (b, a)
        dv = vel[small] - vel[big]
        out.append(Collision(big, small, float(s[a, b]),
                             float(np.hypot(dv[0], dv[1]))))

    out.sort(key=lambda c: c.s)
    return out


def impact_energy(world, c):
    """Kinetic energy of the impact in the pair's centre-of-mass frame.

    Using the reduced mass is what makes this the energy actually available to do
    damage: the part of the kinetic energy that is not just the pair drifting.
    """
    mi, mj = world.mass[c.i], world.mass[c.j]
    total = mi + mj
    if total <= 0.0:
        return 0.0
    reduced = mi * mj / total
    return 0.5 * reduced * c.v_rel * c.v_rel


def binding_of_target(world, c):
    """Gravitational binding energy of the larger body: what holds it together."""
    return units.binding_energy(world.mass[c.i], world.radius[c.i])


def is_catastrophic(world, c):
    """True when the impact carries more energy than the target's self-gravity."""
    return bool(impact_energy(world, c) >= binding_of_target(world, c))


def merge(world, c):
    """Combine the pair into the larger body.  Returns its index afterwards.

    Mass and momentum are conserved exactly.  Density becomes the mass-weighted
    mean of the two, and the radius then follows from mass and density, so the
    result is a physically sized sphere rather than a made-up number.
    """
    i, j = c.i, c.j
    mi, mj = float(world.mass[i]), float(world.mass[j])
    total = mi + mj

    momentum = mi * world.vel[i] + mj * world.vel[j]
    centre = (mi * world.pos[i] + mj * world.pos[j]) / total

    di = units.density_from_mass_radius(mi, float(world.radius[i]))
    dj = units.density_from_mass_radius(mj, float(world.radius[j]))
    density = (mi * di + mj * dj) / total

    survivor = world.bodies[i]
    world.pos[i] = centre
    world.vel[i] = momentum / total
    world.mass[i] = total
    world.radius[i] = units.radius_from_mass_density(total, density)

    if total >= units.STAR_MASS_THRESHOLD:
        survivor.kind = KIND_STAR
    elif survivor.kind == KIND_DEBRIS and total >= units.EARTH_MASS * 0.01:
        survivor.kind = KIND_PLANET

    world.remove(j)
    # remove() swap-removes, so i may no longer be the survivor's row.
    return world.index_of(survivor.id)
