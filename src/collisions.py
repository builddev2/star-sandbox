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

import math
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
    #: Speed at contact, AU/day: ``v_rel`` with the pair's mutual escape
    #: velocity folded in.  A body released from rest at contact separation
    #: still arrives at escape speed, and ``v_rel`` alone -- measured once at
    #: the end of the step -- misses that gravitational focusing entirely.
    v_impact: float = 0.0
    #: Normalised impact parameter at contact: 0 is dead centre, 1 is a graze
    #: along the line where the two surfaces just touch.
    b: float = 0.0


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
        v_rel = float(np.hypot(dv[0], dv[1]))
        sep = float(radii[big] + radii[small])
        v_esc = math.sqrt(2.0 * units.G * float(mass[big] + mass[small]) / sep) \
            if sep > 0.0 else 0.0
        v_impact = math.hypot(v_rel, v_esc)
        miss = math.sqrt(max(float(dist2[a, b]), 0.0))
        impact_b = min(miss / sep, 1.0) if sep > 0.0 else 0.0
        out.append(Collision(big, small, float(s[a, b]), v_rel, v_impact, impact_b))

    out.sort(key=lambda c: c.s)
    return out


def impact_energy(world, c):
    """Kinetic energy of the impact in the pair's centre-of-mass frame, at contact.

    Using the reduced mass is what makes this the energy actually available to do
    damage: the part of the kinetic energy that is not just the pair drifting.
    ``v_impact`` rather than ``v_rel`` is what makes it the energy on arrival,
    not the energy however far apart the two bodies happened to be when the
    swept-collision test last sampled them.
    """
    mi, mj = world.mass[c.i], world.mass[c.j]
    total = mi + mj
    if total <= 0.0:
        return 0.0
    reduced = mi * mj / total
    return 0.5 * reduced * c.v_impact * c.v_impact


def specific_impact_energy(world, c):
    """Impact kinetic energy per unit total mass -- the Q_R of the literature."""
    total = float(world.mass[c.i] + world.mass[c.j])
    return impact_energy(world, c) / total if total > 0.0 else 0.0


def binding_of_target(world, c):
    """Gravitational binding energy of the larger body: what holds it together."""
    return units.binding_energy(world.mass[c.i], world.radius[c.i])


def is_catastrophic(world, c):
    """True when the impact carries more energy than the target's self-gravity.

    Superseded by :func:`classify`, which compares against the target's real
    disruption threshold instead of gravity alone; kept until that lands.
    """
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


#: Fewer than this and a shattering reads as a bad merge rather than a break-up.
MIN_FRAGMENTS = 6
#: More than this and the world fills with debris the O(n^2) kernel cannot afford.
MAX_FRAGMENTS = 12

#: Fragments are placed on a circle this many target-radii out, so they start
#: clear of one another rather than interpenetrating.
FRAGMENT_RING = 1.15
#: Immunity lasts long enough for fragments to travel this many target radii.
IMMUNITY_RADII = 5.0


class CollisionResolver:
    """Owns the RNG and the event counter for collision outcomes.

    Kept as an object rather than module functions so that tests are
    deterministic and independent: two resolvers with the same seed produce the
    same debris field, and no global state leaks between tests.
    """

    def __init__(self, seed=1234):
        self.rng = np.random.default_rng(seed)
        self._next_event_id = 1

    # --- outcomes --------------------------------------------------------------

    def shatter(self, world, c):
        """Break the pair into fragments.  Returns their indices.

        Falls back to :func:`merge` when there is not room for a credible number
        of fragments -- a two-piece "shattering" looks like a bug.
        """
        i, j = c.i, c.j
        mi, mj = float(world.mass[i]), float(world.mass[j])
        total = mi + mj
        r_big = float(world.radius[i])

        momentum = mi * world.vel[i] + mj * world.vel[j]
        centre = (mi * world.pos[i] + mj * world.pos[j]) / total
        drift = momentum / total

        di = units.density_from_mass_radius(mi, r_big)
        dj = units.density_from_mass_radius(mj, float(world.radius[j]))
        density = (mi * di + mj * dj) / total

        # How badly did the impact overshoot what held the target together?
        binding = binding_of_target(world, c)
        excess = impact_energy(world, c) / binding if binding > 0.0 else 1.0
        excess = max(excess, 1.0)

        wanted = int(MIN_FRAGMENTS + 2.0 * np.log2(excess))
        room = world.capacity - (world.n - 2)
        count = int(min(wanted, MAX_FRAGMENTS, room))
        if count < MIN_FRAGMENTS:
            # A two-piece "shattering" reads as a bug, not a break-up.
            merge(world, c)
            return []

        parent = world.bodies[i]
        material = parent.material
        event_id = self._next_event_id
        self._next_event_id += 1

        # Mass split: one dominant remnant, the rest a random partition.
        remnant_share = float(np.clip(0.50 - 0.08 * np.log2(excess), 0.30, 0.50))
        weights = self.rng.random(count - 1) + 0.15
        weights *= (1.0 - remnant_share) / weights.sum()
        shares = np.concatenate(([remnant_share], weights))
        masses = shares * total

        kick = units.escape_velocity(total, r_big)
        angles = (np.arange(count) / count) * 2.0 * np.pi
        angles += self.rng.normal(0.0, 0.25, count)
        directions = np.stack([np.cos(angles), np.sin(angles)], axis=1)
        speeds = kick * self.rng.uniform(0.55, 1.45, count)
        # The remnant keeps most of the mass, so give it least of the kick.
        speeds[0] *= 0.35

        positions = centre + directions * (FRAGMENT_RING * r_big)
        velocities = drift + directions * speeds[:, None]

        # Correct so total momentum is exactly what it was before the impact.
        error = (masses[:, None] * velocities).sum(axis=0) - momentum
        velocities -= error / total

        world.remove_many([i, j])

        made = []
        for k in range(count):
            mass = float(masses[k])
            body = world.add(
                "Fragment", material, mass,
                units.radius_from_mass_density(mass, density),
                positions[k], velocities[k],
                kind=KIND_DEBRIS, seed=int(self.rng.integers(0, 1 << 30)),
            )
            if body is None:
                break
            body.event_id = event_id
            body.no_collide_until = world.time + IMMUNITY_RADII * r_big / max(kick, 1e-12)
            made.append(world.index_of(body.id))
        return made

    # --- driver ----------------------------------------------------------------

    def resolve(self, world, integrator, prev_pos):
        """Detect and resolve every collision from the step just taken.

        Returns the number of events.  Indices shift as bodies are removed, so
        only the first collision of a batch is trusted directly; the rest are
        re-detected on the next call, which is one substep later and invisible.
        """
        hits = find_collisions(world, prev_pos)
        if not hits:
            return 0
        c = hits[0]
        if is_catastrophic(world, c):
            self.shatter(world, c)
        else:
            merge(world, c)
        integrator.refresh(world)
        return 1
