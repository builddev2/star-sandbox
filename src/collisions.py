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
from .body import KIND_BLACK_HOLE, KIND_DEBRIS, KIND_PLANET, KIND_STAR


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


@dataclass(frozen=True)
class ImpactEvent:
    """A plain-data record of one destructive event, for the render layer.

    Built from the pair *before* it is mutated, so it describes what
    happened, not what is left afterward -- the survivor's post-merge radius
    is not "the impact." Holds only floats and tuples, no numpy arrays or
    surfaces, so it stays on the physics side of the render/physics split
    that ``tests/test_layering.py`` enforces.
    """
    pos: tuple            # (x, y) AU, where the impact happened
    vel: tuple            # (vx, vy) AU/day, relative velocity at impact
    energy: float          # sim units, kinetic energy at contact
    outcome: str            # ACCRETE, GRAZE, DISRUPT, "tidal", or "vaporise"
    target_radius: float
    projectile_radius: float
    tones: tuple            # (dark, mid, light) RGB, the target's material
    on_star: bool


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

    # Detection uses the contact radius -- what is drawn, clamped so it can
    # never reach a neighbour's orbit (render/scale.py) -- so aiming at what
    # the eye sees actually connects.  Every outcome computed below (energy,
    # impact parameter, disruption threshold, ...) uses the true radius:
    # only the yes/no question of contact is answered by what is on screen.
    radii = world.radius[:n]
    contact = world.contact_radius[:n]
    touch = contact[:, None] + contact[None, :]
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
        # A black hole is always the target, whatever the masses say: a light
        # hole striking a heavy star would otherwise be absorbed *by the star*.
        a_hole = ba.kind == KIND_BLACK_HOLE
        b_hole = bb.kind == KIND_BLACK_HOLE
        if a_hole != b_hole:
            big, small = (a, b) if a_hole else (b, a)
        else:
            big, small = (a, b) if mass[a] >= mass[b] else (b, a)
        dv = vel[small] - vel[big]
        v_rel = float(np.hypot(dv[0], dv[1]))
        sep = float(radii[big] + radii[small])
        v_esc = math.sqrt(2.0 * units.G * float(mass[big] + mass[small]) / sep) \
            if sep > 0.0 else 0.0
        # A horizon is essentially a point, so sep collapses and the mutual
        # escape velocity runs away -- and impact_energy with it, squared.
        v_impact = min(math.hypot(v_rel, v_esc), units.C)
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


ACCRETE = "accrete"
GRAZE = "graze"
DISRUPT = "disrupt"
#: Not a fourth outcome -- the physics of a swallow *is* accretion.  This is
#: the label ImpactEvent carries so the effects layer can draw a collapse
#: inward rather than an explosion outward.
SWALLOW = "swallow"

#: Below this mass ratio the smaller body cannot survive a graze -- it is a
#: bullet, not a second planet, and simply buries itself regardless of angle.
GRAZE_MIN_MASS_RATIO = 0.02

#: A graze is inelastic: this much of the normal-component relative speed
#: survives it.  Rock does not bounce.
GRAZE_RESTITUTION = 0.3
#: Fraction of the smaller body's mass scraped off and accreted by the larger
#: one in a graze.
GRAZE_MASS_TRANSFER = 0.1


def threshold_of_target(world, c):
    """Specific energy needed to disperse the pair, in simulation units."""
    material = world.bodies[c.i].material
    density = units.density_from_mass_radius(float(world.mass[c.i]),
                                             float(world.radius[c.i]))
    return units.disruption_threshold(float(world.radius[c.i]), density,
                                      material.toughness)


def classify(world, c):
    """Which of the three outcomes this impact is.

    A graze needs both a shallow-enough angle (b past the target's own
    critical impact parameter -- for the Sun against an asteroid that is
    0.99999, correctly ruling a graze out) and a projectile heavy enough to
    survive being deflected rather than simply erased.  Anything else is
    accretion or disruption, decided by the specific impact energy against
    the target's own disruption threshold.
    """
    # Nothing comes back out of a horizon: no graze, no fragments, no ejecta.
    if (world.bodies[c.i].kind == KIND_BLACK_HOLE
            or world.bodies[c.j].kind == KIND_BLACK_HOLE):
        return ACCRETE
    r_i, r_j = float(world.radius[c.i]), float(world.radius[c.j])
    b_crit = r_i / (r_i + r_j) if (r_i + r_j) > 0.0 else 0.0
    ratio = float(world.mass[c.j] / world.mass[c.i]) if world.mass[c.i] > 0.0 else 0.0
    if c.b > b_crit and ratio >= GRAZE_MIN_MASS_RATIO:
        return GRAZE
    if specific_impact_energy(world, c) >= threshold_of_target(world, c):
        return DISRUPT
    return ACCRETE


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

    if survivor.kind == KIND_BLACK_HOLE:
        # Radius follows the horizon, not a mass-weighted density.  The mass
        # ceiling does not apply here: eating is how every hole above the
        # stellar range was actually made.
        world.radius[i] = units.schwarzschild_radius(total)
    else:
        world.radius[i] = units.radius_from_mass_density(total, density)
        if total >= units.STAR_MASS_THRESHOLD:
            survivor.kind = KIND_STAR
        elif survivor.kind == KIND_DEBRIS and total >= units.EARTH_MASS * 0.01:
            survivor.kind = KIND_PLANET

    # Reset rather than leave stale: contact_radius must never sit below the
    # true radius it is stretched from, and the render layer only refreshes
    # it once a frame.
    world.contact_radius[i] = world.radius[i]

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

#: Leinhardt & Stewart (2012): above this multiple of the disruption threshold
#: the largest remnant collapses into a supercatastrophic power law rather
#: than the near-linear regime below it.
SUPERCATASTROPHIC_RATIO = 1.8
#: Fraction of the impact energy *above* what disruption itself consumed that
#: ends up as fragment motion.  The rest goes to heat and to grinding rock
#: into gravel, which is where most of a real impact's energy actually goes.
DISPERSAL_EFFICIENCY = 0.2


def largest_remnant_fraction(q_ratio):
    """Mass of the biggest surviving piece, as a fraction of the pair's total.

    ``q_ratio`` is the specific impact energy divided by the target's
    disruption threshold. At the threshold itself (1.0) half the mass
    survives as one piece; well past it the remnant collapses toward dust.
    """
    if q_ratio <= SUPERCATASTROPHIC_RATIO:
        return float(np.clip(0.5 * (2.0 - q_ratio), 0.05, 1.0))
    return float(np.clip(0.1 * (q_ratio / SUPERCATASTROPHIC_RATIO) ** -1.5,
                         0.01, 0.5))


#: Below this multiple of the disruption threshold, cratering ejecta is a
#: picture for the effects layer to draw, not physics -- the merge stays
#: exact, which is what keeps the O(n^2) kernel from filling up with debris
#: from every gentle bump.
EJECTA_MIN_RATIO = 0.15
#: Ejecta mass as a fraction of the total, growing with how energetic the
#: impact was and capped well short of the point this should have counted as
#: a disruption instead.
EJECTA_MAX_FRACTION = 0.2
EJECTA_SLOPE = 0.3
#: Half-angle of the cone ejecta is thrown into, centred on the direction
#: away from the impact point.
EJECTA_SPREAD = 0.6


def ejecta_mass_fraction(q_ratio):
    """Fraction of the merged mass excavated as real ejecta bodies."""
    return float(np.clip(EJECTA_SLOPE * q_ratio, 0.0, EJECTA_MAX_FRACTION))


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

    def graze(self, world, c):
        """A glancing blow: both bodies survive, deflected and lightly mixed.

        The normal component of the relative velocity (along the line
        joining the centres) is reflected and damped by ``GRAZE_RESTITUTION``
        -- rock does not bounce -- while the tangential component, which is
        most of the relative velocity for a near-grazing hit, passes through
        unchanged. A sliver of the smaller body's mass is scraped onto the
        larger one. Returns ``(i, j)``, unchanged from ``c`` since a graze
        never removes a body.
        """
        i, j = c.i, c.j
        mi, mj = float(world.mass[i]), float(world.mass[j])
        ri, rj = float(world.radius[i]), float(world.radius[j])
        total = mi + mj
        momentum = mi * world.vel[i] + mj * world.vel[j]

        normal = world.pos[j] - world.pos[i]
        dist = float(np.hypot(normal[0], normal[1]))
        normal = normal / dist if dist > 0.0 else np.array([1.0, 0.0])

        dv = world.vel[j] - world.vel[i]
        dv_normal = float(np.dot(dv, normal)) * normal
        dv_tangent = dv - dv_normal
        new_dv = -GRAZE_RESTITUTION * dv_normal + dv_tangent

        di = units.density_from_mass_radius(mi, ri)
        dj = units.density_from_mass_radius(mj, rj)
        transfer = GRAZE_MASS_TRANSFER * mj
        new_mi, new_mj = mi + transfer, mj - transfer

        world.mass[i], world.mass[j] = new_mi, new_mj
        world.radius[i] = units.radius_from_mass_density(new_mi, di)
        world.radius[j] = units.radius_from_mass_density(new_mj, dj)
        world.contact_radius[i] = world.radius[i]
        world.contact_radius[j] = world.radius[j]

        # Solved directly rather than corrected after the fact: this is the
        # unique (vel_i, vel_j) pair whose difference is new_dv and whose
        # mass-weighted sum is the momentum the pair had going in.
        world.vel[i] = (momentum - new_mj * new_dv) / total
        world.vel[j] = world.vel[i] + new_dv

        # Nudge the smaller body clear along the same line so the pair does
        # not still read as touching next frame.
        sep = float(world.radius[i] + world.radius[j])
        world.pos[j] = world.pos[i] + normal * (1.01 * sep)

        event_id = self._next_event_id
        self._next_event_id += 1
        speed = float(np.hypot(new_dv[0], new_dv[1]))
        until = world.time + IMMUNITY_RADII * sep / max(speed, 1e-12)
        for k in (i, j):
            body = world.bodies[k]
            body.event_id = event_id
            body.no_collide_until = until

        return i, j

    def accrete(self, world, c):
        """Merge the pair, throwing off a crater's worth of debris when the
        impact is energetic enough to excavate real ejecta.

        Below ``EJECTA_MIN_RATIO``, or with too little room left in the
        world, the merge stays exact -- a spray at that scale is the effects
        layer's business to draw, not physics to simulate. Returns the
        survivor's index.
        """
        i, j = c.i, c.j
        q_r = specific_impact_energy(world, c)
        q_star = threshold_of_target(world, c)
        q_ratio = q_r / q_star if q_star > 0.0 else 0.0

        # The impact direction, before the pair becomes one body and it is
        # lost: ejecta is thrown back out the way the impactor came in.
        approach = world.pos[j] - world.pos[i]
        dist = float(np.hypot(approach[0], approach[1]))
        direction = approach / dist if dist > 0.0 else np.array([1.0, 0.0])

        idx = merge(world, c)
        if world.bodies[idx].kind == KIND_BLACK_HOLE:
            return idx          # a horizon excavates no crater
        room = world.capacity - world.n
        if q_ratio < EJECTA_MIN_RATIO or room < MIN_FRAGMENTS:
            return idx

        total = float(world.mass[idx])
        r_total = float(world.radius[idx])
        density = units.density_from_mass_radius(total, r_total)
        ejecta_total = ejecta_mass_fraction(q_ratio) * total
        count = min(MIN_FRAGMENTS + int(2.0 * np.log2(q_ratio / EJECTA_MIN_RATIO)),
                   MAX_FRAGMENTS, room)
        if count < MIN_FRAGMENTS:
            return idx

        weights = self.rng.random(count) + 0.15
        weights /= weights.sum()
        masses = weights * ejecta_total

        v_esc = units.escape_velocity(total, r_total)
        base_angle = math.atan2(direction[1], direction[0])
        spread = self.rng.uniform(-EJECTA_SPREAD, EJECTA_SPREAD, count)
        directions = np.stack([np.cos(base_angle + spread), np.sin(base_angle + spread)],
                              axis=1)
        speeds = v_esc * self.rng.uniform(1.0, 1.3, count)

        survivor_pos = world.pos[idx].copy()
        survivor_vel = world.vel[idx].copy()
        positions = survivor_pos + directions * (FRAGMENT_RING * r_total)
        velocities = survivor_vel + directions * speeds[:, None]

        # The survivor keeps the recoil: total momentum is exactly what the
        # merge produced, split between what stays and what leaves.
        ejecta_momentum = (masses[:, None] * velocities).sum(axis=0)
        remaining = total - ejecta_total
        world.mass[idx] = remaining
        world.radius[idx] = units.radius_from_mass_density(remaining, density)
        world.contact_radius[idx] = world.radius[idx]
        world.vel[idx] = (total * survivor_vel - ejecta_momentum) / remaining

        material = world.bodies[idx].material
        event_id = self._next_event_id
        self._next_event_id += 1
        until = world.time + IMMUNITY_RADII * r_total / max(v_esc, 1e-12)
        for k in range(count):
            mass = float(masses[k])
            body = world.add(
                "Ejecta", material, mass,
                units.radius_from_mass_density(mass, density),
                positions[k], velocities[k],
                kind=KIND_DEBRIS, seed=int(self.rng.integers(0, 1 << 30)),
            )
            if body is None:
                break
            body.event_id = event_id
            body.no_collide_until = until
        return idx

    def tidal_disrupt(self, world, i, k):
        """Break body ``i`` apart under star ``k``'s tide.

        Similar in spirit to :meth:`shatter`, but there is only one body: a
        tide pulls without merging into anything, and the star that raised
        it does not move. Returns the fragment indices, or an empty list if
        there is no room to spare for them -- the body is left alone rather
        than silently deleted.
        """
        mass = float(world.mass[i])
        r = float(world.radius[i])
        if mass <= 0.0 or r <= 0.0:
            return []
        density = units.density_from_mass_radius(mass, r)
        material = world.bodies[i].material
        pos_i, vel_i = world.pos[i].copy(), world.vel[i].copy()

        room = world.capacity - (world.n - 1)
        count = min(MIN_FRAGMENTS, MAX_FRAGMENTS, room)
        if count < MIN_FRAGMENTS:
            return []

        weights = self.rng.random(count) + 0.15
        weights /= weights.sum()
        masses = weights * mass

        v_esc = units.escape_velocity(mass, r)
        angles = (np.arange(count) / count) * 2.0 * np.pi
        angles += self.rng.normal(0.0, 0.25, count)
        directions = np.stack([np.cos(angles), np.sin(angles)], axis=1)
        speeds = v_esc * self.rng.uniform(0.5, 1.2, count)

        positions = pos_i + directions * (FRAGMENT_RING * r)
        velocities = vel_i + directions * speeds[:, None]

        world.remove(i)
        event_id = self._next_event_id
        self._next_event_id += 1
        until = world.time + IMMUNITY_RADII * r / max(v_esc, 1e-12)
        made = []
        for m in range(count):
            frag_mass = float(masses[m])
            body = world.add(
                "Debris", material, frag_mass,
                units.radius_from_mass_density(frag_mass, density),
                positions[m], velocities[m],
                kind=KIND_DEBRIS, seed=int(self.rng.integers(0, 1 << 30)),
            )
            if body is None:
                break
            body.event_id = event_id
            body.no_collide_until = until
            made.append(world.index_of(body.id))
        return made

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
        q_r = specific_impact_energy(world, c)
        q_star = threshold_of_target(world, c)
        q_ratio = max(q_r / q_star, 1.0) if q_star > 0.0 else 1.0

        wanted = int(MIN_FRAGMENTS + 2.0 * np.log2(q_ratio))
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

        # Mass split: one dominant remnant (Leinhardt & Stewart), the rest a
        # random partition.
        remnant_share = largest_remnant_fraction(q_ratio)
        weights = self.rng.random(count - 1) + 0.15
        weights *= (1.0 - remnant_share) / weights.sum()
        shares = np.concatenate(([remnant_share], weights))
        masses = shares * total

        angles = (np.arange(count) / count) * 2.0 * np.pi
        angles += self.rng.normal(0.0, 0.25, count)
        directions = np.stack([np.cos(angles), np.sin(angles)], axis=1)

        # Shape the speed distribution first, then scale it to the energy the
        # impact actually has left to disperse the debris with -- above what
        # disruption itself consumed, and discounted by DISPERSAL_EFFICIENCY
        # since most of a real impact's energy goes to heat, not motion.  A
        # marginal disruption then barely separates; an overkill one sprays.
        v_esc = units.escape_velocity(total, r_big)
        speeds = self.rng.uniform(0.55, 1.45, count)
        # The remnant keeps most of the mass, so give it least of the kick.
        speeds[0] *= 0.35
        budget = DISPERSAL_EFFICIENCY * max(q_r - q_star, 0.0) * total
        ke_shape = 0.5 * float((masses * speeds * speeds).sum())
        if ke_shape > 0.0 and budget > 0.0:
            speeds *= math.sqrt(budget / ke_shape)
        # Never below escape speed, or the pieces just fall back together and
        # the break-up is invisible.
        floor = 1.05 * v_esc
        peak = float(speeds.max())
        if peak > 0.0 and peak < floor:
            speeds *= floor / peak

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
            body.no_collide_until = world.time + IMMUNITY_RADII * r_big / max(v_esc, 1e-12)
            made.append(world.index_of(body.id))
        return made

    # --- driver ----------------------------------------------------------------

    def resolve(self, world, integrator, prev_pos):
        """Detect and resolve every collision from the step just taken.

        Returns ``(count, event)`` -- the number of events (0 or 1) and an
        :class:`ImpactEvent` describing it, or ``None``. Indices shift as
        bodies are removed, so only the first collision of a batch is
        trusted directly; the rest are re-detected on the next call, which
        is one substep later and invisible.
        """
        hits = find_collisions(world, prev_pos)
        if not hits:
            return 0, None
        c = hits[0]
        outcome = classify(world, c)
        event = self._build_event(world, c, outcome)
        if outcome == DISRUPT:
            self.shatter(world, c)
        elif outcome == GRAZE:
            self.graze(world, c)
        else:
            self.accrete(world, c)
        integrator.refresh(world)
        return 1, event

    @staticmethod
    def _build_event(world, c, outcome):
        """Snapshot a collision as an :class:`ImpactEvent`, before anything
        about the pair changes."""
        pos = world.pos[c.j]
        vel = world.vel[c.j] - world.vel[c.i]
        target = world.bodies[c.i]
        if target.kind == KIND_BLACK_HOLE:
            outcome = SWALLOW
        return ImpactEvent(
            pos=(float(pos[0]), float(pos[1])),
            vel=(float(vel[0]), float(vel[1])),
            energy=impact_energy(world, c),
            outcome=outcome,
            target_radius=float(world.radius[c.i]),
            projectile_radius=float(world.radius[c.j]),
            tones=target.material.tones,
            on_star=target.kind == KIND_STAR,
        )
