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
