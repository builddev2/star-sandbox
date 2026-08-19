"""Two-body orbital mechanics.

Two jobs.  Building a scenario needs elements -> state vectors; the inspector and the
trajectory preview need state vectors -> elements.  Both are exact for the two-body
problem, which is what makes the predicted orbit line free to draw every frame: no
integration, just algebra.

Everything is planar (2D), so the element set is (a, e, longitude of periapsis, true
anomaly) rather than the full six.
"""

import math

import numpy as np

from . import units
from .units import G


def state_from_elements(mu, a, e, lon_peri=0.0, true_anomaly=0.0):
    """Position and velocity from orbital elements.

    ``mu`` is G*(M + m).  Angles are in radians.  Returns ``(pos, vel)`` as length-2
    numpy arrays in simulation units.
    """
    p = a * (1.0 - e * e)
    nu = true_anomaly
    r = p / (1.0 + e * math.cos(nu))

    # Perifocal frame: periapsis lies along +x.
    pos = np.array([r * math.cos(nu), r * math.sin(nu)])
    k = math.sqrt(mu / p)
    vel = np.array([-k * math.sin(nu), k * (e + math.cos(nu))])

    c, s = math.cos(lon_peri), math.sin(lon_peri)
    rot = np.array([[c, -s], [s, c]])
    return rot @ pos, rot @ vel


def true_anomaly_from_mean(mean_anomaly, e, tol=1e-12, max_iter=40):
    """Solve Kepler's equation M = E - e sin E, then convert E to true anomaly.

    Newton-Raphson from a sensible seed; converges in a handful of iterations for the
    eccentricities the solar system actually uses.
    """
    m = (mean_anomaly + math.pi) % (2.0 * math.pi) - math.pi
    ecc = math.sin(m) * e
    big = m + ecc if e < 0.8 else math.pi
    for _ in range(max_iter):
        f = big - e * math.sin(big) - m
        fp = 1.0 - e * math.cos(big)
        step = f / fp
        big -= step
        if abs(step) < tol:
            break
    return 2.0 * math.atan2(
        math.sqrt(1.0 + e) * math.sin(big * 0.5),
        math.sqrt(1.0 - e) * math.cos(big * 0.5),
    )


def elements_from_state(mu, rel_pos, rel_vel):
    """Osculating elements from a state vector relative to the attractor.

    Returns a dict with ``a``, ``e``, ``p`` (semi-latus rectum), ``period``
    (days, or ``None`` if unbound), ``periapsis``, ``apoapsis`` (or ``None``),
    ``lon_peri``, ``true_anomaly`` and ``bound``.
    """
    rx, ry = float(rel_pos[0]), float(rel_pos[1])
    vx, vy = float(rel_vel[0]), float(rel_vel[1])

    r = math.hypot(rx, ry)
    v2 = vx * vx + vy * vy
    rv = rx * vx + ry * vy

    energy = 0.5 * v2 - mu / r
    # Eccentricity vector.
    c1 = v2 - mu / r
    ex = (c1 * rx - rv * vx) / mu
    ey = (c1 * ry - rv * vy) / mu
    e = math.hypot(ex, ey)

    if abs(energy) < 1e-18:          # parabolic to within roundoff
        a = float("inf")
    else:
        a = -mu / (2.0 * energy)

    # bool(), not the bare comparison: mu is often a numpy scalar, and a
    # leaking np.bool_ breaks every `is True` check downstream.
    bound = bool(energy < 0.0)
    period = 2.0 * math.pi * math.sqrt(a ** 3 / mu) if bound and a > 0 else None

    # Semi-latus rectum from angular momentum.  This is the one form that is
    # correct for every conic, which matters because a * (1 - e) is nan for a
    # parabola and the naive unbound formula r(1-e)/(1+e) goes *negative* once
    # e > 1 -- so a body thrown past escape velocity reported a negative closest
    # approach and drew no predicted path at all.
    h = rx * vy - ry * vx
    p = h * h / mu
    periapsis = p / (1.0 + e) if e > -1.0 else r
    apoapsis = a * (1.0 + e) if bound else None

    lon_peri = math.atan2(ey, ex) if e > 1e-12 else 0.0
    nu = math.atan2(ry, rx) - lon_peri
    nu = (nu + math.pi) % (2.0 * math.pi) - math.pi

    return {
        "a": a,
        "e": e,
        "p": p,
        "period": period,
        "periapsis": periapsis,
        "apoapsis": apoapsis,
        "lon_peri": lon_peri,
        "true_anomaly": nu,
        "bound": bound,
    }


def conic_points(mu, rel_pos, rel_vel, samples=128):
    """The path a body on this state vector will follow, relative to its attractor.

    Returns ``(points, bound)``.  This is algebra, not integration: the whole
    trajectory comes out of the orbital elements in closed form, which is what
    makes it cheap enough to redraw every frame while someone is aiming.

    For a hyperbola the true anomaly is only valid within +-acos(-1/e), so the
    sampling range is narrowed accordingly and the arc is drawn open-ended.
    """
    el = elements_from_state(mu, rel_pos, rel_vel)
    e = el["e"]
    bound = el["bound"]
    p = el["p"]

    if bound:
        nus = np.linspace(-math.pi, math.pi, samples)
    else:
        # Beyond +-acos(-1/e) the conic has no real points; stop just short so
        # the arc goes off to infinity rather than wrapping round.
        limit = (math.acos(min(1.0, max(-1.0, -1.0 / e))) if e > 1.0 else math.pi)
        nus = np.linspace(-limit * 0.985, limit * 0.985, samples)

    if not math.isfinite(p) or p <= 0.0:
        return np.zeros((0, 2)), bound

    denominator = 1.0 + e * np.cos(nus)
    good = denominator > 1e-9
    nus = nus[good]
    radii = p / denominator[good]

    angles = nus + el["lon_peri"]
    return np.stack([radii * np.cos(angles), radii * np.sin(angles)], axis=1), bound


def dominant_attractor(world, index):
    """Index of the body that ``index`` should be considered to orbit.

    **Not the body pulling hardest.**  The Sun pulls on the Moon about twice as
    hard as the Earth does -- that is a real fact about the solar system, not a
    numerical accident -- so raw force gives "Sun" and every moon in the
    inspector reads as orbiting the Sun on a wildly eccentric path.

    The right question is which sphere of influence the body sits inside.  A
    candidate qualifies when it is more massive, when ``index`` lies within its
    Hill radius about the system's primary, and when the two of them are
    gravitationally bound in the two-body sense.  Of the candidates that
    qualify, the most local one wins -- smallest Hill radius, i.e. deepest
    nesting.

    The boundness test is what separates a moon from its neighbours: Io sits
    well inside Ganymede's Hill radius, but the two are racing past each other
    at far above their mutual escape velocity, so Ganymede is correctly
    rejected and Jupiter wins.

    Falls back to the most massive body, and returns None only for a body that
    is itself the most massive or is alone.
    """
    n = world.n
    if n < 2:
        return None

    pos = world.pos[:n]
    vel = world.vel[:n]
    mass = world.mass[:n]

    heaviest = int(np.argmax(mass))
    if heaviest == index:
        return None

    d = pos - pos[index]
    r = np.sqrt(np.einsum("ij,ij->i", d, d))
    r[index] = np.inf

    dv = vel - vel[index]
    v2 = np.einsum("ij,ij->i", dv, dv)

    # Hill radius of every candidate, measured about the system's primary.
    from_primary = pos - pos[heaviest]
    a = np.sqrt(np.einsum("ij,ij->i", from_primary, from_primary))
    hill = units.hill_radius(mass, mass[heaviest], a)

    mu = G * (mass + mass[index])
    bound = (0.5 * v2 - mu / r) < 0.0

    ok = (mass > mass[index]) & (r < hill) & bound
    ok[index] = False
    ok[heaviest] = False

    if not ok.any():
        return heaviest
    candidates = np.flatnonzero(ok)
    return int(candidates[np.argmin(hill[candidates])])


def circular_velocity(world, index, attractor, retrograde=False):
    """Velocity that puts body ``index`` on a circular orbit about ``attractor``.

    Returned in the world frame, so it already includes the attractor's own motion --
    a moon of a moving planet gets the right answer.
    """
    rel = world.pos[index] - world.pos[attractor]
    r = float(np.hypot(rel[0], rel[1]))
    if r <= 0.0:
        return world.vel[attractor].copy()

    mu = G * (world.mass[attractor] + world.mass[index])
    speed = math.sqrt(mu / r)
    # Rotate the radius vector a quarter turn to get the tangential direction.
    tangent = np.array([-rel[1], rel[0]]) / r
    if retrograde:
        tangent = -tangent
    return world.vel[attractor] + speed * tangent
