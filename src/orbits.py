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

    Returns a dict with ``a``, ``e``, ``period`` (days, or ``None`` if unbound),
    ``periapsis``, ``apoapsis`` (or ``None``), ``lon_peri`` and ``true_anomaly``.
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

    bound = energy < 0.0
    period = 2.0 * math.pi * math.sqrt(a ** 3 / mu) if bound and a > 0 else None
    periapsis = a * (1.0 - e) if bound else r * (1.0 - e) / (1.0 + e) if e > 0 else r
    apoapsis = a * (1.0 + e) if bound else None

    lon_peri = math.atan2(ey, ex) if e > 1e-12 else 0.0
    nu = math.atan2(ry, rx) - lon_peri
    nu = (nu + math.pi) % (2.0 * math.pi) - math.pi

    return {
        "a": a,
        "e": e,
        "period": period,
        "periapsis": periapsis,
        "apoapsis": apoapsis,
        "lon_peri": lon_peri,
        "true_anomaly": nu,
        "bound": bound,
    }


def dominant_attractor(world, index):
    """Index of the body pulling hardest on ``index``, or None if it is alone.

    "Hardest" means largest G*m/r^2.  This is what every orbit readout, the trail
    sampler and the circular-orbit helper are all measured against, so it needs to agree
    with intuition: for the Moon it must come out Earth, not the Sun.
    """
    n = world.n
    if n < 2:
        return None
    pos = world.pos[:n]
    mass = world.mass[:n]

    d = pos - pos[index]
    r2 = np.einsum("ij,ij->i", d, d)
    r2[index] = np.inf
    pull = mass / r2
    pull[index] = -1.0
    best = int(np.argmax(pull))
    return best if pull[best] > 0.0 else None


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
