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


def _hyperbolic_mean_from_true(e, nu):
    """Hyperbolic mean anomaly for a true anomaly on an e > 1 conic."""
    big_h = 2.0 * math.atanh(math.sqrt((e - 1.0) / (e + 1.0)) * math.tan(nu * 0.5))
    return e * math.sinh(big_h) - big_h


def _true_from_hyperbolic_mean(mean_h, e, tol=1e-12, max_iter=60):
    """Invert Kepler's hyperbolic equation M = e sinh H - H by Newton's method."""
    big_h = (math.copysign(math.log(2.0 * abs(mean_h) / e + 1.8), mean_h)
            if mean_h != 0.0 else 0.0)
    for _ in range(max_iter):
        f = e * math.sinh(big_h) - big_h - mean_h
        fp = e * math.cosh(big_h) - 1.0
        step = f / fp
        big_h -= step
        if abs(step) < tol:
            break
    return 2.0 * math.atan(math.sqrt((e + 1.0) / (e - 1.0)) * math.tanh(big_h * 0.5))


def propagate(mu, rel_pos, rel_vel, dt):
    """Where a body on this state vector will be after ``dt`` days.

    Exact for a bound orbit or a hyperbolic one: convert to mean (or
    hyperbolic mean) anomaly, advance it linearly -- the one angle that moves
    at a constant rate on either conic -- and convert back. A throw fast
    enough to be on an unbound path is exactly the kind of throw whose target
    most needs leading, since it covers the distance in a hurry; only the
    knife-edge parabolic case (e == 1 to within roundoff) is left unmoved,
    since Barker's equation for that one case is not worth carrying.
    """
    el = elements_from_state(mu, rel_pos, rel_vel)
    a, e = el["a"], el["e"]
    nu0 = el["true_anomaly"]
    if el["bound"] and math.isfinite(a) and a > 0.0:
        big0 = 2.0 * math.atan2(math.sqrt(1.0 - e) * math.sin(nu0 * 0.5),
                                math.sqrt(1.0 + e) * math.cos(nu0 * 0.5))
        mean0 = big0 - e * math.sin(big0)
        mean1 = mean0 + math.sqrt(mu / a ** 3) * dt
        nu1 = true_anomaly_from_mean(mean1, e)
    elif math.isfinite(a) and a < 0.0 and e > 1.0 + 1e-9:
        mean0 = _hyperbolic_mean_from_true(e, nu0)
        mean1 = mean0 + math.sqrt(mu / (-a) ** 3) * dt
        nu1 = _true_from_hyperbolic_mean(mean1, e)
    else:
        return np.array(rel_pos, dtype=float), np.array(rel_vel, dtype=float)
    return state_from_elements(mu, a, e, el["lon_peri"], nu1)


def _orbit_flight_time(mu, a, e, nu0, nu1, prograde):
    """Days for true anomaly to sweep from ``nu0`` to ``nu1`` in the direction
    of actual motion.  Bound orbits take the shortest (zero extra revolution)
    path; a hyperbolic one has no revolutions to begin with.  Returns None for
    the parabolic knife edge, or if ``nu1`` would need to lie behind the body
    on a path with no way back round to it.
    """
    sign = 1.0 if prograde else -1.0
    n0, n1 = sign * nu0, sign * nu1

    if a > 0.0 and e < 1.0:
        def mean_of(nu):
            big = 2.0 * math.atan2(math.sqrt(1.0 - e) * math.sin(nu * 0.5),
                                   math.sqrt(1.0 + e) * math.cos(nu * 0.5))
            return big - e * math.sin(big)
        dm = (mean_of(n1) - mean_of(n0)) % (2.0 * math.pi)
        return dm / math.sqrt(mu / a ** 3)

    if a < 0.0 and e > 1.0 + 1e-9:
        dm = _hyperbolic_mean_from_true(e, n1) - _hyperbolic_mean_from_true(e, n0)
        if dm <= 0.0:
            return None
        return dm / math.sqrt(mu / (-a) ** 3)

    return None


def solve_intercept_angle(mu, rel_pos, speed, target_point, samples=360):
    """The launch direction from ``rel_pos`` at ``speed`` whose two-body conic
    passes exactly through ``target_point``, if one exists at this speed.

    Not a root-find on a smooth function: the semi-major axis a fixed launch
    speed implies is constant (vis-viva), so as the launch angle sweeps a full
    turn, the resulting orbit's radius back along the target's own direction
    sweeps too -- and generically crosses the target's distance either twice,
    once, or not at all, the classic "two firing solutions" of a mortar
    problem. A coarse sweep finds every bracket; each is refined by bisection.
    A near-radial solution can sit inside one coarse interval too narrow for
    the sweep to see the sign change at all -- eccentricity spikes sharply as
    a trajectory approaches purely radial -- so a second pass zooms into the
    sharpest dips of the coarse sweep and looks again there before giving up.
    Of every root found, the one whose flight is shortest wins, since that is
    the one a player expects "aim and it goes" to mean.

    Returns a dict: ``hit`` (bool), ``angle`` (radians), ``velocity``
    (length-2 array), ``miss`` (AU, 0.0 when ``hit``), and ``flight_time``
    (days, or None when the orbit is unbound or nothing hit).
    """
    r1 = np.asarray(rel_pos, dtype=float)
    target = np.asarray(target_point, dtype=float)
    target_mag = float(np.hypot(target[0], target[1]))
    target_angle = math.atan2(target[1], target[0])

    def miss_at(theta):
        """Signed (orbit radius - target radius) along the target's direction."""
        v = speed * np.array([math.cos(theta), math.sin(theta)])
        el = elements_from_state(mu, r1, v)
        nu_t = target_angle - el["lon_peri"]
        denom = 1.0 + el["e"] * math.cos(nu_t)
        if denom <= 1e-9 or not math.isfinite(el["p"]):
            return None
        return el["p"] / denom - target_mag

    def bisect(lo, hi, flo):
        for _ in range(40):
            mid = 0.5 * (lo + hi)
            fm = miss_at(mid)
            if fm is None:
                return mid
            if (fm > 0) == (flo > 0):
                lo, flo = mid, fm
            else:
                hi = mid
        return 0.5 * (lo + hi)

    def brackets(thetas, values, step, wrap):
        """Sign changes between neighbouring samples, each refined to a root.

        Every sample is spaced by exactly ``step``, so the upper edge of a
        bracket is always ``lo + step`` -- never the next array value modulo
        the array length, which for the last sample of a periodic sweep would
        wrap back past zero and hand bisection a negative-width interval.
        """
        found = []
        span = len(thetas) if wrap else len(thetas) - 1
        for k in range(span):
            a, b = values[k], values[(k + 1) % len(thetas)]
            if a is None or b is None or (a > 0) == (b > 0):
                continue
            found.append(bisect(float(thetas[k]), float(thetas[k]) + step, a))
        return found

    thetas = np.linspace(0.0, 2.0 * math.pi, samples, endpoint=False)
    values = [miss_at(float(t)) for t in thetas]
    coarse_step = 2.0 * math.pi / samples
    roots = brackets(thetas, values, coarse_step, wrap=True)

    if not roots:
        dips = [(abs(values[k]), k) for k in range(samples)
               if values[k] is not None
               and values[k - 1] is not None and values[(k + 1) % samples] is not None
               and abs(values[k]) < abs(values[k - 1])
               and abs(values[k]) < abs(values[(k + 1) % samples])]
        dips.sort()
        for _, k in dips[:4]:
            fine = np.linspace(thetas[k] - coarse_step, thetas[k] + coarse_step, 32)
            fine_step = float(fine[1] - fine[0])
            fvalues = [miss_at(float(t)) for t in fine]
            roots.extend(brackets(fine, fvalues, fine_step, wrap=False))

    def result_for(theta):
        v = speed * np.array([math.cos(theta), math.sin(theta)])
        h = r1[0] * v[1] - r1[1] * v[0]
        el = elements_from_state(mu, r1, v)
        flight = None
        if h != 0.0 and math.isfinite(el["a"]):
            nu_t = target_angle - el["lon_peri"]
            flight = _orbit_flight_time(mu, el["a"], el["e"], el["true_anomaly"],
                                        nu_t, h > 0.0)
        return {"hit": True, "angle": theta, "velocity": v, "miss": 0.0,
                "flight_time": flight}

    if roots:
        return min((result_for(t) for t in roots),
                   key=lambda r: r["flight_time"] if r["flight_time"] is not None
                   else math.inf)

    best_theta = 0.0
    best_miss = math.inf
    for t, m in zip(thetas, values):
        if m is not None and abs(m) < best_miss:
            best_miss, best_theta = abs(m), float(t)
    v = speed * np.array([math.cos(best_theta), math.sin(best_theta)])
    return {"hit": False, "angle": best_theta, "velocity": v,
           "miss": best_miss, "flight_time": None}


def solve_intercept(mu, rel_pos, speed, target_rel_pos, target_rel_vel,
                    iterations=6, samples=360):
    """Like :func:`solve_intercept_angle`, but leads a moving target.

    Both bodies' state is relative to the same attractor -- the one the
    projectile is launched from -- which is exact when target and launch
    point share a dominant attractor (the ordinary case: two planets round
    the Sun, or a throw near a moon aimed at another moon of the same planet)
    and a reasonable estimate otherwise. Each pass reads off the flight time
    to the current guess, propagates the target forward by it under its own
    two-body motion, and re-solves; the residual shrinks by roughly 85% a
    pass, so six of them land within a hundredth of an Earth radius even when
    the flight crosses a good fraction of the target's own orbit.
    """
    point = np.array(target_rel_pos, dtype=float)
    result = solve_intercept_angle(mu, rel_pos, speed, point, samples=samples)
    for _ in range(iterations - 1):
        if not result["hit"] or result["flight_time"] is None:
            break
        point, _ = propagate(mu, target_rel_pos, target_rel_vel, result["flight_time"])
        result = solve_intercept_angle(mu, rel_pos, speed, point, samples=samples)
    result["target_point"] = point
    return result


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
    # A body sitting exactly on another divides by zero below.  Infinity is
    # the honest answer -- a separation of nothing is not a sphere of
    # influence anyone is inside -- and it keeps the warning out of the log.
    r[r <= 0.0] = np.inf

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


def circular_velocity(world, attractor, rel, sense=None):
    """Velocity of a circular orbit ``rel`` away from ``attractor``, world frame.

    Takes an offset rather than a body, because both callers ask about a point
    no body occupies yet -- where a comet is about to be released, and where a
    shuttle is about to appear.  The world frame includes the attractor's own
    motion, so an orbit about a moving planet comes out right.

    ``sense`` is any velocity the orbit should share a direction with -- the
    drag that aimed the throw, or the parent's own motion -- so a new orbit
    runs the same way round as what it was launched from.  Without one the
    orbit is anticlockwise.
    """
    r = float(np.hypot(rel[0], rel[1]))
    if r <= 0.0:
        return world.vel[attractor].copy()

    # Rotate the radius vector a quarter turn to get the tangential direction.
    tangent = np.array([-rel[1], rel[0]]) / r
    if sense is not None and float(np.dot(tangent, sense)) < 0.0:
        tangent = -tangent
    speed = math.sqrt(G * float(world.mass[attractor]) / r)
    return world.vel[attractor] + speed * tangent
