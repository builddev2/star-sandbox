"""Steering a craft: a burn, planned before it happens.

A **maneuver node** is a point on the craft's own predicted orbit plus the
impulse to apply there.  That is the whole idea, and it is what turns a
gravity simulator into a spaceflight simulator: you pick a point, say how
hard to burn, and see the new orbit *before* committing to it.

``due_time`` is absolute world days rather than a countdown, so it does not
have to be rewritten every frame as the clock advances; the countdown the
panel shows is ``due_time - world.time``, computed where it is displayed.

Everything here is two-body algebra about whatever ``orbits.dominant_attractor``
says the craft is orbiting.  That is exact for the ordinary case -- a craft
and a planet both going round the same star -- and a good estimate otherwise,
which is the same trade ``LaunchTool``'s aiming preview already makes.

Imports numpy and nothing from pygame -- ``tests/test_layering.py`` asserts it.
"""

import math
from dataclasses import dataclass

import numpy as np

from . import orbits, units


@dataclass(frozen=True)
class Maneuver:
    """One impulsive burn.  Speeds are AU/day; the panel scrubs them in m/s.

    Frozen for the same reason ``Body.locked_fields`` is a frozenset:
    rebinding is then the only way to change a node, so two craft can never
    end up sharing one mutable object.
    """

    due_time: float
    #: Along the velocity vector.  Negative is retrograde.
    prograde: float = 0.0
    #: Along the radius vector, away from the attractor.  Negative is inward.
    radial: float = 0.0

    @property
    def magnitude(self):
        """Total delta-v this burn spends, AU/day."""
        return math.hypot(self.prograde, self.radial)


def craft_state(world, index):
    """``(attractor, mu, rel_pos, rel_vel)`` for a body, or None if it has no
    attractor -- which happens only when it is the most massive thing present.
    """
    attractor = orbits.dominant_attractor(world, index)
    if attractor is None:
        return None
    mu = units.G * (float(world.mass[attractor]) + float(world.mass[index]))
    return (attractor, mu,
            np.array(world.pos[index] - world.pos[attractor], dtype=float),
            np.array(world.vel[index] - world.vel[attractor], dtype=float))


def node_state(world, index, node):
    """``(mu, rel_pos, rel_vel)`` at the node, *before* the burn.

    Propagated on the craft's own two-body orbit, exactly -- ``orbits.propagate``
    converts to mean anomaly, advances it linearly and converts back, which is
    correct for a bound orbit and a hyperbolic one alike.  A node already due
    is evaluated where the craft is now.
    """
    state = craft_state(world, index)
    if state is None:
        return None
    _attractor, mu, rel_pos, rel_vel = state
    dt = float(node.due_time) - float(world.time)
    if dt <= 0.0:
        return mu, rel_pos, rel_vel
    pos, vel = orbits.propagate(mu, rel_pos, rel_vel, dt)
    return mu, np.asarray(pos, dtype=float), np.asarray(vel, dtype=float)


def burn_velocity(rel_pos, rel_vel, node):
    """The relative velocity after ``node``'s impulse is applied.

    Prograde is along the velocity, radial along the position -- the two
    directions a flat orbit offers.  A zero-length vector contributes nothing
    rather than a divide by zero.
    """
    pos = np.asarray(rel_pos, dtype=float)
    vel = np.asarray(rel_vel, dtype=float)
    out = vel.copy()

    speed = float(np.hypot(vel[0], vel[1]))
    if speed > 0.0 and node.prograde != 0.0:
        out = out + (vel / speed) * node.prograde

    distance = float(np.hypot(pos[0], pos[1]))
    if distance > 0.0 and node.radial != 0.0:
        out = out + (pos / distance) * node.radial

    return out


def plan_conics(world, index, plan, samples=160):
    """``(attractor, [(points, bound), ...])`` -- the orbit after each node.

    The plan is sorted by ``due_time`` before it is walked, so a caller may
    hand one over in any order.  Each conic is relative to the attractor, so a
    caller adds the attractor's position before projecting.  Node *k*'s conic is propagated from node
    *k-1*'s post-burn state, which is what makes a two-burn transfer draw as
    the two arcs it really is.
    """
    state = craft_state(world, index)
    if state is None:
        return None, []
    attractor, mu, rel_pos, rel_vel = state

    conics = []
    clock = float(world.time)
    # Sorted here rather than assumed: a plan built out of order would
    # otherwise apply a later-listed, earlier-due burn at the *previous*
    # node's position and return a conic that is silently wrong.
    for node in sorted(plan, key=lambda n: float(n.due_time)):
        dt = float(node.due_time) - clock
        if dt > 0.0:
            rel_pos, rel_vel = orbits.propagate(mu, rel_pos, rel_vel, dt)
            rel_pos = np.asarray(rel_pos, dtype=float)
            rel_vel = np.asarray(rel_vel, dtype=float)
            clock = float(node.due_time)
        rel_vel = burn_velocity(rel_pos, rel_vel, node)
        conics.append(orbits.conic_points(mu, rel_pos, rel_vel, samples=samples))
    return attractor, conics


def closest_approach(mu, craft_pos, craft_vel, target_pos, target_vel,
                     horizon_days, samples=240):
    """How near the craft and the target ever get, within ``horizon_days``.

    Both are propagated on their own two-body orbits about the same attractor
    and sampled at the **same instants**.  Time is the only parameterisation
    that lets two different orbits be compared: sampling each by true anomaly
    would compare a point on one orbit against a point on the other that the
    bodies never occupy together.

    A coarse sweep finds the best sample, then one refinement pass looks
    inside the neighbouring interval, which is enough because separation is
    smooth in time and the coarse grid already brackets the minimum.
    """
    craft_pos = np.asarray(craft_pos, dtype=float)
    craft_vel = np.asarray(craft_vel, dtype=float)
    target_pos = np.asarray(target_pos, dtype=float)
    target_vel = np.asarray(target_vel, dtype=float)

    def separation(dt):
        c_pos, c_vel = orbits.propagate(mu, craft_pos, craft_vel, dt)
        t_pos, t_vel = orbits.propagate(mu, target_pos, target_vel, dt)
        gap = float(np.hypot(c_pos[0] - t_pos[0], c_pos[1] - t_pos[1]))
        return gap, c_pos, c_vel, t_pos, t_vel

    samples = max(8, int(samples))
    horizon = max(0.0, float(horizon_days))
    times = np.linspace(0.0, horizon, samples)
    best_t = 0.0
    best = None
    for t in times:
        found = separation(float(t))
        if best is None or found[0] < best[0]:
            best, best_t = found, float(t)

    step = horizon / (samples - 1) if samples > 1 else 0.0
    if step > 0.0:
        fine = np.linspace(max(0.0, best_t - step), min(horizon, best_t + step), 32)
        for t in fine:
            found = separation(float(t))
            if found[0] < best[0]:
                best, best_t = found, float(t)

    gap, c_pos, c_vel, t_pos, t_vel = best
    return {"time": best_t, "distance": gap,
            "craft_pos": np.asarray(c_pos, dtype=float),
            "craft_vel": np.asarray(c_vel, dtype=float),
            "target_pos": np.asarray(t_pos, dtype=float),
            "target_vel": np.asarray(t_vel, dtype=float)}


def hohmann(mu, r1, r2, theta_craft, theta_target):
    """The two burns of a Hohmann transfer, **and when to start it**.

    The two delta-v figures are the easy half and every textbook has them.
    The half that actually matters when you try to fly one is ``wait_days``:
    a transfer only arrives somewhere useful if the target is where the craft
    is going when it gets there, so the departure has to wait for the phase
    angle to come round.  That is a launch window, and it is why missions to
    Mars leave every twenty-six months rather than whenever they are ready.

    Angles are radians measured about the attractor, in the direction of
    motion.  Returns speeds in AU/day, signed: negative is a retrograde burn,
    which is what going *inward* costs -- or **None** when either orbit is
    degenerate, which is the one case with no transfer to describe.
    """
    r1, r2, mu = float(r1), float(r2), float(mu)
    if r1 <= 0.0 or r2 <= 0.0 or mu <= 0.0:
        # The panel feeds this radii taken from world state, and a craft
        # sitting on its attractor's centre is degenerate but reachable.
        # None says "there is no transfer here", which the caller can show;
        # a ZeroDivisionError says nothing and takes the frame with it.
        return None

    a_transfer = 0.5 * (r1 + r2)

    v1 = math.sqrt(mu / r1)
    v2 = math.sqrt(mu / r2)
    depart = v1 * (math.sqrt(2.0 * r2 / (r1 + r2)) - 1.0)
    arrive = v2 * (1.0 - math.sqrt(2.0 * r1 / (r1 + r2)))
    transfer = math.pi * math.sqrt(a_transfer ** 3 / mu)

    # The target must lead the craft by this much at departure, so that half
    # an orbit of the transfer ellipse later the two arrive together.
    n1 = math.sqrt(mu / r1 ** 3)
    n2 = math.sqrt(mu / r2 ** 3)
    required = math.pi - n2 * transfer
    current = float(theta_target) - float(theta_craft)

    rate = n2 - n1
    if abs(rate) < 1e-15:
        wait = 0.0
    else:
        delta = (required - current) % (2.0 * math.pi)
        if rate < 0.0:
            # The target is falling behind, so the gap has to close the other
            # way round; shifting into (-2pi, 0] keeps the wait positive.
            delta -= 2.0 * math.pi
        wait = delta / rate

    return {"depart_dv": depart, "arrive_dv": arrive,
            "transfer_days": transfer, "wait_days": max(0.0, wait),
            "a_transfer": a_transfer}


def flyby(mu_planet, v_inf, periapsis, side=1.0):
    """How much a planet bends a passing craft, and where it leaves.

    A flyby is a hyperbola about the planet.  The speed the *planet* sees --
    ``v_inf`` -- is exactly the same going out as coming in; all the planet
    does is rotate it, by

        e = 1 + r_p |v_inf|^2 / mu,     delta = 2 asin(1/e)

    and that rotation is the whole gravity assist.  The craft gains nothing
    relative to the planet and everything relative to the star, because the
    rotated ``v_inf`` adds to the planet's own orbital motion differently
    than the original did.  That is why the drawing is a triangle: nothing
    about it is a fudge.

    ``side`` exists because the sense of the rotation is not a free choice --
    it is fixed by which side of the planet the craft passes, and that in
    turn is exactly what decides whether an assist boosts or brakes the
    craft relative to the star.  The sign equals the sign of the flyby's
    angular momentum, ``r_rel x v_inf``: a craft passing on the +y side while
    moving in +x is pulled toward -y, which is a clockwise (negative)
    rotation, so a negative ``side`` matches a negative angular momentum. A
    non-negative ``side`` rotates counter-clockwise, matching the sign
    convention ``math.copysign`` already uses for a positive-signed zero.
    """
    v = np.asarray(v_inf, dtype=float)
    speed2 = float(v[0] * v[0] + v[1] * v[1])
    if speed2 <= 0.0 or periapsis <= 0.0 or mu_planet <= 0.0:
        # e = 1, not infinity: with no speed relative to the planet the
        # formula's own limit is the parabolic case, and an inspector reading
        # this back should say "1" rather than print an infinity.
        return {"e": 1.0, "turn": 0.0, "v_out": v.copy()}

    e = 1.0 + float(periapsis) * speed2 / float(mu_planet)
    turn = 2.0 * math.asin(min(1.0, 1.0 / e))
    # Not copysign: it reads the sign *bit*, so a -0.0 -- which a cross
    # product of two near-parallel vectors really does produce -- would flip
    # the assist over. Only a genuinely negative side rotates clockwise.
    signed_turn = turn if float(side) >= 0.0 else -turn

    cos_t, sin_t = math.cos(signed_turn), math.sin(signed_turn)
    return {"e": e, "turn": turn,
            "v_out": np.array([cos_t * v[0] - sin_t * v[1],
                               sin_t * v[0] + cos_t * v[1]])}


#: One notch of the delta-v steppers, m/s.  Linear rather than a ratio: a
#: burn runs from tens of m/s to a few thousand, which is two decades, not
#: the eight that mass covers.
DV_NOTCH_MS = 10.0
#: One notch of the node-time stepper, as a ratio.  The countdown has to span
#: minutes to years, so it scrubs the way mass does.
TIME_NOTCH_RATIO = 1.06
#: The shortest a node can be set to.
MIN_LEAD_DAYS = 0.01


def scrub_dv(speed, notches):
    """Move a delta-v component by ``notches`` of DV_NOTCH_MS."""
    return speed + units.from_ms(DV_NOTCH_MS * notches)


def scrub_lead(days, notches):
    """Move a countdown by ``notches``, multiplicatively.  Never past zero."""
    days = max(float(days), MIN_LEAD_DAYS)
    return max(MIN_LEAD_DAYS, days * TIME_NOTCH_RATIO ** notches)


# --- firing a planned burn ----------------------------------------------------------


def apply_burn(world, index, node):
    """Fire ``node`` on the body at ``index`` now.  True if anything happened.

    A burn that costs more than the tank holds is **scaled down to what is
    left** rather than refused.  Running out mid-burn is what really happens
    to a real mission, and it teaches more than a control that silently does
    nothing.

    The impulse is added in the **world** frame.  The attractor's own velocity
    appears in both the before and after relative velocities, so it cancels,
    and a craft orbiting a moving planet gets the right answer for free.
    """
    state = craft_state(world, index)
    if state is None:
        return False
    _attractor, _mu, rel_pos, rel_vel = state

    body = world.bodies[index]
    wanted = node.magnitude
    if wanted <= 0.0:
        return False

    budget = float(body.fuel_dv)
    if budget <= 0.0:
        return False
    fraction = 1.0 if wanted <= budget else budget / wanted
    spent = Maneuver(node.due_time, node.prograde * fraction,
                     node.radial * fraction)

    world.vel[index] += burn_velocity(rel_pos, rel_vel, spent) - rel_vel
    body.fuel_dv = max(0.0, budget - spent.magnitude)
    return True


def next_burn_time(world):
    """The soonest ``due_time`` anywhere in the world, or infinity.

    ``Simulation`` caches this so its substep loop is one float comparison
    rather than a scan over every body: at ten years a second the integrator
    takes thousands of substeps per frame, and a Python loop over 200 bodies
    inside that loop would eat the wall-clock budget many times over -- the
    same reasoning that keeps ``thermal.advance`` at once a frame.
    """
    soonest = math.inf
    for i in range(world.n):
        plan = world.bodies[i].plan
        if plan:
            soonest = min(soonest, float(plan[0].due_time))
    return soonest


def execute_due(world, now=None):
    """Fire every node whose time has come.  Returns how many were popped.

    A node is popped whether or not the burn could be applied, so a craft that
    has lost its attractor -- or run dry -- cannot jam the queue forever and
    strand every later burn behind it.
    """
    when = float(world.time) if now is None else float(now)
    fired = 0
    for i in range(world.n):
        body = world.bodies[i]
        while body.plan and float(body.plan[0].due_time) <= when:
            node = body.plan[0]
            body.plan = body.plan[1:]
            apply_burn(world, i, node)
            fired += 1
    return fired


def refine_window(mu, craft_pos, craft_vel, target_pos, target_vel,
                  wait_days, transfer_days, iterations=12):
    """Sharpen a Hohmann departure time against exact propagation.

    :func:`hohmann` works out the launch window from circular mean motions,
    which is the textbook derivation and is what makes the answer explicable.
    It is also only a first guess the moment either orbit is eccentric: Mars
    swings 1.38 to 1.67 AU and its true anomaly runs fast at perihelion and
    slow at aphelion, so the angle it has actually swept by arrival differs
    from ``n * t`` by up to twice its eccentricity in radians.

    So the guess is refined here the same way ``solve_intercept`` leads a
    moving target: propagate both bodies exactly to the arrival instant, read
    off how far ahead or behind the target is, and slide the departure by that
    error over the relative angular rate.  A handful of passes converges,
    because the residual shrinks by roughly the ratio of the rates each time.
    """
    craft_pos = np.asarray(craft_pos, dtype=float)
    craft_vel = np.asarray(craft_vel, dtype=float)
    target_pos = np.asarray(target_pos, dtype=float)
    target_vel = np.asarray(target_vel, dtype=float)

    r1 = float(np.hypot(craft_pos[0], craft_pos[1]))
    r2 = float(np.hypot(target_pos[0], target_pos[1]))
    if r1 <= 0.0 or r2 <= 0.0 or mu <= 0.0:
        return max(0.0, float(wait_days))

    # Relative angular rate, from the mean motions the guess itself used.
    el_c = orbits.elements_from_state(mu, craft_pos, craft_vel)
    el_t = orbits.elements_from_state(mu, target_pos, target_vel)
    a_c = el_c["a"] if el_c["bound"] and el_c["a"] > 0.0 else r1
    a_t = el_t["a"] if el_t["bound"] and el_t["a"] > 0.0 else r2
    rate = math.sqrt(mu / a_t ** 3) - math.sqrt(mu / a_c ** 3)
    if abs(rate) < 1e-15:
        return max(0.0, float(wait_days))

    wait = max(0.0, float(wait_days))
    period = 2.0 * math.pi / abs(rate)          # the synodic period
    for _ in range(iterations):
        departure, _ = orbits.propagate(mu, craft_pos, craft_vel, wait)
        # Half a turn of the transfer ellipse from where the craft departs.
        arrival_angle = math.atan2(departure[1], departure[0]) + math.pi
        there, _ = orbits.propagate(mu, target_pos, target_vel,
                                    wait + transfer_days)
        error = (arrival_angle - math.atan2(there[1], there[0])
                 + math.pi) % (2.0 * math.pi) - math.pi
        if abs(error) < 1e-12:
            break
        wait += error / rate
        # A correction may push the window past a whole synodic period, or
        # before now; both name the same geometry, so fold it back.
        wait = wait % period if period > 0.0 else max(0.0, wait)
    return max(0.0, wait)
