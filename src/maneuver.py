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

    Each conic is relative to the attractor, so a caller adds the attractor's
    position before projecting.  Node *k*'s conic is propagated from node
    *k-1*'s post-burn state, which is what makes a two-burn transfer draw as
    the two arcs it really is.
    """
    state = craft_state(world, index)
    if state is None:
        return None, []
    attractor, mu, rel_pos, rel_vel = state

    conics = []
    clock = float(world.time)
    for node in plan:
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
    which is what going *inward* costs.
    """
    r1, r2 = float(r1), float(r2)
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
