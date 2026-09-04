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
