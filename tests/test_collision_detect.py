import numpy as np

from src import collisions, materials, units
from src.world import World

ROCK = materials.get("rock")


def _world_with(pairs):
    """pairs: list of (mass, radius, pos, vel)."""
    world = World()
    for k, (mass, radius, pos, vel) in enumerate(pairs):
        world.add("b%d" % k, ROCK, mass, radius, np.array(pos, float), np.array(vel, float))
    return world


def _stepped(world, dt):
    """Return prev_pos and advance positions linearly by dt (no gravity)."""
    prev = world.pos.copy()
    world.pos[:world.n] += world.vel[:world.n] * dt
    return prev


def test_separated_bodies_do_not_collide():
    r = units.EARTH_RADIUS
    world = _world_with([
        (units.EARTH_MASS, r, (0.0, 0.0), (0.0, 0.0)),
        (units.EARTH_MASS, r, (1.0, 0.0), (0.0, 0.0)),
    ])
    prev = _stepped(world, 1.0)
    assert collisions.find_collisions(world, prev) == []


def test_overlapping_bodies_collide():
    r = units.EARTH_RADIUS
    world = _world_with([
        (units.EARTH_MASS, r, (0.0, 0.0), (0.0, 0.0)),
        (units.EARTH_MASS, r, (r, 0.0), (0.0, 0.0)),
    ])
    prev = _stepped(world, 1.0)
    hits = collisions.find_collisions(world, prev)
    assert len(hits) == 1


def test_fast_body_passing_clean_through_is_still_caught():
    """The anti-tunnelling case: it starts far left, ends far right, and at no
    point is either endpoint inside the planet."""
    r = units.EARTH_RADIUS
    speed = 200.0 * r          # covers 200 Earth radii in one step
    world = _world_with([
        (units.EARTH_MASS, r, (0.0, 0.0), (0.0, 0.0)),
        (1e-12, r * 0.01, (-100.0 * r, 0.0), (speed, 0.0)),
    ])
    prev = _stepped(world, 1.0)
    assert np.linalg.norm(world.pos[1] - world.pos[0]) > 50 * r  # ended far away
    hits = collisions.find_collisions(world, prev)
    assert len(hits) == 1, "fast body tunnelled through the planet"
    assert hits[0].i == 0, "larger mass must be recorded first"
    assert hits[0].j == 1


def test_near_miss_is_not_a_hit():
    r = units.EARTH_RADIUS
    world = _world_with([
        (units.EARTH_MASS, r, (0.0, 0.0), (0.0, 0.0)),
        (1e-12, r * 0.01, (-100.0 * r, 3.0 * r), (200.0 * r, 0.0)),
    ])
    prev = _stepped(world, 1.0)
    assert collisions.find_collisions(world, prev) == []


def test_hits_are_sorted_earliest_first():
    r = units.EARTH_RADIUS
    world = _world_with([
        (units.EARTH_MASS, r, (0.0, 0.0), (0.0, 0.0)),
        (1e-12, r * 0.01, (-10.0 * r, 0.0), (20.0 * r, 0.0)),   # arrives late
        (1e-12, r * 0.01, (-2.0 * r, 0.0), (20.0 * r, 0.0)),    # arrives early
    ])
    prev = _stepped(world, 1.0)
    hits = collisions.find_collisions(world, prev)
    assert len(hits) >= 2
    assert hits[0].s <= hits[1].s


def test_same_event_fragments_ignore_each_other():
    r = units.EARTH_RADIUS
    world = _world_with([
        (units.EARTH_MASS, r, (0.0, 0.0), (0.0, 0.0)),
        (units.EARTH_MASS, r, (r, 0.0), (0.0, 0.0)),
    ])
    for body in world.bodies:
        body.event_id = 7
        body.no_collide_until = 5.0
    world.time = 1.0
    prev = _stepped(world, 1.0)
    assert collisions.find_collisions(world, prev) == []


def test_event_immunity_expires():
    r = units.EARTH_RADIUS
    world = _world_with([
        (units.EARTH_MASS, r, (0.0, 0.0), (0.0, 0.0)),
        (units.EARTH_MASS, r, (r, 0.0), (0.0, 0.0)),
    ])
    for body in world.bodies:
        body.event_id = 7
        body.no_collide_until = 5.0
    world.time = 9.0
    prev = _stepped(world, 1.0)
    assert len(collisions.find_collisions(world, prev)) == 1
