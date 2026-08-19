import numpy as np
import pytest

from src import collisions, materials, units
from src.body import KIND_STAR
from src.world import World

ROCK = materials.get("rock")


def _pair(m1, r1, v1, m2, r2, v2, gap):
    world = World()
    world.add("big", ROCK, m1, r1, np.zeros(2), np.array(v1, float))
    world.add("small", ROCK, m2, r2, np.array([gap, 0.0]), np.array(v2, float))
    return world


def test_merge_conserves_mass_and_momentum_exactly():
    r = units.EARTH_RADIUS
    world = _pair(units.EARTH_MASS, r, (0.1, 0.0),
                  units.MOON_MASS, r * 0.27, (-0.05, 0.02), r)
    m0 = world.masses.sum()
    p0 = world.total_momentum().copy()

    c = collisions.Collision(0, 1, 0.5, 0.15)
    collisions.merge(world, c)

    assert world.n == 1
    assert world.masses.sum() == pytest.approx(m0, rel=1e-15)
    assert np.allclose(world.total_momentum(), p0, rtol=1e-14, atol=1e-24)


def test_merged_radius_follows_from_mass_and_mean_density():
    r = units.EARTH_RADIUS
    world = _pair(units.EARTH_MASS, r, (0.0, 0.0),
                  units.EARTH_MASS, r, (0.0, 0.0), r)
    d0 = units.density_from_mass_radius(units.EARTH_MASS, r)

    collisions.merge(world, collisions.Collision(0, 1, 0.5, 0.0))

    total = 2 * units.EARTH_MASS
    expected = units.radius_from_mass_density(total, d0)
    assert world.radius[0] == pytest.approx(expected, rel=1e-12)
    assert world.radius[0] > r, "merging two Earths must make something bigger"


def test_merge_keeps_the_larger_body_identity():
    r = units.EARTH_RADIUS
    world = _pair(units.EARTH_MASS, r, (0.0, 0.0),
                  units.MOON_MASS, r * 0.27, (0.0, 0.0), r)
    survivor_id = world.bodies[0].id
    idx = collisions.merge(world, collisions.Collision(0, 1, 0.5, 0.0))
    assert world.bodies[idx].id == survivor_id
    assert world.bodies[idx].name == "big"


def test_merge_returns_a_valid_index_after_the_swap_remove():
    """remove() swap-removes, so the survivor's row can move.  The returned
    index must be the one you can actually use."""
    world = World()
    for k in range(4):
        world.add("b%d" % k, ROCK, units.EARTH_MASS * (k + 1), units.EARTH_RADIUS,
                  np.array([k * 5.0, 0.0]), np.zeros(2))
    survivor_id = world.bodies[1].id
    idx = collisions.merge(world, collisions.Collision(1, 0, 0.5, 0.0))
    assert world.bodies[idx].id == survivor_id
    assert world.index_of(survivor_id) == idx


def test_crossing_the_ignition_threshold_makes_a_star():
    half = units.STAR_MASS_THRESHOLD * 0.6
    r = units.SUN_RADIUS * 0.3
    world = _pair(half, r, (0.0, 0.0), half, r, (0.0, 0.0), r)
    idx = collisions.merge(world, collisions.Collision(0, 1, 0.5, 0.0))
    assert world.mass[idx] > units.STAR_MASS_THRESHOLD
    assert world.bodies[idx].kind == KIND_STAR


def test_gentle_touch_is_not_catastrophic():
    r = units.EARTH_RADIUS
    world = _pair(units.EARTH_MASS, r, (0.0, 0.0),
                  units.MOON_MASS, r * 0.27, (1e-5, 0.0), r)
    c = collisions.Collision(0, 1, 0.5, 1e-5)
    assert collisions.is_catastrophic(world, c) is False


def test_hypervelocity_impact_is_catastrophic():
    r = units.EARTH_RADIUS
    # 500 km/s: far above Earth's 11.2 km/s escape velocity.
    fast = 500.0 / units.AUD_TO_KMS
    world = _pair(units.EARTH_MASS, r, (0.0, 0.0),
                  units.EARTH_MASS * 0.1, r * 0.5, (fast, 0.0), r)
    c = collisions.Collision(0, 1, 0.5, fast, v_impact=fast)
    assert collisions.is_catastrophic(world, c) is True
