import numpy as np
import pytest

from src import collisions, edits, materials, units
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
    c = collisions.Collision(0, 1, 0.5, 1e-5, v_impact=1e-5)
    assert collisions.classify(world, c) != collisions.DISRUPT


def test_hypervelocity_impact_is_catastrophic():
    r = units.EARTH_RADIUS
    # 500 km/s: far above Earth's 11.2 km/s escape velocity.
    fast = 500.0 / units.AUD_TO_KMS
    world = _pair(units.EARTH_MASS, r, (0.0, 0.0),
                  units.EARTH_MASS * 0.1, r * 0.5, (fast, 0.0), r)
    c = collisions.Collision(0, 1, 0.5, fast, v_impact=fast)
    assert collisions.classify(world, c) == collisions.DISRUPT


# --- cratering ejecta on accretion --------------------------------------------------

def _accrete_impact(speed_kms, gap_factor=1.0, ratio=0.1):
    r = units.EARTH_RADIUS
    speed = speed_kms / units.AUD_TO_KMS
    world = _pair(units.EARTH_MASS, r, (0.0, 0.0),
                  units.EARTH_MASS * ratio, r * 0.5, (speed, 0.0), gap_factor * r)
    c = collisions.Collision(0, 1, 0.5, speed, v_impact=speed)
    return world, c


def test_a_feeble_impact_produces_no_ejecta_bodies():
    world, c = _accrete_impact(1e-5)
    resolver = collisions.CollisionResolver(seed=1)
    idx = resolver.accrete(world, c)
    assert world.n == 1
    assert idx == 0


def test_an_energetic_but_sub_catastrophic_impact_throws_real_debris():
    world, c = _accrete_impact(30.0)
    assert collisions.classify(world, c) != collisions.DISRUPT, \
        "test setup must stay sub-catastrophic"
    before = world.n
    resolver = collisions.CollisionResolver(seed=1)
    resolver.accrete(world, c)
    assert world.n > before, "an energetic sub-catastrophic hit must excavate ejecta"


def test_accretion_conserves_mass_and_momentum_with_ejecta():
    world, c = _accrete_impact(30.0)
    m0 = float(world.masses.sum())
    p0 = world.total_momentum().copy()
    resolver = collisions.CollisionResolver(seed=1)
    resolver.accrete(world, c)
    assert float(world.masses.sum()) == pytest.approx(m0, rel=1e-12)
    assert np.allclose(world.total_momentum(), p0, rtol=1e-9,
                       atol=1e-12 * abs(p0).max())


def test_ejecta_is_suppressed_when_the_world_is_nearly_full():
    world, c = _accrete_impact(30.0)
    while world.n < world.capacity - 1:
        world.add("filler", ROCK, 1e-14, units.EARTH_RADIUS * 0.01,
                  np.array([100.0 + world.n, 0.0]), np.zeros(2))
    before = world.n
    resolver = collisions.CollisionResolver(seed=1)
    resolver.accrete(world, c)
    assert world.n == before - 1, "should have merged with no room to spare for ejecta"


def test_the_survivor_keeps_its_own_locks_and_the_merge_ignores_them():
    """A merge is not an edit.

    It writes mass, radius and temperature into the arrays directly rather than
    through edits.set_*, so a locked mass does not stop a body growing by
    eating another -- and the survivor is the same Body object throughout, so
    the locks it was wearing must come out the far side untouched.  Every
    other lock test covers a *fresh* body starting unlocked; this is the one
    body that is neither fresh nor destroyed.
    """
    r = units.EARTH_RADIUS
    world = _pair(units.EARTH_MASS, r, (0.0, 0.0),
                  units.MOON_MASS, r * 0.27, (0.0, 0.0), r)
    world.temperature[0] = 500.0
    world.temperature[1] = 100.0
    survivor = world.bodies[0]
    edits.toggle_lock(survivor, "mass")
    edits.toggle_lock(survivor, "temp")
    mass, temperature = float(world.mass[0]), float(world.temperature[0])

    idx = collisions.merge(world, collisions.Collision(0, 1, 0.5, 0.0))

    assert world.bodies[idx] is survivor
    assert survivor.locked_fields == frozenset({"mass", "temp"})
    assert float(world.mass[idx]) > mass
    assert float(world.temperature[idx]) != pytest.approx(temperature)


def test_a_merge_does_not_hand_the_victims_locks_to_the_survivor():
    """Nothing inherits a lock -- not a fragment, and not the body that ate
    its owner."""
    r = units.EARTH_RADIUS
    world = _pair(units.EARTH_MASS, r, (0.0, 0.0),
                  units.MOON_MASS, r * 0.27, (0.0, 0.0), r)
    edits.toggle_lock(world.bodies[1], "material")

    idx = collisions.merge(world, collisions.Collision(0, 1, 0.5, 0.0))

    assert world.bodies[idx].locked_fields == frozenset()
