import numpy as np
import pytest

from src import collisions, materials, units
from src.body import KIND_DEBRIS
from src.world import World

ROCK = materials.get("rock")


def _impact(target_mass=units.EARTH_MASS, ratio=0.1, speed_kms=500.0):
    r = units.EARTH_RADIUS
    world = World()
    world.add("Target", ROCK, target_mass, r, np.zeros(2), np.zeros(2))
    world.add("Bullet", ROCK, target_mass * ratio, r * 0.5,
              np.array([r, 0.0]), np.array([-speed_kms / units.AUD_TO_KMS, 0.0]))
    c = collisions.Collision(0, 1, 0.5, speed_kms / units.AUD_TO_KMS)
    return world, c


def test_shatter_conserves_mass_exactly():
    world, c = _impact()
    m0 = world.masses.sum()
    resolver = collisions.CollisionResolver(seed=1)
    resolver.shatter(world, c)
    assert world.masses.sum() == pytest.approx(m0, rel=1e-12)


def test_shatter_conserves_momentum_exactly():
    world, c = _impact()
    p0 = world.total_momentum().copy()
    resolver = collisions.CollisionResolver(seed=1)
    resolver.shatter(world, c)
    assert np.allclose(world.total_momentum(), p0, rtol=1e-10,
                       atol=1e-12 * abs(p0).max())


def test_fragment_count_is_within_bounds():
    world, c = _impact()
    resolver = collisions.CollisionResolver(seed=1)
    idx = resolver.shatter(world, c)
    assert collisions.MIN_FRAGMENTS <= len(idx) <= collisions.MAX_FRAGMENTS
    assert world.n == len(idx)


def test_fragments_share_an_event_id_and_an_immunity_window():
    world, c = _impact()
    resolver = collisions.CollisionResolver(seed=1)
    resolver.shatter(world, c)
    event_ids = {b.event_id for b in world.bodies}
    assert len(event_ids) == 1
    assert 0 not in event_ids
    for b in world.bodies:
        assert b.no_collide_until > world.time


def test_fragments_do_not_immediately_recollide():
    """The immunity window is the difference between a shattering and a flicker."""
    world, c = _impact()
    resolver = collisions.CollisionResolver(seed=1)
    resolver.shatter(world, c)
    prev = world.pos.copy()
    world.pos[:world.n] += world.vel[:world.n] * 1e-4
    assert collisions.find_collisions(world, prev) == []


def test_fragments_are_debris():
    world, c = _impact()
    resolver = collisions.CollisionResolver(seed=1)
    resolver.shatter(world, c)
    assert all(b.kind == KIND_DEBRIS for b in world.bodies)


def test_one_fragment_is_clearly_the_remnant():
    world, c = _impact()
    resolver = collisions.CollisionResolver(seed=1)
    resolver.shatter(world, c)
    masses = sorted(world.masses.tolist(), reverse=True)
    assert masses[0] >= 0.25 * sum(masses), "no dominant remnant"


def test_fragment_radii_follow_from_mass_and_density():
    world, c = _impact()
    density = units.density_from_mass_radius(units.EARTH_MASS, units.EARTH_RADIUS)
    resolver = collisions.CollisionResolver(seed=1)
    resolver.shatter(world, c)
    for k in range(world.n):
        actual = units.density_from_mass_radius(world.mass[k], world.radius[k])
        # All fragments share the mean density of the two parents.
        assert actual == pytest.approx(density, rel=0.5)


def test_shatter_falls_back_to_merge_when_the_world_is_full():
    world, c = _impact()
    # Pack the world so fewer than MIN_FRAGMENTS slots remain.
    while world.n < world.capacity - 2:
        world.add("filler", ROCK, 1e-14, units.EARTH_RADIUS * 0.01,
                  np.array([100.0 + world.n, 0.0]), np.zeros(2))
    before = world.n
    resolver = collisions.CollisionResolver(seed=1)
    resolver.shatter(world, c)
    assert world.n == before - 1, "should have merged, not shattered"


def test_resolver_returns_event_count_and_refreshes_the_integrator():
    from src.integrator import Integrator
    world, c = _impact()
    integrator = Integrator(world.capacity)
    integrator.refresh(world)
    prev = world.pos.copy()
    world.pos[:world.n] += world.vel[:world.n] * 1.0
    resolver = collisions.CollisionResolver(seed=1)
    events = resolver.resolve(world, integrator, prev)
    assert events == 1
    # Accelerations must match the post-collision configuration.
    expected = np.zeros_like(world.acc)
    integrator.accelerations(world.pos, world.mass, world.n, expected)
    assert np.allclose(world.acc[:world.n], expected[:world.n])
