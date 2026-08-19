import numpy as np
import pytest

from src import collisions, materials, units
from src.world import World

ROCK = materials.get("rock")


def _grazing_pair():
    """Two equal Earths that only just clip each other -- b close to 1."""
    m = units.EARTH_MASS
    r = units.EARTH_RADIUS
    world = World(8)
    world.add("A", ROCK, m, r, np.array([0.0, 0.0]), np.array([0.0, 0.0]))
    world.add("B", ROCK, m, r, np.array([4.0 * r, 1.9 * r]), np.array([-0.02, 0.0]))
    prev = world.pos.copy()
    world.pos[1] = np.array([-4.0 * r, 1.9 * r])
    c = collisions.find_collisions(world, prev)[0]
    assert c.b > 0.9, "test setup must actually be a graze"
    return world, c


def test_a_graze_leaves_both_bodies_alive():
    world, c = _grazing_pair()
    resolver = collisions.CollisionResolver(seed=1)
    resolver.graze(world, c)
    assert world.n == 2


def test_a_graze_conserves_momentum_and_mass():
    world, c = _grazing_pair()
    m0 = float(world.masses.sum())
    p0 = world.total_momentum().copy()
    resolver = collisions.CollisionResolver(seed=1)
    resolver.graze(world, c)
    assert float(world.masses.sum()) == pytest.approx(m0, rel=1e-12)
    assert np.allclose(world.total_momentum(), p0, rtol=1e-10,
                       atol=1e-12 * abs(p0).max())


def test_a_graze_removes_energy_rather_than_adding_it():
    world, c = _grazing_pair()
    mi, mj = float(world.mass[c.i]), float(world.mass[c.j])
    vi, vj = world.vel[c.i].copy(), world.vel[c.j].copy()
    ke0 = 0.5 * mi * float(np.dot(vi, vi)) + 0.5 * mj * float(np.dot(vj, vj))

    resolver = collisions.CollisionResolver(seed=1)
    resolver.graze(world, c)

    mi2, mj2 = float(world.mass[c.i]), float(world.mass[c.j])
    vi2, vj2 = world.vel[c.i], world.vel[c.j]
    ke1 = 0.5 * mi2 * float(np.dot(vi2, vi2)) + 0.5 * mj2 * float(np.dot(vj2, vj2))
    assert ke1 <= ke0 + 1e-20


def test_a_graze_transfers_some_mass_to_the_larger_body():
    world, c = _grazing_pair()
    mi0, mj0 = float(world.mass[c.i]), float(world.mass[c.j])
    resolver = collisions.CollisionResolver(seed=1)
    resolver.graze(world, c)
    assert world.mass[c.i] > mi0
    assert world.mass[c.j] < mj0


def test_grazed_bodies_do_not_immediately_re_collide():
    world, c = _grazing_pair()
    resolver = collisions.CollisionResolver(seed=1)
    resolver.graze(world, c)
    prev = world.pos.copy()
    world.pos[:world.n] += world.vel[:world.n] * 1e-4
    assert collisions.find_collisions(world, prev) == []


def test_grazed_bodies_end_up_clear_of_each_other():
    world, c = _grazing_pair()
    resolver = collisions.CollisionResolver(seed=1)
    resolver.graze(world, c)
    d = float(np.hypot(*(world.pos[c.j] - world.pos[c.i])))
    touch = float(world.radius[c.i] + world.radius[c.j])
    assert d > touch


def test_resolve_routes_a_graze_through_the_graze_handler():
    """An end-to-end check that classify() -> GRAZE actually reaches
    graze(), not the merge fallback -- both bodies must survive.

    Built already overlapping (rather than swept in from far away, as
    _grazing_pair does) so a small, realistic step is enough for
    find_collisions to catch it near the start of the step.
    """
    from src.integrator import Integrator
    m = units.EARTH_MASS
    r = units.EARTH_RADIUS
    world = World(8)
    world.add("A", ROCK, m, r, np.array([0.0, 0.0]), np.array([0.0, 0.0]))
    world.add("B", ROCK, m, r, np.array([0.3 * r, 1.85 * r]), np.array([-1e-5, 0.0]))
    integrator = Integrator(world.capacity)
    integrator.refresh(world)
    prev = world.pos.copy()
    world.pos[:world.n] += world.vel[:world.n] * 1.0
    resolver = collisions.CollisionResolver(seed=1)
    events = resolver.resolve(world, integrator, prev)
    assert events == 1
    assert world.n == 2
