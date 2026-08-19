import numpy as np

from src import materials, units
from src.world import World


def _add(world, name, mass=units.EARTH_MASS, x=1.0):
    return world.add(name, materials.get("rock"), mass, units.EARTH_RADIUS,
                     np.array([x, 0.0]), np.zeros(2))


def test_ids_survive_swap_removes():
    world = World()
    bodies = [_add(world, "b%d" % i, x=float(i + 1)) for i in range(5)]
    world.remove(1)          # swap-removes: last row moves into slot 1
    world.remove(0)
    alive = [b for b in bodies if world.index_of(b.id) is not None]
    assert len(alive) == 3
    for body in alive:
        i = world.index_of(body.id)
        assert world.bodies[i] is body, "id map desynchronised from rows"


def test_removed_ids_resolve_to_none():
    world = World()
    a = _add(world, "a", x=1.0)
    b = _add(world, "b", x=2.0)
    world.remove(world.index_of(a.id))
    assert world.index_of(a.id) is None
    assert world.index_of(b.id) is not None


def test_remove_many_is_order_independent():
    world = World()
    bodies = [_add(world, "b%d" % i, x=float(i + 1)) for i in range(6)]
    keep = {bodies[0].id, bodies[3].id}
    world.remove_many([1, 2, 4, 5])
    assert world.n == 2
    assert {b.id for b in world.bodies} == keep


def test_positions_view_tracks_n():
    world = World()
    _add(world, "a")
    assert world.positions.shape == (1, 2)
    _add(world, "b", x=2.0)
    assert world.positions.shape == (2, 2)


def test_solar_system_barycentre_is_at_rest():
    from src import scenarios
    world = scenarios.solar_system()
    assert np.allclose(world.total_momentum(), 0.0, atol=1e-18)
