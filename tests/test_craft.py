"""A craft is a one-way kind, like a black hole."""

from src import edits, materials, units
from src.body import KIND_CRAFT
from src.world import World


def _craft(world):
    return world.add("Shuttle", materials.SPACECRAFT, units.EARTH_MASS * 1e-18,
                     units.CRAFT_RADIUS, [1.0, 0.0], [0.0, 0.0],
                     kind=KIND_CRAFT)


def test_a_craft_never_ignites_however_heavy_it_gets():
    world = World()
    body = _craft(world)
    edits.set_mass(world, 0, 5.0)          # five solar masses
    assert body.kind == KIND_CRAFT


def test_metres_per_second_round_trips():
    assert abs(units.to_ms(units.from_ms(7800.0)) - 7800.0) < 1e-6
    # A round trip through x*k/k passes for any nonzero k, even a wrong one;
    # this pins the conversion factor itself.
    assert abs(units.from_ms(units.AUD_TO_MS) - 1.0) < 1e-12


def test_earth_orbital_speed_is_about_30_km_per_second():
    speed = units.circular_speed(units.G, 1.0)
    assert 29000.0 < units.to_ms(speed) < 30000.0


def test_the_spacecraft_material_is_not_reachable_by_cycling():
    assert materials.SPACECRAFT not in materials.MATERIALS
    assert materials.get("spacecraft") is materials.SPACECRAFT


def test_a_craft_starts_with_an_empty_plan_and_an_empty_tank():
    world = World()
    body = _craft(world)
    assert body.plan == ()
    assert body.fuel_dv == 0.0          # World.add does not fuel; the tool does


def test_the_drawn_stand_in_radius_is_far_larger_than_the_true_hull():
    assert units.craft_apparent_radius() > units.CRAFT_RADIUS * 1000.0
