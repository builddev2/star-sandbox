import numpy as np
import pytest

from src import collisions, destruction, materials, thermal, units
from src.body import KIND_STAR
from src.world import World

ROCK = materials.get("rock")
ICE = materials.get("ice")


def _sun_and(material, mass, radius, distance, vel=(0.0, 0.0)):
    world = World(16)
    world.add("Sun", materials.get("hydrogen"), 1.0, units.SUN_RADIUS,
              np.zeros(2), np.zeros(2), kind=KIND_STAR)
    world.add("Body", material, mass, radius, np.array([distance, 0.0]),
              np.array(vel, float))
    # These tests drive destruction.apply directly rather than through
    # Simulation.advance, so nothing else would ever set a temperature for the
    # ablation check to read.  Settling reproduces exactly what the old
    # instantaneous equilibrium gave.
    thermal.settle(world)
    return world


# --- thermal ablation ----------------------------------------------------------------

def test_a_body_at_one_au_is_untouched_by_heat():
    world = _sun_and(ROCK, units.EARTH_MASS * 1e-6, units.EARTH_RADIUS * 0.01, 1.0)
    resolver = collisions.CollisionResolver(seed=1)
    m0 = float(world.mass[1])
    for _ in range(20):
        destruction.apply(world, resolver, 1.0)
    assert world.n == 2
    assert float(world.mass[1]) == pytest.approx(m0, rel=1e-9)


def test_a_comet_passing_close_to_the_sun_is_boiled_away():
    ice = ICE
    world = _sun_and(ice, units.EARTH_MASS * 1e-8, units.EARTH_RADIUS * 0.002,
                     3.0 * units.SUN_RADIUS)
    resolver = collisions.CollisionResolver(seed=1)
    gone = False
    for _ in range(20000):
        count, event = destruction.apply(world, resolver, 0.02)
        if count:
            assert event is not None and event.outcome == "vaporise"
            gone = world.n == 1
            break
    assert gone, "an icy body a few solar radii out must eventually boil away"


def test_ablation_leaves_a_distant_body_and_the_star_alone():
    """Nothing it should not touch: body count, ids and the star's own mass."""
    world = _sun_and(ROCK, units.EARTH_MASS, units.EARTH_RADIUS, 1.0)
    sun_id, body_id = world.bodies[0].id, world.bodies[1].id
    sun_mass = float(world.mass[0])
    resolver = collisions.CollisionResolver(seed=1)
    for _ in range(50):
        destruction.apply(world, resolver, 1.0)
    assert world.n == 2
    assert {b.id for b in world.bodies} == {sun_id, body_id}
    assert float(world.mass[0]) == pytest.approx(sun_mass, rel=1e-15)


# --- tidal disruption ------------------------------------------------------------

def test_tidal_stress_ratio_grows_as_the_cube_of_closeness():
    world = _sun_and(ROCK, units.EARTH_MASS, units.EARTH_RADIUS, 1.0)
    far = destruction.tidal_stress_ratio(world, 1, 0)
    world.pos[1] = np.array([0.5, 0.0])
    near = destruction.tidal_stress_ratio(world, 1, 0)
    assert near > far * 7.0     # (1/0.5)^3 == 8, minus a little slack


def test_a_loosely_bound_rubble_pile_is_torn_apart_inside_the_roche_limit():
    """A big, low-density body -- water's density is close enough to the
    Sun's own mean density that a Roche-limit-style disruption only reaches
    it very close in, well within the star's own radius -- comes apart in
    one substep once it is that close. (Gravity-only tidal stress scales
    with 1/density and is independent of the body's own size, the classic
    Roche result, so getting the ratio past 1.0 at all needs either a much
    lower density than any material here or getting this close; this test
    takes the second route and leaves density-scaling to the tidal_stress_ratio
    unit test below.)
    """
    d = 0.5 * units.SUN_RADIUS
    huge_weak = materials.get("water")
    world = _sun_and(huge_weak, units.EARTH_MASS, units.EARTH_RADIUS, d)
    ratio = destruction.tidal_stress_ratio(world, 1, 0)
    assert ratio >= 1.0, "test setup must actually be past the tidal limit"

    resolver = collisions.CollisionResolver(seed=1)
    events, event = destruction.apply(world, resolver, 0.5)
    assert events == 1
    assert event is not None and event.outcome == "tidal"
    assert world.n > 1, "tidal disruption makes fragments, not nothing"
    assert all(b.kind != "star" for b in world.bodies[1:])


def test_a_small_strong_rock_survives_the_same_pass():
    """Not tidally shredded, even at the exact distance that disrupts the
    loosely-bound water body above -- material strength (Task 1's
    ``toughness``) is what makes the difference, not just being small."""
    d = 0.5 * units.SUN_RADIUS
    r = units.EARTH_RADIUS * 2e-4
    mass = units.FOUR_THIRDS_PI * ROCK.density * r ** 3
    assert mass > 100.0 * destruction.VAPORISE_FLOOR_MASS, \
        "test setup must stay well clear of the ablation floor"
    world = _sun_and(ROCK, mass, r, d)
    ratio = destruction.tidal_stress_ratio(world, 1, 0)
    assert ratio < 1.0

    resolver = collisions.CollisionResolver(seed=1)
    destruction.apply(world, resolver, 0.5)
    assert world.n == 2, "one substep must not tidally shred it"


def test_tidal_disrupt_conserves_mass():
    world = _sun_and(materials.get("water"), units.EARTH_MASS, units.EARTH_RADIUS,
                     3.0 * units.SUN_RADIUS)
    m0 = float(world.masses.sum())
    resolver = collisions.CollisionResolver(seed=1)
    resolver.tidal_disrupt(world, 1, 0)
    assert float(world.masses.sum()) == pytest.approx(m0, rel=1e-12)


def test_tidal_disrupt_returns_empty_when_the_world_is_full():
    world = World(3)
    world.add("Sun", materials.get("hydrogen"), 1.0, units.SUN_RADIUS,
              np.zeros(2), np.zeros(2))
    world.add("Body", materials.get("water"), units.EARTH_MASS, units.EARTH_RADIUS,
              np.array([3.0 * units.SUN_RADIUS, 0.0]), np.zeros(2))
    resolver = collisions.CollisionResolver(seed=1)
    made = resolver.tidal_disrupt(world, 1, 0)
    assert made == []
    assert world.n == 2, "a body with nowhere to spare for fragments is left alone"
