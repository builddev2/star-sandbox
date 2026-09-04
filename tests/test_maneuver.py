"""The arithmetic of a burn.  Headless -- no pygame anywhere in this file."""

import math

import numpy as np

from src import maneuver, materials, orbits, units
from src.body import KIND_CRAFT, KIND_STAR
from src.world import World


def _sun_and_craft(radius=1.0):
    """A one-solar-mass star with a craft on a circular orbit at ``radius``."""
    world = World()
    world.add("Sun", materials.get("hydrogen"), 1.0, units.SUN_RADIUS,
              np.zeros(2), np.zeros(2), kind=KIND_STAR)
    speed = math.sqrt(units.G / radius)
    world.add("Shuttle", materials.SPACECRAFT, units.EARTH_MASS * 1e-18,
              units.CRAFT_RADIUS, np.array([radius, 0.0]),
              np.array([0.0, speed]), kind=KIND_CRAFT)
    return world


def _elements(world, index, rel_pos, rel_vel):
    mu = units.G * world.mass[0]
    return orbits.elements_from_state(mu, rel_pos, rel_vel)


def test_a_prograde_burn_raises_the_far_side_and_leaves_the_near_side_alone():
    world = _sun_and_craft()
    node = maneuver.Maneuver(due_time=0.0, prograde=units.from_ms(1000.0))
    mu, pos, vel = maneuver.node_state(world, 1, node)
    after = maneuver.burn_velocity(pos, vel, node)
    el = orbits.elements_from_state(mu, pos, after)
    assert el["apoapsis"] > 1.05
    assert abs(el["periapsis"] - 1.0) < 1e-6


def test_a_retrograde_burn_lowers_the_far_side():
    world = _sun_and_craft()
    node = maneuver.Maneuver(due_time=0.0, prograde=-units.from_ms(1000.0))
    mu, pos, vel = maneuver.node_state(world, 1, node)
    after = maneuver.burn_velocity(pos, vel, node)
    el = orbits.elements_from_state(mu, pos, after)
    assert el["periapsis"] < 0.95
    assert abs(el["apoapsis"] - 1.0) < 1e-6


def test_a_radial_burn_tilts_the_orbit_without_changing_its_size_much():
    world = _sun_and_craft()
    node = maneuver.Maneuver(due_time=0.0, radial=units.from_ms(500.0))
    mu, pos, vel = maneuver.node_state(world, 1, node)
    after = maneuver.burn_velocity(pos, vel, node)
    el = orbits.elements_from_state(mu, pos, after)
    assert el["e"] > 0.01
    # Energy, and so the semi-major axis, barely moves: the burn is perpendicular.
    assert abs(el["a"] - 1.0) < 0.1


def test_magnitude_combines_both_components():
    node = maneuver.Maneuver(due_time=0.0, prograde=3.0, radial=4.0)
    assert abs(node.magnitude - 5.0) < 1e-12


def test_a_node_in_the_future_is_evaluated_where_the_craft_will_be():
    world = _sun_and_craft()
    period = 2.0 * math.pi / math.sqrt(units.G)
    node = maneuver.Maneuver(due_time=period * 0.5, prograde=0.0)
    _mu, pos, _vel = maneuver.node_state(world, 1, node)
    # Half an orbit on: the far side of the circle.
    assert pos[0] < -0.99


def test_craft_state_finds_the_star_as_the_attractor():
    world = _sun_and_craft()
    attractor, mu, rel_pos, _rel_vel = maneuver.craft_state(world, 1)
    assert attractor == 0
    assert abs(mu - units.G * (1.0 + world.mass[1])) < 1e-18
    assert abs(float(np.hypot(*rel_pos)) - 1.0) < 1e-12


def test_plan_conics_returns_one_conic_per_node():
    world = _sun_and_craft()
    plan = (maneuver.Maneuver(due_time=10.0, prograde=units.from_ms(800.0)),
            maneuver.Maneuver(due_time=200.0, prograde=units.from_ms(600.0)))
    attractor, conics = maneuver.plan_conics(world, 1, plan, samples=64)
    assert attractor == 0
    assert len(conics) == 2
    for points, _bound in conics:
        assert len(points) > 8


def test_two_circular_orbits_come_closest_at_the_difference_of_their_radii():
    """A craft at 1 AU and a target at 1.5 AU, both circular and coplanar:
    the closest they ever get is 0.5 AU, when they line up."""
    mu = units.G
    craft_pos = np.array([1.0, 0.0])
    craft_vel = np.array([0.0, math.sqrt(mu / 1.0)])
    target_pos = np.array([-1.5, 0.0])          # starts on the far side
    target_vel = np.array([0.0, -math.sqrt(mu / 1.5)])

    result = maneuver.closest_approach(mu, craft_pos, craft_vel,
                                       target_pos, target_vel,
                                       horizon_days=1200.0, samples=400)
    assert abs(result["distance"] - 0.5) < 0.02
    assert result["time"] > 0.0


def test_a_craft_that_never_leaves_its_target_reports_a_tiny_separation():
    mu = units.G
    pos = np.array([1.0, 0.0])
    vel = np.array([0.0, math.sqrt(mu)])
    result = maneuver.closest_approach(mu, pos, vel, pos, vel,
                                       horizon_days=365.0, samples=120)
    assert result["distance"] < 1e-9


def test_closest_approach_reports_where_both_bodies_were():
    mu = units.G
    craft_pos = np.array([1.0, 0.0])
    craft_vel = np.array([0.0, math.sqrt(mu)])
    target_pos = np.array([0.0, 1.5])
    target_vel = np.array([-math.sqrt(mu / 1.5), 0.0])
    result = maneuver.closest_approach(mu, craft_pos, craft_vel,
                                       target_pos, target_vel,
                                       horizon_days=800.0, samples=300)
    gap = np.hypot(*(result["craft_pos"] - result["target_pos"]))
    assert abs(gap - result["distance"]) < 1e-9


def test_earth_to_mars_matches_the_textbook():
    """The acceptance test for the whole solver.  Every number here is one
    an eight-year-old can look up and check."""
    result = maneuver.hohmann(units.G, 1.0, 1.524, 0.0, 0.0)
    assert 2900.0 < units.to_ms(result["depart_dv"]) < 3000.0
    assert 2600.0 < units.to_ms(result["arrive_dv"]) < 2700.0
    assert 250.0 < result["transfer_days"] < 270.0


def test_going_inward_costs_a_retrograde_burn():
    result = maneuver.hohmann(units.G, 1.0, 0.72, 0.0, 0.0)
    assert result["depart_dv"] < 0.0
    assert result["arrive_dv"] < 0.0


def test_the_wait_puts_the_target_where_the_craft_will_arrive():
    """The part a naive Hohmann omits: you cannot leave whenever you like."""
    mu = units.G
    r1, r2 = 1.0, 1.524
    theta_craft, theta_target = 0.0, 1.0
    result = maneuver.hohmann(mu, r1, r2, theta_craft, theta_target)

    n1 = math.sqrt(mu / r1 ** 3)
    n2 = math.sqrt(mu / r2 ** 3)
    wait, transfer = result["wait_days"], result["transfer_days"]

    # Where each body is when the craft arrives at the far side.
    craft_arrival_angle = theta_craft + n1 * wait + math.pi
    target_arrival_angle = theta_target + n2 * (wait + transfer)
    gap = (craft_arrival_angle - target_arrival_angle) % (2.0 * math.pi)
    gap = min(gap, 2.0 * math.pi - gap)
    assert gap < 1e-6


def test_the_wait_is_never_negative():
    for target_angle in (0.0, 1.0, 3.0, 5.5, 6.2):
        result = maneuver.hohmann(units.G, 1.0, 1.524, 0.0, target_angle)
        assert result["wait_days"] >= 0.0
        inward = maneuver.hohmann(units.G, 1.0, 0.72, 0.0, target_angle)
        assert inward["wait_days"] >= 0.0


def test_equal_radii_need_no_burn_at_all():
    result = maneuver.hohmann(units.G, 1.0, 1.0, 0.0, 0.0)
    assert abs(result["depart_dv"]) < 1e-12
    assert abs(result["arrive_dv"]) < 1e-12
