"""The arithmetic of a burn.  Headless -- no pygame anywhere in this file."""

import math

import numpy as np
import pytest

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


def test_node_state_reads_due_time_as_absolute_not_a_countdown():
    """``due_time`` is absolute world days (see the module docstring), so a
    node due at day 150 when the clock reads 100 must be evaluated exactly
    like a node due at day 50 when the clock reads 0 -- both are 50 days out.
    A mutant that drops ``- world.time`` treats ``due_time`` as a countdown
    instead and would advance 150 days here rather than 50, landing the
    craft somewhere else on its circular orbit entirely."""
    world_now = _sun_and_craft()
    world_now.time = 100.0
    node_now = maneuver.Maneuver(due_time=150.0, prograde=0.0)
    _mu, pos_now, vel_now = maneuver.node_state(world_now, 1, node_now)

    world_zero = _sun_and_craft()
    node_zero = maneuver.Maneuver(due_time=50.0, prograde=0.0)
    _mu, pos_zero, vel_zero = maneuver.node_state(world_zero, 1, node_zero)

    assert np.allclose(pos_now, pos_zero, atol=1e-12)
    assert np.allclose(vel_now, vel_zero, atol=1e-12)


def test_plan_conics_advances_the_clock_from_world_time_not_from_zero():
    """Same absolute-time promise as ``node_state``, but exercised through
    ``plan_conics``'s own clock bookkeeping across two nodes."""
    world_now = _sun_and_craft()
    world_now.time = 100.0
    plan_now = (maneuver.Maneuver(due_time=110.0, prograde=units.from_ms(800.0)),
                maneuver.Maneuver(due_time=300.0, prograde=units.from_ms(600.0)))
    _attractor, conics_now = maneuver.plan_conics(world_now, 1, plan_now, samples=64)

    world_zero = _sun_and_craft()
    plan_zero = (maneuver.Maneuver(due_time=10.0, prograde=units.from_ms(800.0)),
                 maneuver.Maneuver(due_time=200.0, prograde=units.from_ms(600.0)))
    _attractor, conics_zero = maneuver.plan_conics(world_zero, 1, plan_zero, samples=64)

    for (points_now, bound_now), (points_zero, bound_zero) in zip(conics_now, conics_zero):
        assert bound_now == bound_zero
        assert np.allclose(points_now, points_zero, atol=1e-9)


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


def test_plan_conics_chains_each_node_from_the_previous_post_burn_state():
    """The docstring's central promise: node k's conic is propagated from
    node k-1's POST-burn state, not its pre-burn one. A real two-burn Hohmann
    transfer makes this checkable exactly -- the second conic must be
    circular at the target radius, because that circularisation is the whole
    point of the arrival burn. A mutant that chains from the pre-burn state
    instead gives a lopsided conic ranging ~1.0 to ~1.4385 AU rather than a
    circle at 1.5 AU, which a 1e-3 tolerance catches easily."""
    world = _sun_and_craft(radius=1.0)
    mu = units.G * (1.0 + world.mass[1])
    r1, r2 = 1.0, 1.5
    h = maneuver.hohmann(mu, r1, r2, 0.0, 0.0)
    depart_time = 0.0
    arrive_time = depart_time + h["transfer_days"]
    plan = (maneuver.Maneuver(due_time=depart_time, prograde=h["depart_dv"]),
            maneuver.Maneuver(due_time=arrive_time, prograde=h["arrive_dv"]))

    _attractor, conics = maneuver.plan_conics(world, 1, plan, samples=200)
    points, _bound = conics[1]
    radii = np.hypot(points[:, 0], points[:, 1])
    assert abs(radii.min() - r2) < 1e-3
    assert abs(radii.max() - r2) < 1e-3


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


def test_the_refinement_pass_actually_tightens_a_phased_intercept():
    """A real Earth->Mars Hohmann transfer, phased so the craft and Mars
    actually meet: the coarse 240-sample sweep alone lands ~7.6e-04 AU away
    at best, and the refinement pass narrows that to ~2.0e-05 AU -- 38x
    tighter. A tolerance loose enough to pass on the coarse sweep alone
    would never notice the refinement pass being skipped."""
    mu = units.G
    r1, r2 = 1.0, 1.524
    v1 = math.sqrt(mu / r1)
    v2 = math.sqrt(mu / r2)
    h = maneuver.hohmann(mu, r1, r2, 0.0, 0.0)
    n2 = math.sqrt(mu / r2 ** 3)
    required = math.pi - n2 * h["transfer_days"]

    craft_pos = np.array([r1, 0.0])
    craft_vel = np.array([0.0, v1 + h["depart_dv"]])

    target_theta = required % (2.0 * math.pi)
    target_pos = np.array([r2 * math.cos(target_theta), r2 * math.sin(target_theta)])
    target_vel = v2 * np.array([-math.sin(target_theta), math.cos(target_theta)])

    result = maneuver.closest_approach(mu, craft_pos, craft_vel,
                                       target_pos, target_vel,
                                       horizon_days=400.0, samples=240)
    assert result["distance"] < 1e-4


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


@pytest.mark.parametrize("r2", [1.524, 0.72])
def test_the_wait_puts_the_target_where_the_craft_will_arrive(r2):
    """The part a naive Hohmann omits: you cannot leave whenever you like.

    Parameterised over an outward (r2=1.524) and an inward (r2=0.72) transfer:
    the phase-closure check below is the only thing in the suite that can
    catch a sign error confined to the inward branch, because
    ``test_the_wait_is_never_negative`` is defeated by its own ``max(0.0, ...)``
    clamp -- a wrong sign there just reads as zero, not wrong.
    """
    mu = units.G
    r1 = 1.0
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


def test_a_flyby_never_changes_the_speed_the_planet_sees():
    """The whole trick: |v_inf| out equals |v_inf| in.  Only the direction
    changes, and the sum with the planet's own motion is what gains you speed."""
    mu_planet = units.G * units.JUPITER_MASS
    v_in = np.array([0.005, 0.0])
    result = maneuver.flyby(mu_planet, v_in, periapsis=units.JUPITER_RADIUS * 3.0)
    assert abs(np.hypot(*result["v_out"]) - np.hypot(*v_in)) < 1e-15


def test_a_flyby_actually_rotates_v_inf_by_the_closed_form_angle():
    """Pins the rotation itself, not just that speed is preserved -- a bare
    |v_out| == |v_in| check is satisfied by the identity rotation too.

    mu_planet = periapsis = 1.0 and |v_inf| = 1.0 make the eccentricity term
    ``periapsis * |v_inf|^2 / mu_planet`` exactly 1.0, so ``e == 2.0`` and
    ``turn == 2*asin(0.5) == pi/3`` are exact, not approximate-by-luck.
    """
    mu_planet = 1.0
    periapsis = 1.0
    v_in = np.array([1.0, 0.0])
    result = maneuver.flyby(mu_planet, v_in, periapsis=periapsis)

    assert result["e"] == pytest.approx(2.0)
    assert result["turn"] == pytest.approx(2.0 * math.asin(0.5))

    v_out = result["v_out"]
    swept = math.atan2(v_out[1], v_out[0]) - math.atan2(v_in[1], v_in[0])
    swept = (swept + math.pi) % (2.0 * math.pi) - math.pi
    if swept <= -math.pi:
        swept += 2.0 * math.pi
    assert swept == pytest.approx(result["turn"])


def test_a_close_slow_pass_turns_harder_than_a_distant_fast_one():
    mu_planet = units.G * units.JUPITER_MASS
    slow_close = maneuver.flyby(mu_planet, np.array([0.002, 0.0]),
                                periapsis=units.JUPITER_RADIUS * 1.5)
    fast_far = maneuver.flyby(mu_planet, np.array([0.02, 0.0]),
                              periapsis=units.JUPITER_RADIUS * 40.0)
    assert slow_close["turn"] > fast_far["turn"]


def test_the_trajectory_is_always_a_hyperbola():
    mu_planet = units.G * units.JUPITER_MASS
    result = maneuver.flyby(mu_planet, np.array([0.005, 0.0]),
                            periapsis=units.JUPITER_RADIUS * 2.0)
    assert result["e"] > 1.0


def test_a_flyby_with_no_approach_speed_does_nothing():
    result = maneuver.flyby(units.G * units.JUPITER_MASS, np.array([0.0, 0.0]),
                            periapsis=units.JUPITER_RADIUS)
    assert result["turn"] == 0.0


def test_side_minus_one_rotates_the_opposite_way_but_still_conserves_speed():
    """``side`` picks which side of the planet the craft passes, which is
    what decides whether an assist boosts or brakes it -- so flipping it
    must flip the sense of rotation, not just scale something."""
    mu_planet = units.G * units.JUPITER_MASS
    v_in = np.array([0.005, 0.0])
    periapsis = units.JUPITER_RADIUS * 3.0

    positive = maneuver.flyby(mu_planet, v_in, periapsis=periapsis, side=1.0)
    negative = maneuver.flyby(mu_planet, v_in, periapsis=periapsis, side=-1.0)

    assert abs(np.hypot(*negative["v_out"]) - np.hypot(*v_in)) < 1e-15
    assert negative["turn"] == pytest.approx(positive["turn"])
    # A rotation by +turn and one by -turn are mirror images across v_in's
    # axis: the y-component flips sign, the x-component does not.
    assert negative["v_out"][1] == pytest.approx(-positive["v_out"][1])
    assert negative["v_out"][0] == pytest.approx(positive["v_out"][0])
    assert not np.allclose(negative["v_out"], positive["v_out"])


# --- firing a planned burn ----------------------------------------------------------


def test_a_burn_changes_the_world_velocity_by_exactly_its_delta_v():
    world = _sun_and_craft()
    world.bodies[1].fuel_dv = units.from_ms(9300.0)
    before = np.array(world.vel[1])
    node = maneuver.Maneuver(due_time=0.0, prograde=units.from_ms(1000.0))
    assert maneuver.apply_burn(world, 1, node)
    change = float(np.hypot(*(world.vel[1] - before)))
    assert abs(units.to_ms(change) - 1000.0) < 1.0


def test_a_burn_spends_fuel():
    world = _sun_and_craft()
    world.bodies[1].fuel_dv = units.from_ms(9300.0)
    maneuver.apply_burn(world, 1, maneuver.Maneuver(0.0, units.from_ms(1000.0)))
    assert abs(units.to_ms(world.bodies[1].fuel_dv) - 8300.0) < 1.0


def test_a_burn_bigger_than_the_tank_is_scaled_down_to_what_is_left():
    """Running dry mid-burn is a real thing that happens to real missions,
    and it is a better lesson than a button that silently does nothing."""
    world = _sun_and_craft()
    world.bodies[1].fuel_dv = units.from_ms(500.0)
    before = np.array(world.vel[1])
    maneuver.apply_burn(world, 1, maneuver.Maneuver(0.0, units.from_ms(4000.0)))
    change = float(np.hypot(*(world.vel[1] - before)))
    assert abs(units.to_ms(change) - 500.0) < 1.0
    assert world.bodies[1].fuel_dv == 0.0


def test_an_empty_tank_fires_nothing_at_all():
    world = _sun_and_craft()
    world.bodies[1].fuel_dv = 0.0
    before = np.array(world.vel[1])
    assert not maneuver.apply_burn(world, 1, maneuver.Maneuver(0.0, units.from_ms(100.0)))
    assert float(np.hypot(*(world.vel[1] - before))) == 0.0


def test_execute_due_fires_only_what_is_due():
    world = _sun_and_craft()
    world.bodies[1].fuel_dv = units.from_ms(9300.0)
    world.bodies[1].plan = (maneuver.Maneuver(5.0, units.from_ms(100.0)),
                            maneuver.Maneuver(50.0, units.from_ms(100.0)))
    world.time = 10.0
    assert maneuver.execute_due(world) == 1
    assert len(world.bodies[1].plan) == 1
    assert world.bodies[1].plan[0].due_time == 50.0


def test_execute_due_leaves_a_plan_whose_time_has_not_come():
    world = _sun_and_craft()
    world.bodies[1].fuel_dv = units.from_ms(9300.0)
    world.bodies[1].plan = (maneuver.Maneuver(50.0, units.from_ms(100.0)),)
    world.time = 10.0
    assert maneuver.execute_due(world) == 0
    assert len(world.bodies[1].plan) == 1


def test_next_burn_time_is_infinite_with_nothing_planned():
    world = _sun_and_craft()
    assert maneuver.next_burn_time(world) == math.inf
    world.bodies[1].plan = (maneuver.Maneuver(42.0),)
    assert maneuver.next_burn_time(world) == 42.0


def test_a_craft_that_lost_its_attractor_cannot_jam_the_queue():
    """The node is popped whether or not the burn could be applied."""
    world = World()
    lone = world.add("Shuttle", materials.SPACECRAFT, units.EARTH_MASS * 1e-18,
                     units.CRAFT_RADIUS, np.array([1.0, 0.0]), np.zeros(2),
                     kind=KIND_CRAFT)
    lone.fuel_dv = units.from_ms(9300.0)
    lone.plan = (maneuver.Maneuver(0.0, units.from_ms(100.0)),)
    assert maneuver.execute_due(world) == 1
    assert lone.plan == ()


def test_refining_the_window_puts_the_target_where_the_craft_arrives():
    """The circular-orbit phase angle is only a first guess when the target's
    orbit is eccentric.  Refining it against exact propagation should leave
    the target essentially on top of the arrival point."""
    mu = units.G
    craft_pos = np.array([1.0, 0.0])
    craft_vel = np.array([0.0, math.sqrt(mu / 1.0)])
    # An eccentric target, the way Mars really is.
    tp, tv = orbits.state_from_elements(mu, 1.524, 0.093, 0.4, 1.1)

    guess = maneuver.hohmann(mu, 1.0, 1.524,
                             math.atan2(craft_pos[1], craft_pos[0]),
                             math.atan2(tp[1], tp[0]))
    refined = maneuver.refine_window(mu, craft_pos, craft_vel, tp, tv,
                                     guess["wait_days"], guess["transfer_days"])

    def gap_for(wait):
        arrive = wait + guess["transfer_days"]
        cp, _cv = orbits.propagate(mu, craft_pos, craft_vel, wait)
        # The craft leaves on the transfer ellipse and arrives half a turn on.
        angle = math.atan2(cp[1], cp[0]) + math.pi
        target_at, _ = orbits.propagate(mu, tp, tv, arrive)
        return abs((angle - math.atan2(target_at[1], target_at[0])
                    + math.pi) % (2.0 * math.pi) - math.pi)

    assert gap_for(refined) < gap_for(guess["wait_days"])
    assert gap_for(refined) < 0.02          # radians
    assert refined >= 0.0


# --- defects found by review, fixed 2026-09-04 --------------------------------------


def test_a_flyby_with_no_approach_speed_is_a_straight_line_not_an_infinite_conic():
    """e = inf is not the limit.  A craft with no speed relative to the planet
    is on a parabolic path, e = 1, and an inspector reading e should say so
    rather than printing infinity."""
    result = maneuver.flyby(units.G * units.JUPITER_MASS, np.array([0.0, 0.0]),
                            periapsis=units.JUPITER_RADIUS)
    assert result["e"] == 1.0
    assert result["turn"] == 0.0


def test_a_degenerate_side_still_rotates_one_definite_way():
    """A cross product of two near-parallel vectors can come out as -0.0, and
    copysign reads the sign bit -- so -0.0 used to flip the assist over."""
    mu_planet = units.G * units.JUPITER_MASS
    v_in = np.array([0.005, 0.0])
    p = units.JUPITER_RADIUS * 3.0
    positive = maneuver.flyby(mu_planet, v_in, p, side=0.0)
    negative_zero = maneuver.flyby(mu_planet, v_in, p, side=-0.0)
    assert np.allclose(positive["v_out"], negative_zero["v_out"])


def test_hohmann_refuses_a_radius_of_zero_rather_than_dividing_by_it():
    """The panel feeds this radii taken from world state, and a craft sitting
    on the attractor's centre is degenerate but reachable."""
    assert maneuver.hohmann(units.G, 0.0, 1.5, 0.0, 0.0) is None
    assert maneuver.hohmann(units.G, 1.0, 0.0, 0.0, 0.0) is None
    assert maneuver.hohmann(0.0, 1.0, 1.5, 0.0, 0.0) is None
    assert maneuver.hohmann(units.G, 1.0, 1.5, 0.0, 0.0) is not None


def test_plan_conics_sorts_a_plan_given_out_of_order():
    """Nothing stops a caller building a plan out of order; applying a later
    node's burn at an earlier node's position is silently wrong."""
    world = _sun_and_craft()
    late = maneuver.Maneuver(due_time=200.0, prograde=units.from_ms(600.0))
    early = maneuver.Maneuver(due_time=10.0, prograde=units.from_ms(800.0))
    _a, out_of_order = maneuver.plan_conics(world, 1, (late, early), samples=64)
    _b, in_order = maneuver.plan_conics(world, 1, (early, late), samples=64)
    assert len(out_of_order) == len(in_order) == 2
    for got, want in zip(out_of_order, in_order):
        assert np.allclose(got[0], want[0])


def test_closest_approach_actually_uses_the_sample_count_it_is_given():
    """`samples` was silently floored to 8 and never varied in a test, so an
    implementation ignoring the argument entirely passed."""
    mu = units.G
    craft_pos = np.array([1.0, 0.0])
    craft_vel = np.array([0.0, math.sqrt(mu)])
    target_pos, target_vel = orbits.state_from_elements(mu, 1.9, 0.0, 0.0, 2.2)
    coarse = maneuver.closest_approach(mu, craft_pos, craft_vel,
                                       target_pos, target_vel,
                                       horizon_days=900.0, samples=8)
    fine = maneuver.closest_approach(mu, craft_pos, craft_vel,
                                     target_pos, target_vel,
                                     horizon_days=900.0, samples=600)
    assert fine["distance"] <= coarse["distance"]
    assert fine["distance"] < coarse["distance"] - 1e-9 or coarse["distance"] < 1e-6


def test_the_refinement_pass_is_fine_enough_to_be_worth_having():
    """Only the existence of the refinement was pinned, not its resolution:
    shrinking the fine sweep from 32 samples to 4 still passed."""
    mu = units.G
    craft_pos = np.array([1.0, 0.0])
    craft_vel = np.array([0.0, math.sqrt(mu / 1.0)])
    target_pos, target_vel = orbits.state_from_elements(mu, 1.524, 0.0, 0.0, 1.3)
    found = maneuver.closest_approach(mu, craft_pos, craft_vel,
                                      target_pos, target_vel,
                                      horizon_days=700.0, samples=240)
    # Re-measure at the reported instant: a coarse refinement lands off the
    # true minimum, and the separation there is worse than what it claims.
    c_pos, _ = orbits.propagate(mu, craft_pos, craft_vel, found["time"])
    t_pos, _ = orbits.propagate(mu, target_pos, target_vel, found["time"])
    exact = float(np.hypot(c_pos[0] - t_pos[0], c_pos[1] - t_pos[1]))
    assert abs(exact - found["distance"]) < 1e-12
    # And the minimum really is a minimum: a step either side is no closer.
    for step in (-0.5, 0.5):
        c2, _ = orbits.propagate(mu, craft_pos, craft_vel, found["time"] + step)
        t2, _ = orbits.propagate(mu, target_pos, target_vel, found["time"] + step)
        assert float(np.hypot(c2[0] - t2[0], c2[1] - t2[1])) >= found["distance"] - 1e-9


def test_craft_state_refuses_a_missing_row_instead_of_indexing_with_none():
    """`pos[None]` does not raise -- numpy quietly adds an axis, and the
    failure surfaces frames later inside an einsum with a message that names
    nothing.  A missing row means "no craft", which every caller handles."""
    world = _sun_and_craft()
    assert maneuver.craft_state(world, None) is None
    assert maneuver.node_state(world, None, maneuver.Maneuver(0.0)) is None
    attractor, conics = maneuver.plan_conics(world, None, ())
    assert attractor is None and conics == []
