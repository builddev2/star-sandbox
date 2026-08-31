"""Temperature: what sets it, what changes it, and how fast."""

import math

import numpy as np
import pytest

from src import materials, thermal, units
from src.body import KIND_BLACK_HOLE, KIND_STAR
from src.world import World

ROCK = materials.get("rock")
ICE = materials.get("ice")
HYDROGEN = materials.get("hydrogen")


# --- constants and conversions -------------------------------------------------

def test_stefan_boltzmann_is_the_si_value():
    assert units.STEFAN_BOLTZMANN == pytest.approx(5.670374419e-8, rel=1e-9)


def test_the_microwave_background_is_2_point_725_kelvin():
    assert units.CMB_TEMPERATURE_K == pytest.approx(2.725, rel=1e-6)


def test_celsius_is_kelvin_less_absolute_zero():
    assert units.to_celsius(273.15) == pytest.approx(0.0, abs=1e-9)
    assert units.to_celsius(0.0) == pytest.approx(-273.15, abs=1e-9)
    assert units.to_celsius(373.15) == pytest.approx(100.0, abs=1e-9)


def test_density_converts_to_si_kilograms_per_cubic_metre():
    """Water is 1000 kg/m3 by definition, whichever way round we go."""
    assert units.to_kgm3(units.from_gcm3(1.0)) == pytest.approx(1000.0, rel=1e-9)


def test_the_editable_range_spans_the_background_to_hotter_than_any_star():
    assert units.MIN_TEMPERATURE_K == pytest.approx(units.CMB_TEMPERATURE_K)
    assert units.MAX_TEMPERATURE_K > units.SUN_TEMPERATURE_K * 10.0


# --- material properties -------------------------------------------------------

def test_every_material_has_a_real_albedo_and_heat_capacity():
    for m in materials.MATERIALS:
        assert 0.0 <= m.albedo < 1.0, m.key
        assert m.heat_capacity_j_kgk > 0.0, m.key


def test_ice_is_bright_and_water_is_dark():
    """Albedo is what makes an icy world colder than a rocky one at the same
    distance -- the physical fact, not a tuning knob."""
    assert materials.get("ice").albedo > materials.get("rock").albedo
    assert materials.get("water").albedo < materials.get("rock").albedo


def test_water_stores_far_more_heat_than_iron():
    """4180 against 450 J/kg/K: the real ratio, and why oceans moderate
    climate and a frying pan does not."""
    assert (materials.get("water").heat_capacity_j_kgk
            > 8.0 * materials.get("iron").heat_capacity_j_kgk)


def test_a_singularity_reflects_nothing_and_cannot_be_heated():
    assert materials.SINGULARITY.albedo == 0.0
    assert materials.SINGULARITY.heat_capacity_j_kgk == math.inf


# --- equilibrium ---------------------------------------------------------------

def _sun_and(material, distance, mass=units.EARTH_MASS,
             radius=units.EARTH_RADIUS, sun_mass=1.0):
    """A star at the origin and one body ``distance`` AU away."""
    world = World(16)
    world.add("Sun", HYDROGEN, sun_mass, units.SUN_RADIUS,
              np.zeros(2), np.zeros(2), kind=KIND_STAR)
    world.add("Body", material, mass, radius,
              np.array([distance, 0.0]), np.zeros(2))
    return world


class _Black:
    """A perfectly absorbing stand-in, so a test can check the bare formula
    without an albedo in the way."""
    albedo = 0.0
    heat_capacity_j_kgk = 800.0
    key = "black"
    tones = ((0, 0, 0), (0, 0, 0), (0, 0, 0))
    vapour_temperature_k = 1e9


def test_a_black_body_at_one_au_sits_at_the_textbook_temperature():
    """T = T_star * sqrt(R_star / 2d).  278.3 K is the number every textbook
    quotes for Earth before albedo, and it is what falls out of the sum."""
    world = _sun_and(ROCK, 1.0)
    world.bodies[1].material = _Black()
    assert thermal.equilibrium_temperature(world, 1) == pytest.approx(278.3, abs=0.5)


def test_temperature_falls_as_the_square_root_of_distance():
    """Flux goes as 1/d^2 and temperature as its fourth root, so four times
    further is half as hot."""
    near = _sun_and(ROCK, 1.0)
    far = _sun_and(ROCK, 4.0)
    assert (thermal.equilibrium_temperature(far, 1)
            == pytest.approx(0.5 * thermal.equilibrium_temperature(near, 1), rel=1e-3))


def test_albedo_cools_a_body_by_the_fourth_root_of_what_it_reflects():
    world = _sun_and(ROCK, 1.0)
    black = _sun_and(ROCK, 1.0)
    black.bodies[1].material = _Black()
    expected = (1.0 - ROCK.albedo) ** 0.25
    assert (thermal.equilibrium_temperature(world, 1)
            == pytest.approx(expected * thermal.equilibrium_temperature(black, 1),
                             rel=1e-6))


def test_an_icy_world_is_colder_than_a_rocky_one_at_the_same_distance():
    rocky = _sun_and(ROCK, 3.0)
    icy = _sun_and(ICE, 3.0)
    assert thermal.equilibrium_temperature(icy, 1) < thermal.equilibrium_temperature(rocky, 1)


def test_two_suns_add_their_light_not_their_temperatures():
    """Fluxes add; temperatures do not.  Two identical stars make a body
    2**0.25 = 1.19 times hotter, not twice."""
    one = _sun_and(ROCK, 1.0)
    two = _sun_and(ROCK, 1.0)
    two.add("Sun B", HYDROGEN, 1.0, units.SUN_RADIUS,
            np.array([0.0, 1.0e-9]), np.zeros(2), kind=KIND_STAR)
    assert (thermal.equilibrium_temperature(two, 1)
            == pytest.approx(2.0 ** 0.25 * thermal.equilibrium_temperature(one, 1),
                             rel=1e-3))


def test_a_warm_planet_heats_its_moon():
    """The same sum that makes the Sun heat Mercury.  A 288 K Earth puts about
    26 K on the Moon at its real distance -- small, real, and free: it needs
    no code of its own."""
    world = World(8)
    world.add("Earth", ROCK, units.EARTH_MASS, units.EARTH_RADIUS,
              np.zeros(2), np.zeros(2), temperature=288.0)
    world.bodies[0].material = _Black()
    world.add("Moon", ROCK, units.MOON_MASS, 1737.4 / units.AU_KM,
              np.array([384400.0 / units.AU_KM, 0.0]), np.zeros(2))
    world.bodies[1].material = _Black()
    assert thermal.equilibrium_temperature(world, 1) == pytest.approx(26.2, abs=1.0)


def test_a_molten_planet_heats_its_moon_far_harder():
    world = World(8)
    world.add("Earth", ROCK, units.EARTH_MASS, units.EARTH_RADIUS,
              np.zeros(2), np.zeros(2), temperature=1800.0)
    world.bodies[0].material = _Black()
    world.add("Moon", ROCK, units.MOON_MASS, 1737.4 / units.AU_KM,
              np.array([384400.0 / units.AU_KM, 0.0]), np.zeros(2))
    world.bodies[1].material = _Black()
    assert thermal.equilibrium_temperature(world, 1) == pytest.approx(163.9, abs=3.0)


def test_a_rogue_body_between_the_stars_sits_at_the_microwave_background():
    """Three light years out.

    The approach to the floor is far slower than it looks: flux falls as 1/d^2
    but temperature is its fourth root, so a body is still 0.9% above the
    background at 50,000 AU -- five times past the distance where the Sun's
    contribution and the CMB's are equal -- and only settles onto it out here.
    """
    world = _sun_and(ROCK, 200000.0)
    assert (thermal.equilibrium_temperature(world, 1)
            == pytest.approx(units.CMB_TEMPERATURE_K, rel=1e-3))


def test_nothing_is_ever_colder_than_the_microwave_background():
    """The floor is a floor at every distance, not only in the limit."""
    for distance in (1.0, 100.0, 5000.0, 50000.0, 1.0e7):
        world = _sun_and(ICE, distance)
        assert thermal.equilibrium_temperature(world, 1) >= units.CMB_TEMPERATURE_K


def test_a_star_is_pinned_to_the_temperature_its_mass_gives_it():
    world = _sun_and(ROCK, 1.0)
    assert thermal.pinned_temperature(world, 0) == pytest.approx(
        units.star_temperature(1.0))
    assert thermal.pinned_temperature(world, 1) is None


def test_a_black_hole_is_pinned_to_its_hawking_temperature():
    world = World(4)
    world.add("Hole", materials.SINGULARITY, 10.0,
              units.schwarzschild_radius(10.0), np.zeros(2), np.zeros(2),
              kind=KIND_BLACK_HOLE)
    assert thermal.pinned_temperature(world, 0) == pytest.approx(
        units.hawking_temperature_k(10.0))


def test_a_black_hole_contributes_no_meaningful_heat_to_anything():
    """No special case needed -- 6e-8 K over a horizon a few km wide simply
    evaluates to nothing."""
    world = World(4)
    world.add("Hole", materials.SINGULARITY, 10.0,
              units.schwarzschild_radius(10.0), np.zeros(2), np.zeros(2),
              kind=KIND_BLACK_HOLE)
    world.add("Rock", ROCK, units.EARTH_MASS, units.EARTH_RADIUS,
              np.array([0.01, 0.0]), np.zeros(2))
    assert thermal.equilibrium_temperature(world, 1) == pytest.approx(
        units.CMB_TEMPERATURE_K, rel=1e-6)


def test_equilibrium_of_an_empty_world_is_an_empty_array():
    assert thermal.equilibrium_temperatures(World(4)).shape == (0,)


# --- relaxation ----------------------------------------------------------------

def test_a_hot_rock_near_earths_orbit_cools_over_days_not_millennia():
    """The whole point of the skin: heating a body right through gives a time
    constant near a hundred thousand years, which no one would ever see."""
    world = _sun_and(ROCK, 1.0)
    world.temperature[1] = 255.0
    tau = thermal.relaxation_days(world, 1, 255.0)
    assert 5.0 < tau < 20.0


def test_a_glowing_body_cools_far_faster_than_a_frozen_one():
    """Radiated power goes as T^4, so tau goes as 1/T^3.  Stefan-Boltzmann,
    not a tuning choice."""
    world = _sun_and(ROCK, 1.0)
    world.temperature[1] = 1500.0
    hot = thermal.relaxation_days(world, 1, 1500.0)
    world.temperature[1] = 50.0
    cold = thermal.relaxation_days(world, 1, 50.0)
    assert cold > 1000.0 * hot


def test_a_denser_body_holds_its_heat_longer():
    """rho is the body's own current density, so squeezing a planet in the
    Edit tool really does change how long it stays warm."""
    light = _sun_and(ROCK, 1.0, radius=units.EARTH_RADIUS)
    heavy = _sun_and(ROCK, 1.0, radius=units.EARTH_RADIUS * 0.5)
    light.temperature[1] = heavy.temperature[1] = 300.0
    assert (thermal.relaxation_days(heavy, 1, 300.0)
            > thermal.relaxation_days(light, 1, 300.0))


def test_relaxation_never_overshoots_its_target():
    """Whatever the step, a body lands between where it started and where it is
    heading -- never past.  An overshoot would mean a body cooling below what
    the sunlight on it supports and then oscillating, which is the failure mode
    an explicit Euler step has and the analytic one cannot.
    """
    world = _sun_and(ROCK, 1.0)
    target = thermal.equilibrium_temperature(world, 1)
    for start in (3000.0, 1000.0, 300.0, 10.0):
        for step in (0.001, 0.1, 1.0, 100.0, 1.0e6):
            world.temperature[1] = start
            thermal.advance(world, step)
            landed = float(world.temperature[1])
            lo, hi = min(start, target), max(start, target)
            assert lo - 1e-9 <= landed <= hi + 1e-9, (start, step, landed)


def test_a_hot_body_cools_toward_equilibrium_step_by_step():
    """Strict monotonicity needs a step well inside tau.  Note what tau is
    here: 400 K rock relaxes over about three and a half days, so a 0.05-day
    step moves it roughly 1.4% of the remaining gap.  Take a *whole day* at
    3000 K instead and tau is 0.008 days -- one step is 120 e-foldings, the
    body lands on equilibrium to the last bit of a float64, and there is no
    monotonic tail left to check.
    """
    world = _sun_and(ROCK, 1.0)
    target = thermal.equilibrium_temperature(world, 1)
    world.temperature[1] = 400.0
    previous = 400.0
    for _ in range(20):
        thermal.advance(world, 0.05)
        current = float(world.temperature[1])
        assert current < previous
        assert current > target
        previous = current


def test_a_cold_body_near_a_star_warms_up_instead_of_staying_frozen():
    """tau is linearised at whichever end of the exchange is hotter.  Taken at
    the body's own temperature instead, a body dropped to 3 K next to the Sun
    would take eighteen thousand years to notice."""
    world = _sun_and(ROCK, 1.0)
    world.temperature[1] = 3.0
    thermal.advance(world, 30.0)
    assert float(world.temperature[1]) > 200.0


def test_a_huge_step_lands_exactly_on_equilibrium_and_does_not_explode():
    """The step is integrated analytically, so dt >> tau collapses to T = T_eq.
    An explicit Euler step would need dt < tau, and tau is under an hour for
    anything glowing -- at ten years a second the sandbox would blow up."""
    world = _sun_and(ROCK, 1.0)
    target = thermal.equilibrium_temperature(world, 1)
    world.temperature[1] = 90000.0
    thermal.advance(world, 1.0e6)
    assert float(world.temperature[1]) == pytest.approx(target, rel=1e-6)


def test_settle_puts_everything_at_equilibrium_and_is_a_fixed_point():
    world = _sun_and(ROCK, 1.0)
    thermal.settle(world)
    settled = float(world.temperature[1])
    assert settled == pytest.approx(thermal.equilibrium_temperature(world, 1), rel=1e-9)
    thermal.advance(world, 100.0)
    assert float(world.temperature[1]) == pytest.approx(settled, rel=1e-9)


def test_settling_the_solar_system_reduces_to_the_textbook_law():
    """The end-to-end check, and the one worth making: the whole-world O(n^2)
    sum has to agree with `T = T_star sqrt(R_star / 2d) (1-A)^(1/4)` for every
    planet at the distance it actually is.

    Distances are the J2000 positions, not the semi-major axes: Mercury sits
    near aphelion at 0.4665 AU and comes out at 385 K, where its *average*
    distance would give 440 K.  Quoting a remembered textbook number here
    rather than the body's real distance is how you get a test that is wrong
    and confident about it.

    rel=1e-4 rather than tighter because the expected value below leaves out
    the microwave background, which is a few parts per million of Neptune's
    temperature and less than that for everything closer in.
    """
    from src import scenarios
    world = scenarios.solar_system(moons=False)
    thermal.settle(world)
    for i in range(1, world.n):
        body = world.bodies[i]
        d = float(np.hypot(*(world.pos[i] - world.pos[0])))
        expected = (units.star_temperature(float(world.mass[0]))
                    * math.sqrt(units.SUN_RADIUS / (2.0 * d))
                    * (1.0 - body.material.albedo) ** 0.25)
        assert float(world.temperature[i]) == pytest.approx(expected, rel=1e-4), body.name


def test_the_inner_planets_come_out_hotter_than_the_outer_ones():
    from src import scenarios
    world = scenarios.solar_system(moons=False)
    thermal.settle(world)
    by_name = {b.name: float(world.temperature[world.index_of(b.id)])
               for b in world.bodies}
    assert by_name["Mercury"] > by_name["Earth"] > by_name["Jupiter"] > by_name["Neptune"]
    assert 250.0 < by_name["Earth"] < 290.0
    assert 30.0 < by_name["Neptune"] < 50.0


def test_a_star_never_relaxes_however_long_it_runs():
    world = _sun_and(ROCK, 1.0)
    thermal.advance(world, 1.0e6)
    assert float(world.temperature[0]) == pytest.approx(units.star_temperature(1.0))


def test_advancing_by_nothing_changes_nothing():
    world = _sun_and(ROCK, 1.0)
    thermal.settle(world)
    before = world.temperature[:world.n].copy()
    thermal.advance(world, 0.0)
    assert np.allclose(world.temperature[:world.n], before)


def test_temperature_never_leaves_the_allowed_range():
    world = _sun_and(ROCK, 0.005)      # inside the Sun's own radius
    thermal.settle(world)
    for _ in range(20):
        thermal.advance(world, 10.0)
    assert units.MIN_TEMPERATURE_K <= float(world.temperature[1]) <= units.MAX_TEMPERATURE_K


def test_a_locked_temperature_does_not_relax():
    """The point of the lock: a child pins a temperature and it has to actually
    stay put, not just refuse edits.set_temperature and then slide straight
    back to equilibrium on the next frame.  edits.py is not the only writer --
    advance() is the other one, and it has to honour the same lock."""
    world = _sun_and(ROCK, 1.0)
    world.temperature[1] = 900.0
    world.bodies[1].locked_fields = frozenset({"temp"})
    thermal.advance(world, 10.0)
    assert float(world.temperature[1]) == pytest.approx(900.0)


def test_an_unlocked_neighbour_still_relaxes_normally():
    """The lock reaches only the locked body -- a neighbour with nothing
    locked keeps behaving exactly as it always has."""
    world = _sun_and(ROCK, 1.0)
    world.temperature[1] = 900.0
    thermal.advance(world, 10.0)
    assert float(world.temperature[1]) != pytest.approx(900.0)


def test_a_stale_temperature_lock_does_not_stop_a_star_being_pinned():
    """A planet that locked its temperature and then ignited keeps the lock --
    it is simply inert for a kind that offers no Temp row.  ``advance`` must
    still pin it to its fusion temperature, not freeze it at the last planetary
    reading, the same way ``edits.set_temperature`` already refuses a star
    outright regardless of any lock."""
    world = World(16)
    world.add("Star", HYDROGEN, 1.0, units.SUN_RADIUS,
              np.zeros(2), np.zeros(2), kind=KIND_STAR, temperature=400.0)
    world.bodies[0].locked_fields = frozenset({"temp"})
    thermal.advance(world, 10.0)
    assert float(world.temperature[0]) == pytest.approx(units.star_temperature(1.0))


# --- impact heating ------------------------------------------------------------

def test_impact_heating_scales_with_energy_and_against_heat_capacity():
    a = thermal.impact_heating(ROCK, 1.0)
    b = thermal.impact_heating(ROCK, 2.0)
    assert b == pytest.approx(2.0 * a)
    water = thermal.impact_heating(materials.get("water"), 1.0)
    assert water == pytest.approx(
        a * ROCK.heat_capacity_j_kgk / materials.get("water").heat_capacity_j_kgk)


def test_a_giant_impact_melts_a_planet():
    """About 5e6 J/kg for the Moon-forming impact, which is roughly 1900 K into
    rock.  Earth really was a magma ocean afterwards."""
    q_r = 5.0e6 / units.SPECIFIC_ENERGY_SI
    assert thermal.impact_heating(ROCK, q_r) == pytest.approx(1875.0, rel=0.1)


def test_a_singularity_cannot_be_heated_by_an_impact():
    assert thermal.impact_heating(materials.SINGULARITY, 1.0e9) == 0.0


def test_a_merge_mixes_the_two_temperatures_by_mass_and_adds_the_impact_heat():
    # collisions.Collision has no `distance`/`impact_parameter` keywords (see
    # its definition near ImpactEvent in src/collisions.py) -- build a real
    # pair through find_collisions instead, as the plan's own fallback allows.
    from src import collisions
    world = World(8)
    world.add("Big", ROCK, units.EARTH_MASS, units.EARTH_RADIUS,
              np.zeros(2), np.zeros(2), temperature=300.0)
    world.add("Small", ROCK, units.EARTH_MASS * 0.1, units.EARTH_RADIUS * 0.46,
              np.array([units.EARTH_RADIUS * 1.4, 0.0]),
              np.array([-0.005, 0.0]), temperature=100.0)
    hits = collisions.find_collisions(world, world.pos[:world.n].copy())
    assert hits, "expected a detected collision"
    c = hits[0]
    mixed = (units.EARTH_MASS * 300.0 + units.EARTH_MASS * 0.1 * 100.0) \
        / (units.EARTH_MASS * 1.1)
    heat = thermal.impact_heating(ROCK, collisions.specific_impact_energy(world, c))
    idx = collisions.merge(world, c)
    assert float(world.temperature[idx]) == pytest.approx(mixed + heat, rel=1e-6)


def test_an_impact_makes_the_remnant_hotter_than_either_body_was():
    from src import collisions
    world = World(8)
    world.add("Big", ROCK, units.EARTH_MASS, units.EARTH_RADIUS,
              np.zeros(2), np.zeros(2), temperature=300.0)
    world.add("Small", ROCK, units.EARTH_MASS * 0.1, units.EARTH_RADIUS * 0.46,
              np.array([units.EARTH_RADIUS * 1.4, 0.0]),
              np.array([-0.01, 0.0]), temperature=100.0)
    hits = collisions.find_collisions(world, world.pos[:world.n].copy())
    assert hits, "expected a detected collision"
    c = hits[0]
    idx = collisions.merge(world, c)
    assert float(world.temperature[idx]) > 300.0


def test_tidal_fragments_inherit_the_temperature_of_what_they_were():
    from src import collisions
    world = World(32)
    world.add("Sun", HYDROGEN, 1.0, units.SUN_RADIUS, np.zeros(2), np.zeros(2),
              kind=KIND_STAR)
    world.add("Doomed", ROCK, units.EARTH_MASS, units.EARTH_RADIUS,
              np.array([0.02, 0.0]), np.zeros(2), temperature=900.0)
    resolver = collisions.CollisionResolver(seed=1)
    made = resolver.tidal_disrupt(world, 1, 0)
    assert made, "expected fragments"
    for k in made:
        assert float(world.temperature[k]) == pytest.approx(900.0)


def test_no_collision_outcome_leaves_a_body_at_absolute_zero():
    """Every path that creates a body has to give it a temperature.  A fragment
    left at 0.0 would be colder than the universe allows and would then warm
    up from a temperature it was never at."""
    from src import collisions
    world = World(64)
    world.add("A", ROCK, units.EARTH_MASS, units.EARTH_RADIUS,
              np.zeros(2), np.zeros(2), temperature=400.0)
    world.add("B", ROCK, units.EARTH_MASS * 0.8, units.EARTH_RADIUS * 0.93,
              np.array([units.EARTH_RADIUS * 1.9, 0.0]),
              np.array([-0.05, 0.0]), temperature=400.0)
    hits = collisions.find_collisions(world, world.pos[:world.n].copy())
    assert hits, "expected a detected collision"
    c = hits[0]
    resolver = collisions.CollisionResolver(seed=3)
    resolver.shatter(world, c)
    assert world.n > 0
    for k in range(world.n):
        assert float(world.temperature[k]) >= units.MIN_TEMPERATURE_K
