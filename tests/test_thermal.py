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
