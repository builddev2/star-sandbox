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
