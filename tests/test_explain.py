"""The explain layer: what an edit did, and why.

Pure -- no pygame anywhere in this file, because src/explain.py is physics-tier
and tests/test_layering.py enforces that.
"""

import numpy as np
import pytest

from src import explain, orbits, scenarios, thermal, units


@pytest.fixture
def world():
    """Planets only.  Moons drive the timestep down by two orders of magnitude
    and nothing here needs them."""
    return scenarios.solar_system(moons=False)


def index_of_name(world, name):
    return [b.name for b in world.bodies].index(name)


def id_of(world, name):
    return world.bodies[index_of_name(world, name)].id


def test_snapshot_records_every_body_by_id(world):
    snap = explain.snapshot(world)
    assert len(snap.bodies) == world.n
    for body in world.bodies[:world.n]:
        assert body.id in snap.bodies
        assert snap.bodies[body.id].name == body.name


def test_snapshot_records_the_real_orbit(world):
    earth = explain.snapshot(world).bodies[id_of(world, "Earth")]
    assert earth.attractor == "Sun"
    assert earth.bound is True
    assert earth.a == pytest.approx(1.0, abs=0.01)
    assert earth.e == pytest.approx(0.0167, abs=0.001)
    assert earth.period == pytest.approx(365.3, abs=1.0)


def test_the_sun_gets_a_pinned_temperature_not_an_equilibrium_one(world):
    """thermal.equilibrium_temperatures is meaningless for a radiator: it
    computes what a body absorbs, and a star's temperature comes from fusion.
    pinned_temperature is the one function that knows which bodies are held."""
    sun = explain.snapshot(world).bodies[id_of(world, "Sun")]
    expected = units.star_temperature(float(world.mass[index_of_name(world, "Sun")]))
    assert sun.eq_temperature == pytest.approx(expected, rel=1e-9)
    assert sun.eq_temperature > 5000.0


def test_snapshot_carries_the_substance_and_the_hole_dials(world):
    """Every field a later task reads is populated here, so no task has to
    retrofit the record it was handed."""
    earth = explain.snapshot(world).bodies[id_of(world, "Earth")]
    assert earth.material == "Rock"
    assert earth.material_key == "rock"
    assert earth.vapour_k > 1000.0
    # Not a hole, so both dials read zero rather than being absent.
    assert earth.spin == 0.0 and earth.disk == 0.0


def test_an_unbound_body_has_no_period(world):
    """period is None out of elements_from_state; a Row needs a float, so an
    escaping body reports 0.0 and bound=False rather than crashing the diff."""
    i = index_of_name(world, "Mercury")
    world.vel[i] = world.vel[i] * 5.0        # far past escape
    mercury = explain.snapshot(world).bodies[id_of(world, "Mercury")]
    assert mercury.bound is False
    assert mercury.period == 0.0
