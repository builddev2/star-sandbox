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


# --- the diff -----------------------------------------------------------------------

def double_the_sun(world):
    """The worked example the spec is built around.  Returns (before, after)."""
    from src import edits
    before = explain.snapshot(world)
    i = index_of_name(world, "Sun")
    edits.set_mass(world, i, 2.0 * float(world.mass[i]))
    return before, explain.snapshot(world)


def rows_of(change):
    return {row.label: row for row in change.rows}


def test_doubling_the_sun_reports_earths_real_new_year(world):
    """These numbers came out of the simulation, not out of a textbook: an
    instantaneous doubling leaves Earth's position and speed alone, so the
    circle becomes an ellipse with the old radius as its farthest point."""
    before, after = double_the_sun(world)
    report = explain.diff(before, after, id_of(world, "Sun"), "mass")

    earth = next(c for c in report.changes if c.name == "Earth")
    lap = rows_of(earth)["One lap"]
    assert lap.before == pytest.approx(365.3, abs=1.0)
    assert lap.after == pytest.approx(138.2, abs=1.0)
    stretch = rows_of(earth)["Stretch"]
    assert stretch.before == pytest.approx(0.0167, abs=0.002)
    assert stretch.after == pytest.approx(0.492, abs=0.01)


def test_the_edited_body_leads_and_the_ripple_is_ranked(world):
    before, after = double_the_sun(world)
    report = explain.diff(before, after, id_of(world, "Sun"), "mass")

    assert report.changes[0].name == "Sun"
    assert report.changes[0].is_edited is True
    # Three rippled bodies, ordered by how much they moved -- not by index.
    assert [c.name for c in report.changes[1:]] == ["Mercury", "Venus", "Earth"]
    assert all(c.is_edited is False for c in report.changes[1:])


def test_the_edited_bodys_own_rows_are_the_ones_that_moved(world):
    before, after = double_the_sun(world)
    report = explain.diff(before, after, id_of(world, "Sun"), "mass")
    sun = rows_of(report.changes[0])
    assert "Mass" in sun
    assert sun["Mass"].after == pytest.approx(2.0 * sun["Mass"].before, rel=1e-9)
    assert sun["Mass"].format == "mass"
    # Density was held constant, so radius followed mass.  Density did not move
    # and must not be listed.
    assert "Density" not in sun


def test_editing_a_small_body_ripples_to_nobody(world):
    from src import edits
    before = explain.snapshot(world)
    i = index_of_name(world, "Mercury")
    edits.set_mass(world, i, 1.5 * float(world.mass[i]))
    after = explain.snapshot(world)
    report = explain.diff(before, after, id_of(world, "Mercury"), "mass")
    assert [c.name for c in report.changes] == ["Mercury"]


def test_an_edit_that_moved_nothing_is_no_report(world):
    before = explain.snapshot(world)
    after = explain.snapshot(world)
    assert explain.diff(before, after, id_of(world, "Sun"), "mass") is None


def test_a_body_deleted_before_the_burst_settled_is_no_report(world):
    """Snapshots are keyed by id, so the edited body is simply absent from the
    second one.  Reporting on a body that is gone is worse than saying nothing."""
    before = explain.snapshot(world)
    gone = id_of(world, "Mars")
    world.remove(index_of_name(world, "Mars"))
    after = explain.snapshot(world)
    assert explain.diff(before, after, gone, "mass") is None


def test_the_report_survives_the_world_running_underneath_it(world):
    """The claim the whole design rests on: elements are invariant under motion
    along the orbit, so the sim advancing between baseline and report cannot
    contaminate the diff.

    A tolerance, not equality: osculating elements really do drift a little
    under n-body perturbation -- Jupiter pulls Earth's around over a hundred
    days -- and a test asserting otherwise would assert a falsehood and fail at
    random.  1% is far tighter than the 62% change being reported.
    """
    from src.sim import Simulation
    sim = Simulation(world)
    still = explain.diff(*double_the_sun(world), id_of(world, "Sun"), "mass")

    world2 = scenarios.solar_system(moons=False)
    before = explain.snapshot(world2)
    from src import edits
    i = index_of_name(world2, "Sun")
    edits.set_mass(world2, i, 2.0 * float(world2.mass[i]))
    sim2 = Simulation(world2)
    sim2.refresh()
    sim2.advance(100.0)
    moved = explain.diff(before, explain.snapshot(world2),
                         id_of(world2, "Sun"), "mass")

    a = rows_of(next(c for c in still.changes if c.name == "Earth"))
    b = rows_of(next(c for c in moved.changes if c.name == "Earth"))
    assert b["One lap"].after == pytest.approx(a["One lap"].after, rel=0.01)
    assert b["Stretch"].after == pytest.approx(a["Stretch"].after, rel=0.01)


def test_temperature_rows_use_equilibrium_not_the_instantaneous_value(world):
    """And a guard on the guard: assert that reading world.temperature instead
    *would* have moved, so swapping the source back fails this test."""
    from src import edits
    from src.sim import Simulation
    i = index_of_name(world, "Earth")
    edits.set_temperature(world, i, 3000.0)
    settled = explain.snapshot(world)
    hot = float(world.temperature[i])

    sim = Simulation(world)
    sim.refresh()
    sim.advance(30.0)
    thermal.advance(world, 30.0)
    later = explain.snapshot(world)

    earth_id = id_of(world, "Earth")
    assert later.bodies[earth_id].eq_temperature == pytest.approx(
        settled.bodies[earth_id].eq_temperature, rel=0.01)
    # The instantaneous value, meanwhile, has fallen a long way.
    assert later.bodies[earth_id].temperature < 0.9 * hot
