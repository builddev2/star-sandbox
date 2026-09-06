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
    report = explain.diff(before, after, id_of(world, "Sun"), "mass", max_ripple=8)

    earth = next(c for c in report.changes if c.name == "Earth")
    lap = rows_of(earth)["One lap"]
    assert lap.before == pytest.approx(365.3, abs=1.0)
    assert lap.after == pytest.approx(138.2, abs=1.0)
    stretch = rows_of(earth)["Stretch"]
    assert stretch.before == pytest.approx(0.0167, abs=0.002)
    assert stretch.after == pytest.approx(0.492, abs=0.01)


def test_the_edited_body_leads_and_the_ripple_is_ranked(world):
    """The Sun first, then MAX_RIPPLE planets in non-increasing order of how
    much they moved.

    Deliberately not an exact roster.  Doubling the Sun hits every planet
    almost identically -- every period falls by a factor near 0.38 and every
    orbit ends up near e = 0.5 -- so which three come out on top is settled in
    the third decimal place, by where the planets happened to be at the J2000
    epoch.  Pinning the roster would be pinning a coincidence, and the next
    person to touch the ranking would "fix" the code to match it.
    """
    before, after = double_the_sun(world)
    report = explain.diff(before, after, id_of(world, "Sun"), "mass")

    assert report.changes[0].name == "Sun"
    assert report.changes[0].is_edited is True

    ripple = report.changes[1:]
    assert len(ripple) == explain.MAX_RIPPLE
    assert all(c.is_edited is False for c in ripple)
    others = {b.name for b in world.bodies[:world.n]} - {"Sun"}
    assert {c.name for c in ripple} <= others
    scores = [explain._score(c) for c in ripple]
    assert scores == sorted(scores, reverse=True)


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


# --- why ----------------------------------------------------------------------------

def test_doubling_the_sun_explains_the_ripple_not_the_edit(world):
    report = explain.diff(*double_the_sun(world), id_of(world, "Sun"), "mass")
    said = " ".join(report.why)
    assert "heavier Sun" in said
    assert "pulls harder" in said
    assert "oval" in said


def test_collapse_says_the_orbits_did_not_move(world):
    """The most valuable line in the table: mass is conserved, so the ripple is
    empty, and the sentence is what makes an empty ripple the right answer."""
    from src import edits
    before = explain.snapshot(world)
    i = index_of_name(world, "Sun")
    edits.collapse(world, i)
    report = explain.diff(before, explain.snapshot(world),
                          id_of(world, "Sun"), "collapse")
    said = " ".join(report.why)
    assert "mass did not change" in said
    assert "nothing orbiting it moved" in said


def test_heating_a_world_past_boiling_says_so(world):
    from src import edits
    before = explain.snapshot(world)
    i = index_of_name(world, "Earth")
    edits.set_temperature(world, i, 500.0)
    report = explain.diff(before, explain.snapshot(world),
                          id_of(world, "Earth"), "temp")
    said = " ".join(report.why)
    assert "water boils" in said


def test_every_temperature_edit_admits_it_is_not_pinned(world):
    """edits.set_temperature's docstring calls the relaxation "the whole
    lesson", and nothing on screen has ever said it."""
    from src import edits
    before = explain.snapshot(world)
    i = index_of_name(world, "Earth")
    edits.set_temperature(world, i, 500.0)
    report = explain.diff(before, explain.snapshot(world),
                          id_of(world, "Earth"), "temp")
    said = " ".join(report.why)
    assert "nothing pins it there" in said


def test_spinning_a_hole_produces_a_report_at_all(world):
    """set_spin moves nothing but Body.spin, so this is the edit most likely
    to diff to nothing and leave the panel silent."""
    from src import edits
    i = index_of_name(world, "Sun")
    edits.collapse(world, i)
    before = explain.snapshot(world)
    edits.set_spin(world, i, 0.9)
    report = explain.diff(before, explain.snapshot(world),
                          id_of(world, "Sun"), "spin")
    assert report is not None
    assert any(r.label == "Spin" for r in report.changes[0].rows)
    assert "drags space" in " ".join(report.why)


def test_heating_a_world_reports_the_temperature_he_actually_set(world):
    """Nothing heats itself, so a temperature edit leaves the edited body's
    equilibrium bit-identical.  Reporting that row would diff to nothing and
    the panel would stay silent on the one edit it has most to say about.
    """
    from src import edits
    i = index_of_name(world, "Earth")
    before = explain.snapshot(world)
    edits.set_temperature(world, i, 500.0)
    after = explain.snapshot(world)

    earth_id = id_of(world, "Earth")
    assert (after.bodies[earth_id].eq_temperature
            == before.bodies[earth_id].eq_temperature)

    report = explain.diff(before, after, earth_id, "temp")
    assert report is not None
    temp = next(r for r in report.changes[0].rows if r.label == "Temp")
    assert temp.after == pytest.approx(500.0)


def test_role_is_decided_by_id_when_two_bodies_share_a_name(world):
    """Collision fragments are all minted "Ejecta" or "Debris"
    (src/collisions.py:488 and :545), so several bodies in one world
    routinely share a name.  Deciding the edited body's role by name would
    read some unrelated same-named body's orbit instead of the one the
    report is actually about, and print "a heavier Sun pulls harder" about
    a body the Sun does not hold.

    Built directly rather than through a collision: the point is two bodies
    with one name, and reaching that state through the physics would take a
    shatter and a great deal of setup to say the same thing.
    """
    import dataclasses

    snap = explain.snapshot(world)
    sun_id = id_of(world, "Sun")
    earth_id = id_of(world, "Earth")
    mercury_id = id_of(world, "Mercury")

    # Two bodies called "Debris".  The one the Change is about is bound to
    # Mercury; the decoy, sharing its name, is bound to the Sun.
    loose = dataclasses.replace(snap.bodies[earth_id], name="Debris",
                                attractor_id=mercury_id)
    decoy = dataclasses.replace(snap.bodies[mercury_id], name="Debris",
                                attractor_id=sun_id)
    after = explain.Snapshot(bodies={sun_id: snap.bodies[sun_id],
                                     earth_id: loose, mercury_id: decoy})
    moved = (explain.Row("One lap", 365.0, 300.0, "period"),)

    # The Change is about the loose one, which the Sun does not hold -- so
    # the role is "self", however many same-named bodies the Sun does hold.
    about_loose = explain.Change(body_id=earth_id, name="Debris",
                                 is_edited=False, rows=moved)
    assert explain._role(after, sun_id, (about_loose,)) == "self"

    # And the decoy, which really is the Sun's, still gives the other answer
    # -- so this cannot be passed by a _role that always says "self".
    about_decoy = dataclasses.replace(about_loose, body_id=mercury_id)
    assert explain._role(after, sun_id, (about_decoy,)) == "attractor"


def test_no_edit_falls_through_to_silence():
    """Every (field, direction, role) an edit can produce must find a line.
    A report with no explanation is the failure mode this whole feature exists
    to fix, so it must not be reachable."""
    fields = ("mass", "density", "temp", "material", "spin", "disk", "collapse")
    for field in fields:
        for direction in ("up", "down"):
            for role in ("attractor", "self"):
                assert explain.lookup_why(field, direction, role), \
                    "no line for %s/%s/%s" % (field, direction, role)
