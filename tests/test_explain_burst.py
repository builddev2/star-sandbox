"""One report per burst, not one per scrub notch.

Mass scrubs by ratio at 1.06 a notch, three notches a press, repeating while
held -- doubling the Sun is a dozen-plus separate set_mass calls, and a report
per call would be a strobe.
"""

import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import pygame
import pytest

from src.app.state import AppState, SETTLE_MS
from src.app.tools import EditTool
from src.render.camera import Camera
from src.render.draw import ViewSettings
from src.sim import Simulation


@pytest.fixture(scope="module", autouse=True)
def _pygame():
    # pygame.time.get_ticks() is frozen at 0 until pygame.init() runs, and
    # EditTool._scrub stamps its own note() call with that real clock -- so
    # without this, every burst it opens closes on the first settle() check.
    # Every other pygame-touching test file in this suite has this fixture
    # (test_state.py, test_tools_edit.py, ...); this one needs it too.
    pygame.init()
    yield
    pygame.quit()


@pytest.fixture
def state():
    from src import scenarios
    # Planets only: six moons drive the timestep down two orders of magnitude
    # and nothing here needs them.
    return AppState(Simulation(scenarios.solar_system(moons=False)),
                    Camera(800, 600), ViewSettings())


def select(state, name):
    world = state.sim.world
    index = [b.name for b in world.bodies].index(name)
    state.select(world.bodies[index].id)
    return index


def test_a_held_stepper_produces_exactly_one_report(state):
    """Twelve notches, one lesson.

    Nothing here calls note(): _scrub does that itself, on the real clock.
    The test owns only settle()'s timestamp, which is the one App.step_once
    supplies.
    """
    state.set_tool(EditTool)
    select(state, "Sun")
    for _ in range(12):
        state.tool._scrub(state, 3)

    # Still mid-burst: nothing has gone quiet.
    state.explain.settle(state.sim.world, pygame.time.get_ticks())
    assert state.explain.report is None

    # Quiet for longer than the settle interval, and one report lands.
    state.explain.settle(state.sim.world,
                         pygame.time.get_ticks() + SETTLE_MS + 1)
    assert state.explain.report is not None
    assert state.explain.report.changes[0].name == "Sun"
    mass = [r for r in state.explain.report.changes[0].rows if r.label == "Mass"]
    assert mass and mass[0].after > 1.5 * mass[0].before, \
        "the baseline was a later notch, not the state he started from"


def test_changing_field_mid_hold_closes_the_first_burst(state):
    """A different property is a different lesson, and does not wait for quiet."""
    state.set_tool(EditTool)
    select(state, "Sun")
    state.tool.field_index = 0          # mass
    state.tool._scrub(state, 12)
    assert state.explain.report is None

    state.tool.field_index = 1          # density
    state.tool._scrub(state, 1)
    assert state.explain.report is not None
    assert any(r.label == "Mass" for r in state.explain.report.changes[0].rows)


def test_a_burst_that_never_settles_reports_nothing(state):
    state.set_tool(EditTool)
    select(state, "Sun")
    state.tool._scrub(state, 3)
    state.explain.settle(state.sim.world, pygame.time.get_ticks())
    assert state.explain.report is None


def test_turning_explanations_off_takes_no_snapshot_at_all(state):
    """Off means off -- not recording silently and hiding the result."""
    state.explain.enabled = False
    state.set_tool(EditTool)
    select(state, "Sun")
    state.tool._scrub(state, 12)
    assert state.explain._baseline is None
    state.explain.settle(state.sim.world,
                         pygame.time.get_ticks() + SETTLE_MS + 1)
    assert state.explain.report is None


def test_restart_drops_a_standing_report_and_an_open_burst(state):
    """F5 clears the world and rebuilds every Body, so a standing report and
    an open burst both name bodies that no longer exist."""
    state.set_tool(EditTool)
    select(state, "Sun")
    state.tool._scrub(state, 12)
    state.explain.settle(state.sim.world,
                         pygame.time.get_ticks() + SETTLE_MS + 1)
    assert state.explain.report is not None

    state.reset()
    assert state.explain.report is None
    assert state.explain._baseline is None


def test_scrubbing_through_the_tool_records_without_the_test_calling_note(state):
    """The contract stated plainly: EditTool._scrub calls note() itself, or
    nothing the child does through the buttons or the arrow keys is ever
    explained."""
    state.set_tool(EditTool)
    select(state, "Sun")
    state.tool._scrub(state, 12)
    assert state.explain._baseline is not None


def test_the_world_running_during_a_burst_is_not_blamed_on_the_edit(state):
    """The sim advances between the last write and the settle -- 1278
    simulated days at ten years a second.  Orbital elements survive that;
    temperature does not, and the edited body's Temp row is instantaneous.
    Before this was fixed, heating Earth to 3115 K reported "You cooled
    Earth." at every time scale, and named planets the edit never touched.
    """
    from src import edits, thermal
    state.set_tool(EditTool)
    world = state.sim.world
    i = [b.name for b in world.bodies].index("Earth")
    state.select(world.bodies[i].id)
    state.tool.field_index = list(edits.edit_fields(world.bodies[i])).index("temp")
    for _ in range(14):
        state.tool._scrub(state, 3)
    hot = float(world.temperature[i])
    assert hot > 1000.0

    # _scrub already called record() after every one of those writes, so
    # _after is already frozen at "hot" -- this settle() call is mid-burst
    # and only confirms nothing closes yet.  Then the world runs on.
    state.explain.settle(world, pygame.time.get_ticks())
    assert state.explain.report is None
    for _ in range(21):
        state.sim.advance(60.0)
        thermal.advance(world, 60.0)
    state.explain.settle(world, pygame.time.get_ticks() + SETTLE_MS + 1)

    report = state.explain.report
    assert report is not None
    assert "heated" in report.action, report.action
    temp = next(r for r in report.changes[0].rows if r.label == "Temp")
    assert temp.after == pytest.approx(hot, rel=0.01)


def test_the_after_snapshot_survives_this_frames_own_advance(state):
    """The regression this whole fix is about, reproduced with the app's own
    frame order rather than a settle() call the test controls by hand.

    ``App.step_once`` runs ``sim.advance`` -- which calls ``thermal.advance``
    once, with the *whole* frame's worth of simulated time -- before it calls
    ``explain.settle``.  A snapshot merely deferred to "the next settle()"
    would already have that one frame of physics baked into it, and at ten
    years a second one 60 fps frame is over 60 simulated days: long enough,
    against a relaxation time constant under a thousandth of a day for
    anything glowing, to fully cool a 3115 K edit back to equilibrium before
    any settle()-timed snapshot could ever see it.  This is exactly what a
    version of the fix built on "settle() refreshes _after when dirty" does,
    even though it passes the previous test -- that test never calls
    ``sim.advance`` between the edit and its first ``settle()`` call, so it
    cannot tell the two mechanisms apart.  This one drives ``sim.advance``
    then ``settle`` every iteration, in that order, exactly as
    ``App.step_once`` does.
    """
    from src import edits
    state.set_tool(EditTool)
    world = state.sim.world
    i = [b.name for b in world.bodies].index("Earth")
    state.select(world.bodies[i].id)
    state.tool.field_index = list(edits.edit_fields(world.bodies[i])).index("temp")
    for _ in range(14):
        state.tool._scrub(state, 3)
    hot = float(world.temperature[i])
    assert hot > 1000.0

    days_per_second = 10.0 * 365.25          # ten years a second
    frame_dt = 1.0 / 60.0
    now = pygame.time.get_ticks()
    for _ in range(21):
        state.sim.advance(days_per_second * frame_dt)
        state.sim.update_trails()
        now += 16
        state.explain.settle(world, now)
    state.explain.settle(world, now + SETTLE_MS + 1)

    report = state.explain.report
    assert report is not None
    assert "heated" in report.action, report.action
    temp = next(r for r in report.changes[0].rows if r.label == "Temp")
    assert temp.after == pytest.approx(hot, rel=0.01)


def test_settle_does_nothing_while_disabled(state):
    """A burst left open across the Y toggle must not keep snapshotting, and
    must not install a report that pops into view when it is turned back on.
    The class docstring's "costs nothing at all" covers settle(), not just
    note()."""
    state.set_tool(EditTool)
    select(state, "Sun")
    state.tool._scrub(state, 12)
    assert state.explain._baseline is not None

    state.explain.enabled = False
    state.explain.settle(state.sim.world, pygame.time.get_ticks() + SETTLE_MS + 1)
    assert state.explain.report is None
    assert state.explain._baseline is not None, \
        "settle() must not touch the open burst while disabled"

    state.explain.enabled = True
    state.explain.settle(state.sim.world, pygame.time.get_ticks() + SETTLE_MS + 1)
    assert state.explain.report is not None


def test_collapse_is_the_second_note_choke_point(state):
    """_scrub is not the only place note() must be called -- _collapse is the
    other, and the spec's showcase report (see test_collapse_says... in
    test_explain.py) comes through it, not through a scrub."""
    state.set_tool(EditTool)
    select(state, "Sun")
    assert state.tool._collapse(state)
    state.explain.settle(state.sim.world, pygame.time.get_ticks() + SETTLE_MS + 1)

    report = state.explain.report
    assert report is not None
    assert report.changes[0].name == "Sun"
    assert "collapsed" in report.action
