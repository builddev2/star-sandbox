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
