import os

import numpy as np
import pytest

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import pygame

from src.render.camera import Camera
from src.render.draw import ViewSettings
from src.sim import Simulation


@pytest.fixture(scope="module", autouse=True)
def _pygame():
    pygame.init()
    yield
    pygame.quit()


@pytest.fixture
def state():
    from src.app.state import AppState
    sim = Simulation()
    return AppState(sim, Camera(1000, 700), ViewSettings())


def test_time_ladder_is_monotonic_and_labelled():
    from src.app.state import TIME_LABELS, TIME_SCALES
    assert len(TIME_SCALES) == len(TIME_LABELS)
    assert list(TIME_SCALES) == sorted(TIME_SCALES)
    assert TIME_SCALES[0] < 1.0 < TIME_SCALES[-1]


def test_time_scale_clamps_at_both_ends(state):
    from src.app.state import TIME_SCALES
    for _ in range(50):
        state.nudge_time(-1)
    assert state.days_per_second == TIME_SCALES[0]
    for _ in range(50):
        state.nudge_time(1)
    assert state.days_per_second == TIME_SCALES[-1]


def test_pause_toggles_the_simulation(state):
    assert state.sim.paused is False
    state.toggle_pause()
    assert state.sim.paused is True
    state.toggle_pause()
    assert state.sim.paused is False


def test_selection_survives_bodies_being_removed(state):
    """Selection stores an id, so a merge elsewhere must not silently reselect
    a different planet."""
    earth = next(b for b in state.sim.world.bodies if b.name == "Earth")
    state.select(earth.id)
    state.sim.world.remove(0)          # remove the Sun; rows shuffle
    assert state.sim.world.bodies[state.selected_index].name == "Earth"


def test_selecting_a_removed_body_reports_none(state):
    doomed = state.sim.world.bodies[-1]
    state.select(doomed.id)
    state.sim.world.remove(state.sim.world.index_of(doomed.id))
    assert state.selected_index is None
    assert state.selected_body is None


def test_clicking_on_a_planet_picks_it(state):
    world = state.sim.world
    jupiter = next(b for b in world.bodies if b.name == "Jupiter")
    index = world.index_of(jupiter.id)
    state.camera.centre = world.pos[index].copy()
    sx, sy = state.camera.world_to_screen(world.pos[index])
    assert state.pick_body_at(sx, sy) == index


def test_clicking_empty_space_picks_nothing(state):
    assert state.pick_body_at(3.0, 3.0) is None


def test_clicking_jupiter_does_not_select_one_of_its_moons(state):
    """At system zoom the Galilean moons are a fraction of a pixel from
    Jupiter's centre, so a naive smallest-wins rule makes Jupiter unclickable."""
    world = state.sim.world
    jupiter = world.index_of(next(b for b in world.bodies if b.name == "Jupiter").id)
    state.camera.centre = world.pos[jupiter].copy()
    sx, sy = state.camera.world_to_screen(world.pos[jupiter])
    assert world.bodies[state.pick_body_at(sx, sy)].name == "Jupiter"


def test_a_moon_in_front_of_its_planet_is_still_selectable(state):
    """Picking the smallest match, not the first, is what makes this work."""
    world = state.sim.world
    moon = world.index_of(next(b for b in world.bodies if b.name == "Moon").id)
    state.camera.centre = world.pos[moon].copy()
    state.camera.log_zoom = np.log(1000 / 0.02)   # 0.02 AU across
    sx, sy = state.camera.world_to_screen(world.pos[moon])
    assert world.bodies[state.pick_body_at(sx, sy)].name == "Moon"


def test_cycle_selection_visits_different_bodies(state):
    state.cycle_selection()
    first = state.selected_id
    state.cycle_selection()
    assert state.selected_id != first


def test_follow_selected_sets_the_camera(state):
    mars = next(b for b in state.sim.world.bodies if b.name == "Mars")
    state.select(mars.id)
    state.follow_selected()
    assert state.camera.follow_id == mars.id


def test_delete_selected_removes_it_and_clears_selection(state):
    mars = next(b for b in state.sim.world.bodies if b.name == "Mars")
    before = state.sim.world.n
    state.select(mars.id)
    state.delete_selected()
    assert state.sim.world.n == before - 1
    assert state.selected_id is None
    assert all(b.name != "Mars" for b in state.sim.world.bodies)


def test_stopping_the_earth_makes_it_fall_into_the_sun(state):
    """The single best trick in the toy, so it gets a test."""
    world = state.sim.world
    earth = next(b for b in world.bodies if b.name == "Earth")
    sun = next(b for b in world.bodies if b.name == "Sun")
    state.select(earth.id)
    state.stop_selected()

    def distance():
        return float(np.linalg.norm(world.pos[world.index_of(earth.id)]
                                    - world.pos[world.index_of(sun.id)]))

    start = distance()
    for _ in range(30):
        state.sim.advance(2.0, budget_s=5.0)
    assert distance() < start * 0.9


def test_reset_restores_everything(state):
    state.select(state.sim.world.bodies[2].id)
    state.follow_selected()
    state.sim.advance(500.0, budget_s=5.0)
    state.reset()
    assert state.sim.world.time == 0.0
    assert state.selected_id is None
    assert state.camera.follow_id is None


def test_a_restart_clears_every_lock(state):
    """Restart rebuilds every Body from the scenario, so a lock cannot survive
    it -- the body it belonged to does not.  Right, but only accidentally so
    until something says it out loud: a scenario builder that reused Body
    objects instead of world.clear()-ing them would carry stale locks into a
    world the child asked to have put back."""
    from src import edits
    body = next(b for b in state.sim.world.bodies if b.name == "Earth")
    for field in edits.edit_fields(body):
        edits.toggle_lock(body, field)
    assert body.locked_fields

    state.reset()

    assert all(b.locked_fields == frozenset() for b in state.sim.world.bodies)
    assert all(b is not body for b in state.sim.world.bodies)


def test_elapsed_label_reports_years_and_days(state):
    state.sim.world.time = 400.0
    label = state.elapsed_label
    assert "Year" in label and "Day" in label


def test_hint_falls_back_when_no_tool_is_active(state):
    assert isinstance(state.hint, str) and state.hint


def test_the_inspector_starts_collapsed(state):
    assert state.inspector_expanded is False


def test_selecting_a_new_body_collapses_the_inspector_again(state):
    state.select(state.sim.world.bodies[0].id)
    state.toggle_inspector_expanded()
    assert state.inspector_expanded is True
    state.select(state.sim.world.bodies[1].id)
    assert state.inspector_expanded is False


def test_deselecting_collapses_the_inspector(state):
    state.select(state.sim.world.bodies[0].id)
    state.toggle_inspector_expanded()
    state.select(None)
    assert state.inspector_expanded is False


def test_selecting_stamps_the_selection_time(state):
    state.select(state.sim.world.bodies[0].id, now=1000)
    assert state.selected_at == 1000


def test_reselecting_the_same_body_restamps_the_time(state):
    """The tether line's 3-second countdown has to restart even when the
    click lands on the body that was already selected -- this is what makes
    that possible without watching selected_id for a change that never
    happens."""
    body = state.sim.world.bodies[0]
    state.select(body.id, now=1000)
    state.select(body.id, now=5000)
    assert state.selected_at == 5000


def test_cycling_selection_restamps_the_time(state):
    state.select(state.sim.world.bodies[0].id, now=1000)
    before = pygame.time.get_ticks()
    state.cycle_selection()
    assert state.selected_at >= before
