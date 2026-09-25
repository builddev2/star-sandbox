"""The Measure tool: its panel, its keys, and where it sits."""

import math
import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import pygame
import pytest

from src import scenarios
from src.app.state import AppState
from src.app.tools import MeasureTool
from src.render.camera import Camera
from src.render.draw import ViewSettings
from src.sim import Simulation

SIZE = (640, 360)


@pytest.fixture
def state():
    pygame.init()
    sim = Simulation(world=scenarios.solar_system(moons=False))
    app = AppState(sim, Camera(*SIZE), ViewSettings())
    app.set_tool(MeasureTool)
    return app


def _body(state, name):
    return next(b for b in state.sim.world.bodies if b.name == name)


def _key(key):
    return pygame.event.Event(pygame.KEYDOWN, key=key, mod=0)


def _controls(state):
    return {c.id: c for row in state.tool.controls(state, None) for c in row}


def test_the_tool_registers_under_key_five():
    from src.app.tools import TOOL_BY_NAME, TOOLS
    assert TOOLS[pygame.K_5] is MeasureTool
    assert TOOL_BY_NAME["measure"] is MeasureTool


def test_the_hotbar_has_a_measure_button():
    from src.app.ui import TOOL_BUTTONS
    assert ("tool.measure", "Measure") in {(b[0], b[1]) for b in TOOL_BUTTONS}


def test_timing_is_disabled_and_says_why_with_nothing_selected(state):
    time = _controls(state)["measure.time"]
    assert not time.enabled
    assert time.label == "Pick a body first"
    assert not _controls(state)["measure.stop"].enabled


def test_the_button_starts_timing_the_selection(state):
    # Deleting the measure.time branch of perform makes this fail.
    state.select(_body(state, "Mars").id)
    time = _controls(state)["measure.time"]
    assert time.enabled and time.hotkey == "P"
    assert state.tool.perform(time, state)
    assert state.period_timer.body_id == _body(state, "Mars").id
    assert state.tool.perform(_controls(state)["measure.stop"], state)
    assert state.period_timer is None


def test_the_target_button_arms_picking(state):
    # Deleting the measure.target branch of perform makes this fail.
    target = _controls(state)["measure.target"]
    assert state.tool.perform(target, state)
    assert state.tool.picking_target
    assert state.tool.perform(target, state)
    assert not state.tool.picking_target


def test_p_times_the_selection_and_does_nothing_without_one(state):
    assert state.tool.on_event(_key(pygame.K_p), state)
    assert state.period_timer is None
    state.select(_body(state, "Mars").id)
    state.tool.on_event(_key(pygame.K_p), state)
    assert state.period_timer.body_id == _body(state, "Mars").id


def test_x_then_click_locks_the_target(state):
    earth = _body(state, "Earth")
    state.camera.frame_body(state.sim.world, state.sim.world.index_of(earth.id))
    x, y = state.camera.world_to_screen(state.sim.world.pos[state.sim.world.index_of(earth.id)])
    state.tool.on_event(_key(pygame.K_x), state)
    assert state.tool.picking_target
    state.tool.on_event(pygame.event.Event(
        pygame.MOUSEBUTTONDOWN, button=1, pos=(int(x), int(y))), state)
    assert state.tool.target_id == earth.id
    assert _controls(state)["measure.target"].label == "Target: Earth"


def test_a_plain_click_selects(state):
    mars = _body(state, "Mars")
    index = state.sim.world.index_of(mars.id)
    state.camera.frame_body(state.sim.world, index)
    pos = tuple(int(v) for v in state.camera.world_to_screen(state.sim.world.pos[index]))
    state.tool.on_event(pygame.event.Event(pygame.MOUSEBUTTONDOWN, button=1, pos=pos), state)
    state.tool.on_event(pygame.event.Event(pygame.MOUSEBUTTONUP, button=1, pos=pos), state)
    assert state.selected_id == mars.id


def test_every_control_explains_itself(state):
    state.select(_body(state, "Mars").id)
    state.start_timer()
    for control in _controls(state).values():
        assert control.hint(), control.id


def test_the_keys_are_documented():
    from src.render.hud import KEY_HELP
    combos = {combo for combo, _ in KEY_HELP}
    assert "P" in combos
    assert any("5" in combo.split() for combo in combos)


@pytest.mark.parametrize("name", ["measure", "clock"])
def test_the_new_icons_draw_something(name):
    # Deleting either draw_icon branch leaves the surface blank.
    from src.render.widgets import draw_icon
    surface = pygame.Surface((24, 24))
    surface.fill((0, 0, 0))
    draw_icon(surface, name, pygame.Rect(2, 2, 20, 20), (255, 255, 255))
    assert pygame.mask.from_threshold(surface, (255, 255, 255), (60, 60, 60, 255)).count() > 10


def _two_body_state():
    pygame.init()
    sim = Simulation(world=scenarios.two_body())
    app = AppState(sim, Camera(*SIZE), ViewSettings())
    app.set_tool(MeasureTool)
    return app


def _texts(state):
    return [text for text, _ in state.tool.readout_lines(state)]


def test_nothing_to_say_with_nothing_chosen(state):
    assert state.tool.readout_lines(state) == []


def test_the_ruler_reads_light_minutes_to_the_sun():
    """One AU exactly, from the two-body scenario: 8.3 light-minutes."""
    st = _two_body_state()
    st.select(st.sim.world.bodies[1].id)
    st.tool.target_id = st.sim.world.bodies[0].id
    texts = _texts(st)
    assert texts[0] == "Secondary to Primary"
    assert "Apart  1 AU" in texts
    assert "Light takes 8.3 min" in texts
    assert not any(t.startswith("Gap") for t in texts)      # too far to matter


def test_the_gap_appears_once_they_are_close():
    # Earth to the Moon is well inside 0.02 AU.
    pygame.init()
    sim = Simulation(world=scenarios.solar_system())
    st = AppState(sim, Camera(*SIZE), ViewSettings())
    st.set_tool(MeasureTool)
    st.select(_body(st, "Earth").id)
    st.tool.target_id = _body(st, "Moon").id
    assert any(t.startswith("Gap") and t.endswith("km") for t in _texts(st))


def test_the_timer_reads_measured_against_kepler():
    """After one lap: the measured period, Kepler's, how far apart, and the
    mass it implies.  Removing any of the four lines fails this."""
    st = _two_body_state()
    st.select(st.sim.world.bodies[1].id)
    st.start_timer()
    while st.period_timer.measured is None and st.sim.world.time < 1000.0:
        st.sim.advance(5.0, budget_s=1e9)
        st.sample_timer()
    texts = _texts(st)
    assert texts[0] == "Timing Secondary round Primary"
    assert "Measured    365.3 d" in texts
    assert "Kepler says 365.3 d" in texts
    assert any(t.startswith("Off by 0.0") for t in texts)
    assert "a^3/P^2 weighs 1 Suns" in texts


def test_too_fast_is_said_out_loud():
    st = _two_body_state()
    st.select(st.sim.world.bodies[1].id)
    st.start_timer()
    st.sim.advance(200.0, budget_s=1e9)
    st.sample_timer()
    assert "Too fast to time -- slow down" in _texts(st)


def test_a_line_is_drawn_between_the_two():
    """A pixel a quarter of the way along the line is lit only when there is
    a target.  Deleting the aaline call makes this fail."""
    from src.render.pixelfont import PixelFont
    st = _two_body_state()
    st.show_body_list = False
    st.camera.log_zoom = math.log(SIZE[0] / 3.0)     # pixels per AU: 3 AU across
    st.camera.centre[:] = 0.0
    st.select(st.sim.world.bodies[1].id)
    st.tool.target_id = st.sim.world.bodies[0].id
    world = st.sim.world
    a = st.camera.world_to_screen(world.pos[1])
    b = st.camera.world_to_screen(world.pos[0])
    probe = (int(round(a[0] + 0.25 * (b[0] - a[0]))), int(round(a[1] + 0.25 * (b[1] - a[1]))))
    # A probe off screen would make this test pass or fail for no reason.
    assert pygame.Rect(0, 0, *SIZE).collidepoint(probe)
    surface = pygame.Surface(SIZE)
    surface.fill((0, 0, 0))
    st.tool.draw_overlay(surface, st, PixelFont(12))
    lit = max(surface.get_at((probe[0] + dx, probe[1] + dy))[:3]
              for dx in (-1, 0, 1) for dy in (-1, 0, 1))
    assert sum(lit) > 150
