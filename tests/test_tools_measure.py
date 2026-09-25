"""The Measure tool: its panel, its keys, and where it sits."""

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
