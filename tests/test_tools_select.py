import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import numpy as np
import pygame
import pytest


@pytest.fixture(scope="module", autouse=True)
def _pygame():
    pygame.init()
    pygame.display.set_mode((320, 200))
    yield
    pygame.quit()


@pytest.fixture
def state():
    from src.app.state import AppState
    from src.app.tools import SelectTool
    from src.render.camera import Camera
    from src.render.draw import ViewSettings
    from src.sim import Simulation
    st = AppState(Simulation(), Camera(1000, 700), ViewSettings())
    st.set_tool(SelectTool)
    return st


def _click(x, y, button=1):
    return pygame.event.Event(pygame.MOUSEBUTTONDOWN, pos=(x, y), button=button)


def _release(x, y, button=1):
    return pygame.event.Event(pygame.MOUSEBUTTONUP, pos=(x, y), button=button)


def _motion(x, y, rel, buttons=(1, 0, 0)):
    return pygame.event.Event(pygame.MOUSEMOTION, pos=(x, y), rel=rel, buttons=buttons)


def _screen_of(state, name):
    world = state.sim.world
    index = world.index_of(next(b for b in world.bodies if b.name == name).id)
    state.camera.centre = world.pos[index].copy()
    x, y = state.camera.world_to_screen(world.pos[index])
    return int(x), int(y)


def test_hint_comes_from_the_active_tool(state):
    assert state.hint == state.tool.hint
    assert state.tool_name == "Look"


def test_clicking_a_body_selects_it(state):
    x, y = _screen_of(state, "Jupiter")
    state.tool.on_event(_click(x, y), state)
    state.tool.on_event(_release(x, y), state)
    assert state.sim.world.bodies[state.selected_index].name == "Jupiter"


def test_clicking_empty_space_deselects(state):
    earth = next(b for b in state.sim.world.bodies if b.name == "Earth")
    state.select(earth.id)
    state.tool.on_event(_click(2, 2), state)
    state.tool.on_event(_release(2, 2), state)
    assert state.selected_id is None


def test_dragging_pans_instead_of_selecting(state):
    earth = next(b for b in state.sim.world.bodies if b.name == "Earth")
    state.select(earth.id)
    start = state.camera.centre.copy()
    state.tool.on_event(_click(500, 350), state)
    state.tool.on_event(_motion(560, 350, (60, 0)), state)
    state.tool.on_event(_release(560, 350), state)
    assert state.camera.centre[0] != start[0]
    assert state.selected_id == earth.id, "a drag must not change the selection"


def test_a_tiny_jitter_still_counts_as_a_click(state):
    """A child's hand moves a pixel or two; that must not become a pan."""
    x, y = _screen_of(state, "Saturn")
    state.tool.on_event(_click(x, y), state)
    state.tool.on_event(_motion(x + 2, y + 1, (2, 1)), state)
    state.tool.on_event(_release(x + 2, y + 1), state)
    assert state.sim.world.bodies[state.selected_index].name == "Saturn"


def test_double_click_flies_to_the_body(state):
    x, y = _screen_of(state, "Saturn")
    zoom_before = state.camera.zoom
    for _ in range(2):
        state.tool.on_event(_click(x, y), state)
        state.tool.on_event(_release(x, y), state)
    assert state.camera.zoom > zoom_before
    assert state.camera.follow_id == state.selected_id


def test_number_keys_switch_tools(state):
    from src.app.input import handle_event
    from src.app.tools import SelectTool
    handle_event(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_1, mod=0), state, None)
    assert isinstance(state.tool, SelectTool)


def test_bracket_keys_change_the_time_scale(state):
    from src.app.input import handle_event
    before = state.days_per_second
    handle_event(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_RIGHTBRACKET, mod=0),
                 state, None)
    assert state.days_per_second > before
    handle_event(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_LEFTBRACKET, mod=0),
                 state, None)
    assert state.days_per_second == before


def test_space_pauses(state):
    from src.app.input import handle_event
    handle_event(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_SPACE, mod=0),
                 state, None)
    assert state.sim.paused is True


def test_toggles_flip_view_settings(state):
    from src.app.input import handle_event
    for key, attribute in ((pygame.K_t, "show_trails"),
                           (pygame.K_r, "real_size"),
                           (pygame.K_l, "show_labels")):
        before = getattr(state.view, attribute)
        handle_event(pygame.event.Event(pygame.KEYDOWN, key=key, mod=0), state, None)
        assert getattr(state.view, attribute) is not before


def test_zero_key_stops_the_selected_body(state):
    from src.app.input import handle_event
    earth = next(b for b in state.sim.world.bodies if b.name == "Earth")
    state.select(earth.id)
    handle_event(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_0, mod=0), state, None)
    assert np.allclose(state.sim.world.vel[state.sim.world.index_of(earth.id)], 0.0)


def test_escape_closes_help_before_deselecting(state):
    from src.app.input import handle_event
    earth = next(b for b in state.sim.world.bodies if b.name == "Earth")
    state.select(earth.id)
    state.help_visible = True
    escape = pygame.event.Event(pygame.KEYDOWN, key=pygame.K_ESCAPE, mod=0)
    handle_event(escape, state, None)
    assert state.help_visible is False
    assert state.selected_id == earth.id
    handle_event(escape, state, None)
    assert state.selected_id is None


def test_delete_removes_the_selected_body(state):
    from src.app.input import handle_event
    mars = next(b for b in state.sim.world.bodies if b.name == "Mars")
    state.select(mars.id)
    before = state.sim.world.n
    handle_event(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_DELETE, mod=0),
                 state, None)
    assert state.sim.world.n == before - 1


def test_f5_resets_the_scenario(state):
    from src.app.input import handle_event
    state.sim.advance(400.0, budget_s=5.0)
    handle_event(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_F5, mod=0),
                 state, None)
    assert state.sim.world.time == 0.0


def test_command_q_quits(state):
    from src.app.input import handle_event
    handle_event(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_q,
                                    mod=pygame.KMOD_META), state, None)
    assert state.running is False


def test_the_tool_sees_events_before_the_globals(state):
    """A tool that consumes a drag must stop the camera panning underneath it."""
    from src.app.input import handle_event
    from src.app.tools import Tool

    class Greedy(Tool):
        name = "Greedy"
        def __init__(self):
            self.seen = 0
        def on_event(self, event, state):
            self.seen += 1
            return True

    state.set_tool(Greedy)
    before = state.days_per_second
    handle_event(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_RIGHTBRACKET, mod=0),
                 state, None)
    assert state.tool.seen == 1
    assert state.days_per_second == before, "global key fired despite being consumed"
