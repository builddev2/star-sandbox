import os

import numpy as np
import pygame
import pytest

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


@pytest.fixture(scope="module", autouse=True)
def _pygame():
    pygame.init()
    pygame.display.set_mode((320, 200))
    yield
    pygame.quit()


@pytest.fixture
def state():
    from src.app.state import AppState
    from src.app.tools import EditTool
    from src.render.camera import Camera
    from src.render.draw import ViewSettings
    from src.sim import Simulation
    st = AppState(Simulation(), Camera(1000, 700), ViewSettings())
    st.set_tool(EditTool)
    st.select(next(b for b in st.sim.world.bodies if b.name == "Earth").id)
    return st


def _key(key):
    return pygame.event.Event(pygame.KEYDOWN, key=key, mod=0)


def test_up_and_down_move_between_fields(state):
    from src.app.tools import EDIT_FIELDS
    assert state.tool.field_index == 0
    state.tool.on_event(_key(pygame.K_DOWN), state)
    assert state.tool.field_index == 1
    for _ in range(10):
        state.tool.on_event(_key(pygame.K_DOWN), state)
    assert state.tool.field_index == len(EDIT_FIELDS) - 1
    for _ in range(10):
        state.tool.on_event(_key(pygame.K_UP), state)
    assert state.tool.field_index == 0


def test_right_arrow_increases_the_selected_field(state):
    before = float(state.sim.world.mass[state.selected_index])
    state.tool.on_event(_key(pygame.K_RIGHT), state)
    assert state.sim.world.mass[state.selected_index] > before


def test_left_arrow_decreases_it(state):
    before = float(state.sim.world.mass[state.selected_index])
    state.tool.on_event(_key(pygame.K_LEFT), state)
    assert state.sim.world.mass[state.selected_index] < before


def test_editing_mass_leaves_the_triple_consistent(state):
    from src import units
    for _ in range(40):
        state.tool.on_event(_key(pygame.K_RIGHT), state)
    i = state.selected_index
    mass, radius = float(state.sim.world.mass[i]), float(state.sim.world.radius[i])
    density = units.density_from_mass_radius(mass, radius)
    assert units.radius_from_mass_density(mass, density) == pytest.approx(radius)


def test_editing_density_changes_the_radius(state):
    before = float(state.sim.world.radius[state.selected_index])
    state.tool.on_event(_key(pygame.K_DOWN), state)      # to density
    state.tool.on_event(_key(pygame.K_RIGHT), state)
    assert state.sim.world.radius[state.selected_index] != before


def test_editing_material_changes_it(state):
    before = state.sim.world.bodies[state.selected_index].material.key
    for _ in range(2):
        state.tool.on_event(_key(pygame.K_DOWN), state)  # to material
    state.tool.on_event(_key(pygame.K_RIGHT), state)
    assert state.sim.world.bodies[state.selected_index].material.key != before


def test_making_jupiter_heavy_enough_lights_it_up(state):
    """The headline trick of the tool, driven entirely through key events."""
    from src.body import KIND_STAR
    jupiter = next(b for b in state.sim.world.bodies if b.name == "Jupiter")
    state.select(jupiter.id)
    for _ in range(200):
        state.tool.on_event(_key(pygame.K_RIGHT), state)
        if jupiter.kind == KIND_STAR:
            break
    assert jupiter.kind == KIND_STAR


def test_dragging_horizontally_scrubs(state):
    before = float(state.sim.world.mass[state.selected_index])
    state.tool.on_event(pygame.event.Event(pygame.MOUSEBUTTONDOWN, pos=(2, 2),
                                           button=1), state)
    state.tool.on_event(pygame.event.Event(pygame.MOUSEMOTION, pos=(62, 2),
                                           rel=(60, 0), buttons=(1, 0, 0)), state)
    state.tool.on_event(pygame.event.Event(pygame.MOUSEBUTTONUP, pos=(62, 2),
                                           button=1), state)
    assert state.sim.world.mass[state.selected_index] > before


def test_clicking_a_body_selects_it_for_editing(state):
    world = state.sim.world
    saturn = next(b for b in world.bodies if b.name == "Saturn")
    index = world.index_of(saturn.id)
    state.camera.centre = world.pos[index].copy()
    x, y = state.camera.world_to_screen(world.pos[index])
    state.tool.on_event(pygame.event.Event(pygame.MOUSEBUTTONDOWN,
                                           pos=(int(x), int(y)), button=1), state)
    state.tool.on_event(pygame.event.Event(pygame.MOUSEBUTTONUP,
                                           pos=(int(x), int(y)), button=1), state)
    assert state.selected_id == saturn.id


def test_editing_with_nothing_selected_is_harmless(state):
    state.selected_id = None
    state.tool.on_event(_key(pygame.K_RIGHT), state)


def test_editing_a_destroyed_body_is_harmless(state):
    state.sim.world.remove(state.selected_index)
    state.tool.on_event(_key(pygame.K_RIGHT), state)


def test_overlay_names_the_active_field(state):
    from src.render import palette
    from src.render.pixelfont import PixelFont
    surface = pygame.Surface((1000, 700))
    surface.fill(palette.SPACE)
    state.tool.draw_overlay(surface, state, PixelFont(18))
    ink = sum(1 for x in range(0, 1000, 2) for y in range(0, 700, 2)
              if surface.get_at((x, y))[:3] != palette.SPACE[:3])
    assert ink > 20


def test_overlay_prompts_when_nothing_is_selected(state):
    from src.render import palette
    from src.render.pixelfont import PixelFont
    state.selected_id = None
    surface = pygame.Surface((1000, 700))
    surface.fill(palette.SPACE)
    state.tool.draw_overlay(surface, state, PixelFont(18))
    ink = sum(1 for x in range(0, 1000, 2) for y in range(0, 700, 2)
              if surface.get_at((x, y))[:3] != palette.SPACE[:3])
    assert ink > 20


def test_overlay_survives_every_body_and_every_field(state):
    from src.render.pixelfont import PixelFont
    font = PixelFont(18)
    surface = pygame.Surface((1000, 700))
    for body in list(state.sim.world.bodies):
        state.select(body.id)
        for field in range(3):
            state.tool.field_index = field
            state.tool.draw_overlay(surface, state, font)
