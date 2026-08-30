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
    from src import edits
    fields = edits.edit_fields(state.sim.world.bodies[state.selected_index])
    assert state.tool.field_index == 0
    state.tool.on_event(_key(pygame.K_DOWN), state)
    assert state.tool.field_index == 1
    for _ in range(10):
        state.tool.on_event(_key(pygame.K_DOWN), state)
    assert state.tool.field_index == len(fields) - 1
    for _ in range(10):
        state.tool.on_event(_key(pygame.K_UP), state)
    assert state.tool.field_index == 0


def test_down_stops_at_the_last_field_of_the_selected_body(state):
    """Not at the last field of some other kind of body.  A star has three
    fields and a planet four, so a count taken from a module-level tuple walks
    off the end of one of them -- and a hotkey never passes through
    UiLayer.perform, so nothing else would catch it."""
    from src import edits
    body = state.sim.world.bodies[state.selected_index]
    expected = len(edits.edit_fields(body)) - 1
    for _ in range(10):
        state.tool.on_event(_key(pygame.K_DOWN), state)
    assert state.tool.field_index == expected


def test_the_edit_panel_offers_a_temp_row_for_a_planet(state):
    values = state.tool.field_values(state.sim.world, state.selected_index)
    assert "temp" in values
    assert values["temp"].endswith("K")


def test_the_temp_row_scrubs_the_selected_bodys_temperature(state):
    from src import edits
    fields = edits.edit_fields(state.sim.world.bodies[state.selected_index])
    index = state.selected_index
    before = float(state.sim.world.temperature[index])
    state.tool.field_index = fields.index("temp")
    state.tool.on_event(_key(pygame.K_RIGHT), state)
    assert float(state.sim.world.temperature[state.selected_index]) > before


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
    from src import edits
    fields = edits.edit_fields(state.sim.world.bodies[state.selected_index])
    before = state.sim.world.bodies[state.selected_index].material.key
    for _ in range(fields.index("material")):
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


def _metrics():
    from src.render.pixelfont import PixelFont
    from src.render.widgets import Metrics
    return Metrics(PixelFont(18), 1000, 700)


def test_the_panel_shows_a_row_for_every_field(state):
    from src import edits
    fields = edits.edit_fields(state.sim.world.bodies[state.selected_index])
    rows = state.tool.controls(state, _metrics())
    labels = [c.label for row in rows for c in row if c.id == "edit.field"]
    assert len(labels) == len(fields)
    assert any(label.startswith("Mass") for label in labels)
    assert any(label.startswith("Density") for label in labels)
    assert any(label.startswith("Temp") for label in labels)
    assert any(label.startswith("Material") for label in labels)


def test_the_panel_marks_the_active_field(state):
    state.tool.field_index = 1
    rows = state.tool.controls(state, _metrics())
    active = [c.label for row in rows for c in row
              if c.id == "edit.field" and c.active]
    assert len(active) == 1
    assert active[0].startswith("Density")


def test_the_panel_prompts_when_nothing_is_selected(state):
    state.selected_id = None
    rows = state.tool.controls(state, _metrics())
    assert any("Click a body" in c.label for row in rows for c in row)
    assert not any(c.clickable for row in rows for c in row)


def test_the_panel_reports_how_far_from_igniting(state):
    rows = state.tool.controls(state, _metrics())
    status = [c.label for row in rows for c in row if c.id == "edit.status"]
    assert status and status[0].endswith("heavier to ignite")


def test_the_panel_survives_every_body_and_every_field(state):
    metrics = _metrics()
    for body in list(state.sim.world.bodies):
        state.select(body.id)
        for field in range(3):
            state.tool.field_index = field
            assert state.tool.controls(state, metrics)
