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


# --- locking ----------------------------------------------------------------------

def _locks(rows):
    return [c for row in rows for c in row if c.id == "edit.lock"]


def _steps(rows, field):
    return [c for row in rows for c in row
            if c.id == "edit.step" and c.payload[0] == field]


def test_every_field_row_carries_a_lock_button(state):
    from src import edits
    fields = edits.edit_fields(state.sim.world.bodies[state.selected_index])
    locks = _locks(state.tool.controls(state, _metrics()))
    assert [c.payload for c in locks] == list(fields)


def test_the_lock_button_swaps_its_icon(state):
    """The same icon-swap-on-state the play/pause button already uses: the
    button is the readout."""
    from src import edits
    body = state.sim.world.bodies[state.selected_index]
    lock = next(c for c in _locks(state.tool.controls(state, _metrics()))
                if c.payload == "mass")
    assert lock.icon == "unlock"
    edits.toggle_lock(body, "mass")
    lock = next(c for c in _locks(state.tool.controls(state, _metrics()))
                if c.payload == "mass")
    assert lock.icon == "lock"
    assert lock.active


def test_a_locked_row_greys_out_both_its_steppers(state):
    from src import edits
    body = state.sim.world.bodies[state.selected_index]
    edits.toggle_lock(body, "mass")
    rows = state.tool.controls(state, _metrics())
    assert [c.enabled for c in _steps(rows, "mass")] == [False, False]
    assert all(c.enabled for c in _steps(rows, "density"))


def test_the_row_itself_stays_selectable_when_it_is_locked(state):
    """Only the two steppers go inert.  The value still reads out, and the row
    can still be picked."""
    from src import edits
    edits.toggle_lock(state.sim.world.bodies[state.selected_index], "mass")
    rows = state.tool.controls(state, _metrics())
    field = next(c for row in rows for c in row
                 if c.id == "edit.field" and c.payload == "mass")
    assert field.clickable
    assert field.label.startswith("Mass")


def test_the_lock_button_is_never_itself_disabled(state):
    """Otherwise locking a field would be one-way through the mouse."""
    from src import edits
    body = state.sim.world.bodies[state.selected_index]
    for field in edits.edit_fields(body):
        edits.toggle_lock(body, field)
    assert all(c.enabled for c in _locks(state.tool.controls(state, _metrics())))


def test_pressing_a_lock_toggles_it_and_picks_that_row(state):
    """A click on a row's padlock also selects the row, so Up/Down and
    Left/Right carry on from where the mouse left off -- the same thing
    edit.step already does."""
    from src import edits
    body = state.sim.world.bodies[state.selected_index]
    fields = edits.edit_fields(body)
    lock = next(c for c in _locks(state.tool.controls(state, _metrics()))
                if c.payload == fields[-1])
    assert state.tool.perform(lock, state)
    assert edits.is_locked(body, fields[-1])
    assert state.tool.field_index == len(fields) - 1
    assert state.tool.perform(lock, state)
    assert not edits.is_locked(body, fields[-1])


def test_unlock_all_is_disabled_until_something_is_locked(state):
    from src import edits
    rows = state.tool.controls(state, _metrics())
    unlock = next(c for row in rows for c in row if c.id == "edit.unlock_all")
    assert not unlock.enabled
    edits.toggle_lock(state.sim.world.bodies[state.selected_index], "density")
    rows = state.tool.controls(state, _metrics())
    unlock = next(c for row in rows for c in row if c.id == "edit.unlock_all")
    assert unlock.enabled


def test_unlock_all_clears_every_lock(state):
    from src import edits
    body = state.sim.world.bodies[state.selected_index]
    for field in edits.edit_fields(body):
        edits.toggle_lock(body, field)
    rows = state.tool.controls(state, _metrics())
    unlock = next(c for row in rows for c in row if c.id == "edit.unlock_all")
    assert state.tool.perform(unlock, state)
    assert body.locked_fields == frozenset()


def test_unlock_all_shares_the_collapse_row(state):
    """A seventh row would draw over the transport bar at the interface floor;
    tests/test_layout.py is what holds this in place."""
    rows = state.tool.controls(state, _metrics())
    row = next(row for row in rows
               if any(c.id == "edit.unlock_all" for c in row))
    assert {c.id for c in row} == {"edit.collapse", "edit.unlock_all"}


def test_a_black_hole_gets_unlock_all_on_a_row_of_its_own(state):
    """It has no Collapse button to share with."""
    from src import edits
    index = state.selected_index
    edits.set_mass(state.sim.world, index, 10.0)
    edits.collapse(state.sim.world, index)
    rows = state.tool.controls(state, _metrics())
    row = next(row for row in rows
               if any(c.id == "edit.unlock_all" for c in row))
    assert [c.id for c in row] == ["edit.unlock_all"]
    assert [c.payload for c in _locks(rows)] == ["mass", "spin", "disk"]


def test_k_locks_the_selected_field_and_locks_it_again(state):
    from src import edits
    body = state.sim.world.bodies[state.selected_index]
    assert state.tool.on_event(_key(pygame.K_k), state)
    assert edits.is_locked(body, "mass")
    assert state.tool.on_event(_key(pygame.K_k), state)
    assert not edits.is_locked(body, "mass")


def test_k_locks_the_row_the_arrows_navigated_to(state):
    """It mirrors Left/Right, which act on self.field_index, not on row zero."""
    from src import edits
    body = state.sim.world.bodies[state.selected_index]
    fields = edits.edit_fields(body)
    state.tool.on_event(_key(pygame.K_DOWN), state)
    state.tool.on_event(_key(pygame.K_k), state)
    assert edits.is_locked(body, fields[1])
    assert not edits.is_locked(body, fields[0])


def test_a_locked_field_ignores_left_and_right(state):
    """The point of the whole feature: a stray arrow key cannot nudge a pinned
    value.  Enforced in edits.py, because a key never passes through the panel
    that greyed the buttons out."""
    index = state.selected_index
    before = float(state.sim.world.mass[index])
    state.tool.on_event(_key(pygame.K_k), state)
    for _ in range(20):
        state.tool.on_event(_key(pygame.K_RIGHT), state)
        state.tool.on_event(_key(pygame.K_LEFT), state)
    assert float(state.sim.world.mass[state.selected_index]) == pytest.approx(before)


def test_a_locked_field_ignores_a_drag(state):
    index = state.selected_index
    before = float(state.sim.world.mass[index])
    state.tool.on_event(_key(pygame.K_k), state)
    state.tool.on_event(pygame.event.Event(pygame.MOUSEBUTTONDOWN, pos=(2, 2),
                                           button=1), state)
    state.tool.on_event(pygame.event.Event(pygame.MOUSEMOTION, pos=(122, 2),
                                           rel=(120, 0), buttons=(1, 0, 0)), state)
    state.tool.on_event(pygame.event.Event(pygame.MOUSEBUTTONUP, pos=(122, 2),
                                           button=1), state)
    assert float(state.sim.world.mass[state.selected_index]) == pytest.approx(before)


def test_locking_one_field_leaves_the_others_scrubbable(state):
    index = state.selected_index
    mass = float(state.sim.world.mass[index])
    radius = float(state.sim.world.radius[index])
    state.tool.on_event(_key(pygame.K_k), state)        # lock mass
    state.tool.on_event(_key(pygame.K_DOWN), state)     # to density
    state.tool.on_event(_key(pygame.K_RIGHT), state)
    assert float(state.sim.world.mass[state.selected_index]) == pytest.approx(mass)
    assert float(state.sim.world.radius[state.selected_index]) != radius


def test_u_unlocks_everything_on_the_body(state):
    from src import edits
    body = state.sim.world.bodies[state.selected_index]
    for field in edits.edit_fields(body):
        edits.toggle_lock(body, field)
    assert state.tool.on_event(_key(pygame.K_u), state)
    assert body.locked_fields == frozenset()


def test_u_with_nothing_locked_does_nothing_and_says_so(state):
    """It matches its own disabled button: nothing to undo, so the event is not
    consumed either."""
    assert not state.tool.on_event(_key(pygame.K_u), state)


def test_k_and_u_with_nothing_selected_are_harmless(state):
    state.selected_id = None
    assert not state.tool.on_event(_key(pygame.K_k), state)
    assert not state.tool.on_event(_key(pygame.K_u), state)


def test_k_and_u_on_a_destroyed_body_are_harmless(state):
    state.sim.world.remove(state.selected_index)
    state.tool.on_event(_key(pygame.K_k), state)
    state.tool.on_event(_key(pygame.K_u), state)


def test_the_padlock_tooltip_does_not_promise_K_acts_on_this_row(state):
    """Control.hint() appends "[K]" to anything carrying a hotkey, and K acts on
    the *selected* row, not the hovered one.  So hovering the Density padlock
    while Mass is picked must not read as a promise that K will lock density.
    """
    from src import edits
    body = state.sim.world.bodies[state.selected_index]
    for field in edits.edit_fields(body):
        edits.toggle_lock(body, field)          # check the locked wording too
    for lock in _locks(state.tool.controls(state, _metrics())):
        assert "the row that is picked" in lock.tooltip, lock.tooltip
    edits.unlock_all(body)
    for lock in _locks(state.tool.controls(state, _metrics())):
        assert "the row that is picked" in lock.tooltip, lock.tooltip


# --- a control from a panel the body has outgrown ---------------------------------

def _stale_planet_controls(state):
    """The Edit panel for a planet, and then a body that is no longer one.

    The panel is built at the end of a frame from the body as it was then; the
    press it answers arrives at the start of the next one, or -- with a held
    stepper, which UiLayer keeps a reference to across frames -- many frames
    later.  A merge in between can ignite the body, and edit_fields then stops
    offering Temp.
    """
    from src import edits, units
    rows = state.tool.controls(state, _metrics())
    stale = {
        "step": next(c for row in rows for c in row
                     if c.id == "edit.step" and c.payload[0] == "temp"),
        "field": next(c for row in rows for c in row
                      if c.id == "edit.field" and c.payload == "temp"),
        "lock": next(c for row in rows for c in row
                     if c.id == "edit.lock" and c.payload == "temp"),
    }
    edits.set_mass(state.sim.world, state.selected_index,
                   units.STAR_MASS_THRESHOLD * 1.1)
    assert "temp" not in edits.edit_fields(
        state.sim.world.bodies[state.selected_index])
    return stale


def test_a_stale_control_is_ignored_rather_than_raising(state):
    """fields.index() raises ValueError on a row that has gone.  Saying "that
    row is gone" and doing nothing is right: the panel is rebuilt from the new
    kind before the next press anyway."""
    stale = _stale_planet_controls(state)
    # Read *after* igniting: _refresh_kind pins a star's temperature the moment
    # it lights up, so the pre-ignition value is not what a no-op must preserve.
    temperature = float(state.sim.world.temperature[state.selected_index])
    for control in stale.values():
        assert state.tool.perform(control, state) is False
    assert float(state.sim.world.temperature[state.selected_index]) == \
        pytest.approx(temperature)


def test_a_stale_control_leaves_the_selected_row_alone(state):
    """A press that does nothing must not move the cursor either."""
    state.tool.field_index = 1
    for control in _stale_planet_controls(state).values():
        state.tool.perform(control, state)
    assert state.tool.field_index == 1


def test_a_stale_lock_does_not_lock_the_wrong_field(state):
    """The worst outcome of guessing: silently pinning whatever row happens to
    sit at that index in the new tuple."""
    stale = _stale_planet_controls(state)
    body = state.sim.world.bodies[state.selected_index]
    state.tool.perform(stale["lock"], state)
    assert body.locked_fields == frozenset()


def test_pressing_an_edit_control_with_nothing_selected_is_harmless(state):
    """selected_index is None once the selected body is destroyed, and
    world.bodies[None] raises TypeError -- which edit.step and edit.field did."""
    rows = state.tool.controls(state, _metrics())
    controls = [c for row in rows for c in row
                if c.id in ("edit.step", "edit.field", "edit.lock")]
    state.selected_id = None
    for control in controls:
        assert state.tool.perform(control, state) is False
