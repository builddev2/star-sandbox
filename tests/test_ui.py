"""The clickable interface: what is on screen, and what pressing it does.

Two things are worth stating about how these tests work.  Controls are found by
id rather than by coordinate, so moving a panel does not break a test about what
a button *does*; and clicking is always a real press and release through
``on_event``, because half the behaviour worth testing lives in the gap between
those two events.
"""

import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import numpy as np
import pygame
import pytest

SIZE = (1000, 700)


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
    st = AppState(Simulation(), Camera(*SIZE), ViewSettings())
    st.set_tool(SelectTool)
    return st


@pytest.fixture
def ui(state):
    from src.app.ui import UiLayer
    from src.render.hud import Hud
    from src.render.pixelfont import PixelFont
    font = PixelFont(18)
    layer = UiLayer(font, Hud(font))
    layer.build(state, SIZE)
    return layer


def find(ui, cid, payload=None):
    for control in ui.controls:
        if control.id == cid and (payload is None or control.payload == payload):
            return control
    return None


def ids(ui):
    return {control.id for control in ui.controls}


def click(ui, state, control, ctx=None, size=SIZE):
    """A real press and release on a control, through the event path.

    Rebuilds afterwards at the same size, because that is what the frame loop
    does and because half these tests then look at the rebuilt controls.
    """
    pos = control.rect.center
    ui.on_event(pygame.event.Event(pygame.MOUSEBUTTONDOWN, pos=pos, button=1),
                state, ctx)
    ui.on_event(pygame.event.Event(pygame.MOUSEBUTTONUP, pos=pos, button=1),
                state, ctx)
    ui.build(state, size, ctx)


def click_id(ui, state, cid, payload=None, ctx=None):
    control = find(ui, cid, payload)
    assert control is not None, "no control %r on screen" % cid
    click(ui, state, control, ctx)
    return control


# --- what is on screen ------------------------------------------------------------

def test_the_whole_interface_is_present_without_a_selection(ui):
    assert {"tool.look", "tool.throw", "tool.edit", "panel.bodies",
            "time.pause", "time.slower", "time.faster", "sim.reset",
            "view.trails", "view.sizes", "view.labels", "view.grid",
            "view.fullscreen", "view.help"} <= ids(ui)


def test_no_control_hangs_off_the_edge_of_the_screen(ui):
    screen = pygame.Rect(0, 0, *SIZE)
    for control in ui.controls:
        assert screen.contains(control.rect), "%s is off screen" % control.id


def test_controls_do_not_sit_on_top_of_each_other(ui):
    """Overlapping buttons mean one of them cannot be clicked."""
    seen = []
    for control in ui.controls:
        for other in seen:
            assert not control.rect.colliderect(other.rect), \
                "%s overlaps %s" % (control.id, other.id)
        seen.append(control)


def test_the_active_tool_is_the_lit_button(state, ui):
    from src.app.tools import LaunchTool
    assert find(ui, "tool.look").active
    state.set_tool(LaunchTool)
    ui.build(state, SIZE)
    assert find(ui, "tool.throw").active
    assert not find(ui, "tool.look").active


# --- tools ------------------------------------------------------------------------

@pytest.mark.parametrize("cid,name", [("tool.look", "Look"),
                                      ("tool.throw", "Throw"),
                                      ("tool.edit", "Edit")])
def test_each_tool_button_switches_tool(state, ui, cid, name):
    click_id(ui, state, cid)
    assert state.tool_name == name


def test_the_throw_presets_are_buttons(state, ui):
    from src.app.tools import PRESETS
    click_id(ui, state, "tool.throw")
    for k in range(len(PRESETS)):
        assert find(ui, "throw.preset", payload=k) is not None
    click_id(ui, state, "throw.preset", payload=3)
    assert state.tool.preset_index == 3


def test_the_edit_steppers_change_the_body(state, ui):
    earth = next(b for b in state.sim.world.bodies if b.name == "Earth")
    state.select(earth.id)
    click_id(ui, state, "tool.edit")

    before = float(state.sim.world.mass[state.selected_index])
    click_id(ui, state, "edit.step", payload=("mass", 3))
    assert state.sim.world.mass[state.selected_index] > before

    click_id(ui, state, "edit.step", payload=("mass", -3))
    assert state.sim.world.mass[state.selected_index] == pytest.approx(before)


def test_editing_by_button_keeps_radius_derived(state, ui):
    """The buttons go through the same edits module the keys do."""
    from src import units
    earth = next(b for b in state.sim.world.bodies if b.name == "Earth")
    state.select(earth.id)
    click_id(ui, state, "tool.edit")
    density = units.density_from_mass_radius(
        float(state.sim.world.mass[state.selected_index]),
        float(state.sim.world.radius[state.selected_index]))

    for _ in range(5):
        click_id(ui, state, "edit.step", payload=("mass", 3))

    index = state.selected_index
    assert float(state.sim.world.radius[index]) == pytest.approx(
        units.radius_from_mass_density(float(state.sim.world.mass[index]), density),
        rel=1e-9)


def test_the_material_arrows_walk_through_the_substances(state, ui):
    earth = next(b for b in state.sim.world.bodies if b.name == "Earth")
    state.select(earth.id)
    click_id(ui, state, "tool.edit")
    first = state.sim.world.bodies[state.selected_index].material.key
    click_id(ui, state, "edit.step", payload=("material", 1))
    assert state.sim.world.bodies[state.selected_index].material.key != first


# --- time -------------------------------------------------------------------------

def test_the_pause_button_toggles_and_relabels(state, ui):
    assert not state.sim.paused
    assert find(ui, "time.pause").label == "Pause"
    click_id(ui, state, "time.pause")
    assert state.sim.paused
    assert find(ui, "time.pause").label == "Play"
    click_id(ui, state, "time.pause")
    assert not state.sim.paused


def test_the_speed_buttons_step_and_stop_at_the_ends(state, ui):
    start = state.time_index
    click_id(ui, state, "time.faster")
    assert state.time_index == start + 1
    click_id(ui, state, "time.slower")
    assert state.time_index == start

    state.time_index = 0
    ui.build(state, SIZE)
    assert not find(ui, "time.slower").enabled
    state.time_index = state.time_index_max
    ui.build(state, SIZE)
    assert not find(ui, "time.faster").enabled


def test_the_readout_shows_the_current_speed_and_cannot_be_clicked(state, ui):
    assert find(ui, "time.readout").label == state.time_label
    assert not find(ui, "time.readout").clickable
    assert ui.control_at(find(ui, "time.readout").rect.center) is None


def test_restart_puts_the_solar_system_back(state, ui):
    state.sim.world.remove(3)
    state.select(state.sim.world.bodies[0].id)
    before = state.sim.world.n
    click_id(ui, state, "sim.reset")
    assert state.sim.world.n > before
    assert state.selected_id is None


# --- view toggles -----------------------------------------------------------------

@pytest.mark.parametrize("cid,attr", [("view.trails", "show_trails"),
                                      ("view.sizes", "real_size"),
                                      ("view.labels", "show_labels")])
def test_a_view_toggle_flips_its_flag_and_lights_up(state, ui, cid, attr):
    before = getattr(state.view, attr)
    click_id(ui, state, cid)
    assert getattr(state.view, attr) is not before
    assert find(ui, cid).active is getattr(state.view, attr)


def test_there_is_a_way_out_without_the_keyboard(state, ui):
    """Fullscreen has no window close box, so this button is the only way out."""
    assert find(ui, "app.quit") is not None


def test_quitting_takes_two_presses(state, ui):
    click_id(ui, state, "app.quit")
    assert state.quit_armed
    assert state.running, "one stray click must not end the program"
    assert find(ui, "app.quit").label == "Really quit?"

    click_id(ui, state, "app.quit")
    assert not state.running


def test_anything_else_answers_no_to_a_waiting_quit(state, ui):
    for cid in ("time.pause", "view.trails", "tool.throw"):
        state.quit_armed = True
        ui.build(state, SIZE)
        click_id(ui, state, cid)
        assert not state.quit_armed
        assert state.running


def test_a_press_in_empty_space_answers_no_as_well(state, ui):
    state.quit_armed = True
    ui.build(state, SIZE)
    ui.on_event(pygame.event.Event(pygame.MOUSEBUTTONDOWN, pos=(500, 400), button=1),
                state)
    assert not state.quit_armed
    assert state.running


def test_escape_calls_off_a_waiting_quit_before_anything_else(state, ui):
    from src.app.input import handle_event
    state.quit_armed = True
    state.help_visible = True
    state.select(state.sim.world.bodies[3].id)
    handle_event(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_ESCAPE, mod=0),
                 state, None, ui)
    assert not state.quit_armed
    assert state.help_visible, "escape was spent on the quit, not the help"
    assert state.running


def test_the_keyboard_still_quits_outright(state, ui):
    from src.app.input import handle_event
    handle_event(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_q,
                                    mod=pygame.KMOD_META), state, None, ui)
    assert not state.running


def test_the_zoom_buttons_work_without_a_wheel(state, ui):
    before = state.camera.zoom
    click_id(ui, state, "view.zoom", payload=1)
    assert state.camera.zoom > before
    click_id(ui, state, "view.zoom", payload=-1)
    assert state.camera.zoom == pytest.approx(before)


def test_the_zoom_slider_exists_and_shows_the_current_zoom(state, ui):
    slider = find(ui, "view.zoom_slider")
    assert slider is not None
    assert slider.value == pytest.approx(state.camera.zoom_fraction)


def test_clicking_the_top_of_the_slider_zooms_all_the_way_in(state, ui):
    slider = find(ui, "view.zoom_slider")
    ui.on_event(pygame.event.Event(pygame.MOUSEBUTTONDOWN,
                                   pos=(slider.rect.centerx, slider.rect.top),
                                   button=1), state)
    ui.on_event(pygame.event.Event(pygame.MOUSEBUTTONUP,
                                   pos=(slider.rect.centerx, slider.rect.top),
                                   button=1), state)
    assert state.camera.zoom_fraction == pytest.approx(1.0, abs=0.02)


def test_clicking_the_bottom_of_the_slider_zooms_all_the_way_out(state, ui):
    slider = find(ui, "view.zoom_slider")
    bottom = slider.rect.bottom - 1  # collidepoint excludes the bottom edge
    ui.on_event(pygame.event.Event(pygame.MOUSEBUTTONDOWN,
                                   pos=(slider.rect.centerx, bottom),
                                   button=1), state)
    ui.on_event(pygame.event.Event(pygame.MOUSEBUTTONUP,
                                   pos=(slider.rect.centerx, bottom),
                                   button=1), state)
    assert state.camera.zoom_fraction == pytest.approx(0.0, abs=0.02)


def test_dragging_the_slider_off_its_own_rect_keeps_tracking(state, ui):
    """A drag that leaves the track must still move the thumb, the way the
    button drag above stays swallowed once it starts on a panel."""
    slider = find(ui, "view.zoom_slider")
    ui.on_event(pygame.event.Event(pygame.MOUSEBUTTONDOWN,
                                   pos=slider.rect.center, button=1), state)
    moved = pygame.event.Event(pygame.MOUSEMOTION,
                               pos=(slider.rect.centerx, slider.rect.top - 500),
                               rel=(0, -500), buttons=(1, 0, 0))
    assert ui.on_event(moved, state)
    assert state.camera.zoom_fraction == pytest.approx(1.0, abs=1e-9)


def test_releasing_the_slider_does_not_fire_a_second_notch(state, ui):
    """Unlike a button, a slider must not act again on release -- it already
    acted continuously while held.

    ``set_zoom_fraction`` is idempotent, so a before/after equality check would
    pass even with the release guard removed -- a call count is what actually
    tells the two apart.
    """
    calls = []
    original = state.camera.set_zoom_fraction
    state.camera.set_zoom_fraction = lambda frac: (calls.append(frac), original(frac))[1]

    slider = find(ui, "view.zoom_slider")
    ui.on_event(pygame.event.Event(pygame.MOUSEBUTTONDOWN,
                                   pos=(slider.rect.centerx, slider.rect.top),
                                   button=1), state)
    assert len(calls) == 1
    ui.on_event(pygame.event.Event(pygame.MOUSEBUTTONUP,
                                   pos=(slider.rect.centerx, slider.rect.top),
                                   button=1), state)
    assert len(calls) == 1


def test_the_zoom_slider_rebuilds_to_match_the_wheel_and_the_buttons(state, ui):
    state.camera.zoom_by(20)
    ui.build(state, SIZE)
    assert find(ui, "view.zoom_slider").value == pytest.approx(
        state.camera.zoom_fraction)
    click_id(ui, state, "view.zoom", payload=1)
    assert find(ui, "view.zoom_slider").value == pytest.approx(
        state.camera.zoom_fraction)


def test_going_home_undoes_being_lost(state, ui):
    state.camera.centre[:] = (900.0, -400.0)
    state.camera.zoom_by(-60)
    state.camera.follow_id = state.sim.world.bodies[2].id
    click_id(ui, state, "view.home")
    assert tuple(state.camera.centre) == (0.0, 0.0)
    assert state.camera.follow_id is None
    assert state.camera.zoom == pytest.approx(
        SIZE[0] / state.camera.DEFAULT_AU_ACROSS)


def test_help_opens_and_closes_from_its_button(state, ui):
    click_id(ui, state, "view.help")
    assert state.help_visible
    assert ui.modal is not None


def test_controls_needing_a_display_are_disabled_without_one(ui):
    assert not find(ui, "view.grid").enabled
    assert not find(ui, "view.fullscreen").enabled


def test_a_disabled_control_does_nothing_when_clicked(state, ui):
    grid = find(ui, "view.grid")
    assert ui.control_at(grid.rect.center) is None
    assert not ui.perform(grid, state, None)


# --- the body list ----------------------------------------------------------------

def test_the_list_offers_every_body_by_name(state, ui):
    names = [c.label for c in ui.controls if c.id == "select.body"]
    assert "Earth" in names and "Jupiter" in names


def test_clicking_a_name_selects_that_body(state, ui):
    earth = next(b for b in state.sim.world.bodies if b.name == "Earth")
    click_id(ui, state, "select.body", payload=earth.id)
    assert state.selected_id == earth.id


def test_picking_a_name_closes_the_drawer(state, ui):
    """Borrowed space, not a permanent column -- picking a name is what the
    drawer is for, so picking one closes it, the same as its own X does."""
    earth = next(b for b in state.sim.world.bodies if b.name == "Earth")
    assert state.show_body_list
    click_id(ui, state, "select.body", payload=earth.id)
    assert not state.show_body_list
    assert find(ui, "select.body") is None


def test_the_list_closes_and_the_rail_button_brings_it_back(state, ui):
    click_id(ui, state, "panel.bodies.close")
    assert not state.show_body_list
    assert find(ui, "select.body") is None
    click_id(ui, state, "panel.bodies")
    assert state.show_body_list
    assert find(ui, "select.body") is not None


def test_the_list_scrolls_and_the_offset_stays_inside_it(state, ui):
    # A short window is what forces a scroll; a tall one shows everything.
    short = (1000, 260)
    ui.build(state, short)
    scroll_down = find(ui, "list.scroll", payload=1)
    assert scroll_down is not None

    click(ui, state, scroll_down, size=short)
    assert state.body_list_offset == 1

    for _ in range(50):
        state.scroll_body_list(1)
    ui.build(state, short)
    assert state.body_list_offset < state.sim.world.n
    assert not find(ui, "list.scroll", payload=1).enabled
    assert find(ui, "list.scroll", payload=-1).enabled


def test_a_window_tall_enough_for_every_body_has_no_scroll_buttons(state, ui):
    ui.build(state, (1000, 1400))
    assert find(ui, "list.scroll") is None
    assert state.body_list_offset == 0


def test_the_wheel_over_a_panel_scrolls_instead_of_zooming(state, ui):
    ui.build(state, (1000, 260))
    row = find(ui, "select.body")
    ui.mouse = row.rect.center
    zoom = state.camera.zoom
    assert ui.on_event(pygame.event.Event(pygame.MOUSEWHEEL, y=-1), state)
    assert state.body_list_offset == 1
    assert state.camera.zoom == zoom


def test_bodies_destroyed_under_an_open_list_do_not_break_it(state, ui):
    for _ in range(4):
        state.scroll_body_list(1)
    while state.sim.world.n > 2:
        state.sim.world.remove(state.sim.world.n - 1)
    ui.build(state, SIZE)
    assert state.body_list_offset == 0
    assert len([c for c in ui.controls if c.id == "select.body"]) == 2


# --- the selected body ------------------------------------------------------------

def test_the_action_buttons_appear_only_with_a_selection(state, ui):
    assert find(ui, "body.follow") is None
    state.select(state.sim.world.bodies[3].id)
    ui.build(state, SIZE)
    assert {"body.follow", "body.flyto", "body.stop", "body.delete"} <= ids(ui)


def test_follow_toggles_rather_than_only_switching_on(state, ui):
    body = state.sim.world.bodies[3]
    state.select(body.id)
    ui.build(state, SIZE)
    click_id(ui, state, "body.follow")
    assert state.camera.follow_id == body.id
    assert find(ui, "body.follow").active
    click_id(ui, state, "body.follow")
    assert state.camera.follow_id is None


def test_fly_to_zooms_in_and_follows(state, ui):
    body = state.sim.world.bodies[3]
    state.select(body.id)
    ui.build(state, SIZE)
    before = state.camera.zoom
    click_id(ui, state, "body.flyto")
    assert state.camera.zoom > before
    assert state.camera.follow_id == body.id


def test_stop_takes_all_the_speed_away(state, ui):
    body = next(b for b in state.sim.world.bodies if b.name == "Earth")
    state.select(body.id)
    ui.build(state, SIZE)
    assert np.hypot(*state.sim.world.vel[state.selected_index]) > 0
    click_id(ui, state, "body.stop")
    index = state.sim.world.index_of(body.id)
    assert np.allclose(state.sim.world.vel[index], 0.0)


def test_delete_removes_the_body_and_clears_the_selection(state, ui):
    body = state.sim.world.bodies[3]
    state.select(body.id)
    ui.build(state, SIZE)
    before = state.sim.world.n
    click_id(ui, state, "body.delete")
    assert state.sim.world.n == before - 1
    assert state.sim.world.index_of(body.id) is None
    assert state.selected_id is None


# --- press, release and what the world sees ---------------------------------------

def test_a_press_on_a_button_never_reaches_the_world(state, ui):
    button = find(ui, "time.pause")
    assert ui.blocks(button.rect.center)
    assert ui.on_event(pygame.event.Event(pygame.MOUSEBUTTONDOWN,
                                          pos=button.rect.center, button=1), state)


def test_a_press_on_bare_panel_is_swallowed_too(state, ui):
    """Otherwise the camera pans when you miss a button by two pixels."""
    panel = ui.panels[0][0]
    spot = (panel.right - 1, panel.bottom - 1)
    assert ui.control_at(spot) is None
    assert ui.over_panel(spot)
    assert ui.on_event(pygame.event.Event(pygame.MOUSEBUTTONDOWN, pos=spot,
                                          button=1), state)


def test_empty_space_is_left_alone_for_the_world(state, ui):
    spot = (SIZE[0] // 2, SIZE[1] // 2)
    assert not ui.blocks(spot)
    assert not ui.on_event(pygame.event.Event(pygame.MOUSEBUTTONDOWN, pos=spot,
                                              button=1), state)


def test_a_click_dragged_off_a_button_is_cancelled(state, ui):
    button = find(ui, "time.pause")
    ui.on_event(pygame.event.Event(pygame.MOUSEBUTTONDOWN, pos=button.rect.center,
                                   button=1), state)
    ui.on_event(pygame.event.Event(pygame.MOUSEBUTTONUP, pos=(500, 400), button=1),
                state)
    assert not state.sim.paused


def test_a_drag_that_began_on_a_panel_keeps_being_swallowed(state, ui):
    button = find(ui, "time.pause")
    ui.on_event(pygame.event.Event(pygame.MOUSEBUTTONDOWN, pos=button.rect.center,
                                   button=1), state)
    moved = pygame.event.Event(pygame.MOUSEMOTION, pos=(500, 400), rel=(30, 0),
                               buttons=(1, 0, 0))
    assert ui.on_event(moved, state)


def test_a_drag_that_began_in_space_is_never_stolen(state, ui):
    ui.on_event(pygame.event.Event(pygame.MOUSEBUTTONDOWN, pos=(500, 400), button=1),
                state)
    over_button = find(ui, "time.pause").rect.center
    moved = pygame.event.Event(pygame.MOUSEMOTION, pos=over_button, rel=(1, 1),
                               buttons=(1, 0, 0))
    assert not ui.on_event(moved, state)


def test_clicking_a_button_does_not_also_throw_a_body(state, ui):
    """The whole reason the interface is routed before the tools."""
    from src.app.input import handle_event
    from src.app.tools import LaunchTool
    state.set_tool(LaunchTool)
    ui.build(state, SIZE)
    before = state.sim.world.n
    button = find(ui, "time.pause")
    for event in (pygame.event.Event(pygame.MOUSEBUTTONDOWN, pos=button.rect.center,
                                     button=1),
                  pygame.event.Event(pygame.MOUSEBUTTONUP, pos=button.rect.center,
                                     button=1)):
        handle_event(event, state, None, ui)
    assert state.sim.world.n == before
    assert state.sim.paused


# --- holding a stepper ------------------------------------------------------------

def test_a_held_stepper_repeats_but_only_after_a_pause(state, ui):
    from src.app import ui as ui_module
    earth = next(b for b in state.sim.world.bodies if b.name == "Earth")
    state.select(earth.id)
    click_id(ui, state, "tool.edit")
    plus = find(ui, "edit.step", payload=("mass", 3))

    ui.on_event(pygame.event.Event(pygame.MOUSEBUTTONDOWN, pos=plus.rect.center,
                                   button=1), state)
    start = float(state.sim.world.mass[state.selected_index])
    now = pygame.time.get_ticks()

    assert ui.update(state, None, now=now) == 0
    assert float(state.sim.world.mass[state.selected_index]) == start

    fired = ui.update(state, None, now=now + ui_module.REPEAT_DELAY_MS
                      + 3 * ui_module.REPEAT_INTERVAL_MS)
    assert fired >= 3
    assert float(state.sim.world.mass[state.selected_index]) > start


def test_a_long_stall_cannot_spend_the_frame_repeating(state, ui):
    from src.app import ui as ui_module
    state.select(next(b for b in state.sim.world.bodies if b.name == "Earth").id)
    click_id(ui, state, "tool.edit")
    plus = find(ui, "edit.step", payload=("mass", 3))
    ui.on_event(pygame.event.Event(pygame.MOUSEBUTTONDOWN, pos=plus.rect.center,
                                   button=1), state)
    fired = ui.update(state, None, now=pygame.time.get_ticks() + 100_000)
    assert fired == ui_module.MAX_REPEATS_PER_FRAME


def test_releasing_stops_the_repeat(state, ui):
    state.select(next(b for b in state.sim.world.bodies if b.name == "Earth").id)
    click_id(ui, state, "tool.edit")
    plus = find(ui, "edit.step", payload=("mass", 3))
    ui.on_event(pygame.event.Event(pygame.MOUSEBUTTONDOWN, pos=plus.rect.center,
                                   button=1), state)
    ui.on_event(pygame.event.Event(pygame.MOUSEBUTTONUP, pos=plus.rect.center,
                                   button=1), state)
    assert ui.update(state, None, now=pygame.time.get_ticks() + 100_000) == 0


def test_a_stepper_acts_on_the_way_down(state, ui):
    """Waiting for the release would make a stepper feel broken."""
    state.select(next(b for b in state.sim.world.bodies if b.name == "Earth").id)
    click_id(ui, state, "tool.edit")
    plus = find(ui, "edit.step", payload=("mass", 3))
    before = float(state.sim.world.mass[state.selected_index])
    ui.on_event(pygame.event.Event(pygame.MOUSEBUTTONDOWN, pos=plus.rect.center,
                                   button=1), state)
    assert float(state.sim.world.mass[state.selected_index]) > before


# --- the help panel ---------------------------------------------------------------

def test_help_is_modal_and_a_click_outside_puts_it_away(state, ui):
    state.help_visible = True
    ui.build(state, SIZE)
    assert ui.modal is not None
    outside = (5, SIZE[1] - 5)
    assert ui.on_event(pygame.event.Event(pygame.MOUSEBUTTONDOWN, pos=outside,
                                          button=1), state)
    assert not state.help_visible


def test_a_click_outside_the_help_is_spent_closing_it(state, ui):
    """It must not also press whatever button was underneath."""
    state.help_visible = True
    ui.build(state, SIZE)
    pause = find(ui, "time.pause")
    ui.on_event(pygame.event.Event(pygame.MOUSEBUTTONDOWN, pos=pause.rect.center,
                                   button=1), state)
    assert not state.help_visible
    assert not state.sim.paused


def test_the_help_close_button_closes_it(state, ui):
    state.help_visible = True
    ui.build(state, SIZE)
    close = ui._modal_control
    assert close is not None and ui.modal.contains(close.rect)
    click(ui, state, close)
    assert not state.help_visible


# --- everything at once -----------------------------------------------------------

def test_every_control_of_every_tool_can_be_pressed_without_raising(state, ui):
    from src.app.tools import EditTool, LaunchTool, SelectTool
    for tool in (SelectTool, LaunchTool, EditTool):
        state.set_tool(tool)
        state.select(state.sim.world.bodies[2].id)
        ui.build(state, SIZE)
        for control in list(ui.controls):
            if control.clickable:
                ui.perform(control, state, None)
            ui.build(state, SIZE)


def test_the_interface_draws_over_every_tool(state, ui):
    from src.app.tools import EditTool, LaunchTool, SelectTool
    from src.render import palette
    for tool in (SelectTool, LaunchTool, EditTool):
        state.set_tool(tool)
        state.select(state.sim.world.bodies[3].id)
        state.help_visible = tool is EditTool
        ui.build(state, SIZE)
        surface = pygame.Surface(SIZE)
        surface.fill(palette.SPACE)
        ui.draw(surface, state)
        ink = sum(1 for x in range(0, SIZE[0], 3) for y in range(0, SIZE[1], 3)
                  if surface.get_at((x, y))[:3] != palette.SPACE[:3])
        assert ink > 200


def test_hovering_a_control_raises_a_tooltip(state, ui):
    from src.render import palette
    button = find(ui, "time.pause")
    ui.mouse = button.rect.center
    surface = pygame.Surface(SIZE)
    surface.fill(palette.SPACE)
    ui.draw(surface, state)
    above = pygame.Rect(button.rect.x, button.rect.y - 40, button.rect.width, 30)
    ink = sum(1 for x in range(above.left, above.right)
              for y in range(max(0, above.top), above.bottom)
              if surface.get_at((x, y))[:3] != palette.SPACE[:3])
    assert ink > 20


def test_a_tiny_window_still_lays_out_without_raising(state, ui):
    state.select(state.sim.world.bodies[1].id)
    for size in ((320, 200), (480, 320), (2560, 1440)):
        ui.build(state, size)
        assert ui.controls


# --- the promise this interface makes ---------------------------------------------

#: The entries in the help table that are not keys at all, or whose mouse
#: equivalent is a gesture rather than a button: the wheel, dragging and
#: clicking *are* the mouse; Tab's equivalent is picking a name out of the list;
#: Esc's is clicking away from whatever is open.
NOT_EXPECTED_TO_BE_BUTTONS = {"Wheel", "Drag", "Click", "Double-click", "Tab", "Esc"}


def documented_shortcuts():
    """Every key the help panel promises, one per token.

    Read out of the help table rather than listed here, so that adding a
    shortcut and documenting it -- which is the same commit -- is what makes
    this test demand a button for it.
    """
    from src.render.hud import KEY_HELP
    keys = set()
    for combo, _ in KEY_HELP:
        keys |= set(combo.split())
    return keys - NOT_EXPECTED_TO_BE_BUTTONS


def test_every_shortcut_is_also_a_button(state, ui):
    from src.app.tools import EditTool, FlyTool, LaunchTool, SelectTool
    keys = set()
    for tool in (SelectTool, LaunchTool, EditTool, FlyTool):
        state.set_tool(tool)
        state.select(state.sim.world.bodies[3].id)
        ui.build(state, SIZE)
        keys |= {c.hotkey for c in ui.controls if c.hotkey}
    missing = documented_shortcuts() - keys
    assert not missing, "no button for %s" % sorted(missing)


def test_the_parity_check_is_actually_checking_something(state, ui):
    """A guard on the guard: if the help table stops being readable, say so."""
    found = documented_shortcuts()
    assert len(found) > 15
    assert {"Space", "Del", "Cmd+Q", "F5"} <= found


def test_selecting_a_body_never_requires_the_keyboard(state, ui):
    """Tab has a mouse equivalent: the list, by name."""
    assert [c for c in ui.controls if c.id == "select.body"]


def test_every_button_says_what_it_does(state, ui):
    from src.app.tools import EditTool, LaunchTool, SelectTool
    for tool in (SelectTool, LaunchTool, EditTool):
        state.set_tool(tool)
        state.select(state.sim.world.bodies[3].id)
        ui.build(state, SIZE)
        for control in ui.controls:
            if control.clickable:
                assert control.hint(), "%s explains nothing" % control.id


# --- locking a property -----------------------------------------------------------

def _edit_earth(state, ui):
    earth = next(b for b in state.sim.world.bodies if b.name == "Earth")
    state.select(earth.id)
    click_id(ui, state, "tool.edit")
    return earth


def test_locking_a_row_greys_out_its_steppers(state, ui):
    """The second line of defence.  The first is in edits.py, and both have to
    be there: perform() is the door the held repeat and the tests come
    through."""
    _edit_earth(state, ui)
    assert find(ui, "edit.step", payload=("mass", 3)).enabled

    click_id(ui, state, "edit.lock", payload="mass")

    plus = find(ui, "edit.step", payload=("mass", 3))
    assert not plus.enabled
    before = float(state.sim.world.mass[state.selected_index])
    assert not ui.perform(plus, state, None)
    assert float(state.sim.world.mass[state.selected_index]) == pytest.approx(before)


def test_the_padlock_button_unlocks_as_well_as_locks(state, ui):
    from src import edits
    earth = _edit_earth(state, ui)
    click_id(ui, state, "edit.lock", payload="density")
    assert edits.is_locked(earth, "density")
    click_id(ui, state, "edit.lock", payload="density")
    assert not edits.is_locked(earth, "density")
    assert find(ui, "edit.step", payload=("density", 1)).enabled


def test_unlock_all_is_dead_until_something_is_locked(state, ui):
    earth = _edit_earth(state, ui)
    assert not find(ui, "edit.unlock_all").enabled

    click_id(ui, state, "edit.lock", payload="mass")
    click_id(ui, state, "edit.lock", payload="material")
    unlock = find(ui, "edit.unlock_all")
    assert unlock.enabled

    click(ui, state, unlock)
    assert earth.locked_fields == frozenset()
    assert not find(ui, "edit.unlock_all").enabled


def test_a_lock_outlives_leaving_the_tool_and_the_selection(state, ui):
    """It is a standing preference for this body, not a mode the Edit tool is
    in."""
    from src import edits
    earth = _edit_earth(state, ui)
    click_id(ui, state, "edit.lock", payload="mass")
    click_id(ui, state, "tool.look")
    state.select(state.sim.world.bodies[1].id)
    click_id(ui, state, "tool.edit")
    state.select(earth.id)
    ui.build(state, SIZE)
    assert edits.is_locked(earth, "mass")
    assert not find(ui, "edit.step", payload=("mass", 3)).enabled


def test_both_lock_keys_are_documented_and_have_buttons(state, ui):
    """The parity test above reads KEY_HELP; this one says which two keys this
    feature put in it, so a table edit that drops them is caught by name."""
    from src.render.hud import KEY_HELP
    documented = {combo for combo, _ in KEY_HELP}
    assert "K" in documented and "U" in documented
    _edit_earth(state, ui)
    assert {c.hotkey for c in ui.controls} >= {"K", "U"}


# --- the bottom hotbar --------------------------------------------------------------

def test_the_tool_picker_and_transport_share_the_bottom_hotbar(ui):
    tool_bottoms = {c.rect.bottom for c in ui.controls
                    if c.id.startswith("tool.") or c.id == "panel.bodies"}
    transport_tops = {c.rect.top for c in ui.controls
                      if c.id in ("sim.reset", "time.pause")}
    # Adjacent, not merely "somewhere above" -- the old rail already sat
    # above the transport bar spatially, so ordering alone would pass without
    # the merge.  A gap smaller than a button's own height is what "directly
    # above, same bar" actually means.
    gap = min(transport_tops) - max(tool_bottoms)
    assert 0 <= gap < ui.metrics.btn_h


def test_no_control_sits_on_the_left_edge_with_the_list_closed(state, ui):
    """The whole reason for this redesign: the left edge is open canvas by
    default.  It is legitimate for the Bodies drawer to use that space while
    it is open (a later test covers that) -- this is the *closed* baseline.
    """
    state.show_body_list = False
    ui.build(state, SIZE)
    left_band = ui.metrics.pad + ui.metrics.gap * 2
    assert not any(c.rect.left < left_band and c.rect.top > ui.metrics.top_h
                  for c in ui.controls)


def test_the_action_buttons_sit_above_the_inspector_stats(state, ui):
    body = state.sim.world.bodies[3]
    state.select(body.id)
    ui.build(state, SIZE)
    follow = find(ui, "body.follow")
    stats_top = ui.hud.inspector_rect(state, *SIZE).top
    assert follow.rect.bottom <= stats_top


def test_the_expand_chevron_opens_and_closes_the_full_stats(state, ui):
    body = state.sim.world.bodies[3]
    state.select(body.id)
    ui.build(state, SIZE)
    assert not state.inspector_expanded
    short = ui.hud.inspector_rect(state, *SIZE).height

    click_id(ui, state, "inspector.expand")
    assert state.inspector_expanded
    tall = ui.hud.inspector_rect(state, *SIZE).height
    assert tall > short


def test_there_is_no_expand_chevron_without_a_selection(state, ui):
    assert find(ui, "inspector.expand") is None


def test_the_throw_panel_sits_under_the_inspector_when_something_is_selected(state, ui):
    from src.app.tools import LaunchTool
    body = state.sim.world.bodies[3]
    state.select(body.id)
    state.set_tool(LaunchTool)
    ui.build(state, SIZE)
    preset = find(ui, "throw.preset", payload=0)
    inspector_bottom = ui.hud.inspector_rect(state, *SIZE).bottom
    assert preset.rect.top >= inspector_bottom


def test_the_throw_panel_sits_near_the_top_with_nothing_selected(state, ui):
    from src.app.tools import LaunchTool
    state.set_tool(LaunchTool)
    ui.build(state, SIZE)
    preset = find(ui, "throw.preset", payload=0)
    assert preset.rect.top < SIZE[1] // 3


def test_the_bodies_list_now_opens_on_the_left(state, ui):
    ui.build(state, SIZE)
    row = find(ui, "select.body")
    assert row.rect.centerx < SIZE[0] // 2


def test_a_selected_bodys_tether_reaches_toward_the_inspector(state, ui):
    from src.render import palette
    body = state.sim.world.bodies[3]
    state.select(body.id)
    ui.build(state, SIZE)
    surface = pygame.Surface(SIZE)
    surface.fill(palette.SPACE)
    ui.draw(surface, state)

    inspector = ui.hud.inspector_rect(state, *SIZE)
    # A strip just left of the inspector's left edge, spanning its height:
    # the tether has to cross this strip on its way from the body to the
    # panel, wherever exactly the body sits.
    strip_x = inspector.left - 4
    hit = any(surface.get_at((strip_x, y))[:3] == palette.SELECT
             for y in range(inspector.top, inspector.bottom))
    assert hit


def test_no_tether_without_a_selection(state, ui):
    from src.render import palette
    ui.build(state, SIZE)
    surface = pygame.Surface(SIZE)
    surface.fill(palette.SPACE)
    ui.draw(surface, state)


def _tether_visible(ui, state):
    from src.render import palette
    surface = pygame.Surface(SIZE)
    surface.fill(palette.SPACE)
    ui.draw(surface, state)
    inspector = ui.hud.inspector_rect(state, *SIZE)
    strip_x = inspector.left - 4
    return any(surface.get_at((strip_x, y))[:3] == palette.SELECT
              for y in range(inspector.top, inspector.bottom))


def test_the_tether_fades_a_few_seconds_after_a_selection(state, ui):
    from src.app.ui import TETHER_VISIBLE_MS
    body = state.sim.world.bodies[3]
    now = pygame.time.get_ticks()
    state.select(body.id, now=now)
    ui.build(state, SIZE)
    assert _tether_visible(ui, state)

    state.selected_at = now - TETHER_VISIBLE_MS - 1
    assert not _tether_visible(ui, state)


def test_reselecting_the_same_body_brings_the_tether_back(state, ui):
    from src.app.ui import TETHER_VISIBLE_MS
    body = state.sim.world.bodies[3]
    now = pygame.time.get_ticks()
    state.select(body.id, now=now - TETHER_VISIBLE_MS - 1)
    ui.build(state, SIZE)
    assert not _tether_visible(ui, state)

    state.select(body.id, now=now)
    assert _tether_visible(ui, state)


# --- the explain panel --------------------------------------------------------------

def _make_a_report(state):
    from src.app.state import SETTLE_MS
    from src.app.tools import EditTool
    state.set_tool(EditTool)
    world = state.sim.world
    index = [b.name for b in world.bodies].index("Sun")
    state.select(world.bodies[index].id)
    state.tool.field_index = 0          # mass
    state.tool._scrub(state, 12)
    state.explain.settle(world, pygame.time.get_ticks() + SETTLE_MS + 1)
    assert state.explain.report is not None


def test_the_explain_panel_has_a_close_button(state, ui):
    _make_a_report(state)
    ui.build(state, SIZE)
    assert find(ui, "explain.close") is not None
    ui.perform(find(ui, "explain.close"), state)
    assert state.explain.report is None


def test_the_maths_button_toggles_and_disappears_with_the_report(state, ui):
    _make_a_report(state)
    ui.build(state, SIZE)
    assert state.show_maths is False
    ui.perform(find(ui, "explain.maths"), state)
    assert state.show_maths is True
    state.explain.dismiss()
    ui.build(state, SIZE)
    assert find(ui, "explain.maths") is None


def test_y_is_documented_and_has_a_button(state, ui):
    from src.render.hud import KEY_HELP
    assert "Y" in {combo for combo, _ in KEY_HELP}
    ui.build(state, SIZE)
    assert find(ui, "view.explain").hotkey == "Y"


@pytest.mark.parametrize("s", (1, 2, 3))
def test_view_cluster_labels_fit_their_button_at_every_grid(s):
    """Every view-cluster button is drawn at the fixed square ``icon_btn_w``,
    and drawing clips to it -- so a label wider than the box is not a
    neighbour-spilling bug, it is a silently half-erased glyph (this is what
    "Why" did at the default grid: 25px of label in an 18px button).  A drawn
    icon fits its box by construction (see the "explain" icon test), so this
    only has teeth for a button that still draws a label instead -- which is
    exactly the case this guards against reintroducing.
    """
    from src.app import config
    from src.app.ui import VIEW_BUTTONS
    from src.render.pixelfont import PixelFont
    from src.render.widgets import Metrics
    font = PixelFont(config.font_px(s))
    w, h = config.min_internal(s)
    m = Metrics(font, w, h)
    for cid, icon, label, key, tip in VIEW_BUTTONS:
        if icon or not label:
            continue
        assert font.width(label) <= m.icon_btn_w, (
            "%s's label %r is %dpx wide, wider than the %dpx button at "
            "grid %d" % (cid, label, font.width(label), m.icon_btn_w, s))


def test_turning_explanations_off_hides_a_standing_report(state, ui):
    _make_a_report(state)
    ui.build(state, SIZE)
    ui.perform(find(ui, "view.explain"), state)
    assert state.explain.enabled is False
    ui.build(state, SIZE)
    assert find(ui, "explain.close") is None
