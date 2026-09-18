"""The window: resizing it, the size presets, and the pixels inside it.

The bug the old version of these existed for was invisible from inside the
program: ``SCALED`` pinned the window's minimum size to the buffer and every
second ``set_mode`` failed silently.  The lesson survives the move to
``pygame.Window``: check the *outcome* -- the buffer, the window, the
coordinates -- and never merely that a call returned.

The dummy driver reports one pixel per point.  Tests that need a Retina
display say so with ``pixels_per_point=2``, which is what the constructor
offers in place of measuring.
"""

import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import pygame
import pytest

from src.app import config
from src.app.config import (DisplayContext, MIN_WINDOW, WINDOW_FRACTIONS,
                            clamp_window, font_px, internal_size,
                            min_internal, preset_size)

#: A 16" MacBook Pro, the machine the measurements in config.py come from.
LAPTOP = (1728, 1117)


@pytest.fixture
def ctx():
    pygame.display.init()
    context = DisplayContext(size=(800, 600))
    yield context
    context.close()
    pygame.display.quit()


@pytest.fixture
def retina():
    """A context that believes the display has two pixels per point."""
    pygame.display.init()
    context = DisplayContext(size=(800, 600), pixels_per_point=2)
    yield context
    context.close()
    pygame.display.quit()


# --- the numbers ----------------------------------------------------------------

def test_the_smallest_window_is_smaller_than_a_third_of_a_laptop_screen():
    """Otherwise the one-third preset clamps back up and stops being one third."""
    third = preset_size(LAPTOP, 1 / 3)
    assert MIN_WINDOW[0] < LAPTOP[0] // 3
    assert third[0] == round(LAPTOP[0] / 3)


def test_a_preset_leaves_room_for_the_menu_bar_and_the_title_bar():
    for fraction, _ in WINDOW_FRACTIONS:
        w, h = preset_size(LAPTOP, fraction)
        assert h < LAPTOP[1] - config.MENU_BAR_POINTS
        assert w <= LAPTOP[0]


def test_the_presets_are_the_fractions_they_say_they_are():
    widths = [preset_size(LAPTOP, f)[0] for f, _ in WINDOW_FRACTIONS]
    assert widths == sorted(widths, reverse=True)
    assert widths[0] == LAPTOP[0]
    assert widths[2] == LAPTOP[0] // 2


def test_clamp_holds_a_window_between_the_floor_and_the_screen():
    assert clamp_window((10, 10)) == MIN_WINDOW
    assert clamp_window((9999, 9999), LAPTOP) == LAPTOP
    assert clamp_window((900, 700), LAPTOP) == (900, 700)


# --- the window itself -----------------------------------------------------------

def test_the_window_is_resizable_and_not_pinned_to_the_screen(ctx):
    assert ctx.resizable
    assert not ctx.fullscreen
    assert ctx.window[0] < ctx.desktop[0] or ctx.window[1] < ctx.desktop[1]


def test_resizing_moves_the_buffer_with_the_window(ctx):
    ctx.resize((640, 480))
    assert ctx.window == (640, 480)
    assert (ctx.width, ctx.height) == (640, 480)
    ctx.resize((1000, 700))
    assert (ctx.width, ctx.height) == (1000, 700)


def test_a_drag_below_the_floor_is_pushed_back_out_to_it(ctx):
    ctx.resize((100, 80))
    assert ctx.window == MIN_WINDOW
    assert (ctx.width, ctx.height) == MIN_WINDOW


def test_every_preset_lands_on_a_window_the_interface_fits_in(ctx):
    seen = []
    for _ in range(len(WINDOW_FRACTIONS)):
        label = ctx.cycle_window_size()
        seen.append(label)
        assert ctx.width >= min_internal(ctx.s)[0]
        assert ctx.height >= min_internal(ctx.s)[1]
    assert sorted(seen) == sorted(name for _, name in WINDOW_FRACTIONS)


def test_the_presets_cycle_back_round(ctx):
    first = ctx.cycle_window_size()
    for _ in range(len(WINDOW_FRACTIONS) - 1):
        ctx.cycle_window_size()
    assert ctx.cycle_window_size() == first


def test_a_preset_leaves_fullscreen(ctx):
    ctx.toggle_fullscreen()
    assert ctx.fullscreen
    ctx.cycle_window_size()
    assert not ctx.fullscreen


def test_fullscreen_and_back_returns_to_the_size_it_left(ctx):
    ctx.resize((720, 540))
    before = ctx.window
    ctx.toggle_fullscreen()
    assert ctx.fullscreen
    ctx.toggle_fullscreen()
    assert not ctx.fullscreen
    assert ctx.window == before


def test_size_changed_notices_a_resize_the_context_was_not_told_about(ctx):
    """The once-a-frame safety poll: a maximise or a macOS tile reaches the
    window without an event we listen for."""
    assert not ctx.size_changed()
    ctx._window.size = (900, 650)
    pygame.event.pump()
    assert ctx.size_changed()
    ctx.resize()
    assert not ctx.size_changed()
    assert ctx.window == (900, 650)


# --- pixels per point ----------------------------------------------------------------

def test_the_dummy_driver_has_one_pixel_per_point(ctx):
    assert ctx.native == 1
    assert ctx.s == 1
    assert ctx.usable_grids == (1,)
    assert ctx.surface.get_size() == ctx.window


def test_a_retina_display_opens_at_native_density(retina):
    assert retina.native == 2
    assert retina.s == 2
    assert retina.usable_grids == (2, 1)
    assert (retina.width, retina.height) == internal_size(800, 600, 2)
    assert retina.surface.get_size() == (1600, 1200)
    assert (retina.texture.width, retina.texture.height) == (1600, 1200)


def test_standard_density_is_the_escape_hatch(retina):
    """Half the pixels for when the physics budget is saturated at fullscreen."""
    retina.set_grid(1)
    assert retina.s == 1
    assert retina.surface.get_size() == (800, 600)
    assert (retina.texture.width, retina.texture.height) == (800, 600)
    assert retina.upscale == 1.0


def test_a_density_the_display_does_not_have_is_refused(retina):
    retina.set_grid(3)
    assert retina.s == 2
    retina.set_grid(1)
    retina.set_grid(7)
    assert retina.s == 2, "refused, and back to native rather than left wherever it was"


def test_a_resize_rebuilds_the_texture_to_match(retina):
    retina.resize((900, 650))
    assert retina.surface.get_size() == (1800, 1300)
    assert (retina.texture.width, retina.texture.height) == (1800, 1300)


def test_present_runs_after_every_kind_of_change(retina):
    retina.present()
    retina.resize((640, 480))
    retina.present()
    retina.set_grid(1)
    retina.present()
    retina.toggle_fullscreen()
    retina.present()


def test_the_font_and_the_floor_follow_the_density(retina):
    assert retina.font_px == 2 * font_px(1)
    assert min_internal(retina.s) == (2 * MIN_WINDOW[0], 2 * MIN_WINDOW[1])


# --- window points to internal pixels ----------------------------------------------

def test_at_one_pixel_per_point_a_click_needs_no_translation(ctx):
    assert ctx.to_internal((37, 41)) == (37, 41)


def test_at_two_pixels_per_point_a_click_is_doubled(retina):
    assert retina.to_internal((100, 50)) == (200, 100)


def test_a_mouse_event_is_translated_on_its_way_in(retina):
    event = pygame.event.Event(pygame.MOUSEBUTTONDOWN, {"pos": (60, 80), "button": 1})
    assert retina.map_event(event).pos == (120, 160)


def test_a_translated_event_keeps_everything_else_it_carried(retina):
    event = pygame.event.Event(pygame.MOUSEMOTION,
                               {"pos": (100, 100), "rel": (4, 8), "buttons": (1, 0, 0)})
    mapped = retina.map_event(event)
    assert mapped.buttons == (1, 0, 0)
    assert mapped.rel == (8, 16)


def test_an_event_with_no_position_passes_straight_through(retina):
    event = pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_SPACE, "mod": 0})
    assert retina.map_event(event) is event


def test_at_one_pixel_per_point_an_event_is_not_copied(ctx):
    event = pygame.event.Event(pygame.MOUSEBUTTONDOWN, {"pos": (60, 80), "button": 1})
    assert ctx.map_event(event) is event


# --- the keys ---------------------------------------------------------------------

def _key(key, mod=0):
    return pygame.event.Event(pygame.KEYDOWN, {"key": key, "mod": mod, "unicode": ""})


def test_w_steps_the_window_size(ctx):
    from src.app.input import handle_event
    from src.app.state import AppState
    from src.render.camera import Camera
    from src.render.draw import ViewSettings
    from src.sim import Simulation

    state = AppState(Simulation(), Camera(ctx.width, ctx.height), ViewSettings())
    before = ctx.window
    handle_event(_key(pygame.K_w), state, ctx)
    assert ctx.window != before
    assert (state.camera.width, state.camera.height) == (ctx.width, ctx.height)


def test_g_toggles_the_density_on_a_retina_display(retina):
    from src.app.input import handle_event
    from src.app.state import AppState
    from src.render.camera import Camera
    from src.render.draw import ViewSettings
    from src.sim import Simulation

    state = AppState(Simulation(), Camera(retina.width, retina.height), ViewSettings())
    handle_event(_key(pygame.K_g), state, retina)
    assert retina.s == 1
    assert (state.camera.width, state.camera.height) == (800, 600)
    handle_event(_key(pygame.K_g), state, retina)
    assert retina.s == 2
    assert (state.camera.width, state.camera.height) == (1600, 1200)


def test_command_w_asks_to_quit_rather_than_resizing(ctx):
    """Cmd+W closes a window everywhere else on this platform, and nothing here
    is saved, so falling through to the resize would be the one unrecoverable
    guess."""
    from src.app.input import handle_event
    from src.app.state import AppState
    from src.render.camera import Camera
    from src.render.draw import ViewSettings
    from src.sim import Simulation

    state = AppState(Simulation(), Camera(ctx.width, ctx.height), ViewSettings())
    before = ctx.window
    handle_event(_key(pygame.K_w, pygame.KMOD_META), state, ctx)
    assert ctx.window == before
    assert state.quit_armed
    assert state.running, "one press arms it, it does not go"


def test_a_drag_is_not_capped_at_the_primary_display(ctx):
    """``desktop`` is the *primary* screen.  A window dragged onto a bigger
    external monitor is entitled to be bigger than the laptop it started on, so
    only the presets -- which are fractions of a screen -- are capped to one."""
    huge = (ctx.desktop[0] + 400, ctx.desktop[1] + 300)
    ctx.resize(huge)
    assert ctx.window_size == huge


def test_close_destroys_the_window_and_can_be_called_twice(ctx):
    ctx.close()
    ctx.close()
    assert ctx._window is None
