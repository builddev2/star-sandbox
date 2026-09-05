"""The window: resizing it, the size presets, and the grid that lives inside it.

The bug these exist for was invisible from inside the program.  ``SCALED``
pinned the window's minimum size to a desktop-sized buffer, so the window could
not be made smaller than the whole screen -- and every second ``set_mode``
raised ``failed to create renderer``, which ``config._open`` caught and turned
into "keep the display we had".  The fullscreen toggle and the grid switch both
appeared to work and did nothing at all.  So these tests check the *outcome* --
the buffer, the window, the coordinates -- and never merely that a call
returned.
"""

import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import pygame
import pytest

from src.app import config
from src.app.config import (DisplayContext, GRID_CHOICES, MIN_WINDOW,
                            WINDOW_FRACTIONS, clamp_window, font_px,
                            grid_fits, internal_size, min_internal,
                            preset_size)

#: A 16" MacBook Pro, the machine the measurements in config.py come from.
LAPTOP = (1728, 1117)


@pytest.fixture
def ctx():
    pygame.display.init()
    context = DisplayContext(size=(800, 600))
    yield context
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


def test_a_coarser_grid_needs_fewer_internal_pixels_but_never_more():
    for finer, coarser in zip(GRID_CHOICES, GRID_CHOICES[1:]):
        assert min_internal(coarser) <= min_internal(finer)


def test_the_finest_grid_fits_wherever_the_window_is_legal():
    assert grid_fits(MIN_WINDOW[0], MIN_WINDOW[1], 1)


def test_a_coarse_grid_does_not_fit_a_small_window():
    """It is a divisor: a third of a small window is a very small buffer."""
    assert not grid_fits(MIN_WINDOW[0], MIN_WINDOW[1], GRID_CHOICES[-1])
    assert grid_fits(LAPTOP[0], LAPTOP[1], GRID_CHOICES[-1])


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


# --- the grid inside the window ---------------------------------------------------

def test_the_default_grid_draws_straight_into_the_window(ctx):
    """No copy and no upscale in the common case: the window is the surface."""
    assert ctx.s == 1
    assert ctx.surface is ctx.display
    assert ctx.origin == (0, 0)
    assert ctx.upscale == 1.0


def test_a_coarser_grid_draws_into_its_own_smaller_buffer(ctx):
    ctx.resize((1000, 700))
    ctx.set_grid(2)
    assert ctx.s == 2
    assert ctx.surface is not ctx.display
    assert ctx.surface.get_size() == internal_size(1000, 700, 2)
    assert ctx.upscale == 2.0


def test_the_upscale_is_always_an_exact_integer(ctx):
    """A fractional one is what smears the pixel grid it exists to preserve."""
    ctx.resize((1001, 701))
    for s in GRID_CHOICES:
        if not grid_fits(1001, 701, s):
            continue
        ctx.set_grid(s)
        assert ctx.upscale == float(s)
        assert ctx.width * s <= ctx.window[0]
        assert ctx.height * s <= ctx.window[1]
        assert ctx.window[0] - ctx.width * s < s
        assert ctx.window[1] - ctx.height * s < s


def test_a_window_too_small_for_a_grid_refuses_it(ctx):
    ctx.resize(MIN_WINDOW)
    ctx.set_grid(GRID_CHOICES[-1])
    assert ctx.s in ctx.usable_grids
    assert grid_fits(ctx.window[0], ctx.window[1], ctx.s)


def test_shrinking_the_window_steps_the_grid_back_down(ctx):
    ctx.resize((1024, 768))
    ctx.set_grid(GRID_CHOICES[-1])
    coarse = ctx.s
    ctx.resize(MIN_WINDOW)
    assert ctx.s < coarse
    assert ctx.width >= min_internal(ctx.s)[0]


def test_the_finest_grid_is_always_on_offer(ctx):
    ctx.resize(MIN_WINDOW)
    assert ctx.usable_grids[0] == GRID_CHOICES[0]


# --- window points to internal pixels ----------------------------------------------

def test_at_the_default_grid_a_click_needs_no_translation(ctx):
    assert ctx.to_internal((37, 41)) == (37, 41)


def test_a_coarser_grid_divides_the_click_by_the_grid(ctx):
    ctx.resize((1000, 700))
    ctx.set_grid(2)
    ox, oy = ctx.origin
    assert ctx.to_internal((ox + 100, oy + 50)) == (50, 25)


def test_a_click_in_the_letterbox_lands_outside_the_buffer(ctx):
    """Clamping it would slide it onto whichever control sits at the edge.

    1022 and 767 are both two more than a multiple of three, which is what
    leaves a letterbox to click in: the buffer is blown up by exactly three and
    the two spare points along each edge are painted black rather than drawn
    into.
    """
    ctx.resize((1022, 767))
    ctx.set_grid(3)
    assert ctx.s == 3, "this window was meant to be big enough for the coarse grid"
    assert ctx.origin != (0, 0)
    x, y = ctx.to_internal((0, 0))
    assert x < 0 and y < 0


def test_a_mouse_event_is_translated_on_its_way_in(ctx):
    ctx.resize((1000, 700))
    ctx.set_grid(2)
    ox, oy = ctx.origin
    event = pygame.event.Event(pygame.MOUSEBUTTONDOWN,
                               {"pos": (ox + 60, oy + 80), "button": 1})
    assert ctx.map_event(event).pos == (30, 40)


def test_a_translated_event_keeps_everything_else_it_carried(ctx):
    ctx.resize((1000, 700))
    ctx.set_grid(2)
    event = pygame.event.Event(pygame.MOUSEMOTION,
                               {"pos": (100, 100), "rel": (4, 8), "buttons": (1, 0, 0)})
    mapped = ctx.map_event(event)
    assert mapped.buttons == (1, 0, 0)
    assert mapped.rel == (2, 4)


def test_an_event_with_no_position_passes_straight_through(ctx):
    ctx.resize((1000, 700))
    ctx.set_grid(2)
    event = pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_SPACE, "mod": 0})
    assert ctx.map_event(event) is event


def test_the_font_grows_back_as_the_grid_coarsens(ctx):
    """Apparent text size is what stays constant, not the pixel count."""
    for s in GRID_CHOICES:
        assert font_px(s) * s >= config.TARGET_TEXT_POINTS - 2


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
