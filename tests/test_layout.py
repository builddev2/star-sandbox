"""The interface fits inside the window, at every size the window can be.

Now that the window resizes, ``MIN_INTERNAL`` is a promise: below it the
interface is not expected to fit, at or above it every control and every panel
must be inside the buffer.  A button laid out off the edge is not a cosmetic
problem -- it is a button that cannot be pressed, and the whole point of the
interface is that a mouse can reach everything.

This is what stops ``MIN_INTERNAL`` drifting: add a row to the transport bar or
a line to the help panel and, if it no longer fits, this fails rather than the
control quietly walking off screen.
"""

import os

import pygame
import pytest

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

from src.app import config
from src.app.state import AppState
from src.app.tools import EditTool, LaunchTool, SelectTool
from src.app.ui import UiLayer
from src.render.camera import Camera
from src.render.draw import ViewSettings
from src.render.hud import Hud
from src.render.pixelfont import PixelFont
from src.sim import Simulation

#: Window shapes worth checking: the floor, each preset on a 16" MacBook Pro,
#: and that laptop's fullscreen window.
LAPTOP = (1728, 1117)
WINDOWS = [config.MIN_WINDOW] \
    + [config.preset_size(LAPTOP, fraction) for fraction, _ in config.WINDOW_FRACTIONS] \
    + [(1728, 1080)]

TOOLS = (SelectTool, LaunchTool, EditTool)


@pytest.fixture(scope="module", autouse=True)
def display():
    pygame.display.init()
    pygame.display.set_mode((64, 64))
    yield
    pygame.display.quit()


def cases():
    for window in WINDOWS:
        for s in config.GRID_CHOICES:
            if not config.grid_fits(window[0], window[1], s):
                continue
            for tool in TOOLS:
                yield window, s, tool


def build(size, s, tool, show_list, show_help):
    """Lay the whole interface out and hand back the controls and the panels."""
    font = PixelFont(config.font_px(s))
    hud = Hud(font)
    internal = config.internal_size(size[0], size[1], s)
    state = AppState(Simulation(), Camera(*internal), ViewSettings())
    state.set_tool(tool)
    state.show_body_list = show_list
    state.help_visible = show_help
    state.select(state.sim.world.bodies[4].id)
    ui = UiLayer(font, hud)
    ui.build(state, internal, None)
    return ui, pygame.Rect(0, 0, *internal)


@pytest.mark.parametrize("window,s,tool", list(cases()),
                         ids=lambda v: getattr(v, "name", str(v)))
def test_every_control_is_inside_the_window(window, s, tool):
    for show_list in (False, True):
        for show_help in (False, True):
            ui, screen = build(window, s, tool, show_list, show_help)
            outside = sorted({c.id for c in ui.controls
                              if not screen.contains(c.rect)})
            assert not outside, \
                "%s off screen at %s grid %d (list=%s help=%s)" % (
                    outside, screen.size, s, show_list, show_help)


@pytest.mark.parametrize("window,s,tool", list(cases()),
                         ids=lambda v: getattr(v, "name", str(v)))
def test_every_panel_is_inside_the_window(window, s, tool):
    for show_list in (False, True):
        for show_help in (False, True):
            ui, screen = build(window, s, tool, show_list, show_help)
            outside = [tuple(rect) for rect, _ in ui.panels
                       if not screen.contains(rect)]
            assert not outside, \
                "panels %s off screen at %s grid %d" % (outside, screen.size, s)


@pytest.mark.parametrize("window,s,tool", list(cases()),
                         ids=lambda v: getattr(v, "name", str(v)))
def test_no_two_controls_overlap(window, s, tool):
    """"Inside the window" is not "reachable" -- a control drawn under
    another one is inside the buffer and still cannot be clicked.  The right
    column (actions, the inspector, the live tool's own controls) grew tall
    enough at the window floor to reach down into the zoom cluster's corner;
    this is the regression guard for that class of bug, not just this one
    instance of it.
    """
    for show_list in (False, True):
        ui, _ = build(window, s, tool, show_list, False)
        controls = ui.controls
        for i, a in enumerate(controls):
            for b in controls[i + 1:]:
                assert not a.rect.colliderect(b.rect), \
                    "%s overlaps %s at %s grid %d (list=%s)" % (
                        a.id, b.id, window, s, show_list)


def test_the_help_panel_never_outgrows_the_window():
    """It is the one panel sized by its contents rather than by the window."""
    for s in config.GRID_CHOICES:
        hud = Hud(PixelFont(config.font_px(s)))
        internal = config.min_internal(s)
        rect = hud.help_rect(*internal)
        assert pygame.Rect(0, 0, *internal).contains(rect)


def test_a_window_one_pixel_under_the_floor_is_not_promised_anything():
    """The floor is a real boundary, not a number picked with room to spare.

    If the interface still fitted well below ``MIN_INTERNAL`` the constant
    would be too generous, and the one-third preset would be needlessly wide.
    """
    assert config.MIN_INTERNAL[0] < LAPTOP[0] // 3


def test_the_edit_panel_stays_clear_of_the_transport_bar():
    """The Edit panel grows a row per field, and now a padlock on each of them.

    Nothing else enforces that it stops above the transport bar: the two tests
    above only ask that every control is inside the buffer, and a panel that
    reaches over Play is inside the buffer.  At the floor there are six rows of
    room and no more, which is why "Unlock all" shares the Collapse row instead
    of taking one of its own.
    """
    from src import edits
    from src.render.widgets import Metrics

    for s in config.GRID_CHOICES:
        internal = config.min_internal(s)
        font = PixelFont(config.font_px(s))
        metrics = Metrics(font, *internal)
        state = AppState(Simulation(), Camera(*internal), ViewSettings())
        state.set_tool(EditTool)
        ui = UiLayer(font, Hud(font))

        def deepest(name):
            index = next(i for i, b in enumerate(state.sim.world.bodies)
                         if b.name == name)
            state.select(state.sim.world.bodies[index].id)
            ui.build(state, internal, None)
            return index, max(c.rect.bottom for c in ui.controls
                              if c.id.startswith("edit."))

        for name in ("Earth", "Sun"):          # a planet: four rows; a star: three
            _, bottom = deepest(name)
            assert bottom <= metrics.transport_y, \
                "the Edit panel reaches %d, past the transport bar at %d " \
                "(grid %d, %s)" % (bottom, metrics.transport_y, s, name)

        index, _ = deepest("Sun")
        edits.collapse(state.sim.world, index)
        _, bottom = deepest("Sun")
        assert bottom <= metrics.transport_y, \
            "the black hole panel reaches %d, past %d (grid %d)" % (
                bottom, metrics.transport_y, s)
