"""The drawing primitives behind every button."""

import os

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
def font():
    from src.render.pixelfont import PixelFont
    return PixelFont(18)


def _ink(surface, palette_space=None):
    from src.render import palette
    background = palette_space or palette.SPACE[:3]
    w, h = surface.get_size()
    return sum(1 for x in range(w) for y in range(h)
               if surface.get_at((x, y))[:3] != background)


# --- metrics ----------------------------------------------------------------------

def test_every_size_grows_with_the_font(font):
    from src.render.pixelfont import PixelFont
    from src.render.widgets import Metrics
    small = Metrics(PixelFont(9), 800, 600)
    large = Metrics(PixelFont(24), 800, 600)
    assert large.btn_h > small.btn_h
    assert large.pad >= small.pad
    assert large.icon_btn_w > small.icon_btn_w


def test_the_bands_do_not_overlap(font):
    from src.render.widgets import Metrics
    m = Metrics(font, 800, 600)
    assert 0 < m.top_h < m.content_top
    assert m.content_top < m.content_bottom < m.transport_y + m.transport_h
    assert m.transport_y + m.transport_h + m.hint_h == 600


def test_button_width_pays_for_what_it_carries(font):
    from src.render.widgets import Control, Metrics
    m = Metrics(font, 800, 600)
    bare = Control("x", label="Follow")
    with_icon = Control("x", label="Follow", icon="follow")
    with_key = Control("x", label="Follow", icon="follow", hotkey="F")
    assert m.button_width(with_icon) > m.button_width(bare)
    assert m.button_width(with_key) > m.button_width(with_icon)
    assert m.button_width(Control("x", label="Follow", compact=True)) == m.icon_btn_w


# --- control state ----------------------------------------------------------------

def test_disabled_beats_every_other_state():
    from src.render.widgets import Control, STATE_DISABLED
    control = Control("x", enabled=False, active=True)
    assert control.state(hovered=True, pressed=True) == STATE_DISABLED


def test_state_priority_runs_down_to_rest():
    from src.render import widgets
    from src.render.widgets import Control
    assert Control("x").state(pressed=True) == widgets.STATE_DOWN
    assert Control("x", active=True).state() == widgets.STATE_ACTIVE
    assert Control("x").state(hovered=True) == widgets.STATE_HOVER
    assert Control("x").state() == widgets.STATE_REST


def test_the_tooltip_teaches_the_hotkey():
    from src.render.widgets import Control
    assert Control("x", tooltip="Freeze it", hotkey="Space").hint() == \
        "Freeze it   [Space]"
    # With nothing else to say, the label is the tooltip.
    assert Control("x", label="Follow").hint() == "Follow"


def test_a_readout_is_not_clickable():
    from src.render.widgets import Control
    assert not Control("x", label="1 yr/s", readout=True).clickable
    assert Control("x", label="Follow").clickable
    assert not Control("x", label="Follow", enabled=False).clickable


# --- drawing ----------------------------------------------------------------------

ICON_NAMES = ("look", "throw", "edit", "play", "pause", "slower", "faster", "reset",
              "trails", "sizes", "labels", "grid", "fullscreen", "follow", "flyto",
              "stop", "trash", "plus", "minus", "left", "right", "up", "down",
              "list", "close", "circle", "star", "help", "home", "power")


@pytest.mark.parametrize("name", ICON_NAMES)
def test_every_icon_draws_something_and_stays_inside_its_box(name, font):
    from src.render import widgets
    surface = pygame.Surface((40, 40))
    surface.fill((0, 0, 0))
    rect = pygame.Rect(10, 10, 16, 16)
    widgets.draw_icon(surface, name, rect, (255, 255, 255))
    drawn = [(x, y) for x in range(40) for y in range(40)
             if surface.get_at((x, y))[:3] != (0, 0, 0)]
    assert drawn, "%s drew nothing" % name
    for x, y in drawn:
        assert rect.inflate(2, 2).collidepoint(x, y), "%s escaped its box" % name


def test_an_unknown_icon_is_ignored_rather_than_fatal(font):
    from src.render import widgets
    surface = pygame.Surface((40, 40))
    surface.fill((0, 0, 0))
    widgets.draw_icon(surface, "not-an-icon", pygame.Rect(10, 10, 16, 16),
                      (255, 255, 255))
    assert _ink(surface, (0, 0, 0)) == 0


def test_a_button_draws_in_every_state(font):
    from src.render.widgets import Control, draw_button
    for kwargs in ({}, {"active": True}, {"enabled": False}, {"danger": True},
                   {"compact": True}, {"readout": True}):
        surface = pygame.Surface((120, 40))
        surface.fill((0, 0, 0))
        control = Control("x", label="Delete", icon="trash", hotkey="Del", **kwargs)
        control.rect = pygame.Rect(2, 2, 116, 30)
        draw_button(surface, font, control, hovered=True, pressed=True)
        assert _ink(surface, (0, 0, 0)) > 20


def test_a_button_without_a_rect_is_skipped(font):
    from src.render.widgets import Control, draw_button
    surface = pygame.Surface((40, 40))
    surface.fill((0, 0, 0))
    draw_button(surface, font, Control("x", label="Nowhere"))
    assert _ink(surface, (0, 0, 0)) == 0


def test_a_hotkey_cap_is_dropped_rather_than_overflowing(font):
    """A narrow button loses its key cap instead of painting outside itself."""
    from src.render.widgets import Control, draw_button
    surface = pygame.Surface((200, 40))
    surface.fill((0, 0, 0))
    control = Control("x", label="Fullscreen", hotkey="Cmd+F")
    control.rect = pygame.Rect(2, 2, 14, 30)
    draw_button(surface, font, control)
    for x in range(20, 200):
        for y in range(40):
            assert surface.get_at((x, y))[:3] == (0, 0, 0)


@pytest.mark.parametrize("anchor", [
    pygame.Rect(0, 0, 20, 20),                  # top-left corner
    pygame.Rect(180, 180, 20, 20),              # bottom-right corner
    pygame.Rect(90, 90, 20, 20),                # middle
])
def test_a_tooltip_is_always_fully_on_screen(anchor, font):
    from src.render.widgets import draw_tooltip
    surface = pygame.Surface((200, 200))
    rect = draw_tooltip(surface, font, "Keep the camera on it   [F]", anchor)
    assert rect is not None
    assert surface.get_rect().contains(rect)


def test_an_empty_tooltip_draws_nothing(font):
    from src.render.widgets import draw_tooltip
    surface = pygame.Surface((200, 200))
    surface.fill((0, 0, 0))
    assert draw_tooltip(surface, font, "", pygame.Rect(10, 10, 10, 10)) is None
    assert _ink(surface, (0, 0, 0)) == 0
