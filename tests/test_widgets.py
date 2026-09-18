"""The drawing primitives behind every button."""

import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import pygame
import pytest

from src.app import config


@pytest.fixture(scope="module", autouse=True)
def _pygame():
    pygame.init()
    pygame.display.set_mode((320, 200))
    yield
    pygame.quit()


@pytest.fixture
def font():
    from src.render.pixelfont import PixelFont
    # The size the interface actually renders at one pixel per point.
    return PixelFont(config.font_px(1))


def _ink(surface, palette_space=None):
    from src.render import palette
    background = palette_space or palette.SPACE[:3]
    w, h = surface.get_size()
    return sum(1 for x in range(w) for y in range(h)
               if surface.get_at((x, y))[:3] != background)


# --- palette tiers ------------------------------------------------------------------

def test_the_elevation_ladder_only_ever_gets_lighter():
    from src.render import palette
    ladder = [palette.SPACE, palette.PANEL, palette.PANEL_ROW, palette.BUTTON]
    for dim, brighter in zip(ladder, ladder[1:]):
        assert sum(brighter) > sum(dim), "%s is not lighter than %s" % (brighter, dim)


def test_text_faint_is_dimmer_than_text_dim():
    from src.render import palette
    assert sum(palette.TEXT_FAINT) < sum(palette.TEXT_DIM)


def test_raised_edge_is_lighter_than_the_ordinary_button_edge():
    from src.render import palette
    assert sum(palette.RAISED_EDGE) > sum(palette.BUTTON_EDGE)


def test_a_bar_fills_by_fraction():
    from src.render import palette, widgets
    surface = pygame.Surface((100, 10))
    surface.fill((0, 0, 0))
    widgets.draw_bar(surface, pygame.Rect(0, 0, 100, 10), 0.5, palette.ACCENT)
    lit = [x for x in range(100) if surface.get_at((x, 5))[:3] == palette.ACCENT]
    assert 40 <= len(lit) <= 55


def test_a_bar_clamps_outside_zero_and_one():
    from src.render import palette, widgets
    for fraction, expected_min in ((-1.0, 0), (5.0, 95)):
        surface = pygame.Surface((100, 10))
        surface.fill((0, 0, 0))
        widgets.draw_bar(surface, pygame.Rect(0, 0, 100, 10), fraction, palette.ACCENT)
        lit = sum(1 for x in range(100) if surface.get_at((x, 5))[:3] == palette.ACCENT)
        assert lit >= expected_min or lit == 0


def test_a_resting_button_has_a_raised_top_highlight(font):
    from src.render import palette
    from src.render.widgets import Control, draw_button
    surface = pygame.Surface((40, 40))
    surface.fill((0, 0, 0))
    control = Control("x", label="Follow")
    control.rect = pygame.Rect(4, 4, 32, 20)
    draw_button(surface, font, control)
    top_row = [surface.get_at((x, 4))[:3] for x in range(5, 35)]
    assert palette.RAISED_EDGE in top_row


def test_a_pressed_icon_button_nudges_its_icon_down_and_right(font):
    """The fill and edge blanket the whole rect regardless of press state, so
    this isolates pixels drawn in ``palette.TEXT`` -- the icon stroke colour a
    hovered, non-active button uses -- rather than "anything non-background"."""
    from src.render import palette
    from src.render.widgets import Control, draw_button

    def icon_bbox(pressed):
        surface = pygame.Surface((40, 40))
        surface.fill((0, 0, 0))
        control = Control("x", icon="plus", compact=True)
        control.rect = pygame.Rect(4, 4, 32, 32)
        draw_button(surface, font, control, hovered=True, pressed=pressed)
        points = [(x, y) for x in range(40) for y in range(40)
                  if surface.get_at((x, y))[:3] == palette.TEXT]
        xs = [x for x, _ in points]
        ys = [y for _, y in points]
        return min(xs), min(ys)

    rest_x, rest_y = icon_bbox(False)
    down_x, down_y = icon_bbox(True)
    assert (down_x, down_y) == (rest_x + 1, rest_y + 1)


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
              "list", "close", "circle", "star", "blackhole", "help", "home", "power",
              "lock", "unlock", "explain")


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


def test_the_lock_and_unlock_icons_read_as_a_pair_at_button_size():
    """These two are the whole readout on the padlock button, and they have to
    work at the size the +/- glyphs already draw at -- eight pixels, at the
    coarsest grid in the smallest window.  Which is also why neither is drawn
    with an arc: pygame renders nothing at all for an arc in a box this small.
    """
    from src.render import widgets
    drawn = {}
    for name in ("lock", "unlock"):
        surface = pygame.Surface((24, 24))
        surface.fill((0, 0, 0))
        rect = pygame.Rect(8, 8, 8, 8)
        widgets.draw_icon(surface, name, rect, (255, 255, 255))
        drawn[name] = {(x, y) for x in range(24) for y in range(24)
                       if surface.get_at((x, y))[:3] != (0, 0, 0)}
        assert len(drawn[name]) >= 8, "%s is invisible at eight pixels" % name
        for point in drawn[name]:
            assert rect.inflate(2, 2).collidepoint(point), \
                "%s escaped its box" % name
    assert drawn["lock"] != drawn["unlock"], \
        "locked and unlocked draw the same thing"


@pytest.mark.parametrize("s", (1, 2, 3))
def test_the_explain_icon_fits_the_view_button_at_every_grid(s):
    """``view.explain`` used to draw the label "Why" -- 25px wide in the 18px
    button the default grid gives it, so roughly half the W and half the y
    were clipped away.  A drawn icon is supposed to fit its box by
    construction; render it at the exact size ``_draw_button_content`` gives a
    compact icon button at each grid and confirm it actually paints, and
    paints nothing outside that box.
    """
    from src.app import config
    from src.render import widgets
    from src.render.pixelfont import PixelFont
    font = PixelFont(config.font_px(s))
    gap = max(2, font.height // 6)
    btn_h = font.height + 2 * gap + 2
    icon_side = max(8, font.height - 2)
    side = min(icon_side, btn_h - 4, btn_h - 4)
    surface = pygame.Surface((btn_h + 8, btn_h + 8))
    surface.fill((0, 0, 0))
    rect = pygame.Rect(0, 0, side, side)
    rect.center = surface.get_rect().center
    widgets.draw_icon(surface, "explain", rect, (255, 255, 255))
    drawn = [(x, y) for x in range(surface.get_width())
             for y in range(surface.get_height())
             if surface.get_at((x, y))[:3] != (0, 0, 0)]
    assert drawn, "explain drew nothing at grid %d" % s
    for point in drawn:
        assert rect.collidepoint(point), \
            "explain escaped its %dpx box at grid %d" % (side, s)


def test_a_text_field_draws_its_text_and_a_cursor(font):
    from src.render import palette, widgets
    surface = pygame.Surface((200, 40))
    surface.fill(palette.SPACE)
    control = widgets.Control("slot.name", label="Mars trip", text_field=True,
                              rect=pygame.Rect(4, 4, 190, 30))
    widgets.draw_button(surface, font, control)
    with_text = _ink(surface)
    assert with_text > 100
    # The focus ring is the selection colour: a text field is the thing on
    # screen you are editing, the same way a selected planet is.
    assert surface.get_at((4, 20))[:3] == palette.SELECT
    empty = widgets.Control("slot.name", label="", text_field=True,
                            rect=pygame.Rect(4, 4, 190, 30))
    surface.fill(palette.SPACE)
    widgets.draw_text_field(surface, font, empty, empty.rect, now_ms=0)
    lit = _ink(surface)
    surface.fill(palette.SPACE)
    widgets.draw_text_field(surface, font, empty, empty.rect, now_ms=500)
    dark = _ink(surface)
    assert lit > dark, "the cursor blinks: on at 0 ms, off at 500 ms"


def test_a_text_field_is_clickable_but_a_readout_is_not():
    from src.render.widgets import Control
    assert Control("x", text_field=True).clickable
    assert not Control("x", readout=True).clickable


def test_the_save_icon_draws_something(font):
    from src.render import palette, widgets
    surface = pygame.Surface((24, 24))
    surface.fill(palette.SPACE)
    widgets.draw_icon(surface, "save", pygame.Rect(2, 2, 20, 20), palette.TEXT)
    assert _ink(surface) > 20
