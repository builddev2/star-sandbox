"""Clickable chrome: metrics, buttons, icons, tooltips.

**One description drives everything.**  A :class:`Control` is a plain record --
where it is, what it says, what it means -- and the same record is used to draw
the button, to hit-test the mouse against it, and to write its tooltip.  There
is no second list of rectangles anywhere, so a control cannot be drawn in one
place and clickable in another.

**Every size is derived from the font.**  The pixel grid is user-switchable
(``G``), which changes the font height, and any button measured in raw pixels
would stop matching its own label the moment the grid moved.  :class:`Metrics`
turns one font into the whole layout, and it is the only place a spacing
constant may live.

**Icons are drawn, not typed.**  The font is whatever SDL hands us, so glyphs
like the pause bars or a rewind chevron may simply not exist in it.  Drawing
them from lines and polygons is a few more lines of code and cannot fall back
to a box.  It also keeps them hard-edged, which is the look everything else
here is built around.
"""

from dataclasses import dataclass, field

import pygame

from . import palette

#: Interaction states a button can be drawn in, in the order they take priority.
STATE_DISABLED = "disabled"
STATE_DOWN = "down"
STATE_ACTIVE = "active"
STATE_HOVER = "hover"
STATE_REST = "rest"

_FILLS = {
    STATE_DISABLED: palette.BUTTON,
    STATE_DOWN: palette.BUTTON_DOWN,
    STATE_ACTIVE: palette.BUTTON_ACTIVE,
    STATE_HOVER: palette.BUTTON_HOVER,
    STATE_REST: palette.BUTTON,
}


@dataclass
class Control:
    """One clickable thing on screen.

    ``id`` is what the application dispatches on and ``payload`` carries the
    subject when one id covers many controls -- every body row in the list is
    ``select.body`` with a different body id.  ``rect`` is filled in by the
    layout pass; nothing outside that pass should set it.
    """

    id: str
    label: str = ""
    icon: str = ""
    hotkey: str = ""
    tooltip: str = ""
    payload: object = None
    active: bool = False
    enabled: bool = True
    #: Steppers fire again while the button is held; one click per notch would
    #: make growing Jupiter into a star an eighty-click job.
    repeat: bool = False
    #: Destructive controls are drawn in the danger colour.
    danger: bool = False
    #: Compact controls show the icon only; the label lives in the tooltip.
    compact: bool = False
    #: A readout is laid out and drawn like a button but is not clickable --
    #: the speed display sitting between the slower and faster buttons.
    readout: bool = False
    #: A slider is a draggable track+thumb instead of a button: ``value`` (0..1)
    #: is where the thumb sits, and dragging it re-fires ``id`` with the new
    #: fraction as ``payload`` every motion event, not just on release.
    slider: bool = False
    value: float = 0.0
    rect: pygame.Rect = field(default=None)

    @property
    def clickable(self):
        return self.enabled and not self.readout

    def state(self, hovered=False, pressed=False):
        if not self.enabled:
            return STATE_DISABLED
        if pressed:
            return STATE_DOWN
        if self.active:
            return STATE_ACTIVE
        if hovered:
            return STATE_HOVER
        return STATE_REST

    def hint(self):
        """Tooltip text with the hotkey appended, for the power user."""
        text = self.tooltip or self.label
        if self.hotkey:
            return "%s   [%s]" % (text, self.hotkey)
        return text


class Metrics:
    """Every spacing number in the interface, derived from one font."""

    def __init__(self, font, width, height):
        self.font = font
        self.width = width
        self.height = height
        h = font.height
        self.pad = max(3, h // 3)
        self.gap = max(2, h // 6)
        self.line = h + 2
        self.icon = max(8, h - 2)
        self.btn_h = h + 2 * self.gap + 2
        #: Icon-only buttons are square, so a row of them reads as a toolbar.
        self.icon_btn_w = self.btn_h
        self.top_h = self.btn_h + 2 * self.gap
        self.hint_h = self.line + self.pad
        self.transport_h = self.btn_h + 2 * self.gap
        self.radius = 2

    # --- reserved bands ---------------------------------------------------------

    @property
    def transport_y(self):
        """Top of the bottom control bar."""
        return self.height - self.hint_h - self.transport_h

    @property
    def content_top(self):
        """First y a floating panel may use without covering the top bar."""
        return self.top_h + self.gap

    @property
    def content_bottom(self):
        """Last y a floating panel may use without covering the control bar."""
        return self.transport_y - self.gap

    # --- measuring --------------------------------------------------------------

    def button_width(self, control):
        """Width a button needs to show everything it carries."""
        if control.compact:
            return self.icon_btn_w
        width = 2 * self.pad
        if control.icon:
            width += self.icon + self.gap
        width += self.font.width(control.label)
        if control.hotkey:
            width += self.gap * 2 + self.font.width(control.hotkey) + 2 * self.gap
        return width


# --- panels ---------------------------------------------------------------------

def draw_panel(surface, rect, title=None, font=None):
    """A translucent slab with a hairline border, optionally titled."""
    box = pygame.Surface(rect.size, pygame.SRCALPHA)
    box.fill(tuple(palette.PANEL) + (236,))
    surface.blit(box, rect.topleft)
    pygame.draw.rect(surface, palette.PANEL_EDGE, rect, 1)
    if title and font is not None:
        font.draw(surface, rect.x + 4, rect.y + 2, title, palette.TEXT_DIM)


# --- buttons --------------------------------------------------------------------

def draw_button(surface, font, control, hovered=False, pressed=False):
    """Draw one control in whichever state it is in.

    Everything inside the button is clipped to the button.  A label wider than
    the space it was given is a real possibility -- body names come from the
    world, and someone will throw a planet with a long name -- and text spilling
    across a neighbouring control looks like a crash.
    """
    rect = control.rect
    if rect is None:
        return
    previous_clip = surface.get_clip()
    surface.set_clip(rect.clip(previous_clip) if previous_clip else rect)
    try:
        _draw_button_content(surface, font, control, rect, hovered, pressed)
    finally:
        surface.set_clip(previous_clip)


def _draw_button_content(surface, font, control, rect, hovered, pressed):
    if control.slider:
        draw_slider(surface, control, rect, pressed)
        return
    if control.readout:
        draw_panel(surface, rect)
        x = rect.x + (rect.width - font.width(control.label)) // 2
        font.draw(surface, max(rect.x, x), rect.y + (rect.height - font.height) // 2,
                  control.label, palette.TEXT)
        return
    state = control.state(hovered, pressed)

    fill = _FILLS[state]
    box = pygame.Surface(rect.size, pygame.SRCALPHA)
    box.fill(tuple(fill) + (238 if state != STATE_DISABLED else 150,))
    surface.blit(box, rect.topleft)

    edge = palette.BUTTON_EDGE
    if state == STATE_ACTIVE:
        edge = palette.BUTTON_EDGE_ACTIVE
    elif state == STATE_DISABLED:
        edge = palette.DISABLED
    elif control.danger and state in (STATE_HOVER, STATE_DOWN):
        edge = palette.DANGER
    pygame.draw.rect(surface, edge, rect, 1)

    if state not in (STATE_DISABLED, STATE_DOWN):
        # The elevation cue: a raised element's own top edge reads one step
        # lighter than the rest of its border.  Skipped when pressed -- a
        # button being pushed in is the one moment it should not look raised.
        pygame.draw.line(surface, palette.RAISED_EDGE,
                         (rect.x + 1, rect.y), (rect.right - 2, rect.y))

    if state == STATE_ACTIVE:
        # A lit edge alone is easy to miss on a small button; the stripe is what
        # makes the current tool obvious without reading anything.
        pygame.draw.line(surface, palette.BUTTON_EDGE_ACTIVE,
                         (rect.x + 1, rect.y + 1), (rect.x + 1, rect.bottom - 2), 1)

    # A 1px press-down offset on the drawn content only -- the fill and edge
    # stay exactly where the layout put them, so the button does not visibly
    # resize, but the icon/label/hotkey nudge the way a real key does when
    # struck.  This is the cheapest of the flat-vs-depth fixes: no shadow, no
    # new colour, one moved rectangle.
    content = rect.move(1, 1) if state == STATE_DOWN else rect

    if state == STATE_DISABLED:
        text_colour = palette.DISABLED
    elif control.danger:
        text_colour = palette.DANGER if state != STATE_REST else palette.BUTTON_TEXT_OFF
    elif control.active:
        text_colour = palette.TEXT
    else:
        text_colour = palette.BUTTON_TEXT_OFF if state == STATE_REST else palette.TEXT

    pad = max(3, font.height // 3)
    gap = max(2, font.height // 6)
    icon_side = max(8, font.height - 2)

    if control.compact:
        if control.icon:
            side = min(icon_side, content.width - 4, content.height - 4)
            icon_rect = pygame.Rect(0, 0, side, side)
            icon_rect.center = content.center
            draw_icon(surface, control.icon, icon_rect, text_colour)
        elif control.label:
            x = content.x + (content.width - font.width(control.label)) // 2
            font.draw(surface, x, content.y + (content.height - font.height) // 2,
                      control.label, text_colour)
        return

    x = content.x + pad
    if control.icon:
        icon_rect = pygame.Rect(x, content.y + (content.height - icon_side) // 2,
                                icon_side, icon_side)
        draw_icon(surface, control.icon, icon_rect, text_colour)
        x += icon_side + gap

    font.draw(surface, x, content.y + (content.height - font.height) // 2,
              control.label, text_colour)

    if control.hotkey:
        draw_hotkey(surface, font, control.hotkey, content,
                    dim=state in (STATE_REST, STATE_DISABLED))


def draw_hotkey(surface, font, key, rect, dim=True):
    """The little key cap on the right of a button.

    It is what teaches the shortcut: the child clicks the button for a week and
    reads the letter next to it every time, and one day stops clicking.
    """
    width = font.width(key)
    gap = max(2, font.height // 6)
    cap = pygame.Rect(rect.right - width - 2 * gap - gap,
                      rect.y + (rect.height - font.height) // 2 - 1,
                      width + 2 * gap, font.height + 2)
    if cap.x <= rect.x:
        return
    pygame.draw.rect(surface, palette.DISABLED if dim else palette.BUTTON_EDGE_ACTIVE,
                     cap, 1)
    font.draw(surface, cap.x + gap, cap.y + 1, key,
              palette.DISABLED if dim else palette.TEXT_DIM)


#: The thumb's travel is inset from the control's rect by half its own height,
#: so it stays fully drawn at both ends instead of hanging off the edge.
SLIDER_INSET = 4


def slider_track(rect):
    """The span a vertical slider's thumb centre travels, as ``(top, height)``.

    Drawing and hit-testing both come through here.  A slider whose thumb is
    painted somewhere other than where a click on it lands is exactly the
    "drawn in one place, clickable in another" fault this module exists to
    prevent -- and it shows up as the thumb jumping the moment you grab it.
    """
    height = max(1, rect.height - 2 * SLIDER_INSET)
    return rect.y + (rect.height - height) // 2, height


def draw_slider(surface, control, rect, pressed=False):
    """A vertical track with a thumb at ``control.value`` -- 0 at the bottom, 1
    at the top, matching a stack of buttons read top-to-bottom as fast-to-slow.
    """
    top, height = slider_track(rect)
    track = pygame.Rect(0, 0, 2, height)
    track.centerx = rect.centerx
    track.y = top
    pygame.draw.rect(surface, palette.BUTTON_EDGE, track)

    frac = min(1.0, max(0.0, control.value))
    thumb = pygame.Rect(0, 0, max(4, rect.width - 6), 5)
    thumb.centerx = rect.centerx
    thumb.centery = int(round(top + height - frac * height))
    pygame.draw.rect(surface, palette.BUTTON_DOWN if pressed else palette.BUTTON_ACTIVE, thumb)
    pygame.draw.rect(surface, palette.BUTTON_EDGE_ACTIVE, thumb, 1)


def draw_bar(surface, rect, fraction, colour, bg=None):
    """A horizontal filled bar, clamped to 0..1 -- a bounded quantity read at a
    glance (how close to ignition, how close to boiling) instead of only as a
    number nobody has an intuition for.
    """
    pygame.draw.rect(surface, bg if bg is not None else palette.BUTTON, rect)
    frac = min(1.0, max(0.0, fraction))
    if frac > 0.0:
        fill = pygame.Rect(rect.x, rect.y, max(1, int(round(rect.width * frac))),
                           rect.height)
        pygame.draw.rect(surface, colour, fill)
    pygame.draw.rect(surface, palette.BUTTON_EDGE, rect, 1)


# --- tooltips -------------------------------------------------------------------

def draw_tooltip(surface, font, text, anchor, prefer_above=True):
    """A one-line explanation pinned near ``anchor``, kept on screen.

    Drawn last, over everything, because a tooltip that is covered by the panel
    it describes explains nothing.
    """
    if not text:
        return None
    pad = max(3, font.height // 3)
    width = font.width(text) + 2 * pad
    height = font.height + pad
    x = anchor.centerx - width // 2
    y = anchor.top - height - 4 if prefer_above else anchor.bottom + 4
    if y < 0:
        y = anchor.bottom + 4
    if y + height > surface.get_height():
        y = max(0, anchor.top - height - 4)
    x = max(2, min(x, surface.get_width() - width - 2))

    rect = pygame.Rect(x, y, width, height)
    box = pygame.Surface(rect.size, pygame.SRCALPHA)
    box.fill(tuple(palette.TOOLTIP) + (242,))
    surface.blit(box, rect.topleft)
    pygame.draw.rect(surface, palette.BUTTON_EDGE, rect, 1)
    font.draw(surface, rect.x + pad, rect.y + pad // 2, text, palette.TEXT)
    return rect


# --- icons ----------------------------------------------------------------------

def _chevron(surface, colour, cx, cy, size, direction):
    """A single > drawn as two lines, pointing whichever way is asked."""
    dx = {"left": -1, "right": 1}.get(direction, 0)
    dy = {"up": -1, "down": 1}.get(direction, 0)
    half = max(2, size // 2)
    if dx:
        pygame.draw.lines(surface, colour, False,
                          [(cx - dx * half // 2, cy - half),
                           (cx + dx * half // 2, cy),
                           (cx - dx * half // 2, cy + half)], 1)
    else:
        pygame.draw.lines(surface, colour, False,
                          [(cx - half, cy - dy * half // 2),
                           (cx, cy + dy * half // 2),
                           (cx + half, cy - dy * half // 2)], 1)


def draw_icon(surface, name, rect, colour):
    """Draw the named glyph inside ``rect``.

    Unknown names draw nothing rather than raising: a typo in an icon name must
    not take the frame down mid-flight.
    """
    x, y, w, h = rect
    cx, cy = rect.centerx, rect.centery
    side = min(w, h)
    r = max(2, side // 2 - 1)

    if name == "look":
        pygame.draw.ellipse(surface, colour, pygame.Rect(x, cy - r // 2 - 1, w, r + 2), 1)
        pygame.draw.circle(surface, colour, (cx, cy), max(1, r // 3))
    elif name == "throw":
        pygame.draw.line(surface, colour, (x + 1, rect.bottom - 2), (rect.right - 3, y + 2), 1)
        pygame.draw.circle(surface, colour, (rect.right - 3, y + 2), max(1, r // 3))
        pygame.draw.line(surface, colour, (x + 1, rect.bottom - 2), (x + 1 + r // 2, rect.bottom - 2), 1)
        pygame.draw.line(surface, colour, (x + 1, rect.bottom - 2), (x + 1, rect.bottom - 2 - r // 2), 1)
    elif name == "edit":
        for k in range(3):
            ly = y + 2 + k * max(2, (h - 4) // 2)
            pygame.draw.line(surface, colour, (x + 1, ly), (rect.right - 2, ly), 1)
            knob = x + 2 + (k * 5) % max(1, w - 5)
            pygame.draw.line(surface, colour, (knob, ly - 1), (knob, ly + 1), 1)
    elif name == "play":
        pygame.draw.polygon(surface, colour,
                            [(cx - r + 1, cy - r), (cx + r, cy), (cx - r + 1, cy + r)])
    elif name == "pause":
        bar = max(1, r // 2)
        pygame.draw.rect(surface, colour, pygame.Rect(cx - r + 1, cy - r, bar, 2 * r))
        pygame.draw.rect(surface, colour, pygame.Rect(cx + r - bar, cy - r, bar, 2 * r))
    elif name == "slower":
        _chevron(surface, colour, cx - r // 2, cy, side - 2, "left")
        _chevron(surface, colour, cx + r // 2, cy, side - 2, "left")
    elif name == "faster":
        _chevron(surface, colour, cx - r // 2, cy, side - 2, "right")
        _chevron(surface, colour, cx + r // 2, cy, side - 2, "right")
    elif name == "reset":
        pygame.draw.arc(surface, colour, pygame.Rect(cx - r, cy - r, 2 * r, 2 * r),
                        0.9, 5.6, 1)
        pygame.draw.line(surface, colour, (cx + r - 1, cy - r + 1), (cx + r - 1, cy - 1), 1)
        pygame.draw.line(surface, colour, (cx + r - 1, cy - r + 1), (cx + 1, cy - r + 1), 1)
    elif name == "home":
        # A sun with one planet on its orbit: "show me the whole system again".
        # Deliberately not the reset arrow, which already means restart time.
        pygame.draw.ellipse(surface, colour,
                            pygame.Rect(x, cy - r // 2 - 1, w - 1, r + 2), 1)
        pygame.draw.circle(surface, colour, (cx, cy), max(1, r // 3))
        pygame.draw.circle(surface, colour, (rect.right - 2, cy), 1)
    elif name == "trails":
        pygame.draw.arc(surface, colour, pygame.Rect(x, y + 1, w, h), 0.4, 2.6, 1)
        pygame.draw.circle(surface, colour, (rect.right - 3, cy), max(1, r // 3))
    elif name == "sizes":
        pygame.draw.circle(surface, colour, (cx - r // 2, cy), max(2, r - 1), 1)
        pygame.draw.circle(surface, colour, (cx + r // 2 + 1, cy), 1)
    elif name == "labels":
        pygame.draw.rect(surface, colour, pygame.Rect(x + 1, cy - r + 1, w - 3, 2 * r - 2), 1)
        pygame.draw.line(surface, colour, (x + 3, cy), (rect.right - 5, cy), 1)
    elif name == "grid":
        step = max(2, side // 3)
        for gx in range(3):
            for gy in range(3):
                surface.set_at((min(x + 1 + gx * step, rect.right - 1),
                                min(y + 1 + gy * step, rect.bottom - 1)), colour)
    elif name == "window":
        # A screen with a window filling part of it.  Deliberately not the
        # fullscreen corner brackets next door: that one is a mode, this one is
        # a size, and at 26 pixels the two have to be told apart at a glance.
        pygame.draw.rect(surface, colour,
                         pygame.Rect(x + 1, cy - r, max(4, w - 3), 2 * r), 1)
        pygame.draw.rect(surface, colour,
                         pygame.Rect(x + 2, cy - r + 1,
                                     max(2, (w - 5) // 2), max(1, 2 * r - 2)))
    elif name == "fullscreen":
        # Four corner brackets.  The arms stay short so they read as corners
        # rather than joining up into a plain box.
        arm = max(2, r // 2)
        for sx, sy in ((0, 0), (1, 0), (0, 1), (1, 1)):
            px = x + 1 if sx == 0 else rect.right - 2
            py = y + 1 if sy == 0 else rect.bottom - 2
            ex = px + (arm if sx == 0 else -arm)
            ey = py + (arm if sy == 0 else -arm)
            pygame.draw.line(surface, colour, (px, py), (ex, py), 1)
            pygame.draw.line(surface, colour, (px, py), (px, ey), 1)
    elif name == "follow":
        pygame.draw.circle(surface, colour, (cx, cy), r, 1)
        pygame.draw.line(surface, colour, (cx - r - 1, cy), (cx - r + 1, cy), 1)
        pygame.draw.line(surface, colour, (cx + r - 1, cy), (cx + r + 1, cy), 1)
        pygame.draw.line(surface, colour, (cx, cy - r - 1), (cx, cy - r + 1), 1)
        pygame.draw.line(surface, colour, (cx, cy + r - 1), (cx, cy + r + 1), 1)
        pygame.draw.circle(surface, colour, (cx, cy), max(1, r // 3))
    elif name == "flyto":
        # A magnifier with a plus in it: the universal "get closer to this".
        lens = max(2, r - 1)
        lx, ly = cx - 1, cy - 1
        pygame.draw.circle(surface, colour, (lx, ly), lens, 1)
        pygame.draw.line(surface, colour, (lx - lens // 2, ly), (lx + lens // 2, ly), 1)
        pygame.draw.line(surface, colour, (lx, ly - lens // 2), (lx, ly + lens // 2), 1)
        pygame.draw.line(surface, colour, (lx + lens - 1, ly + lens - 1),
                         (rect.right - 2, rect.bottom - 2), 1)
    elif name == "stop":
        pygame.draw.rect(surface, colour, pygame.Rect(cx - r + 1, cy - r + 1,
                                                      2 * r - 1, 2 * r - 1))
    elif name == "trash":
        pygame.draw.line(surface, colour, (cx - r, y + 2), (cx + r, y + 2), 1)
        pygame.draw.rect(surface, colour,
                         pygame.Rect(cx - r + 1, y + 4, 2 * r - 1, h - 6), 1)
        pygame.draw.line(surface, colour, (cx - 1, y + 6), (cx - 1, rect.bottom - 3), 1)
        pygame.draw.line(surface, colour, (cx + 1, y + 6), (cx + 1, rect.bottom - 3), 1)
    elif name == "plus":
        pygame.draw.line(surface, colour, (cx - r, cy), (cx + r, cy), 1)
        pygame.draw.line(surface, colour, (cx, cy - r), (cx, cy + r), 1)
    elif name == "minus":
        pygame.draw.line(surface, colour, (cx - r, cy), (cx + r, cy), 1)
    elif name in ("left", "right", "up", "down"):
        _chevron(surface, colour, cx, cy, side - 2, name)
    elif name == "list":
        for k in range(3):
            ly = cy - r + k * max(2, r)
            pygame.draw.line(surface, colour, (x + 1, ly), (rect.right - 2, ly), 1)
    elif name == "close":
        pygame.draw.line(surface, colour, (cx - r, cy - r), (cx + r, cy + r), 1)
        pygame.draw.line(surface, colour, (cx - r, cy + r), (cx + r, cy - r), 1)
    elif name in ("lock", "unlock"):
        # A padlock: a body with a keyhole, and a shackle above it.  Drawn from
        # straight lines rather than an arc because an arc inside a box this
        # small renders nothing at all, and this button is eleven pixels square
        # at the coarsest grid.  The two states differ by where the shackle
        # sits -- seated in the body, or lifted and slid off to one side -- so
        # they read as one object in two states rather than two glyphs.
        body_box = pygame.Rect(0, 0, max(5, 2 * r - 1), max(4, r + 1))
        body_box.centerx = cx
        body_box.bottom = min(rect.bottom - 1, cy + r)
        pygame.draw.rect(surface, colour, body_box, 1)
        surface.set_at((body_box.centerx, body_box.centery), colour)
        top = max(y, body_box.top - max(2, r - 1))
        left = body_box.left + 2
        right = max(left + 2, body_box.right - 3)
        if name == "lock":
            pygame.draw.line(surface, colour, (left, top), (right, top), 1)
            pygame.draw.line(surface, colour, (left, top), (left, body_box.top), 1)
            pygame.draw.line(surface, colour, (right, top), (right, body_box.top), 1)
        else:
            shift = max(1, r // 2)
            lifted, hinge = left + shift, min(rect.right - 1, right + shift)
            pygame.draw.line(surface, colour, (lifted, top), (hinge, top), 1)
            pygame.draw.line(surface, colour, (hinge, top), (hinge, body_box.top), 1)
            pygame.draw.line(surface, colour, (lifted, top),
                             (lifted, top + max(1, (body_box.top - top) // 2)), 1)
    elif name == "circle":
        pygame.draw.circle(surface, colour, (cx, cy), r, 1)
    elif name == "star":
        pygame.draw.line(surface, colour, (cx - r, cy), (cx + r, cy), 1)
        pygame.draw.line(surface, colour, (cx, cy - r), (cx, cy + r), 1)
        pygame.draw.line(surface, colour, (cx - r + 1, cy - r + 1), (cx + r - 1, cy + r - 1), 1)
        pygame.draw.line(surface, colour, (cx - r + 1, cy + r - 1), (cx + r - 1, cy - r + 1), 1)
    elif name == "blackhole":
        radius = max(2, min(rect.width, rect.height) // 2 - 1)
        pygame.draw.circle(surface, colour, rect.center, radius, 1)
        pygame.draw.circle(surface, colour, rect.center, max(1, radius - 2))
    elif name == "target":
        pygame.draw.circle(surface, colour, (cx, cy), r, 1)
        pygame.draw.circle(surface, colour, (cx, cy), max(1, r - 2), 1)
        surface.set_at((cx, cy), colour)
    elif name == "power":
        # The gap sits at the top, where the stem goes through it: pygame's
        # angles run counterclockwise from three o'clock, so the top is pi/2.
        pygame.draw.arc(surface, colour, pygame.Rect(cx - r, cy - r + 1, 2 * r, 2 * r),
                        2.2, 7.2, 1)
        pygame.draw.line(surface, colour, (cx, cy - r + 1), (cx, cy), 1)
    elif name == "help":
        pygame.draw.arc(surface, colour, pygame.Rect(cx - r, y + 1, 2 * r, r + 1),
                        0.2, 3.3, 1)
        pygame.draw.line(surface, colour, (cx, cy), (cx, cy + max(1, r // 2)), 1)
        surface.set_at((cx, min(rect.bottom - 1, cy + r)), colour)
