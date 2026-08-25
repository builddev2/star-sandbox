"""Display setup and the constants that depend on the pixel grid.

Measured on a 16" MacBook Pro (macOS 15, pygame 2.6.1, SDL 2.28.4, cocoa driver)
on 2026-08-18::

    panel                        3456 x 2234   (system_profiler)
    get_desktop_sizes()          1728 x 1117   logical points
    fullscreen window            1728 x 1084   logical points (notch strip lost)
    set_mode((800,500), SCALED)  surface 800x500, window 1600x1000

Three findings shape this module.

**SDL reports logical points, never device pixels.**  Neither
``get_desktop_sizes`` nor ``get_window_size`` can see the 3456x2234 panel, and
``WINDOW_ALLOW_HIGHDPI`` does not exist in pygame 2.6.  So there is no way to
address individual panel pixels, and no reason to want to: at one internal pixel
per logical point macOS renders each of ours as an exact 2x2 block of panel
pixels, which is a clean integer upscale and exactly the pixel-art look we are
after.  Hence ``DEFAULT_S = 1`` -- the finest grid available is also the right
one.

**Text is therefore sized in logical points**, not device pixels, because points
are the unit the whole platform is calibrated in.

**A fullscreen window is not the desktop size.**  The menu-bar/notch strip costs
33 points of height, so asking for a 1117-tall buffer gets it scaled by 0.97 --
a non-integer factor, which smears the pixel grid it exists to preserve.  So the
buffer is derived from the window we actually got, in a second pass.

Two further findings, measured 2026-08-24, are why the upscale is ours and not
SDL's::

    set_mode((800,500), SCALED|RESIZABLE)   surface 800x500, window 1600x974
    window dragged down to 640x360          window clamps at 800x500
    a second set_mode(..., SCALED, ...)     pygame.error: failed to create renderer
    set_mode((900,560), RESIZABLE)          surface 900x560, window 900x560
    a second set_mode(..., RESIZABLE, ...)  works, every time, at any size

**``SCALED`` pins the window's minimum size to the buffer.**  With a buffer the
size of the desktop, the smallest window SDL would allow *was* the desktop --
which is what nailed this program to the whole screen, with no title bar, so no
resize handle, no green button and no minimise.

**``SCALED`` also cannot be rebuilt.**  A second ``set_mode`` raises "failed to
create renderer" on the cocoa driver, so the grid switch and the fullscreen
toggle both fell into the ``pygame.error`` branch and silently did nothing.

A plain ``RESIZABLE`` window has neither problem: the surface is the window,
one internal pixel per point, and ``set_mode`` may be called as often as we
like.  What ``SCALED`` was buying us -- the upscale at coarser grids, and the
mouse coordinates that go with it -- this module now does itself, in
``present`` and ``to_internal``, by an exact integer factor.

**Nothing here may touch ``pygame._sdl2.video.Window``.**  It is the only way
to move, minimise or set a minimum size on a window in pygame 2.6, and it
crashes: the object ``Window.from_display_module()`` hands back destroys the
real SDL window when it is garbage-collected, and the program segfaults some
frames later, inside ``pygame.event.get``, a long way from the call that did
it.  So the size presets set a size and leave the window where it is -- macOS
moves windows perfectly well by itself now that this one can be resized at all
-- and the minimum size is enforced by pushing back in ``resize``.
"""

import pygame

#: Finest grid: one internal pixel per logical point, i.e. a 2x2 block of panel
#: pixels on a Retina display.
DEFAULT_S = 1
GRID_CHOICES = (1, 2, 3)

#: Target text height in logical points.  Sized for a child reading alongside an
#: adult on a laptop screen, not for maximum information density.
TARGET_TEXT_POINTS = 18
#: Below this a scalable font stops being legible whatever the grid.
MIN_FONT_PX = 8

#: Height macOS keeps for itself above a window, and for the window's own title
#: bar.  A preset that ignores them puts the transport bar under the Dock.
MENU_BAR_POINTS = 37
TITLE_BAR_POINTS = 28

#: The smallest internal buffer the whole interface lays itself out inside, at
#: the finest grid.  ``tests/test_layout.py`` is what keeps these two numbers
#: honest: it builds every panel of every tool at exactly this size and fails
#: if a single control or panel lands off the edge.
#:
#: The width has to stay below a third of a laptop desktop (1728 / 3 = 576) or
#: the one-third preset would clamp straight back up and stop being one third.
#: The height is set by the help panel, which is the tallest thing here.
MIN_INTERNAL = (480, 440)

#: At the finest grid one internal pixel is one point, so this is also the
#: smallest window.  Coarser grids are handled by ``usable_grids``, not by
#: growing the window under the user.
MIN_WINDOW = MIN_INTERNAL

#: The window sizes offered as presets, as a fraction of the desktop *width*.
#: Height is always the whole usable desktop, because that is what half-screen
#: means to everyone who has ever dragged a window against the side of a
#: screen -- half as wide, full height -- and what macOS's own tiling does.
WINDOW_FRACTIONS = (
    (1.0, "Full width"),
    (2 / 3, "Two thirds"),
    (1 / 2, "Half screen"),
    (1 / 3, "One third"),
)
#: Opens at two thirds: unmistakably a window, with room for the whole interface.
DEFAULT_FRACTION_INDEX = 1


def internal_size(display_w, display_h, s):
    """Internal buffer size for a display and pixel-grid divisor."""
    return (max(1, display_w // s), max(1, display_h // s))


def font_px(s):
    """Font height in internal pixels, so apparent size stays roughly constant."""
    return max(MIN_FONT_PX, round(TARGET_TEXT_POINTS / max(s, 1)))


def min_internal(s):
    """Smallest buffer the interface lays out in, at pixel grid ``s``.

    Every size in the interface is derived from the font and the font shrinks
    with the grid, so a coarser grid fits the same interface into
    proportionally fewer internal pixels.  Scaled by ``font_px`` rather than by
    ``s`` because the font stops shrinking at ``MIN_FONT_PX``, and past that
    point a coarser grid buys no room at all -- it costs it.
    """
    ratio = font_px(s) / font_px(1)
    return (max(1, round(MIN_INTERNAL[0] * ratio)),
            max(1, round(MIN_INTERNAL[1] * ratio)))


def grid_fits(window_w, window_h, s):
    """Does grid ``s`` leave a buffer the interface can be laid out in?"""
    buffer_w, buffer_h = internal_size(window_w, window_h, s)
    need_w, need_h = min_internal(s)
    return buffer_w >= need_w and buffer_h >= need_h


def clamp_window(size, desktop=None):
    """A window size the interface can actually be drawn in, and the screen hold."""
    w, h = max(MIN_WINDOW[0], int(size[0])), max(MIN_WINDOW[1], int(size[1]))
    if desktop is not None:
        w, h = min(w, max(1, desktop[0])), min(h, max(1, desktop[1]))
    return (w, h)


def preset_size(desktop, fraction):
    """Window size for one of ``WINDOW_FRACTIONS`` on this desktop."""
    return clamp_window(
        (round(desktop[0] * fraction),
         desktop[1] - MENU_BAR_POINTS - TITLE_BAR_POINTS),
        desktop)


class DisplayContext:
    """The window plus the internal surface everything draws into.

    Two surfaces, or one.  At the default grid the window *is* the drawing
    surface -- one internal pixel per logical point, nothing copied, which is
    the common case and costs nothing.  At a coarser grid we draw into a
    smaller surface of our own and blow it up by exactly ``s`` on the way to
    the screen.  Exactly ``s``, never "whatever fills the window": a fractional
    factor is the one thing that smears the pixel grid the grid exists to
    preserve, so up to ``s - 1`` points along each edge are left black rather
    than stretched into.
    """

    def __init__(self, s=DEFAULT_S, fullscreen=False, size=None,
                 fraction_index=DEFAULT_FRACTION_INDEX):
        self.fullscreen = fullscreen
        self.desktop = self._desktop_size()
        self.s = s
        self.fraction_index = fraction_index % len(WINDOW_FRACTIONS)
        self.surface = None
        self.display = None
        self._scaled = None
        self.origin = (0, 0)
        if size is None:
            size = preset_size(self.desktop, WINDOW_FRACTIONS[self.fraction_index][0])
        self.window_size = clamp_window(size, self.desktop)
        self._open()

    @staticmethod
    def _desktop_size():
        sizes = pygame.display.get_desktop_sizes()
        return sizes[0] if sizes else (1920, 1080)

    def _flags(self):
        flags = pygame.RESIZABLE
        if self.fullscreen:
            flags |= pygame.FULLSCREEN
        return flags

    def _open(self):
        """Build the window, then derive the buffer from the size we got.

        The second pass is what keeps the upscale an exact integer: a fullscreen
        window is shorter than the desktop by the menu-bar strip, and a window
        manager is free to hand back a size other than the one asked for.
        """
        wanted = (0, 0) if self.fullscreen else self.window_size
        previous = self.display
        try:
            self.display = pygame.display.set_mode(wanted, self._flags(), vsync=1)
            if not self.fullscreen and pygame.display.get_window_size() != wanted:
                # On the way out of fullscreen the first call can come back
                # still the size of the whole screen; a second one lands.
                # Measured on the dummy driver, and one spare call on any other.
                self.display = pygame.display.set_mode(wanted, self._flags(), vsync=1)
        except pygame.error:
            # A driver that will not build this mode must not take the whole
            # app down mid-flight; keep the display we already had.
            if previous is None:
                raise
            self.display = previous
        self._adopt()

    @property
    def usable_grids(self):
        """The pixel grids this window is big enough for, finest first.

        A grid is a divisor: at ``s = 3`` the same window holds a third as many
        internal pixels each way, and past a point the transport bar no longer
        reaches across the bottom of it.  So the grids on offer are a property
        of the window, not a fixed list -- a third-of-the-screen window has
        fewer of them than a fullscreen one.  The finest is always offered,
        because it is the one that asks least of the window.
        """
        window = getattr(self, "window", None) or self.window_size
        fits = tuple(s for s in GRID_CHOICES if grid_fits(window[0], window[1], s))
        return fits or (GRID_CHOICES[0],)

    def _adopt(self):
        """Re-derive the internal buffer from the window that is actually open."""
        self.window = pygame.display.get_window_size()
        if not self.fullscreen:
            self.window_size = self.window
        # Dragging a window small enough steps the grid back down rather than
        # letting the interface lay itself out off its own edges.
        if self.s not in self.usable_grids:
            self.s = self.usable_grids[-1]
        self.width, self.height = internal_size(self.window[0], self.window[1], self.s)
        if self.s == 1:
            self.surface = self.display
            self._scaled = None
            self.origin = (0, 0)
        else:
            self.surface = pygame.Surface((self.width, self.height)).convert()
            self._scaled = pygame.Surface((self.width * self.s,
                                           self.height * self.s)).convert()
            self.origin = ((self.window[0] - self._scaled.get_width()) // 2,
                           (self.window[1] - self._scaled.get_height()) // 2)

    # --- the frame's last two lines ---------------------------------------------

    def present(self):
        """Put the internal surface on the screen.  Replaces ``display.flip``."""
        if self.surface is not self.display:
            self.display.fill((0, 0, 0))
            pygame.transform.scale(self.surface, self._scaled.get_size(), self._scaled)
            self.display.blit(self._scaled, self.origin)
        pygame.display.flip()

    # --- window points to internal pixels ----------------------------------------

    @property
    def _identity(self):
        return self.s == 1 and self.origin == (0, 0)

    def to_internal(self, pos):
        """A window point in internal pixels.

        Deliberately not clamped: a click in the letterbox belongs to nothing,
        and clamping it would slide it onto whichever control sits at the edge.
        """
        if self._identity:
            return (int(pos[0]), int(pos[1]))
        return ((int(pos[0]) - self.origin[0]) // self.s,
                (int(pos[1]) - self.origin[1]) // self.s)

    def map_event(self, event):
        """The same event with its mouse coordinates in internal pixels.

        SDL's ``SCALED`` used to do this for us.  Every reader of ``event.pos``
        -- the interface, all three tools, the effects -- works in internal
        pixels, so the translation happens once, here, rather than at each of
        the two dozen places that would otherwise have to remember it.
        """
        if self._identity or not hasattr(event, "pos"):
            return event
        fields = dict(event.__dict__)
        fields["pos"] = self.to_internal(event.pos)
        if "rel" in fields:
            fields["rel"] = (fields["rel"][0] // self.s, fields["rel"][1] // self.s)
        return pygame.event.Event(event.type, fields)

    def mouse_pos(self):
        """Where the pointer is, in internal pixels."""
        return self.to_internal(pygame.mouse.get_pos())

    @property
    def resizable(self):
        """True while the window has a drag handle and a title bar of its own."""
        return bool(self._flags() & pygame.RESIZABLE) and not self.fullscreen

    @property
    def upscale(self):
        """Window points per internal pixel.  Exactly integer, by construction."""
        return float(self.s)

    @property
    def font_px(self):
        return font_px(self.s)

    # --- changing the window -----------------------------------------------------

    def set_grid(self, s):
        """Change the pixel grid live.  Callers must clear any sprite cache.

        No new window: the grid only decides how big a buffer we draw into and
        by how much it is blown up, and the window it lands in is unaffected.
        A grid this window is too small for is refused, not honoured -- see
        ``usable_grids``.
        """
        self.s = s if s in self.usable_grids else self.usable_grids[-1]
        self._adopt()

    def resize(self, size=None):
        """Take a new window size from the window manager.

        Below ``MIN_WINDOW`` the drag hits a floor and the window is pushed back
        out to it, because the interface has a smallest size it can be laid out
        in and a window under it would hide its own controls.  There is no
        ceiling: see below.
        """
        if self.fullscreen:
            self._adopt()
            return
        current = pygame.display.get_window_size()
        # Floored, but deliberately not capped at ``self.desktop``: that is the
        # *primary* display, and a window dragged onto a bigger external monitor
        # is entitled to be bigger than the laptop screen.  Only the presets,
        # which are computed as fractions of a screen, are capped to one.
        wanted = clamp_window(size if size is not None else current)
        self.window_size = wanted
        if wanted != tuple(current):
            self._open()
        else:
            self._adopt()

    def set_window_fraction(self, index):
        """Snap the window to one of the presets.  Leaves fullscreen behind.

        Size only: the window stays where it is.  Moving it would need
        ``pygame._sdl2.video.Window``, and that is not an option -- see the
        module docstring.
        """
        self.fraction_index = index % len(WINDOW_FRACTIONS)
        self.fullscreen = False
        self.window_size = preset_size(
            self.desktop, WINDOW_FRACTIONS[self.fraction_index][0])
        self._open()

    def cycle_window_size(self):
        """Step to the next preset.  Returns the label of the one now in force."""
        self.set_window_fraction(self.fraction_index + 1)
        return self.window_size_label

    @property
    def window_size_label(self):
        return WINDOW_FRACTIONS[self.fraction_index][1]

    def toggle_fullscreen(self):
        """Fullscreen and back.  Windowed returns to the size it left at."""
        self.fullscreen = not self.fullscreen
        self._open()


def open_display(s=DEFAULT_S, fullscreen=False, size=None):
    pygame.display.init()
    pygame.display.set_caption("Solar System Sandbox")
    return DisplayContext(s, fullscreen, size)
