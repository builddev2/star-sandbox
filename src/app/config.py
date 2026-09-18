"""Display setup and the constants that depend on the pixel density.

Measured on a 16" MacBook Pro (macOS 26, pygame-ce 2.5.8, SDL 2.32, cocoa
driver) on 2026-09-18::

    panel                                   3456 x 2234
    get_desktop_sizes()                     1728 x 1117   logical points
    Window((1152, 720), allow_high_dpi)     1152 x 720    points
    renderer.coordinates_from_window(...)   2304 x 1440   device pixels
    fullscreen                              1728 x 1084   points, 3456 x 2168 pixels

**Only the renderer sees device pixels.**  A window's *surface* is always in
points, on every SDL2 platform, so drawing into ``Window.get_surface()`` could
never be sharper than one pixel per point.  The frame is therefore drawn into
our own surface at ``window * s`` and uploaded into a streaming ``Texture``
each frame (3.4 ms at 2304x1440; measured 7.8 ms for a whole realistic frame).
The original pygame (not the -ce fork) cannot set ``SDL_WINDOW_ALLOW_HIGHDPI``
at all -- the string does not occur in its source -- which is why the
dependency is pygame-ce.

**``s`` is internal pixels per point**, ``native`` is what the display has.
It used to mean the opposite -- points per pixel, the coarse "pixel grid" of
the pixel-art era.  ``G`` now toggles between native and 1: half the pixels is
the escape hatch when the physics budget is saturated at fullscreen.

**Text is sized in points** because points are the unit the platform is
calibrated in; ``font_px`` multiplies by ``s``.

**A fullscreen window is not the desktop size.**  The menu-bar/notch strip
costs 33 points of height, so the buffer is derived from the window we
actually got, after the fact, in ``_adopt``.

**Nothing here may call ``pygame.Window.from_display_module()``.**  The object
it hands back destroys the real SDL window when it is garbage-collected, and
the program segfaults some frames later inside ``pygame.event.get``, a long
way from the call that did it.  This context creates the one ``Window`` for
the process itself, keeps it for its whole life, and destroys it in ``close``.
"""

import os

import pygame
from pygame._sdl2.video import Renderer, Texture

# When the buffer is drawn at standard density onto a Retina window the GPU
# scales it up; this asks SDL for linear filtering rather than nearest, so
# the picture is soft rather than blocky.  Read when the texture is created.
os.environ.setdefault("SDL_RENDER_SCALE_QUALITY", "1")

#: Text height in points at one pixel per point -- the font's point size.
#: Sized for a child reading alongside an adult on a laptop screen, not for
#: maximum information density.  (Atkinson Hyperlegible at 11 renders a 14 px
#: line, a shade taller and narrower than the built-in face used to be.)
TARGET_TEXT_POINTS = 11

#: Height macOS keeps for itself above a window, and for the window's own title
#: bar.  A preset that ignores them puts the transport bar under the Dock.
MENU_BAR_POINTS = 37
TITLE_BAR_POINTS = 28

#: The smallest window the whole interface lays itself out inside, in points.
#: ``tests/test_layout.py`` is what keeps these two numbers honest: it builds
#: every panel of every tool at exactly this size (at every pixel density) and
#: fails if a single control or panel lands off the edge.
#:
#: The width has to stay below a third of a laptop desktop (1728 / 3 = 576) or
#: the one-third preset would clamp straight back up and stop being one third.
#: The height is set by the help panel, which is the tallest thing here.
MIN_INTERNAL = (480, 440)
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

WINDOW_TITLE = "Solar System Sandbox"


def internal_size(window_w, window_h, s):
    """Internal buffer size for a window in points at ``s`` pixels per point."""
    return (max(1, int(window_w) * s), max(1, int(window_h) * s))


def font_px(s):
    """Font size in internal pixels: the point size times the pixel density,
    so the text is the same physical size at every density."""
    return TARGET_TEXT_POINTS * max(1, int(s))


def min_internal(s):
    """Smallest buffer the interface lays out in, at ``s`` pixels per point."""
    return (MIN_INTERNAL[0] * max(1, int(s)), MIN_INTERNAL[1] * max(1, int(s)))


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
    """The window, the renderer that reaches its pixels, and the one software
    surface everything draws into.

    ``surface`` is a plain ``pygame.Surface`` sized ``window * s``; every
    frame ``present`` uploads it into a streaming ``Texture`` and the renderer
    draws that over the whole window.  At native density the texture is the
    window's pixel size exactly, so nothing is scaled; at standard density
    (``s = 1`` on a Retina display) the GPU scales it up with linear filtering.

    ``s`` is **internal pixels per point**.  ``native`` is what the display
    actually has -- measured from the renderer, never guessed -- and the only
    values ever offered are ``native`` and 1.
    """

    def __init__(self, s=None, fullscreen=False, size=None,
                 fraction_index=DEFAULT_FRACTION_INDEX, pixels_per_point=None):
        self.fullscreen = fullscreen
        self.desktop = self._desktop_size()
        self.fraction_index = fraction_index % len(WINDOW_FRACTIONS)
        #: Tests pass this to stand in for a display the dummy driver lacks.
        self._pixels_per_point = pixels_per_point
        self._window = None
        self.renderer = None
        self.texture = None
        self.surface = None
        self.native = 1
        self.s = s
        if size is None:
            size = preset_size(self.desktop, WINDOW_FRACTIONS[self.fraction_index][0])
        self.window_size = clamp_window(size, self.desktop)
        self.window = self.window_size
        self._open()

    @staticmethod
    def _desktop_size():
        sizes = pygame.display.get_desktop_sizes()
        return sizes[0] if sizes else (1920, 1080)

    def _open(self):
        """Build the window once, then put it in the state asked for and derive
        the buffer from the size we actually got."""
        if self._window is None:
            self._window = pygame.Window(WINDOW_TITLE, self.window_size,
                                         resizable=True, allow_high_dpi=True)
            try:
                self.renderer = Renderer(self._window, vsync=True)
            except pygame.error:
                # The dummy driver, and any other that cannot promise vsync.
                self.renderer = Renderer(self._window, vsync=False)
        if self.fullscreen:
            self._window.set_fullscreen(True)
        else:
            self._window.set_windowed()
            self._window.size = self.window_size
        pygame.event.pump()
        self._adopt()

    def _measure_native(self):
        """Pixels per point, from the renderer.  1 on the dummy driver."""
        if self._pixels_per_point is not None:
            return max(1, int(self._pixels_per_point))
        w, h = self.window
        dx, dy = self.renderer.coordinates_from_window((w, h))
        return max(1, int(round(float(dx) / max(1, w))))

    @property
    def usable_grids(self):
        """The densities on offer, native first: the display's own, and 1."""
        return (self.native, 1) if self.native > 1 else (1,)

    def _adopt(self):
        """Re-derive the buffer from the window that is actually open."""
        self.window = tuple(int(v) for v in self._window.size)
        if not self.fullscreen:
            self.window_size = self.window
        self.native = self._measure_native()
        if self.s not in self.usable_grids:
            self.s = self.usable_grids[0]
        self.width, self.height = internal_size(self.window[0], self.window[1], self.s)
        if self.surface is None or self.surface.get_size() != (self.width, self.height):
            self.surface = pygame.Surface((self.width, self.height))
            self.texture = Texture(self.renderer, (self.width, self.height),
                                   streaming=True)

    # --- the frame's last line ---------------------------------------------------

    def present(self):
        """Put the internal surface on the screen.  Replaces ``display.flip``."""
        self.texture.update(self.surface)
        self.renderer.clear()
        self.texture.draw()
        self.renderer.present()

    # --- window points to internal pixels ----------------------------------------

    @property
    def _identity(self):
        return self.s == 1

    def to_internal(self, pos):
        """A window point in internal pixels."""
        if self._identity:
            return (int(pos[0]), int(pos[1]))
        return (int(pos[0]) * self.s, int(pos[1]) * self.s)

    def map_event(self, event):
        """The same event with its mouse coordinates in internal pixels.

        Every reader of ``event.pos`` -- the interface, all the tools, the
        effects -- works in internal pixels, so the translation happens once,
        here, rather than at each of the two dozen places that would otherwise
        have to remember it.
        """
        if self._identity or not hasattr(event, "pos"):
            return event
        fields = dict(event.__dict__)
        fields["pos"] = self.to_internal(event.pos)
        if "rel" in fields:
            fields["rel"] = (int(fields["rel"][0]) * self.s,
                             int(fields["rel"][1]) * self.s)
        return pygame.event.Event(event.type, fields)

    def mouse_pos(self):
        """Where the pointer is, in internal pixels."""
        return self.to_internal(pygame.mouse.get_pos())

    @property
    def resizable(self):
        """True while the window has a drag handle and a title bar of its own."""
        return not self.fullscreen

    @property
    def upscale(self):
        """Window points per internal pixel."""
        return 1.0 / self.s

    @property
    def font_px(self):
        return font_px(self.s)

    # --- changing the window -----------------------------------------------------

    def size_changed(self):
        """Has the window been resized behind our back?  One cheap call a frame."""
        return tuple(int(v) for v in self._window.size) != self.window

    def set_grid(self, s):
        """Change the pixel density live.  Callers must clear any sprite cache.

        Only the display's native density and 1 are ever on offer; anything
        else is refused and lands on native.
        """
        self.s = s if s in self.usable_grids else self.usable_grids[0]
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
        current = tuple(int(v) for v in self._window.size)
        # Floored, but deliberately not capped at ``self.desktop``: that is the
        # *primary* display, and a window dragged onto a bigger external monitor
        # is entitled to be bigger than the laptop screen.  Only the presets,
        # which are computed as fractions of a screen, are capped to one.
        wanted = clamp_window(size if size is not None else current)
        self.window_size = wanted
        if wanted != current:
            self._window.size = wanted
            pygame.event.pump()
        self._adopt()

    def set_window_fraction(self, index):
        """Snap the window to one of the presets.  Leaves fullscreen behind."""
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

    def close(self):
        """Drop the texture and renderer, then destroy the window.  Idempotent."""
        self.texture = None
        self.renderer = None
        if self._window is not None:
            self._window.destroy()
            self._window = None


def open_display(s=None, fullscreen=False, size=None):
    pygame.display.init()
    return DisplayContext(s, fullscreen, size)
