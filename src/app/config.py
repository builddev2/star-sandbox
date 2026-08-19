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
"""

import pygame

#: Finest grid: one internal pixel per logical point, i.e. a 2x2 block of panel
#: pixels on a Retina display.
DEFAULT_S = 1
GRID_CHOICES = (1, 2, 3)

#: Target text height in logical points.
TARGET_TEXT_POINTS = 14
#: Below this a scalable font stops being legible whatever the grid.
MIN_FONT_PX = 8


def internal_size(display_w, display_h, s):
    """Internal buffer size for a display and pixel-grid divisor."""
    return (max(1, display_w // s), max(1, display_h // s))


def font_px(s):
    """Font height in internal pixels, so apparent size stays roughly constant."""
    return max(MIN_FONT_PX, round(TARGET_TEXT_POINTS / max(s, 1)))


class DisplayContext:
    """The window plus the internal surface everything draws into."""

    def __init__(self, s=DEFAULT_S, fullscreen=True):
        self.fullscreen = fullscreen
        self.desktop = self._desktop_size()
        self.s = s
        self.surface = None
        self._open()

    @staticmethod
    def _desktop_size():
        sizes = pygame.display.get_desktop_sizes()
        return sizes[0] if sizes else (1920, 1080)

    def _flags(self):
        flags = pygame.SCALED
        if self.fullscreen:
            flags |= pygame.FULLSCREEN
        return flags

    def _open(self):
        """Open the window, then re-derive the buffer from the size we got.

        The second pass is what keeps the upscale an exact integer: a fullscreen
        window is shorter than the desktop by the menu-bar strip, and sizing the
        buffer to the desktop would leave SDL scaling by 0.97.
        """
        flags = self._flags()
        wanted = internal_size(self.desktop[0], self.desktop[1], self.s)
        self.surface = pygame.display.set_mode(wanted, flags, vsync=1)

        actual = pygame.display.get_window_size()
        corrected = internal_size(actual[0], actual[1], self.s)
        if corrected != wanted:
            self.surface = pygame.display.set_mode(corrected, flags, vsync=1)

        self.width, self.height = self.surface.get_size()
        self.window = pygame.display.get_window_size()

    @property
    def upscale(self):
        """Window points per internal pixel.  Should be an exact integer."""
        return self.window[0] / self.width if self.width else 1.0

    @property
    def font_px(self):
        return font_px(self.s)

    def set_grid(self, s):
        """Change the pixel grid live.  Callers must clear any sprite cache."""
        self.s = s
        self._open()

    def toggle_fullscreen(self):
        self.fullscreen = not self.fullscreen
        self._open()


def open_display(s=DEFAULT_S, fullscreen=True):
    pygame.display.init()
    pygame.display.set_caption("Solar System Sandbox")
    return DisplayContext(s, fullscreen)
