"""Text rendering.

Named for the pixel-art era, when antialiasing was deliberately off; the
class keeps its name because forty callers know it and the interface --
``height``, ``width``, ``render``, ``draw`` -- is all any of them use, which is
exactly what let the face and the antialiasing change without touching one.

The face is Atkinson Hyperlegible (OFL, ``assets/fonts``), chosen for a child
reading numbers at a glance.  If the file is missing the built-in face stands
in, scaled so the line height matches: pygame sizes its built-in face by an
undocumented 0.6875 and it is a shorter face besides, so a request for 11
points becomes 21 to land on the same 14 px line (measured: 11 -> 14 px
bundled, 21 -> 13 px built-in).
"""

import pathlib

import pygame

FONT_PATH = (pathlib.Path(__file__).resolve().parents[2]
             / "assets" / "fonts" / "AtkinsonHyperlegible-Regular.ttf")
#: The built-in face needs this many times the point size to match the
#: bundled face's line height.
DEFAULT_FONT_RATIO = 1.9


class PixelFont:
    def __init__(self, px, path=FONT_PATH):
        if not pygame.font.get_init():
            pygame.font.init()
        self.px = px
        try:
            self._font = pygame.font.Font(str(path), px)
        except (OSError, pygame.error, FileNotFoundError):
            self._font = pygame.font.Font(None, max(1, round(px * DEFAULT_FONT_RATIO)))
        self.height = self._font.get_height()
        self._cache = {}

    def width(self, text):
        if not text:
            return 0
        return self._font.size(text)[0]

    def render(self, text, colour):
        """Cached, antialiased surface for one string in one colour."""
        key = (text, colour)
        surface = self._cache.get(key)
        if surface is None:
            surface = self._font.render(text, True, colour)
            if len(self._cache) > 512:
                self._cache.clear()
            self._cache[key] = surface
        return surface

    def draw(self, target, x, y, text, colour):
        """Blit ``text`` at ``(x, y)``.  Returns the horizontal advance."""
        if not text:
            return 0
        surface = self.render(text, colour)
        target.blit(surface, (x, y))
        return surface.get_width()
