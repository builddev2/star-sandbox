"""Text rendering with the antialiasing off.

Antialiased glyphs blur against a pixel grid and break the look the moment SDL
scales the buffer up.  Rendering hard-edged into the internal buffer and letting
the upscale enlarge whole pixels is what makes the text read as pixel art.

The interface is deliberately narrow -- height, width, draw -- so a real bitmap
font can replace the implementation later without touching a single caller.
"""

import pygame


class PixelFont:
    def __init__(self, px):
        if not pygame.font.get_init():
            pygame.font.init()
        self.px = px
        self._font = pygame.font.Font(None, px)
        self.height = self._font.get_height()
        self._cache = {}

    def width(self, text):
        if not text:
            return 0
        return self._font.size(text)[0]

    def render(self, text, colour):
        """Cached, non-antialiased surface for one string in one colour."""
        key = (text, colour)
        surface = self._cache.get(key)
        if surface is None:
            surface = self._font.render(text, False, colour)
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
