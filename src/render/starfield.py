"""The background.

Pre-rendered once per buffer size into ``sky`` -- ``App._build_render_stack``
makes a new one whenever the buffer changes shape, so nothing here caches a
size across frames -- and blitted up to four times a frame at the wrapped
parallax offset.  That replaces three hundred ``set_at`` calls with four
blits, and lets the sky hold things a per-pixel loop never could: stars of
several brightnesses and colours, and a faint band of Milky Way.

Real stars are so far away they would not parallax at all across a solar
system, but a very small factor gives a depth cue when panning without
implying they are nearby.  Positions come from a fixed seed so the sky does
not reshuffle itself every time the window is resized.

Stars are laid out across exactly one screen and the offset wraps modulo the
screen, so the sky tiles seamlessly and every star is always on screen.
"""

import numpy as np
import pygame

from . import palette, shade

PARALLAX = 0.02
#: Stars per 640x360 of buffer, and the floor and cap on the total.
STARS_PER_REF = 300
MIN_STARS = 100
MAX_STARS = 3000
#: The haze is rendered at this fraction of the buffer and scaled up: at
#: 3456x2168 a full-resolution fbm is a quarter of a second per resize.
HAZE_SCALE = 4
HAZE_BRIGHTNESS = 22


def star_count(width, height):
    """How many stars a buffer of this size gets."""
    n = int(round(STARS_PER_REF * width * height / (640.0 * 360.0)))
    return max(MIN_STARS, min(MAX_STARS, n))


class Starfield:
    def __init__(self, width, height, count=None, seed=7):
        self.width = max(1, int(width))
        self.height = max(1, int(height))
        self.count = star_count(self.width, self.height) if count is None else int(count)
        self.seed = seed
        self.sky = self._render(np.random.default_rng(seed))

    def _render(self, rng):
        w, h = self.width, self.height
        sky = pygame.Surface((w, h))
        sky.fill(palette.SPACE)

        # The Milky Way: a diagonal band of low-frequency noise.
        hw, hh = max(2, w // HAZE_SCALE), max(2, h // HAZE_SCALE)
        xs, ys = np.meshgrid(np.linspace(0, 1, hw), np.linspace(0, 1, hh))
        angle = rng.uniform(0.4, 1.1)
        across = (xs - 0.5) * np.sin(angle) - (ys - 0.5) * np.cos(angle)
        band = np.exp(-(across / 0.18) ** 2)
        noise = shade.fbm(xs, ys, self.seed + 3, octaves=3, cells=3)
        haze = band * (0.4 + 0.6 * noise) * HAZE_BRIGHTNESS
        rgb = np.empty((hh, hw, 3), dtype=np.float64)
        rgb[..., 0] = palette.SPACE[0] + haze * 1.0
        rgb[..., 1] = palette.SPACE[1] + haze * 0.95
        rgb[..., 2] = palette.SPACE[2] + haze * 0.9
        small = pygame.image.frombuffer(
            np.clip(rgb, 0, 255).astype(np.uint8).tobytes(), (hw, hh), "RGB").copy()
        sky.blit(pygame.transform.smoothscale(small, (w, h)), (0, 0))

        # Stars: a power law of brightness, blackbody colours half-desaturated.
        xy = rng.random((self.count, 2)) * (w, h)
        u = rng.random(self.count)
        brightness = 40 + 215 * u ** 3
        kelvin = np.exp(rng.uniform(np.log(3000.0), np.log(15000.0), self.count))
        for (x, y), value, temp in zip(xy, brightness, kelvin):
            r, g, b = palette.blackbody_rgb(float(temp))
            grey = (r + g + b) / 3.0
            r, g, b = (grey + (r - grey) * 0.5), (grey + (g - grey) * 0.5), (grey + (b - grey) * 0.5)
            k = float(value) / 255.0
            colour = (int(r * k), int(g * k), int(b * k))
            xi, yi = int(x) % w, int(y) % h
            if value > 215:
                radius = 1 if value < 240 else 2
                halo = tuple(int(c * 0.3) for c in colour)
                pygame.draw.aacircle(sky, halo, (xi, yi), radius + 1)
                pygame.draw.aacircle(sky, colour, (xi, yi), radius)
            else:
                sky.set_at((xi, yi), colour)
        return sky

    def draw(self, surface, camera):
        shift = camera.centre * camera.zoom * PARALLAX
        ox = int(shift[0]) % self.width
        oy = int(-shift[1]) % self.height
        w, h = self.width, self.height
        for dx in (-ox, w - ox):
            for dy in (-oy, h - oy):
                if dx < surface.get_width() and dy < surface.get_height() \
                        and dx + w > 0 and dy + h > 0:
                    surface.blit(self.sky, (dx, dy))
