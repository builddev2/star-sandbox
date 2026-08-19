"""The background.

Real stars are so far away they would not parallax at all across a solar system,
but a very small factor gives a depth cue when panning without implying they are
nearby.  Positions come from a fixed seed so the sky does not reshuffle itself
every time the window is resized.

Stars are laid out across exactly one screen and the parallax offset wraps
modulo the screen, so the sky tiles seamlessly and every one of ``count`` stars
is always on screen.  Scattering them over a larger area and cropping would mean
asking for 300 and drawing 33.
"""

import numpy as np

PARALLAX = 0.02


class Starfield:
    def __init__(self, width, height, count=300, seed=7):
        rng = np.random.default_rng(seed)
        self.width = max(1, width)
        self.height = max(1, height)
        self._xy = rng.random((count, 2)) * (self.width, self.height)
        self._brightness = rng.integers(40, 190, count).astype(int)

    def draw(self, surface, camera):
        shift = camera.centre * camera.zoom * PARALLAX
        xs = (self._xy[:, 0] - shift[0]) % self.width
        ys = (self._xy[:, 1] + shift[1]) % self.height
        set_at = surface.set_at
        for x, y, value in zip(xs.astype(int), ys.astype(int), self._brightness):
            value = int(value)
            set_at((int(x), int(y)), (value, value, min(255, value + value // 10)))
