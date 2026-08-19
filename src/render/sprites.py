"""Procedurally drawn pixel bodies.

Nothing here is authored art.  The child creates bodies at arbitrary radii out of
arbitrary materials, so there is no sprite sheet that could cover the space --
every disc is generated on demand and cached.

Shading is three offset circles rather than per-pixel lighting: a dark base, a
mid-tone circle pushed toward the light, and a small light-tone highlight pushed
further.  Three draw calls instead of tens of thousands of Python-level pixel
operations, which matters because a zoom can request a 300px body.

The lighting direction is bucketed to 16 compass points.  A continuous angle
would miss the cache on every frame the camera moves, which would mean
regenerating every sprite, every frame.
"""

import math
from collections import OrderedDict

import numpy as np
import pygame

#: Lighting directions the cache distinguishes.
LIT_BUCKETS = 16

#: How far the mid-tone and highlight circles shift toward the light, as a
#: fraction of the radius.
MID_OFFSET = 0.22
HIGHLIGHT_OFFSET = 0.50
HIGHLIGHT_SCALE = 0.62

#: Below this many pixels there is no room for texture; draw a flat dot.
DETAIL_THRESHOLD = 4
#: Glow radius as a multiple of a star's disc radius.
GLOW_SCALE = 2.2
#: Peak brightness of the halo just outside the disc, as a fraction of the
#: star's own colour.  Additive, so this is how much it lifts the background.
GLOW_STRENGTH = 0.55


def _shade(tone, factor):
    return tuple(max(0, min(255, int(c * factor))) for c in tone)


def _draw_detail(surface, material, rng, px, centre):
    """Style-specific surface texture, drawn inside the disc."""
    dark, mid, light = material.tones
    style = material.style

    if style == "cratered":
        for _ in range(int(rng.integers(3, 9))):
            angle = float(rng.random()) * 2 * math.pi
            distance = float(rng.random()) ** 0.5 * px * 0.78
            radius = max(1, int(px * float(rng.uniform(0.08, 0.20))))
            x = int(centre + math.cos(angle) * distance)
            y = int(centre + math.sin(angle) * distance)
            pygame.draw.circle(surface, _shade(dark, 1.15), (x, y), radius)

    elif style == "banded":
        y = -px
        while y < px:
            thickness = max(1, int(px * float(rng.uniform(0.10, 0.22))))
            half = int(math.sqrt(max(0.0, px * px - y * y)))
            tone = _shade(mid, float(rng.uniform(0.78, 1.22)))
            pygame.draw.rect(surface, tone,
                             (centre - half, centre + y, 2 * half, thickness))
            y += thickness + max(1, int(px * float(rng.uniform(0.04, 0.12))))

    elif style == "cracked":
        for _ in range(int(rng.integers(2, 6))):
            angle = float(rng.random()) * 2 * math.pi
            length = px * float(rng.uniform(0.5, 1.4))
            x0 = centre + math.cos(angle) * px * 0.6
            y0 = centre + math.sin(angle) * px * 0.6
            x1 = x0 - math.cos(angle + 0.7) * length
            y1 = y0 - math.sin(angle + 0.7) * length
            pygame.draw.line(surface, light, (int(x0), int(y0)), (int(x1), int(y1)), 1)

    elif style == "turbulent":
        for _ in range(int(rng.integers(6, 14))):
            angle = float(rng.random()) * 2 * math.pi
            distance = float(rng.random()) ** 0.5 * px * 0.8
            radius = max(1, int(px * float(rng.uniform(0.10, 0.26))))
            x = int(centre + math.cos(angle) * distance)
            y = int(centre + math.sin(angle) * distance)
            tone = light if rng.random() < 0.5 else _shade(mid, 0.85)
            pygame.draw.circle(surface, tone, (x, y), radius)


def _mask_to_circle(surface, px, centre):
    """Clip everything to the disc, so no detail can spill past the silhouette."""
    mask = pygame.Surface(surface.get_size(), pygame.SRCALPHA)
    mask.fill((0, 0, 0, 0))
    pygame.draw.circle(mask, (255, 255, 255, 255), (centre, centre), px)
    surface.blit(mask, (0, 0), special_flags=pygame.BLEND_RGBA_MULT)


def _make_body(material, seed, px, lit_angle, is_star):
    px = max(1, int(px))
    dark, mid, light = material.tones

    if is_star:
        size = int(px * GLOW_SCALE) * 2 + 1
        centre = size // 2
        surface = pygame.Surface((size, size), pygame.SRCALPHA)

        # The glow is blitted with BLEND_ADD, and **BLEND_ADD ignores alpha** --
        # it adds raw channel values.  So the falloff has to live in the colour
        # itself; encoding it as alpha produces a solid white disc that swallows
        # everything within a couple of AU of the star.
        steps = max(6, px)
        for k in range(steps, 0, -1):
            fraction = k / steps
            radius = int(px + (px * (GLOW_SCALE - 1.0)) * fraction)
            falloff = (1.0 - fraction) ** 2.2 * GLOW_STRENGTH
            pygame.draw.circle(surface, _shade(light, falloff) + (255,),
                               (centre, centre), radius)

        pygame.draw.circle(surface, tuple(light) + (255,), (centre, centre), px)
        pygame.draw.circle(surface, (255, 255, 255, 255), (centre, centre),
                           max(1, int(px * 0.72)))
        return surface

    size = px * 2 + 1
    centre = px
    surface = pygame.Surface((size, size), pygame.SRCALPHA)

    pygame.draw.circle(surface, dark, (centre, centre), px)
    if px >= 2:
        dx = math.cos(lit_angle) * px * MID_OFFSET
        dy = -math.sin(lit_angle) * px * MID_OFFSET
        pygame.draw.circle(surface, mid, (int(centre + dx), int(centre + dy)), px)

    if px >= DETAIL_THRESHOLD:
        rng = np.random.default_rng(seed if seed else 1)
        _draw_detail(surface, material, rng, px, centre)
        hx = math.cos(lit_angle) * px * HIGHLIGHT_OFFSET
        hy = -math.sin(lit_angle) * px * HIGHLIGHT_OFFSET
        pygame.draw.circle(surface, light,
                           (int(centre + hx), int(centre + hy)),
                           max(1, int(px * (1.0 - HIGHLIGHT_SCALE))))

    _mask_to_circle(surface, px, centre)
    return surface


def ring_surface(px, tones):
    """A flat ring system, drawn behind and in front of the planet.  Saturn only."""
    dark, mid, light = tones
    outer = max(3, int(px * 2.3))
    size = outer * 2 + 1
    centre = outer
    surface = pygame.Surface((size, size), pygame.SRCALPHA)
    for radius, tone in ((outer, light), (max(2, int(px * 1.95)), mid),
                         (max(1, int(px * 1.6)), dark)):
        rect = pygame.Rect(centre - radius, centre - max(1, int(radius * 0.30)),
                           radius * 2, max(2, int(radius * 0.60)))
        pygame.draw.ellipse(surface, tone, rect, max(1, px // 6))
    return surface


class SpriteCache:
    """LRU of rendered bodies, keyed by everything that changes the pixels."""

    def __init__(self, limit=512):
        self.limit = limit
        self._entries = OrderedDict()

    def __len__(self):
        return len(self._entries)

    def clear(self):
        self._entries.clear()

    def get(self, material, seed, px, lit_angle, is_star):
        px = max(1, int(px))
        bucket = int(round(lit_angle / (2 * math.pi) * LIT_BUCKETS)) % LIT_BUCKETS
        key = (material.key, seed, px, bucket, bool(is_star))
        hit = self._entries.get(key)
        if hit is not None:
            self._entries.move_to_end(key)
            return hit
        angle = bucket * 2 * math.pi / LIT_BUCKETS
        surface = _make_body(material, seed, px, angle, is_star)
        self._entries[key] = surface
        if len(self._entries) > self.limit:
            self._entries.popitem(last=False)
        return surface
