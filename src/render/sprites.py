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

from .. import units
from . import palette

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

#: Below this a body's own thermal emission is far too faint to see against the
#: light it reflects; by GLOW_FULL_K there is nothing else left to see.  Real
#: numbers: iron starts to visibly glow around 800 K and is white by 3000 K.
GLOW_MIN_K = 800.0
GLOW_FULL_K = 3000.0
#: Temperature is quantised to this many kelvin before it reaches the cache
#: key.  A body cooling continuously would otherwise miss the cache on every
#: frame, which is precisely what the cache exists to prevent.
GLOW_TEMP_STEP = 100.0

# --- black holes --------------------------------------------------------------
#
# Everything below is a multiple of the drawn *shadow* radius -- the dark disc
# an observer actually sees, which light bending magnifies to sqrt(27)/2 horizon
# radii.  The horizon itself is never drawn: it is 2.6x smaller and, at any zoom
# a child uses, well under a pixel.

#: How far the accretion disk reaches, in shadow radii.
BH_DISK_OUTER = 2.6
#: Vertical squash of the disk: near-edge-on, which is the view that makes a
#: black hole legible instead of a plain ring.
BH_DISK_SQUASH = 0.22
#: The disk's far side, lensed up over the top of the shadow and down under it.
#: Much rounder than the disk itself, which is what lifts it clear.
BH_LENS_ARC = 1.6
BH_LENS_SQUASH = 0.62
#: Jet length at full spin, in shadow radii.
BH_JET_LENGTH = 4.0
#: Brightness ratio between the approaching and receding halves of the disk --
#: relativistic beaming.  About right for gas orbiting at a few tenths of c.
BH_DOPPLER = 2.5
#: Concentric ellipses the disk is built from.
BH_DISK_RINGS = 7
#: Below this many pixels there is no room for a disk; draw shadow and rim only.
BH_DETAIL_PX = 5

#: Where the disk's inner edge is drawn, in shadow radii, at zero spin and at
#: the Thorne limit.
#:
#: The true ISCO/shadow ratio runs 1.155 down to 0.238 over that range -- but a
#: black hole draws at 6 px, so the whole span is under one pixel, and past
#: a* = 0.87 the true inner edge passes *inside* the shadow and is occluded
#: outright.  Widened to a legible range, exactly the trade ``scale._floor``
#: already makes for radii: the ordering and the direction stay honest, and the
#: dark band between the photon ring and the disk really does close as spin
#: rises.
BH_INNER_MAX = 2.00
BH_INNER_MIN = 1.05

#: Disk colours, inner to outer: white, blue-white, orange, deep red.
_DISK_RAMP = (
    (0xFF, 0xFF, 0xF4),
    (0xCF, 0xE4, 0xFF),
    (0xFF, 0xD2, 0x8A),
    (0xE8, 0x82, 0x28),
    (0x9A, 0x30, 0x10),
)
#: The photon ring: the sharpest edge in the picture.
_PHOTON_RING = (255, 250, 235)
#: Jets are synchrotron light -- blue-white, and dimmer than the disk.
_JET_COLOUR = (170, 205, 255)


def _inner_disk_px(px, spin):
    """Drawn inner edge of the accretion disk, in pixels.  See BH_INNER_MAX."""
    t = min(abs(float(spin)), units.MAX_SPIN) / units.MAX_SPIN
    return px * (BH_INNER_MAX + (BH_INNER_MIN - BH_INNER_MAX) * t)


def _ramp_colour(t):
    """A point along the disk's inner-to-outer colour ramp."""
    t = 0.0 if t < 0.0 else (1.0 if t > 1.0 else t)
    span = len(_DISK_RAMP) - 1
    k = min(int(t * span), span - 1)
    local = t * span - k
    a, b = _DISK_RAMP[k], _DISK_RAMP[k + 1]
    return tuple(int(a[i] + (b[i] - a[i]) * local) for i in range(3))


def _draw_ellipse_ring(surface, colour, alpha, centre, rx, ry, width):
    """One concentric ellipse of the disk, centred on ``centre``."""
    rx, ry = max(1, int(rx)), max(1, int(ry))
    rect = pygame.Rect(centre[0] - rx, centre[1] - ry, rx * 2, ry * 2)
    pygame.draw.ellipse(surface, tuple(colour) + (alpha,), rect,
                        max(1, int(width)))


def _make_black_hole(px, spin, disk, seed):
    """A black hole, painted back to front.

    Blitted **normally, never with BLEND_ADD**.  A star's sprite is additive
    because a star only ever lightens what is behind it; a black hole's whole
    point is an opaque hole in the sky, and BLEND_ADD cannot draw black.  The
    outer falloff is therefore carried in alpha rather than baked into colour,
    which is the opposite of the arrangement the star path warns against.
    """
    px = max(1, int(px))
    spin = min(max(float(spin), 0.0), units.MAX_SPIN)
    disk = min(max(float(disk), 0.0), 1.0)

    detailed = px >= BH_DETAIL_PX
    outer = int(px * BH_DISK_OUTER) + 2
    jet = int(px * BH_JET_LENGTH * spin) if (detailed and disk > 0.0) else 0
    half = max(outer, jet)
    size = half * 2 + 1
    centre = (half, half)
    surface = pygame.Surface((size, size), pygame.SRCALPHA)

    if detailed and disk > 0.0:
        # 1. Jets, first, so everything else covers their roots.
        if jet > 0:
            width = max(1, int(px * 0.28))
            alpha = int(150 * disk)
            for sign in (-1, 1):
                tip = centre[1] + sign * jet
                pygame.draw.polygon(
                    surface, _JET_COLOUR + (alpha,),
                    [(centre[0] - width, centre[1]),
                     (centre[0] + width, centre[1]),
                     (centre[0] + max(1, width // 3), tip),
                     (centre[0] - max(1, width // 3), tip)])

        inner = _inner_disk_px(px, spin)
        reach = px * BH_DISK_OUTER * (0.6 + 0.4 * disk)
        approaching_left = bool(int(seed) & 1)

        # 2. The far side of the disk, bent up over the shadow and down under
        #    it.  Rounder than the disk, which is what lifts it clear.
        arc_rx = px * BH_LENS_ARC
        _draw_ellipse_ring(surface, _ramp_colour(0.35), int(190 * disk),
                           centre, arc_rx, arc_rx * BH_LENS_SQUASH,
                           max(1, px // 5))

        # 3. The disk itself, inner to outer, with 4. Doppler beaming applied
        #    by redrawing the approaching half brighter.
        for k in range(BH_DISK_RINGS):
            frac = k / max(1, BH_DISK_RINGS - 1)
            rx = inner + (reach - inner) * frac
            if rx <= 0:
                continue
            ry = rx * BH_DISK_SQUASH
            base = _ramp_colour(frac)
            # A fast-spinning hole radiates 32% of what it eats against a
            # still one's 5.7%, so the inner rings brighten with spin too.
            gain = 1.0 + 0.6 * (spin / units.MAX_SPIN) * (1.0 - frac)
            alpha = int(min(255, 230 * disk * (1.0 - 0.45 * frac)))
            dim = tuple(min(255, int(c * gain / BH_DOPPLER)) for c in base)
            bright = tuple(min(255, int(c * gain)) for c in base)
            width = max(1, int(px * 0.22))

            _draw_ellipse_ring(surface, dim, alpha, centre, rx, ry, width)
            clip = pygame.Rect(0 if approaching_left else centre[0], 0,
                               centre[0] + 1, size)
            previous = surface.get_clip()
            surface.set_clip(clip)
            _draw_ellipse_ring(surface, bright, alpha, centre, rx, ry, width)
            surface.set_clip(previous)

    # 5. The shadow, opaque, over everything drawn so far -- which is what
    #    makes the disk's far side emerge from behind it.
    pygame.draw.circle(surface, (0, 0, 0, 255), centre, px)

    # 6. The photon ring, right at the shadow's edge.
    pygame.draw.circle(surface, _PHOTON_RING + (255,), centre, px,
                       1 if detailed else 0)
    if not detailed:
        # Too small for a one-pixel outline to survive; draw the rim as a
        # filled disc and punch the shadow back out of it.
        pygame.draw.circle(surface, _PHOTON_RING + (255,), centre, px + 1)
        pygame.draw.circle(surface, (0, 0, 0, 255), centre, px)
    return surface


# --- spacecraft ---------------------------------------------------------------
#
# Not authored art and not a glyph: the same rule render/widgets.py follows for
# its icons.  There is no font with a shuttle in it, and a missing glyph would
# render as a box.

#: Below this many pixels there is no room for a wing; draw a bright dot.
CRAFT_DETAIL_PX = 7
#: However far out the camera is, a craft is drawn at least this big, so the
#: silhouette stays readable.  A render-only floor: it never reaches
#: ``contact_radius``, so enlarging the picture does not enlarge the target.
CRAFT_MIN_PX = 9

#: White upper surface, black thermal tile below, grey trim.
_CRAFT_HULL = (0xF2, 0xF5, 0xFA)
_CRAFT_TILE = (0x22, 0x26, 0x30)
_CRAFT_TRIM = (0xB8, 0xBE, 0xC8)
_CRAFT_GLASS = (0x6C, 0xC4, 0xE8)


def _make_craft(px, heading):
    """A shuttle orbiter, nose along ``heading``.

    Drawn nose-along-+x from polygons and then rotated, which is both cheaper
    and steadier than re-deriving every vertex per angle: the outline is
    authored once in units of the body length and scaled.

    Blitted **normally, never with BLEND_ADD**.  Additive blending cannot draw
    the black underside, and the tile pattern is most of what makes the
    silhouette read as a shuttle rather than as a paper dart -- the same
    reasoning the black hole path follows, not the star's.
    """
    px = max(1, int(px))
    length = px * 2
    half = length
    size = half * 2 + 1
    centre = half
    surface = pygame.Surface((size, size), pygame.SRCALPHA)

    if px < CRAFT_DETAIL_PX:
        pygame.draw.circle(surface, _CRAFT_HULL, (centre, centre),
                           max(1, px // 2))
        return surface

    def point(along, across):
        """Body coordinates in fractions of the half-length, to pixels."""
        return (int(centre + along * px), int(centre - across * px))

    # Delta wing first, so the fuselage sits on top of it.
    pygame.draw.polygon(surface, _CRAFT_TRIM, [
        point(0.30, 0.0), point(-0.70, 1.05), point(-0.95, 1.05),
        point(-0.95, -1.05), point(-0.70, -1.05),
    ])
    # Vertical stabiliser, seen from the side the flat model gives us.
    pygame.draw.polygon(surface, _CRAFT_TRIM, [
        point(-0.62, 0.0), point(-0.95, 0.62), point(-1.00, 0.60),
        point(-0.86, 0.0),
    ])
    # Fuselage: blunt nose, straight body.
    pygame.draw.polygon(surface, _CRAFT_HULL, [
        point(1.00, 0.10), point(0.86, 0.26), point(-0.86, 0.30),
        point(-0.95, 0.0), point(-0.86, -0.30), point(0.86, -0.26),
        point(1.00, -0.10),
    ])
    # Thermal tile along the underside, and the flight deck windows.
    pygame.draw.polygon(surface, _CRAFT_TILE, [
        point(1.00, -0.10), point(0.86, -0.26), point(-0.86, -0.30),
        point(-0.95, 0.0), point(-0.90, -0.02), point(0.90, -0.02),
    ])
    if px >= CRAFT_DETAIL_PX + 4:
        pygame.draw.line(surface, _CRAFT_GLASS,
                         point(0.80, 0.16), point(0.62, 0.20), 1)

    if heading:
        surface = pygame.transform.rotate(surface, math.degrees(heading))
    return surface


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


def glow_fraction(kelvin):
    """How far a body has gone from its own colours to its blackbody glow, 0..1."""
    if kelvin <= GLOW_MIN_K:
        return 0.0
    return min(1.0, (kelvin - GLOW_MIN_K) / (GLOW_FULL_K - GLOW_MIN_K))


def glow_key(kelvin):
    """The quantised temperature the cache is keyed on.  Zero means no glow, so
    every ordinary body shares one key however its temperature drifts."""
    if glow_fraction(kelvin) <= 0.0:
        return 0
    return int(round(float(kelvin) / GLOW_TEMP_STEP))


def _blend_toward(tone, target, weight):
    return tuple(int(round(tone[k] + (target[k] - tone[k]) * weight))
                 for k in range(3))


def _make_body(material, seed, px, lit_angle, is_star, glow_k=0.0):
    px = max(1, int(px))
    dark, mid, light = material.tones

    if glow_k > 0.0:
        # Its own light, mixed into the light it reflects.  Applied before the
        # star branch as well as after it, so a red dwarf really is red and a
        # blue giant really is blue -- the same Planck locus, the same helper.
        target = palette.blackbody_rgb(glow_k)
        weight = glow_fraction(glow_k)
        dark = _blend_toward(dark, target, weight)
        mid = _blend_toward(mid, target, weight)
        light = _blend_toward(light, target, weight)

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

    def get(self, material, seed, px, lit_angle, is_star,
            temperature=units.CMB_TEMPERATURE_K):
        px = max(1, int(px))
        bucket = int(round(lit_angle / (2 * math.pi) * LIT_BUCKETS)) % LIT_BUCKETS
        heat = glow_key(temperature)
        key = (material.key, seed, px, bucket, bool(is_star), heat)
        hit = self._entries.get(key)
        if hit is not None:
            self._entries.move_to_end(key)
            return hit
        angle = bucket * 2 * math.pi / LIT_BUCKETS
        surface = _make_body(material, seed, px, angle, is_star,
                             heat * GLOW_TEMP_STEP)
        self._entries[key] = surface
        if len(self._entries) > self.limit:
            self._entries.popitem(last=False)
        return surface

    def get_black_hole(self, seed, px, spin, disk):
        """A black hole sprite, cached like any other.

        Spin and disk are bucketed for the same reason ``get`` buckets the
        lighting angle: a continuous value would miss the cache on every frame
        the child holds a stepper down, and regenerating the sprite every frame
        is exactly what the cache exists to avoid.
        """
        px = max(1, int(px))
        spin_bucket = int(round(min(abs(float(spin)), units.MAX_SPIN)
                                / units.MAX_SPIN * 8))
        disk_bucket = int(round(min(max(float(disk), 0.0), 1.0) * 8))
        key = ("bh", int(seed) & 1, px, spin_bucket, disk_bucket)
        hit = self._entries.get(key)
        if hit is not None:
            self._entries.move_to_end(key)
            return hit
        surface = _make_black_hole(px, spin_bucket / 8.0 * units.MAX_SPIN,
                                   disk_bucket / 8.0, seed)
        self._entries[key] = surface
        if len(self._entries) > self.limit:
            self._entries.popitem(last=False)
        return surface

    def get_craft(self, px, heading):
        """A shuttle sprite, cached like any other.

        Heading is bucketed into the same 16 compass points the lighting angle
        uses, and for exactly the same reason: a craft turning continuously
        along its orbit would miss the cache on every single frame, which is
        precisely what the cache exists to prevent.
        """
        px = max(1, int(px))
        bucket = int(round(heading / (2 * math.pi) * LIT_BUCKETS)) % LIT_BUCKETS
        key = ("craft", px, bucket)
        hit = self._entries.get(key)
        if hit is not None:
            self._entries.move_to_end(key)
            return hit
        surface = _make_craft(px, bucket * 2 * math.pi / LIT_BUCKETS)
        self._entries[key] = surface
        if len(self._entries) > self.limit:
            self._entries.popitem(last=False)
        return surface
